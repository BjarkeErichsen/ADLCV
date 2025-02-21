import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets
import tqdm
from transformer import TransformerClassifier, to_device

NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512
RESULTS_FILE = "results/model_results2.txt"

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data_iter(sampled_ratio=0.2, batch_size=16):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
    print("Dataset size:", len(tdata))

    reduced_tdata, _ = tdata.split(split_ratio=sampled_ratio)
    train, test = reduced_tdata.split(split_ratio=0.8)
    print('training:', len(train), 'test:', len(test))
    TEXT.build_vocab(train, max_size=VOCAB_SIZE - 2)
    LABEL.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits((train, test), 
                                                       batch_size=batch_size, 
                                                       device=to_device())
    return train_iter, test_iter

def train_and_evaluate(embed_dim, num_heads, num_layers, num_epochs=3, 
                       pos_enc='fixed', pool='max', dropout=0.0, fc_dim=None,
                       batch_size=16, lr=1e-4, warmup_steps=625, 
                       weight_decay=1e-4, gradient_clipping=1):

    train_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO, batch_size=batch_size)
    
    model = TransformerClassifier(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                                  pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim,
                                  max_seq_len=MAX_SEQ_LEN, num_tokens=VOCAB_SIZE, num_classes=NUM_CLS)
    
    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))
    loss_function = nn.CrossEntropyLoss()

    epoch_accuracies = []

    # Training loop
    for e in range(num_epochs):
        print(f'\nEpoch {e + 1}/{num_epochs}')
        model.train()
        for batch in tqdm.tqdm(train_iter, desc=f"Training Epoch {e + 1}"):
            opt.zero_grad()
            input_seq = batch.text[0]
            batch_size, seq_len = input_seq.size()
            label = batch.label - 1
            if seq_len > MAX_SEQ_LEN:
                input_seq = input_seq[:, :MAX_SEQ_LEN]
            out = model(input_seq)
            loss = loss_function(out, label)  
            loss.backward()  
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

        # Evaluation after each epoch
        with torch.no_grad():
            model.eval()
            tot, cor = 0.0, 0.0
            for batch in test_iter:
                input_seq = batch.text[0]
                batch_size, seq_len = input_seq.size()
                label = batch.label - 1
                if seq_len > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                out = model(input_seq).argmax(dim=1)
                tot += float(input_seq.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            epoch_accuracies.append(acc)
            print(f'-- Validation Accuracy after Epoch {e + 1}: {acc:.3f}')
    
    return epoch_accuracies

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")

    # Define different configurations for 10 models
    model_configs = [
        {"embed_dim": 128, "num_heads": 4, "num_layers": 4},
        {"embed_dim": 256, "num_heads": 4, "num_layers": 4},
        {"embed_dim": 128, "num_heads": 8, "num_layers": 4},
        {"embed_dim": 128, "num_heads": 4, "num_layers": 6},
        {"embed_dim": 256, "num_heads": 8, "num_layers": 4},
        {"embed_dim": 256, "num_heads": 4, "num_layers": 6},
        {"embed_dim": 128, "num_heads": 8, "num_layers": 6},
        {"embed_dim": 256, "num_heads": 8, "num_layers": 6},
        {"embed_dim": 512, "num_heads": 8, "num_layers": 4},
        {"embed_dim": 512, "num_heads": 8, "num_layers": 6},
    ]
    
    num_runs = 2  # Number of runs per configuration for computing confidence intervals
    
    with open(RESULTS_FILE, "w") as f:
        for idx, config in enumerate(model_configs):
            run_final_accuracies = []
            for run in range(num_runs):
                print(f"\nTraining Model {idx + 1} Run {run + 1}/{num_runs}: {config}")
                # Set a different seed for each run to vary initialization and data shuffling
                set_seed(seed=run)
                accuracies = train_and_evaluate(**config)
                # Record the final epoch's accuracy for this run
                run_final_accuracies.append(accuracies[-1])
            
            # Compute statistics
            mean_acc = np.mean(run_final_accuracies)
            std_acc = np.std(run_final_accuracies)
            # Compute a 95% confidence interval (using 1.96 as the z-value)
            ci_lower = mean_acc - 1.96 * (std_acc / np.sqrt(num_runs))
            ci_upper = mean_acc + 1.96 * (std_acc / np.sqrt(num_runs))
            
            f.write(f"Model {idx + 1}: {config}\n")
            f.write(f"  Final Accuracy over {num_runs} runs: Mean = {mean_acc:.3f}, Std = {std_acc:.3f}\n")
            f.write(f"  95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]\n\n")
            print(f"Model {idx + 1} - Mean Final Accuracy: {mean_acc:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    print(f"\nAll results saved to {RESULTS_FILE}")
