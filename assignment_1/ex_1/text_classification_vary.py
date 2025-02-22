import numpy as np
import os
import random
import torch
import json
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets
import tqdm
from transformer import TransformerClassifier, to_device

NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512
RESULTS_FILE = "results/all_results2.json"  


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

def save_results(config, accuracies, model_id):
    """Save all model results into a single JSON file."""
    results_data = {}

    # Load existing results to prevent overwriting
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                results_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("Warning: Couldn't read existing results file. Creating a new one.")
            results_data = {}

    # Append new results
    results_data[model_id] = {
        "config": config,
        "accuracies": accuracies
    }

    # Save back to JSON file
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_data, f, indent=4)



def train_and_evaluate(embed_dim, num_heads, num_layers, num_epochs=20, 
                       pos_enc='fixed', pool='max', dropout=0.0, fc_dim=None,
                       batch_size=16, lr=1e-4, warmup_steps=625, 
                       weight_decay=1e-4, gradient_clipping=1, model_id=""):

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
            label = batch.label - 1  # Assuming labels start from 1
            if input_seq.size(1) > MAX_SEQ_LEN:
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
                label = batch.label - 1
                if input_seq.size(1) > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                
                out = model(input_seq).argmax(dim=1)
                tot += float(input_seq.size(0))
                cor += float((label == out).sum().item())
            
            acc = cor / tot
            epoch_accuracies.append(acc)
            print(f'-- Validation Accuracy after Epoch {e + 1}: {acc:.3f}')

    # Save accuracy per epoch for later visualization
    save_results({"embed_dim": embed_dim, "num_heads": num_heads, "num_layers": num_layers}, epoch_accuracies, model_id)

    return epoch_accuracies

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")

    set_seed(seed=1) 

    # Define different configurations for 12 models
    model_configs = [
        # Embedding Dimension: 128
        {"embed_dim": 128, "num_heads": 4, "num_layers": 4},
        {"embed_dim": 128, "num_heads": 8, "num_layers": 4},
        {"embed_dim": 128, "num_heads": 4, "num_layers": 6},
        {"embed_dim": 128, "num_heads": 8, "num_layers": 6},

        # Embedding Dimension: 256
        {"embed_dim": 256, "num_heads": 4, "num_layers": 4},
        {"embed_dim": 256, "num_heads": 8, "num_layers": 4},
        {"embed_dim": 256, "num_heads": 4, "num_layers": 6},
        {"embed_dim": 256, "num_heads": 8, "num_layers": 6},

        # Embedding Dimension: 512
        {"embed_dim": 512, "num_heads": 4, "num_layers": 4},
        {"embed_dim": 512, "num_heads": 8, "num_layers": 4},
        {"embed_dim": 512, "num_heads": 4, "num_layers": 6},
        {"embed_dim": 512, "num_heads": 8, "num_layers": 6},
    ]

    
    num_runs = 5  # Number of runs per model

    for idx, config in enumerate(model_configs):
        for run in range(num_runs):
            print(f"\nTraining Model {idx + 1} Run {run + 1}/{num_runs}: {config}")
            
            set_seed(seed=run)  # Ensure different random states per run

            accuracies = train_and_evaluate(**config, model_id=f"{idx}_run{run}")

    print("\nTraining completed. Results saved in JSON files.")

