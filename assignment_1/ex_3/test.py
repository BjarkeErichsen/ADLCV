import torch
from gpt import AndersenGPT
from train import (
    EMBED_DIM,
    MAX_SEQ_LEN,
    MODEL_SAVE_PATH,
    NUM_HEADS,
    NUM_LAYERS,
    POS_ENC,
    PRETRAINED_TOKENIZER,
)
from transformers import AutoTokenizer


import torch
import torch.nn.functional as F

import sys



# def generate_text(model, tokenizer, prompt, max_gen_len=500, device="cpu"):
#     """
#     Given a prompt string, generate a continuation using greedy decoding.
#     The prompt is encoded using the pretrained tokenizer.
#     """
#     # Encode the prompt (returns a list of token ids)
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

#     for _ in range(max_gen_len):
#         ####################### insert code here #####################################################
#         # Ensure we work with the last MAX_SEQ_LEN tokens if the sequence gets too long.
#         # If the sequence is longer than MAX_SEQ_LEN, keep only the last MAX_SEQ_LEN tokens.
#         # Check if input_ids is too long, if it is crop it
#         ...

#         # Forward pass: get logits for all tokens in the sequence.
#         logits = ...
        
#         # Get the logits for the last token only: shape [batch_size, vocab_size]
#         next_token_logits = ...

#         # You will implement two strategies for generating the next token:
#         strategy = "greedy"
#         if strategy == "greedy":
#             # Greedy: choose the token with highest probability.
#             next_token_id = ...
#         elif strategy == "sampling":
#             # Multinomial Sampling: Sample from the probability distribution.
#             # The temperature parameter controls the randomness of the sampling.
#             temperature = 0.8
#             probabilities = ... # softmax with temperature
#             next_token_id = ... # multinomial sampling

#         # Append predicted token to input_ids. Concatenate
#         input_ids = ...

#         # Stop early if the model generates the EOS token.
#         # Check if next_token_id == tokenizer.eos_token_id
#         # If next_token is end of sentence token, it should stop
#         ...
#         ################################################################################################

#     # Decode the full sequence to text.
#     output_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
#     return output_text

def generate_text(model, tokenizer, prompt, max_gen_len=500, device="cpu", max_seq_len=1024):
    """
    Given a prompt string, generate a continuation using greedy decoding or sampling.
    The prompt is encoded using the pretrained tokenizer.
    """
    # Encode the prompt (returns a list of token ids)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_gen_len):
        ####################### inserted code #####################################################
        # Ensure we work with the last MAX_SEQ_LEN tokens if the sequence gets too long.
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]  # Keep only last max_seq_len tokens

        # Forward pass: get logits for all tokens in the sequence.
        with torch.no_grad():
            logits = model(input_ids)  # Shape: [batch_size, seq_len, vocab_size]

        # Get the logits for the last token only: shape [batch_size, vocab_size]
        next_token_logits = logits[:, -1, :]  # Get last token's logits

        # You will implement two strategies for generating the next token:
        if len(sys.argv) > 1:
            if sys.argv[1] in ["greedy", "sampling"]:
                strategy = sys.argv[1]
            else:
                print("Invalid strategy. Choose 'greedy' or 'sampling'.")
                return

        if strategy == "sampling" and len(sys.argv) > 2:
            try:
                temperature = float(sys.argv[2])
                if temperature <= 0:
                    raise ValueError
            except ValueError:
                print("Invalid temperature value. Must be a positive float.")
                return
            
        if strategy == "greedy":
            # Greedy: choose the token with the highest probability.
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
        elif strategy == "sampling":
            # Multinomial Sampling: Sample from the probability distribution.
            # The temperature parameter controls the randomness of the sampling.
            probabilities = F.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

        # Append predicted token to input_ids.
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)

        # Stop early if the model generates the EOS token.
        if next_token_id.item() == tokenizer.eos_token_id:
            break
        ################################################################################################

    # Decode the full sequence to text.
    output_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    return output_text


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device} ...")

    # Load the same pretrained tokenizer used during training.
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER)

    # GPT2 does not have a PAD token by default; set it to the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Instantiate the GPT-style model with the same hyperparameters as during training.
    model = AndersenGPT(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        pos_enc=POS_ENC,
        dropout=0.0,
        fc_dim=None,
        num_tokens=tokenizer.vocab_size,
    ).to(device)

    # Load the model checkpoint.
    state_dict = torch.load(MODEL_SAVE_PATH + "/best.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.\n")

    print("Enter a prompt and the model will generate a continuation.")
    print("Type 'quit' or 'exit' to stop.\n")
    while True:
        prompt = input("Prompt: ").strip()  # Stripping is for tokenization weirdness
        if prompt.lower() in ["quit", "exit"]:
            break
        generated_text = generate_text(
            model, tokenizer, prompt, max_gen_len=500, device=device
        )
        print("\n--- Generated Text ---")
        print(generated_text)
        print("----------------------\n")


if __name__ == "__main__":
    main()
