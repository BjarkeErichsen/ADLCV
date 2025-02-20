import random
from datasets import load_dataset

# Dataset name
DATASET_NAME = "monurcan/andersen_fairy_tales"

# Load the dataset
dataset = load_dataset(DATASET_NAME)

# Extract stories from the training set
print(dataset.keys())
train = dataset['train']
stories = train['story']

# Sample some fairy tales
#save as txt file
with open('fairy_tales.txt', 'w') as f:
    for story in stories:
        f.write(story + '\n\n')
        
# Print the number of stories
print(f"Number of stories: {len(stories)}")
# Print the first story 
print(stories[0])