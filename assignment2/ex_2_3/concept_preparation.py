import os
from PIL import Image
import torchvision.transforms as transforms

# Define paths
raw_input_folder = "raw_baymax_concept"  # Change if needed
processed_folder = ["concept_folder"]  # concept folder for training

# Ensure processed folder exists
os.makedirs(processed_folder, exist_ok=True)

# Define transformations: Resize, Normalize, and Convert to Tensor
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to 512x512
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Process each image in the raw input folder
for image_name in os.listdir(raw_input_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        img_path = os.path.join(raw_input_folder, image_name)
        
        try:
            # Open and process the image
            image = Image.open(img_path).convert("RGB")  # Ensure RGB format
            processed_image = transform(image)  # Apply transformations
            processed_image = transforms.ToPILImage()(processed_image)  # Convert back to PIL Image

            # Save the processed image
            save_path = os.path.join(processed_folder, image_name)
            processed_image.save(save_path)

            print(f"Processed and saved: {save_path}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print(f"All images processed and saved to: {processed_folder}")
