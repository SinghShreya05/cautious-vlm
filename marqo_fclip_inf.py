import open_clip
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
# tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionSigLIP')
from torch.nn.functional import cosine_similarity

import torch
from PIL import Image

image_path_1 = 'image.png'
image_path_2 = 'image copy.png'
image_path_3 = 'image copy 2.png'
image_path_4 = 'image copy 3.png'

image_1 = preprocess_val(Image.open(image_path_1)).unsqueeze(0)
image_2 = preprocess_val(Image.open(image_path_2)).unsqueeze(0)
image_3 = preprocess_val(Image.open(image_path_3)).unsqueeze(0)
image_4 = preprocess_val(Image.open(image_path_4)).unsqueeze(0)
# text = tokenizer(["a hat", "a t-shirt", "shoes"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image1_features = model.encode_image(image_1)
    image1_features /= image1_features.norm(dim=-1, keepdim=True)

    image2_features = model.encode_image(image_2)
    image2_features /= image2_features.norm(dim=-1, keepdim=True)

    image3_features = model.encode_image(image_3)
    image3_features /= image3_features.norm(dim=-1, keepdim=True)

    image4_features = model.encode_image(image_4)
    image4_features /= image4_features.norm(dim=-1, keepdim=True)


# print(image_features)
# print()
# print(len(image_features))

# print("Label probs:", text_probs)
# Calculate cosine similarity between the two embeddings
similarity = cosine_similarity(image1_features, image2_features)
print(f"Cosine similarity: {similarity.item():.4f}")

similarity = cosine_similarity(image1_features, image3_features)
print(f"Cosine similarity: {similarity.item():.4f}")

similarity = cosine_similarity(image1_features, image4_features)
print(f"Cosine similarity: {similarity.item():.4f}")
