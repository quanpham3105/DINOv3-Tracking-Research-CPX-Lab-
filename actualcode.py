import torch
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

pipe = pipeline(
    task="image-feature-extraction",
    model="facebook/dinov3-vits16-pretrain-lvd1689m",
    dtype=torch.bfloat16,
)

# Save the output to a variable
temp = pipe("dog.jpg")

vec = temp[0]
size = int(np.ceil(np.sqrt(len(vec))))  # find nearest square
img = np.zeros((size*size,))
img[:len(vec)] = vec
img = img.reshape(size, size)

plt.imshow(img, cmap="viridis")
plt.title("DINOv3 Embedding Heatmap")
plt.axis("off")
plt.show()