import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.model.baseline_backbone.siglip.siglip_inference import SiglipModel
from PIL import Image
import torch

model_name = 'google/siglip-so400m-patch14-384'
model = SiglipModel(model_name=model_name)
model.eval()

# Image + Text -> Text
img = Image.open('assets/example.jpg').convert('RGB')

qry_output = model.get_fused_embeddings(
    texts=['Represent the given image with the following question: What is in the image'],
    images=[img],
)

string = 'A cat and a dog'
tgt_output = model.get_fused_embeddings(texts=[string], images=[None])
sim = (qry_output * tgt_output).sum().item()
print(f'{string} = {sim:.4f}')

string = 'A cat and a tiger'
tgt_output = model.get_fused_embeddings(texts=[string], images=[None])
sim = (qry_output * tgt_output).sum().item()
print(f'{string} = {sim:.4f}')


# Batch processing
qry_output = model.get_fused_embeddings(
    texts=[
        'Represent the given image with the following question: What is in the image',
        'Represent the given image with the following question: What is in the image',
    ],
    images=[
        Image.open('assets/example.jpg').convert('RGB'),
        Image.open('assets/example.jpg').convert('RGB'),
    ],
)

tgt_output = model.get_fused_embeddings(
    texts=['A cat and a dog', 'A cat and a tiger'],
    images=[None, None],
)

sim_matrix = torch.matmul(qry_output, tgt_output.T)
print(sim_matrix)
