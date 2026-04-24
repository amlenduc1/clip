from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import io
import numpy as np
import logging
from typing import List
from pydantic import BaseModel
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Pydantic models for request/response
class TextEmbedRequest(BaseModel):
    texts: List[str]

class TextEmbedResponse(BaseModel):
    embeddings: List[List[float]]

def get_image_embedding(image: Image.Image):
    logger.info(f"Processing image of size: {image.size}")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    logger.info(f"Pixel values shape: {pixel_values.shape}")
    with torch.no_grad():
        # Explicitly use vision model pooler_output -> project to 512-dim
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        logger.info(f"Vision pooler_output shape: {vision_outputs.pooler_output.shape}")
        image_features = model.visual_projection(vision_outputs.pooler_output)
        logger.info(f"Projected image_features shape: {image_features.shape}")
    embedding = image_features[0].cpu().numpy().flatten().tolist()
    # Ensure 512-dim vector
    if len(embedding) != 512:
        raise ValueError(f"CLIP embedding is not 512-dim, got {len(embedding)}")
    logger.info(f"Generated embedding of length: {len(embedding)}")
    return embedding

def get_text_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate 512-dim embeddings for text labels using CLIP's text encoder."""
    logger.info(f"Processing {len(texts)} text inputs: {texts}")
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        # Use text model pooler_output -> project to 512-dim
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logger.info(f"Text pooler_output shape: {text_outputs.pooler_output.shape}")
        text_features = model.text_projection(text_outputs.pooler_output)
        logger.info(f"Projected text_features shape: {text_features.shape}")
        
        # Normalize embeddings (CLIP uses cosine similarity)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    embeddings = text_features.cpu().numpy().tolist()
    logger.info(f"Generated {len(embeddings)} text embeddings, each of length: {len(embeddings[0])}")
    return embeddings

def get_attention_map(image: Image.Image):
    """
    Extract attention map from CLIP's Vision Transformer.
    Returns attention weights showing which image patches the model focuses on.
    """
    logger.info(f"Generating attention map for image of size: {image.size}")
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    
    # Forward pass with attention output
    with torch.no_grad():
        # Get vision encoder outputs with attentions
        vision_outputs = model.vision_model(
            pixel_values=pixel_values,
            output_attentions=True  # Request attention weights
        )
        
        # Extract attention weights from all layers
        # attentions is a tuple of (num_layers) tensors
        # Each tensor shape: [batch_size, num_heads, num_patches, num_patches]
        attentions = vision_outputs.attentions
        
        logger.info(f"Number of attention layers: {len(attentions)}")
        logger.info(f"Attention shape: {attentions[0].shape}")
        
        # Use attention from the last layer
        last_layer_attention = attentions[-1]  # Shape: [1, num_heads, num_patches, num_patches]
        
        # Average attention across all heads
        attention = last_layer_attention.mean(dim=1)  # Shape: [1, num_patches, num_patches]
        attention = attention[0]  # Shape: [num_patches, num_patches]
        
        # Get attention from CLS token (first token) to all other patches
        # This shows which patches CLS token attends to (i.e., important for classification)
        cls_attention = attention[0, 1:]  # Skip CLS token itself, shape: [num_patches-1]
        
        # ViT-B/16 with 224x224 input has 14x14 = 196 patches (+ 1 CLS token)
        num_patches = int(np.sqrt(cls_attention.shape[0]))
        logger.info(f"Number of patches per dimension: {num_patches}")
        
        # Reshape to 2D grid
        attention_map = cls_attention.reshape(num_patches, num_patches)
        attention_map = attention_map.cpu().numpy()
        
        # Normalize to [0, 1]
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        logger.info(f"Attention map shape: {attention_map.shape}")
        
        return attention_map

def create_attention_overlay(image: Image.Image, attention_map: np.ndarray, alpha=0.5):
    """
    Create a visualization with attention heat map overlaid on the original image.
    """
    # Resize image to 224x224 (CLIP's input size) for alignment
    img_resized = image.resize((224, 224))
    img_np = np.array(img_resized)
    
    # Resize attention map to match image size using interpolation
    attention_resized = cv2.resize(attention_map, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    # Create heat map using OpenCV's COLORMAP_JET
    # Convert to 0-255 range
    heatmap = (attention_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    
    # Convert back to PIL Image
    overlay_image = Image.fromarray(overlay)
    
    # Resize back to original image size
    overlay_image = overlay_image.resize(image.size, Image.BILINEAR)
    
    return overlay_image

@app.post("/embed")
def embed_image(file: UploadFile = File(...)):
    logger.info(f"Received embedding request for file: {file.filename}")
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    embedding = get_image_embedding(image)
    logger.info("Successfully generated embedding")
    return {"embedding": embedding}

@app.post("/embed-text", response_model=TextEmbedResponse)
def embed_text(request: TextEmbedRequest):
    """Generate embeddings for text labels (for zero-shot classification)."""
    logger.info(f"Received text embedding request for {len(request.texts)} texts")
    embeddings = get_text_embeddings(request.texts)
    logger.info("Successfully generated text embeddings")
    return TextEmbedResponse(embeddings=embeddings)

@app.post("/attention-map")
def generate_attention_map(file: UploadFile = File(...)):
    """
    Generate attention map visualization showing which parts of the image
    CLIP focuses on. Returns an image with heat map overlay.
    """
    logger.info(f"Received attention map request for file: {file.filename}")
    
    # Read image
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    
    # Extract attention map
    attention_map = get_attention_map(image)
    
    # Create overlay visualization
    overlay_image = create_attention_overlay(image, attention_map, alpha=0.4)
    
    # Convert to bytes for response
    buf = io.BytesIO()
    overlay_image.save(buf, format='PNG')
    buf.seek(0)
    
    logger.info("Successfully generated attention map visualization")
    
    return StreamingResponse(buf, media_type="image/png")
