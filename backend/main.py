from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import requests
import io
import numpy as np
from opensearchpy import OpenSearch
import logging
from typing import List, Optional
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Default labels for zero-shot classification (can be customized)
DEFAULT_LABELS = [
    "diatom microscopic structure",
    "pollen particle contamination", 
    "tapetal cell defect",
    "crystalline defect",
    "surface contamination",
    "geometric irregularity",
    "Patterned_surface_defect",
    "Foreign_particle_contamination",
    "particle",
    "surface"
]

# Pydantic models
class ZeroShotRequest(BaseModel):
    labels: Optional[List[str]] = None  # If None, use DEFAULT_LABELS

app = FastAPI()

# OpenSearch client
os_client = OpenSearch(
    hosts=[{"host": "opensearch", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
)
INDEX_NAME = "sem-defects"

# Create index with knn mapping if it does not exist
embedding_dim = 512  # CLIP ViT base patch16 outputs 512-dim vectors
if not os_client.indices.exists(index=INDEX_NAME):
    index_body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim
                },
                "label": {"type": "keyword"}
            }
        }
    }
    os_client.indices.create(index=INDEX_NAME, body=index_body)

@app.post("/index-image/")
def index_image(file: UploadFile = File(...), label: str = Form("unknown")):
    logger.info(f"Received indexing request - filename: {file.filename}, label: {label}")
    logger.info(f"Indexing image with label: {label}")
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    logger.info(f"Image size: {image.size}")
    # Get embedding from CLIP model service
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    resp = requests.post("http://clip_model:5001/embed", files={"file": buf})
    embedding = resp.json()["embedding"]
    logger.info(f"Received embedding length: {len(embedding)}")
    # Robustly flatten embedding to 1D list
    embedding = np.array(embedding).flatten().tolist()
    logger.info(f"Flattened embedding length: {len(embedding)}")
    doc = {"embedding": embedding, "label": label}
    result = os_client.index(index=INDEX_NAME, body=doc)
    logger.info(f"Indexed document ID: {result['_id']}, result: {result['result']}")
    return {"status": "indexed", "doc_id": result['_id']}

@app.post("/classify-image/")
def classify_image(file: UploadFile = File(...)):
    """Few-shot classification using k-NN search on indexed images."""
    logger.info("Starting few-shot image classification (k-NN)")
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    logger.info(f"Image size: {image.size}")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    resp = requests.post("http://clip_model:5001/embed", files={"file": buf})
    embedding = resp.json()["embedding"]
    logger.info(f"Received embedding length: {len(embedding)}")
    # Robustly flatten embedding to 1D list
    embedding = np.array(embedding).flatten().tolist()
    logger.info(f"Flattened embedding length: {len(embedding)}")
    
    # Check index document count
    count = os_client.count(index=INDEX_NAME)
    logger.info(f"Total documents in index: {count['count']}")
    
    # Vector search in OpenSearch
    query = {
        "size": 1,
        "query": {
            "knn": {
                "embedding": {
                    "vector": embedding,
                    "k": 1
                }
            }
        }
    }
    logger.info(f"Executing k-NN search with k=1")
    res = os_client.search(index=INDEX_NAME, body=query)
    logger.info(f"Search returned {len(res['hits']['hits'])} results")
    
    if res["hits"]["hits"]:
        top_hit = res["hits"]["hits"][0]
        label = top_hit["_source"]["label"]
        score = top_hit["_score"]
        logger.info(f"Top match: label={label}, score={score}")
        return {"defect_type": label, "confidence_score": score, "mode": "few-shot"}
    
    logger.warning("No matches found in index")
    return {"defect_type": "unknown", "confidence_score": 0.0, "mode": "few-shot"}

@app.post("/classify-image-zeroshot/")
def classify_image_zeroshot(file: UploadFile = File(...), labels: Optional[str] = Form(None)):
    """
    Zero-shot classification using CLIP's text encoder.
    No training images required - classifies based on text label similarity.
    
    Args:
        file: Image file to classify
        labels: Comma-separated list of labels (e.g., "diatom,pollen,tapetal")
                If not provided, uses DEFAULT_LABELS
    """
    logger.info("Starting zero-shot image classification (text-image similarity)")
    
    # Parse labels
    if labels:
        label_list = [l.strip() for l in labels.split(",")]
        logger.info(f"Using custom labels: {label_list}")
    else:
        label_list = DEFAULT_LABELS
        logger.info(f"Using default labels: {label_list}")
    
    # Get image embedding
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    logger.info(f"Image size: {image.size}")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    resp = requests.post("http://clip_model:5001/embed", files={"file": buf})
    image_embedding = np.array(resp.json()["embedding"])
    logger.info(f"Image embedding length: {len(image_embedding)}")
    
    # Get text embeddings for all labels
    text_resp = requests.post(
        "http://clip_model:5001/embed-text",
        json={"texts": label_list}
    )
    text_embeddings = np.array(text_resp.json()["embeddings"])
    logger.info(f"Generated {len(text_embeddings)} text embeddings")
    
    # Normalize image embedding
    image_embedding = image_embedding / np.linalg.norm(image_embedding)
    
    # Compute cosine similarity (text embeddings are already normalized)
    similarities = np.dot(text_embeddings, image_embedding)
    logger.info(f"Similarities: {similarities}")
    
    # Get best match
    best_idx = np.argmax(similarities)
    best_label = label_list[best_idx]
    best_score = float(similarities[best_idx])
    
    logger.info(f"Best match: label={best_label}, score={best_score}")
    
    # Return all scores for transparency
    all_scores = {label: float(score) for label, score in zip(label_list, similarities)}
    
    return {
        "defect_type": best_label,
        "confidence_score": best_score,
        "mode": "zero-shot",
        "all_scores": all_scores
    }

@app.get("/index-stats/")
def get_index_stats():
    """Get statistics about the indexed training images."""
    try:
        # Get total document count
        count_result = os_client.count(index=INDEX_NAME)
        total_docs = count_result['count']
        
        # Get label distribution using aggregation
        agg_query = {
            "size": 0,
            "aggs": {
                "labels": {
                    "terms": {
                        "field": "label",
                        "size": 100  # Get up to 100 unique labels
                    }
                }
            }
        }
        
        agg_result = os_client.search(index=INDEX_NAME, body=agg_query)
        
        # Extract label distribution
        label_counts = {}
        if "aggregations" in agg_result and "labels" in agg_result["aggregations"]:
            for bucket in agg_result["aggregations"]["labels"]["buckets"]:
                label_counts[bucket["key"]] = bucket["doc_count"]
        
        return {
            "total_images": total_docs,
            "unique_labels": len(label_counts),
            "label_distribution": label_counts,
            "index_name": INDEX_NAME
        }
    except Exception as e:
        logger.error(f"Error fetching index stats: {e}")
        return {
            "total_images": 0,
            "unique_labels": 0,
            "label_distribution": {},
            "error": str(e)
        }

@app.post("/attention-map/")
def get_attention_map(file: UploadFile = File(...)):
    """
    Generate attention visualization showing which parts of the image
    CLIP focuses on during classification.
    
    Returns: PNG image with heat map overlay
    """
    logger.info(f"Received attention map request for file: {file.filename}")
    
    try:
        # Read the uploaded file
        image_bytes = file.file.read()
        
        # Forward to CLIP service
        files = {"file": (file.filename, io.BytesIO(image_bytes), "image/png")}
        resp = requests.post("http://clip_model:5001/attention-map", files=files)
        
        if resp.ok:
            logger.info("Successfully generated attention map")
            # Return the image directly
            return StreamingResponse(io.BytesIO(resp.content), media_type="image/png")
        else:
            logger.error(f"CLIP service error: {resp.text}")
            return {"error": "Failed to generate attention map"}
            
    except Exception as e:
        logger.error(f"Error generating attention map: {e}")
        return {"error": str(e)}
