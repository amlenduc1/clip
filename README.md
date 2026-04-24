# SEM Defect Image Classification using CLIP

A dual-mode defect classification system for Scanning Electron Microscope (SEM) images using OpenAI's CLIP (Contrastive Language-Image Pre-training) model, supporting both **zero-shot** (text-based) and **few-shot** (image similarity) classification approaches.

---

##  Problem Statement

### The Challenge in Modern Semiconductor Manufacturing

In today's advanced manufacturing landscape—particularly in semiconductor fabrication, MEMS devices, and nanomaterial production—the **Scanning Electron Microscope (SEM)** serves as the cornerstone imaging instrument for determining critical surface attributes including compositions, morphologies, and geometric defects at the nanometer scale. SEM technology overcomes the fundamental diffraction limits of optical microscopy, achieving magnifications up to 1,000,000× and resolutions below 1 nanometer, making it indispensable for quality control in processes where even atomic-level defects can cascade into catastrophic yield loss.

### Traditional CNN-Based Approach: Limitations at Scale

Historically, data scientists and process engineers have deployed **Convolutional Neural Networks (CNNs)** for Automated Defect Classification (ADC) systems. While CNNs have demonstrated success in controlled environments, they face critical limitations in real-world manufacturing:

1. **Data Hunger Problem**: CNNs require thousands of labeled examples per defect class. In semiconductor fabs, where new defect modes emerge with each process node shrink (e.g., 7nm → 5nm → 3nm), collecting sufficient labeled data for rare defects (e.g., stacking faults, micro-voids, or contamination particles) is prohibitively expensive and time-consuming.

2. **Catastrophic Forgetting**: When retraining CNNs to recognize new defect types (e.g., novel metallization defects in advanced packaging), they often degrade performance on previously learned classes, requiring complex continual learning strategies.

3. **Domain Shift Brittleness**: A CNN trained on defects from one tool vendor or process recipe often fails when deployed on slightly different equipment or materials, despite visual similarities apparent to human experts. This necessitates expensive domain adaptation or separate models per manufacturing line.

4. **Long Development Cycles**: Training a production-grade CNN model for defect classification typically requires 3-6 months of data collection, labeling, model development, and validation—by which time process parameters may have already evolved.

5. **Class Imbalance**: Critical defects are inherently rare (often <0.01% of inspected dies). CNNs struggle with this imbalance, leading to high false-negative rates for the very defects that matter most for yield.

### The CLIP Advantage: Dual-Mode Classification

This project demonstrates how **CLIP-based generative AI-powered ADC** overcomes these fundamental limitations through **two complementary approaches**:

#### Mode 1: Zero-Shot Classification (No Training Images Required)

CLIP's vision-language pre-training enables defect classification **without ANY labeled training images**:

- **How it works**: Uses CLIP's text encoder to generate embeddings for defect type descriptions (e.g., "pollen particle contamination"), then compares them with the query image embedding using cosine similarity
- **Ideal for**: New defect types, exploratory analysis, rapid prototyping
- **Advantages**: 
  - Add new defect categories instantly by just typing text labels
  - No need to collect, label, or index training images
  - Leverages CLIP's semantic understanding from 400M+ image-text pairs
- **Example**: Classify an image into "diatom", "pollen", or "tapetal" without showing the model any example images

#### Mode 2: Few-Shot Classification (Image Similarity Search)

Uses CLIP's vision encoder with k-NN similarity search on indexed reference images:

- **How it works**: Stores embeddings of labeled training images in OpenSearch vector DB, classifies new images by finding the most similar indexed image
- **Ideal for**: Production deployment, high-accuracy requirements, domain-specific defects
- **Advantages**:
  - Needs only 1-5 examples per defect class (vs. thousands for CNNs)
  - Higher accuracy when reference images closely match production data
  - Enables visual similarity search and defect clustering
- **Example**: Index 5 "stacking fault" images, then automatically classify new occurrences

#### Key Benefits Across Both Modes

- **Cross-Domain Generalization**: CLIP embeddings capture high-level semantic features (e.g., "crystalline structure," "surface contamination," "geometric irregularity") that generalize across different SEM instruments, imaging conditions, and material systems.

- **Rapid Deployment**: New defect classes can be onboarded in **hours** (zero-shot: type labels; few-shot: index reference images) rather than months of CNN training.

- **Unified Embedding Space**: All defect types—whether described by text or images—are mapped into a shared 512-dimensional vector space, enabling similarity-based retrieval, anomaly detection, and intelligent defect clustering for root-cause analysis.

This approach represents a paradigm shift from **discriminative learning** (CNN: "learn decision boundaries between known classes") to **metric learning** (CLIP: "measure semantic similarity in embedding space"), making ADC systems more agile, scalable, and aligned with the rapid innovation cycles of modern semiconductor manufacturing.

---

##  Dataset Sources

This project uses publicly available SEM image datasets for research and demonstration purposes:

- **Primary Dataset**: [3DSEM - A Dataset for 3D SEM Surface Reconstruction](https://b2share.eudat.eu/records/19cc2afd23e34b92b36a1dfd0113a89f)  
  *EUDAT B2SHARE Repository*

- **Alternative Dataset**: [3DSEM Dataset on Kaggle](https://www.kaggle.com/datasets/kmader/3dsem-a-dataset-for-3d-sem-surface-reconstruction)  
  *Kaggle - Accessible mirror of the EUDAT dataset*

**Dataset Characteristics**:
- Multiple defect categories including diatom structures, pollen particles, and tapetal cells
- High-resolution SEM images (2560×1920 pixels)
- Suitable for testing cross-domain generalization in microscopy image analysis

---

##  Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI (Port 8501)                                   │
│  - Multi-image upload & batch classification                │
│  - Mode selector: Zero-Shot vs Few-Shot                     │
│  - Custom label input for zero-shot                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Backend (Port 8000)                                │
│  - /classify-image-zeroshot/ → Text-image similarity        │
│  - /classify-image/ → k-NN search (few-shot)                │
│  - /index-image/ → Index training images                    │
└────────┬────────────────────────┬───────────────────────────┘
         │                        │
         ▼                        ▼
┌────────────────────┐   ┌───────────────────┐
│  CLIP Service      │   │  OpenSearch       │
│  (Port 5001)       │   │  Vector DB        │
│  ViT-B/16          │   │  (Port 9200)      │
├────────────────────┤   └───────────────────┘
│ Vision Encoder     │    k-NN search for
│ → 512-dim image    │    few-shot mode
│   embeddings       │
│                    │
│ Text Encoder       │
│ → 512-dim text     │
│   embeddings       │
└────────────────────┘
  Zero-shot & Few-shot
```

### Key Components

1. **CLIP Model Service** (`clip_model/`)
   - Model: `openai/clip-vit-base-patch16`
   - **Vision Encoder**: Generates 512-dimensional image embeddings using ViT + projection layer
   - **Text Encoder**: Generates 512-dimensional text embeddings for zero-shot classification
   - Endpoints:
     - `/embed`: Image → 512-dim vector
     - `/embed-text`: Text labels → 512-dim vectors (batch)

2. **FastAPI Backend** (`backend/`)
   - **Zero-Shot Mode** (`/classify-image-zeroshot/`):
     - Accepts image + optional text labels
     - Computes cosine similarity between image and all label embeddings
     - Returns best matching label with scores for all labels
     - **No training images required**
   - **Few-Shot Mode** (`/classify-image/`):
     - Performs k-NN similarity search (k=1) on indexed images
     - Returns label of nearest neighbor image
     - **Requires indexed training images**
   - **Indexing** (`/index-image/`):
     - Indexes training images with labels into OpenSearch
     - Used only for few-shot mode

3. **OpenSearch Vector Database**
   - Stores image embeddings as `knn_vector` type (dimension=512)
   - Enables fast approximate nearest neighbor search using HNSW algorithm
   - **Used only for few-shot mode**

4. **Streamlit Frontend** (`frontend/`)
   - **Mode Selection**: Toggle between Zero-Shot and Few-Shot
   - **Custom Labels**: In zero-shot mode, enter custom defect categories
   - **Batch Classification**: Upload and classify multiple images at once
   - **Results Visualization**: Shows confidence scores, all label similarities (zero-shot), defect types

---

##  Setup Instructions

### Prerequisites

- **Docker Desktop** installed and running
- **Windows 10/11** with WSL2 enabled (or Linux/macOS)
- At least **8GB RAM** available for Docker
- **Dataset** (optional): Required only for few-shot mode. Zero-shot mode works without any training images!

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd clip
   ```

2. **Prepare the dataset**:
   - Download dataset from [EUDAT](https://b2share.eudat.eu/records/19cc2afd23e34b92b36a1dfd0113a89f) or [Kaggle](https://www.kaggle.com/datasets/kmader/3dsem-a-dataset-for-3d-sem-surface-reconstruction)
   - Extract and organize images into folders by defect type:
     ```
     data/
     └── images/
         ├── diatom/
         │   ├── diatom-01.TIF
         │   └── ...
         ├── pollen/
         │   └── ...
         └── tapetal/
             └── ...
     ```

3. **Build and start all services**:
   ```powershell
   docker compose up -d --build
   ```

4. **Wait for services to initialize** (~30-60 seconds):
   ```powershell
   docker compose logs -f backend
   # Wait until you see: "Application startup complete"
   ```

5. **Verify OpenSearch index mapping**:
   ```powershell
   docker compose exec opensearch curl -s "localhost:9200/sem-defects/_mapping?pretty"
   ```
   Expected output:
   ```json
   {
     "sem-defects": {
       "mappings": {
         "properties": {
           "embedding": {
             "type": "knn_vector",
             "dimension": 512
           },
           "label": {
             "type": "keyword"
           }
         }
       }
     }
   }
   ```

6. **Index training images** (required only for few-shot mode):
   ```powershell
   docker compose exec backend python index_images.py
   ```
   Expected output:
   ```
   Found 17 images. Starting indexing...
   [OK] /app/data/images/diatom/diatom-01.TIF -> label: diatom
   [OK] /app/data/images/pollen/Pollen1001.jpg -> label: pollen
   ...
   Indexing complete. Success: 17, Failed: 0
   ```
   
   **Note**: Skip this step if you only want to use zero-shot mode.

7. **Access the application**:
   - **Streamlit UI**: http://localhost:8501
   - **Backend API**: http://localhost:8000/docs
   - **OpenSearch Dashboards**: http://localhost:5601

8. **Try it out**:
   - **Zero-Shot**: Select "Zero-Shot" mode in the UI, upload an image, classify (no training images needed!)
   - **Few-Shot**: Select "Few-Shot" mode (requires step 6 to be completed)

---

##  Usage

### Web Interface (Recommended)

1. Open http://localhost:8501 in your browser
2. **Select Classification Mode** in the sidebar:
   - **Zero-Shot**: No training images needed, classify using text labels
   - **Few-Shot**: Uses indexed training images for similarity search
3. **Enable Attention Visualization** (optional):
   - Check **"Show Attention Maps"** in the sidebar
   - This overlays a heat map showing which image regions CLIP focuses on
   - Red/yellow areas = high attention, blue areas = low attention
4. **Configure (Zero-Shot only)**:
   - Use default labels or enter custom labels (comma-separated)
   - Example: `diatom,pollen,tapetal,contamination,crystal defect`
5. **Upload Images**: Drag & drop one or more SEM images
6. Click **"Classify All Images"**
7. **View Results**:
   - Defect type and confidence score for each image
   - **Attention maps** (if enabled): Side-by-side view of original image and attention heat map
   - **Zero-shot mode**: Shows similarity scores for all labels
   - **Few-shot mode**: Shows nearest neighbor match score

### API Usage

**Zero-Shot Classification** (no training images required):
```bash
# With default labels
curl -X POST "http://localhost:8000/classify-image-zeroshot/" \
  -F "file=@/path/to/test-image.jpg"

# With custom labels
curl -X POST "http://localhost:8000/classify-image-zeroshot/" \
  -F "file=@/path/to/test-image.jpg" \
  -F "labels=diatom,pollen,tapetal,contamination"
```

Response:
```json
{
  "defect_type": "pollen particle contamination",
  "confidence_score": 0.8543,
  "mode": "zero-shot",
  "all_scores": {
    "diatom microscopic structure": 0.6234,
    "pollen particle contamination": 0.8543,
    "tapetal cell defect": 0.5891,
    "crystalline defect": 0.4523
  }
}
```

**Few-Shot Classification** (requires indexed training images):
```bash
curl -X POST "http://localhost:8000/classify-image/" \
  -F "file=@/path/to/test-image.jpg"
```

Response:
```json
{
  "defect_type": "pollen",
  "confidence_score": 0.9876,
  "mode": "few-shot"
}
```

**Index a new training image** (for few-shot mode):
```bash
curl -X POST "http://localhost:8000/index-image/" \
  -F "file=@/path/to/reference-image.jpg" \
  -F "label=new_defect_type"
```

### Adding Training Images for Few-Shot Mode

There are **three ways** to add training images to the index:

#### Method 1: Web UI (Recommended for Interactive Use)

1. Open http://localhost:8501
2. Scroll to **"Index Training Images (Few-Shot Mode)"** section
3. Click **"➕ Add Training Images to Index"** expander
4. Upload one or more images
5. Choose labeling method:
   - **Same label for all**: Enter one label for all images
   - **Individual labels**: Assign different labels to each image
6. Click **" Index Training Images"**
7. View indexing results and success/failure status

**Benefits**: Visual interface, batch upload, immediate feedback, preview thumbnails

#### Method 2: Bulk Indexing Script (Best for Large Datasets)

1. Organize images by category in the data folder:
   ```
   data/images/
   ├── diatom/
   │   ├── example1.jpg
   │   └── example2.jpg
   ├── pollen/
   │   └── ...
   └── new_defect_type/   ← Add new category folder
       ├── image1.jpg
       ├── image2.jpg
       └── image3.jpg
   ```

2. Run the indexing script:
   ```powershell
   docker compose exec backend python index_images.py
   ```

**Benefits**: Handles hundreds/thousands of images, automatic label extraction from folder names, resumable

#### Method 3: Direct API Call (For Automation/Scripts)

```bash
# Single image
curl -X POST "http://localhost:8000/index-image/" \
  -F "file=@/path/to/reference-image.jpg" \
  -F "label=defect_type"

# Batch script (PowerShell example)
Get-ChildItem "C:\defect_images\contamination\*.jpg" | ForEach-Object {
    curl -X POST "http://localhost:8000/index-image/" `
      -F "file=@$($_.FullName)" `
      -F "label=contamination"
}
```

**Benefits**: Automation, integration with CI/CD, programmatic control

### View Index Statistics

**Web UI**:
1. Open http://localhost:8501
2. Scroll to **" View Index Statistics"** section
3. Click **" Refresh Stats"**
4. View total images, unique labels, and distribution chart

**API**:
```bash
curl http://localhost:8000/index-stats/
```

Response:
```json
{
  "total_images": 25,
  "unique_labels": 4,
  "label_distribution": {
    "diatom": 8,
    "pollen": 10,
    "tapetal": 5,
    "contamination": 2
  }
}
```

---

## 🔧 Technical Details

### CLIP Dual-Encoder Architecture

CLIP uses **two separate encoders** that map different modalities (images and text) into a shared 512-dimensional embedding space:

#### Image Embedding (Vision Encoder)

```python
# Image preprocessing
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Vision encoder (ViT-B/16) → 768-dim pooler output
vision_outputs = model.vision_model(pixel_values)

# Project to CLIP's shared embedding space → 512-dim
image_features = model.visual_projection(vision_outputs.pooler_output)

# Normalize for cosine similarity
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
```

#### Text Embedding (Text Encoder)

```python
# Text preprocessing
inputs = processor(text=["diatom defect", "pollen contamination"], 
                   return_tensors="pt", padding=True)

# Text encoder (Transformer) → 512-dim pooler output
text_outputs = model.text_model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"]
)

# Project to CLIP's shared embedding space → 512-dim
text_features = model.text_projection(text_outputs.pooler_output)

# Normalize for cosine similarity
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
```

**Why 512 dimensions?**  
CLIP's vision-language alignment is learned in 512-dimensional space through contrastive learning on 400M+ image-text pairs. This dimensionality balances expressiveness (capturing complex visual and semantic concepts) with computational efficiency (fast similarity search).

### Zero-Shot Classification Implementation

**True zero-shot** - no training images required:

```python
# 1. Get text embeddings for all possible labels
labels = ["diatom", "pollen", "tapetal", "contamination"]
text_embeddings = get_text_embeddings(labels)  # Shape: (4, 512)

# 2. Get image embedding for query image
image_embedding = get_image_embedding(query_image)  # Shape: (512,)

# 3. Compute cosine similarity (embeddings are normalized)
similarities = text_embeddings @ image_embedding  # Dot product = cosine similarity

# 4. Get best match
best_idx = argmax(similarities)
predicted_label = labels[best_idx]
confidence = similarities[best_idx]
```

**Key insight**: CLIP was trained to maximize similarity between matching image-text pairs. We leverage this by treating defect type labels as "text descriptions" of what the image should contain.

### Few-Shot Classification (k-NN Search)

Classification via **1-nearest-neighbor search** in the embedding space:

```python
# Index phase (offline)
for image, label in training_images:
    embedding = get_image_embedding(image)
    opensearch.index({"embedding": embedding, "label": label})

# Query phase (online)
query_embedding = get_image_embedding(test_image)
result = opensearch.knn_search(vector=query_embedding, k=1)
predicted_label = result.hits[0].label
```

OpenSearch uses the **HNSW (Hierarchical Navigable Small World)** algorithm for approximate nearest neighbor search, achieving sub-linear query time complexity even with millions of indexed vectors.

### Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| **Image embedding** | 50-100ms (CPU)<br>10-20ms (GPU) | ViT-B/16 forward pass |
| **Text embedding** | 10-30ms (CPU)<br>5-10ms (GPU) | Transformer forward pass (batch) |
| **Zero-shot classification** | 60-130ms | Image encoding + text encoding (5 labels) + similarity |
| **Few-shot k-NN search** | 50-110ms | Image encoding + OpenSearch query (<10K vectors) |
| **Indexing throughput** | 5-10 images/sec | Limited by embedding generation |
| **Memory per indexed image** | ~2KB | 512 float32 values + metadata |
| **Attention map generation** | 60-120ms | Vision encoder forward pass + attention extraction |

### Attention Visualization

The system includes **attention map visualization** to show which parts of an image CLIP focuses on during classification. This provides interpretability and helps debug classification decisions.

#### How It Works

CLIP's Vision Transformer (ViT-B/16) processes images as follows:

1. **Image → Patches**: Input image (224×224) is divided into 196 patches (14×14 grid), each 16×16 pixels
2. **Transformer Layers**: 12 self-attention layers process relationships between patches
3. **CLS Token**: A special classification token aggregates information from all patches
4. **Attention Extraction**: We extract attention weights from the CLS token to all patches in the last layer

#### Technical Implementation

```python
# Extract attention from CLIP's vision transformer
vision_outputs = model.vision_model(
    pixel_values=pixel_values,
    output_attentions=True  # Request attention weights
)

# Get last layer attention: [batch_size, num_heads, num_patches, num_patches]
last_layer_attention = vision_outputs.attentions[-1]

# Average across all attention heads
attention = last_layer_attention.mean(dim=1)

# Extract CLS token attention to image patches
cls_attention = attention[0, 0, 1:]  # [196] for ViT-B/16

# Reshape to 2D grid and normalize
attention_map = cls_attention.reshape(14, 14)
attention_map = (attention_map - min) / (max - min)

# Create heat map overlay using OpenCV
heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
overlay = alpha * heatmap + (1 - alpha) * original_image
```

#### Interpretation Guide

**Heat Map Colors**:
- 🔴 **Red/Yellow**: High attention - CLIP focuses heavily on these regions
- 🟢 **Green**: Medium attention - Moderately important features
- 🔵 **Blue/Purple**: Low attention - Background or less relevant areas

**Use Cases**:
- **Debug misclassifications**: See if CLIP is focusing on the right features
- **Validate defect detection**: Confirm attention aligns with defect location
- **Understand model behavior**: Learn what visual patterns CLIP associates with each label
- **Data quality check**: Identify if model focuses on artifacts or irrelevant areas

**Example Insights**:
- If classifying "contamination", attention should focus on particle regions
- If classifying "crystal structure", attention should highlight lattice patterns
- Unexpected attention patterns may indicate:
  - Misleading image features
  - Need for better training examples (few-shot mode)
  - Need for more descriptive labels (zero-shot mode)

**API Endpoint**:
```bash
curl -X POST "http://localhost:8000/attention-map/" \
  -F "file=@/path/to/image.jpg" \
  --output attention_overlay.png
```

---

## ⚖️ Zero-Shot vs Few-Shot: When to Use Each Mode

| Criteria | Zero-Shot | Few-Shot |
|----------|-----------|----------|
| **Training Images** |  None required |  1-5 per class minimum |
| **Setup Time** |  Instant (type labels) |  Minutes to hours (index images) |
| **Accuracy** |  Good for general categories |  Excellent for domain-specific defects |
| **Flexibility** |  Add new classes instantly |  Requires indexing new examples |
| **Best For** | Exploration, new defect types, prototyping | Production, high-accuracy requirements |
| **Example Use Case** | "Is this contamination or a crystal defect?" | "Match this to our library of known defects" |
| **Computational Cost** |  Lower (no database required) |  Higher (OpenSearch + indexing) |
| **Interpretability** |  Shows similarity to all label descriptions |  Shows most similar reference image |

### Recommended Workflow

1. **Start with Zero-Shot** to quickly explore your dataset and identify defect categories
2. **Collect representative examples** of each defect type based on zero-shot results
3. **Switch to Few-Shot** by indexing 3-10 examples per category for higher accuracy
4. **Iterate**: Use zero-shot for new/rare defects, few-shot for common defects

---

## 🐳 Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| `opensearch` | 9200 | Vector database with k-NN plugin |
| `opensearch-dashboards` | 5601 | Data visualization & index management |
| `clip_model` | 5001 | CLIP embedding generation service |
| `backend` | 8000 | FastAPI REST API |
| `frontend` | 8501 | Streamlit web interface |

**Network**: All services communicate via Docker bridge network `semnet`

---

##  Testing & Validation

### Verify CLIP Embeddings

Check that embeddings are correctly generated as 512-dimensional vectors:

```powershell
docker compose logs clip_model | Select-String "Final embedding length"
# Expected: Final embedding length: 512
```

### Inspect Indexed Data

Query OpenSearch to verify indexed documents:

```powershell
docker compose exec opensearch curl -s "localhost:9200/sem-defects/_search?size=3&pretty"
```

### Check Classification Accuracy

Upload test images from each defect category and verify the predicted labels match the ground truth.

---

## 🔍 Troubleshooting

### Issue: "Field 'embedding' is not knn_vector type"

**Cause**: Index was created without k-NN mapping  
**Solution**: Delete and recreate index:
```powershell
docker compose exec opensearch curl -X DELETE "localhost:9200/sem-defects"
docker compose restart backend
```

### Issue: All classifications return "unknown"

**Cause**: Labels not passed correctly during indexing  
**Solution**: Ensure `Form(...)` is used in backend endpoint and rebuild:
```powershell
docker compose up -d --build backend
docker compose exec backend python index_images.py
```

### Issue: Embedding dimension mismatch (151296 instead of 512)

**Cause**: Using raw vision encoder output instead of projected embeddings  
**Solution**: Verify `clip_model/serve_clip.py` uses `visual_projection` layer

---

##  References

- **CLIP Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- **OpenSearch k-NN**: [k-NN Plugin Documentation](https://opensearch.org/docs/latest/search-plugins/knn/index/)
- **Vision Transformer**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

---

##  License

This project is intended for research and educational purposes. Please comply with the dataset licenses and OpenAI's CLIP model terms of use.

---

##  Acknowledgments

- Dataset providers: EUDAT B2SHARE and Kaggle contributors
- OpenAI for the CLIP model
- OpenSearch community for the k-NN vector search plugin

---

##  Contributing

Contributions are welcome! Areas for improvement:

**Zero-Shot Enhancements**:
- Automatic prompt engineering (e.g., "a SEM image showing {defect_type}")
- Multi-label zero-shot classification
- Confidence calibration for zero-shot predictions

**Few-Shot Enhancements**:
- Active learning pipeline for label-efficient training
- k-NN with k>1 for ensemble predictions
- Weighted voting based on similarity scores

**General Improvements**:
- Support for additional SEM defect datasets
- Integration with real-time microscope image streams
- GPU acceleration for CLIP encoding
- Quantitative evaluation metrics (precision, recall, F1, mAP@k)
- Hybrid mode: Use zero-shot for rare defects, few-shot for common ones
- Export/import of indexed defect libraries

---