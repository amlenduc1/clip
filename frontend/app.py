import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="SEM Defect Classifier", layout="wide")

st.title("🔬 SEM Defect Image Classifier")
st.markdown("Upload one or more SEM images to classify defect types using CLIP")

# Classification mode selection
st.sidebar.header("⚙️ Configuration")
mode = st.sidebar.radio(
    "Classification Mode",
    options=["Zero-Shot", "Few-Shot"],
    help="""
    **Zero-Shot**: No training images needed. Uses text descriptions of defect types.
    **Few-Shot**: Uses similarity to indexed training images.
    """
)

# Visualization options
st.sidebar.subheader("🔍 Visualization")
show_attention = st.sidebar.checkbox(
    "Show Attention Maps",
    value=False,
    help="Visualize which parts of the image CLIP focuses on (shows heat map overlay)"
)

# Zero-shot configuration
custom_labels = None
if mode == "Zero-Shot":
    st.sidebar.subheader("Zero-Shot Labels")
    use_custom_labels = st.sidebar.checkbox("Use custom labels", value=False)
    if use_custom_labels:
        custom_labels_text = st.sidebar.text_area(
            "Enter labels (comma-separated)",
            value="diatom,pollen,tapetal",
            help="Enter defect type labels separated by commas"
        )
        custom_labels = custom_labels_text
    else:
        st.sidebar.info("Using default labels:\n- diatom microscopic structure\n- pollen particle contamination\n- tapetal cell defect\n- crystalline defect\n- surface contamination\n- geometric irregularity")

# Display mode information
if mode == "Zero-Shot":
    st.info("🚀 **Zero-Shot Mode**: No training images required! Classifies based on text similarity.")
else:
    st.info("📚 **Few-Shot Mode**: Uses k-NN search on indexed training images.")

# File uploader - always visible, supports multiple files
uploaded_files = st.file_uploader(
    "Choose SEM images", 
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    accept_multiple_files=True,
    help="Upload one or more images for classification"
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)} image(s) uploaded**")
    
    # Classify button
    if st.button("🚀 Classify All Images", type="primary"):
        # Create columns for results
        results = []
        
        # Determine endpoint based on mode
        if mode == "Zero-Shot":
            endpoint = "http://backend:8000/classify-image-zeroshot/"
        else:
            endpoint = "http://backend:8000/classify-image/"
        
        with st.spinner(f"Classifying {len(uploaded_files)} image(s) using {mode} mode..."):
            for uploaded_file in uploaded_files:
                try:
                    # Read and prepare image
                    uploaded_file.seek(0)  # Reset file pointer
                    image = Image.open(uploaded_file)
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    buf.seek(0)
                    
                    # Call backend API
                    files = {"file": (uploaded_file.name, buf, "image/png")}
                    
                    # Add custom labels for zero-shot mode
                    if mode == "Zero-Shot" and custom_labels:
                        data = {"labels": custom_labels}
                        resp = requests.post(endpoint, files=files, data=data)
                    else:
                        resp = requests.post(endpoint, files=files)
                    
                    # Get attention map if requested
                    attention_image = None
                    if show_attention and resp.ok:
                        try:
                            buf.seek(0)  # Reset buffer
                            files_attn = {"file": (uploaded_file.name, buf, "image/png")}
                            attn_resp = requests.post("http://backend:8000/attention-map/", files=files_attn)
                            if attn_resp.ok:
                                attention_image = Image.open(io.BytesIO(attn_resp.content))
                        except Exception as attn_error:
                            logger.info(f"Failed to generate attention map: {attn_error}")
                    
                    if resp.ok:
                        result = resp.json()
                        results.append({
                            "filename": uploaded_file.name,
                            "image": image,
                            "attention_image": attention_image,
                            "defect_type": result['defect_type'],
                            "confidence": result['confidence_score'],
                            "mode": result.get('mode', mode),
                            "all_scores": result.get('all_scores', {}),
                            "status": "success"
                        })
                    else:
                        results.append({
                            "filename": uploaded_file.name,
                            "image": None,
                            "attention_image": None,
                            "defect_type": "Error",
                            "confidence": 0.0,
                            "mode": mode,
                            "all_scores": {},
                            "status": "error"
                        })
                except Exception as e:
                    results.append({
                        "filename": uploaded_file.name,
                        "image": None,
                        "attention_image": None,
                        "defect_type": f"Error: {str(e)}",
                        "confidence": 0.0,
                        "mode": mode,
                        "all_scores": {},
                        "status": "error"
                    })
        
        # Display results
        st.success(f"✅ Classification complete for {len(results)} image(s)")
        
        # Show results in a grid
        for idx, res in enumerate(results):
            with st.expander(f"📷 {res['filename']} - **{res['defect_type']}** (Score: {res['confidence']:.4f})", expanded=True):
                if res['status'] == 'success':
                    # Show images side-by-side if attention map is available
                    if res.get('attention_image'):
                        col_img1, col_img2, col_info = st.columns([1, 1, 2])
                        with col_img1:
                            st.image(res['image'], caption="Original Image", use_column_width=True)
                        with col_img2:
                            st.image(res['attention_image'], caption="Attention Heat Map", use_column_width=True)
                        with col_info:
                            st.metric("Defect Type", res['defect_type'])
                            st.metric("Confidence Score", f"{res['confidence']:.4f}")
                            st.caption(f"Mode: {res['mode']}")
                            st.info("🔥 **Attention Map**: Red areas show where CLIP focuses most when classifying this image")
                    else:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if res['image']:
                                st.image(res['image'], caption=res['filename'], use_column_width=True)
                        with col2:
                            st.metric("Defect Type", res['defect_type'])
                            st.metric("Confidence Score", f"{res['confidence']:.4f}")
                            st.caption(f"Mode: {res['mode']}")
                    
                    # Show all scores for zero-shot mode
                    if res['all_scores']:
                        st.subheader("All Label Scores")
                        score_df = {
                            "Label": list(res['all_scores'].keys()),
                            "Similarity Score": [f"{v:.4f}" for v in res['all_scores'].values()]
                        }
                        st.table(score_df)
                else:
                    st.error(f"Classification failed: {res['defect_type']}")
    
    # Preview uploaded images
    else:
        st.info("👆 Click 'Classify All Images' button to start classification")
        
        # Show thumbnails
        cols = st.columns(min(len(uploaded_files), 4))
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % 4]:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_column_width=True)

else:
    st.info("👆 Upload images using the file uploader above")

# Separator
st.markdown("---")

# Training Image Indexing Section (for Few-Shot mode)
st.header("📚 Index Training Images (Few-Shot Mode)")
st.markdown("Add new labeled images to the index for improved few-shot classification accuracy")

with st.expander("➕ Add Training Images to Index", expanded=False):
    st.info("💡 **Tip**: Index 3-10 examples per defect category for best few-shot performance")
    
    # File uploader for training images
    training_files = st.file_uploader(
        "Upload training images",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        help="Select one or more images to add to the training index",
        key="training_uploader"
    )
    
    if training_files:
        st.write(f"**{len(training_files)} training image(s) selected**")
        
        # Label input options
        label_input_method = st.radio(
            "How to assign labels?",
            options=["Same label for all", "Individual labels"],
            horizontal=True,
            help="Choose whether all images have the same label or each needs a different label"
        )
        
        labels_dict = {}
        
        if label_input_method == "Same label for all":
            common_label = st.text_input(
                "Defect Type Label",
                placeholder="e.g., diatom, pollen, contamination",
                help="Enter the defect category for all uploaded images"
            )
            if common_label:
                labels_dict = {f.name: common_label.strip() for f in training_files}
        else:
            st.markdown("**Assign labels to each image:**")
            cols = st.columns(3)
            for idx, training_file in enumerate(training_files):
                with cols[idx % 3]:
                    # Show thumbnail
                    img = Image.open(training_file)
                    st.image(img, caption=training_file.name, use_column_width=True)
                    # Label input
                    label = st.text_input(
                        f"Label for {training_file.name}",
                        key=f"label_{idx}",
                        placeholder="defect type"
                    )
                    if label:
                        labels_dict[training_file.name] = label.strip()
        
        # Index button
        if st.button("📥 Index Training Images", type="primary", key="index_button"):
            if not labels_dict or len(labels_dict) != len(training_files):
                st.error("⚠️ Please provide labels for all images before indexing")
            else:
                index_results = []
                
                with st.spinner(f"Indexing {len(training_files)} image(s)..."):
                    for training_file in training_files:
                        label = labels_dict.get(training_file.name, "unknown")
                        
                        try:
                            # Read and prepare image
                            training_file.seek(0)  # Reset file pointer
                            image = Image.open(training_file)
                            buf = io.BytesIO()
                            image.save(buf, format="PNG")
                            buf.seek(0)
                            
                            # Call indexing API
                            files = {"file": (training_file.name, buf, "image/png")}
                            data = {"label": label}
                            resp = requests.post("http://backend:8000/index-image/", files=files, data=data)
                            
                            if resp.ok:
                                result = resp.json()
                                index_results.append({
                                    "filename": training_file.name,
                                    "label": label,
                                    "doc_id": result.get('doc_id', 'N/A'),
                                    "status": "success"
                                })
                            else:
                                index_results.append({
                                    "filename": training_file.name,
                                    "label": label,
                                    "doc_id": None,
                                    "status": f"error: {resp.text}"
                                })
                        except Exception as e:
                            index_results.append({
                                "filename": training_file.name,
                                "label": label,
                                "doc_id": None,
                                "status": f"error: {str(e)}"
                            })
                
                # Display indexing results
                success_count = sum(1 for r in index_results if r['status'] == 'success')
                failed_count = len(index_results) - success_count
                
                if success_count > 0:
                    st.success(f"✅ Successfully indexed {success_count} image(s)")
                if failed_count > 0:
                    st.error(f"❌ Failed to index {failed_count} image(s)")
                
                # Show detailed results
                st.subheader("Indexing Results")
                for res in index_results:
                    if res['status'] == 'success':
                        st.success(f"✓ **{res['filename']}** → Label: `{res['label']}` | Doc ID: `{res['doc_id']}`")
                    else:
                        st.error(f"✗ **{res['filename']}** → {res['status']}")
                
                st.info("🎉 Training images indexed! They will now be used for few-shot classification.")

# Get index statistics
with st.expander("📊 View Index Statistics", expanded=False):
    if st.button("🔍 Refresh Stats"):
        try:
            resp = requests.get("http://backend:8000/index-stats/")
            if resp.ok:
                stats = resp.json()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Indexed Images", stats.get('total_images', 0))
                with col2:
                    st.metric("Unique Labels", stats.get('unique_labels', 0))
                
                # Show label distribution
                if stats.get('label_distribution'):
                    st.subheader("Label Distribution")
                    label_dist = stats['label_distribution']
                    
                    # Create a simple bar chart using st.bar_chart
                    import pandas as pd
                    df = pd.DataFrame({
                        'Label': list(label_dist.keys()),
                        'Count': list(label_dist.values())
                    })
                    df = df.sort_values('Count', ascending=False)
                    st.bar_chart(df.set_index('Label'))
                    
                    # Also show as table
                    st.table(df)
                else:
                    st.info("No indexed images yet. Start by adding training images above.")
            else:
                st.error("Failed to fetch index statistics")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.markdown("""
    **Manual Check:**
    - Open [OpenSearch Dashboards](http://localhost:5601)
    - Navigate to Dev Tools
    - Run: `GET sem-defects/_count`
    """)
