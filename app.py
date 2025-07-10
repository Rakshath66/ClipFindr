import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- Load CLIP ---
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

st.markdown(
    """
    <h1 style="
        background: linear-gradient(to right, #00FF00, #c2f0ff, #00FF00);
        -webkit-background-clip: text;
        color: transparent;
        font-size: 3rem;
        text-align: center;
        padding: 1rem 0;
    ">
         🖼️ Find Similar Image From Gallery
    </h1>
    """,
    unsafe_allow_html=True
)
model, processor = load_clip_model()



# Dummy Gallery Images (upload some manually or simulate here)
st.markdown("### Upload some Gallery Images:")
gallery_images = st.file_uploader(type=["jpg", "png"], accept_multiple_files=True)

# --- Upload and Compare ---
st.markdown("### Query:")
uploaded_image = st.file_uploader(type=["jpg", "png", "jpeg"])

with st.expander("⚠️ &nbsp;  Technology Used &nbsp;  ⚠️ ", expanded=False):
    st.markdown("""
    **🔍 Upload a query image and find the most similar images from your gallery using CLIP!**

    ---
    ### 🛠️ Technology Used:
    - 🧠 CLIP (Contrastive Language–Image Pretraining)
    - 📊 Cosine Similarity (scikit-learn)
    - 🎨 PIL (Image Processing)
    - 🖼️ Streamlit File Upload + Display
    - 📚 Transformers (Hugging Face)
    - 🧮 PyTorch + TorchVision

    ---
    ### ✅ Example Use Cases:
    - *"Find similar product images."*
    - *"Match uploaded screenshot with gallery."*
    - *"Reverse visual search from a dataset."*

    ---
    ### ⚠️ Notes:
    - Works best when gallery images are visually distinct.
    - Results are based on visual feature similarity, not file names.
    """)

gallery_dir = "gallery"
os.makedirs(gallery_dir, exist_ok=True)

for img in gallery_images:
    with open(os.path.join(gallery_dir, img.name), "wb") as f:
        f.write(img.read())

def get_embedding(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
        return embedding

if uploaded_image and gallery_images:
    query_img = Image.open(uploaded_image).convert("RGB")
    query_embedding = get_embedding(query_img)

    similarities = []
    for file in os.listdir(gallery_dir):
        path = os.path.join(gallery_dir, file)
        img = Image.open(path).convert("RGB")
        emb = get_embedding(img)
        score = cosine_similarity(query_embedding, emb)[0][0]
        similarities.append((file, score))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:3]

    st.image(query_img, caption="Query Image", use_container_width=True)
    st.markdown("### 🔍 Top 3 Matches (with scores):")

    for filename, score in top_matches:
        img = Image.open(os.path.join(gallery_dir, filename)).convert("RGB")
        st.image(img, caption=f"{filename} (Score: {score:.2f})", use_column_width=True)
