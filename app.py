"""
Image Recognition App
Classifies an image using a pretrained Hugging Face model.
"""

import streamlit as st
from PIL import Image
from transformers import pipeline

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Image Recognition",
    page_icon="🖼️",
    layout="centered"
)

# -------------------------------------------------
# Available Models
# -------------------------------------------------
MODELS = {
    "ViT Base (Google)": "google/vit-base-patch16-224",
    "ResNet-50 (Microsoft)": "microsoft/resnet-50"
}


# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_classifier(model_id):
    return pipeline(
        task="image-classification",
        model=model_id
    )


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Choose Model",
    list(MODELS.keys())
)

top_k = st.sidebar.slider(
    "Number of Predictions",
    min_value=1,
    max_value=10,
    value=5
)

st.sidebar.markdown("---")

st.sidebar.info(
    "These models are trained on ImageNet "
    "and can recognize many common objects."
)


# -------------------------------------------------
# Main Title
# -------------------------------------------------
st.title("🖼️ Image Recognition")

st.write(
    "Upload an image or take a photo. "
    "The AI model will analyze the image and show its predictions."
)


# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab_upload, tab_camera = st.tabs(
    ["📁 Upload Image", "📷 Camera"]
)

image = None


# -------------------------------------------------
# Upload Image
# -------------------------------------------------
with tab_upload:

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp"
        ]
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

        except Exception as error:
            st.error(
                f"Unable to open the image: {error}"
            )


# -------------------------------------------------
# Camera
# -------------------------------------------------
with tab_camera:

    camera_photo = st.camera_input(
        "Take a picture"
    )

    if camera_photo is not None:

        try:
            image = Image.open(camera_photo).convert("RGB")

        except Exception as error:
            st.error(
                f"Unable to open the camera image: {error}"
            )


# -------------------------------------------------
# Display Image
# -------------------------------------------------
if image is not None:

    st.image(
        image,
        caption="Selected Image",
        use_container_width=True
    )

    st.markdown("---")

    # -------------------------------------------------
    # Recognize Button
    # -------------------------------------------------
    if st.button(
        "🔍 Recognize Image",
        type="primary",
        use_container_width=True
    ):

        try:

            # Load selected model
            model_id = MODELS[model_name]

            with st.spinner(
                "Loading AI model and analyzing image..."
            ):

                classifier = load_classifier(
                    model_id
                )

                results = classifier(
                    image,
                    top_k=top_k
                )

            # -------------------------------------------------
            # Results
            # -------------------------------------------------
            if results:

                best_prediction = results[0]

                label = best_prediction["label"]
                confidence = best_prediction["score"] * 100

                st.success(
                    f"🎯 Top Prediction: **{label}** "
                    f"({confidence:.2f}%)"
                )

                st.markdown("---")

                st.subheader("📊 All Predictions")

                for index, result in enumerate(
                    results,
                    start=1
                ):

                    result_label = result["label"]
                    score = result["score"]

                    percentage = score * 100

                    st.write(
                        f"**{index}. {result_label}**"
                    )

                    st.progress(
                        float(score)
                    )

                    st.caption(
                        f"Confidence: {percentage:.2f}%"
                    )

            else:

                st.warning(
                    "No prediction was returned by the model."
                )

        except Exception as error:

            st.error(
                "❌ An error occurred while analyzing the image."
            )

            st.exception(error)

else:

    st.info(
        "👆 Upload an image or use the camera to get started."
    )


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")

st.caption(
    "🖼️ Image Recognition App • Powered by Hugging Face Transformers"
)