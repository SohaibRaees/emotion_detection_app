# =====================================================
# EMOTION DETECTION PAGE
# =====================================================

import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from utils.location_service import (
    get_city_name,
    update_user_location,
    get_user_location
)

from database.db_connection import execute_insert, execute_query        

# =====================================================
# LOGIN CHECK
# =====================================================

if (
    "user_id" not in st.session_state
    or st.session_state.user_id is None
):

    st.warning(
        "Please login first."
    )

    st.stop()

# =====================================================
# PAGE CONFIG
# =====================================================

# st.title("🎭 Emotion Detection")
from utils.ui import (
    load_ui,
    kpi_card,
    page_intro
)

load_ui("Emotion Detection")

page_intro(

    "Emotion Detection",

    "Upload or capture an image so our Vision Transformer can detect your emotion.",

    "🎭"

)

# =====================================================
# CURRENT LOCATION
# =====================================================

st.markdown("---")
st.subheader("📍 Current Location")

# Detect only once per session
if "location_updated" not in st.session_state:

    with st.spinner("Detecting your current location..."):

        success = update_user_location(
            st.session_state.user_id
        )

    if success:

        st.session_state.location_updated = True

        lat, lon = get_user_location(
            st.session_state.user_id
        )

        city = get_city_name(
            lat,
            lon
        
        )

        st.success("📍 Current Location Updated Successfully")

        st.info(f"🏙 City: {city}")

        col1, col2 = st.columns(2)

        with col1:
            kpi_card(
                "📍",
                "Current City",
                city
            )

        with col2:
            kpi_card(
                "👤",
                "Current User",
                st.session_state.user_name
            )

    else:

        st.warning(
            """
            ⚠ Please allow browser location access.

            Restaurant recommendations require your current location.
            """
        )

        st.stop()

else:

    lat, lon = get_user_location(
        st.session_state.user_id
    )

    if lat is not None:

        st.success("📍 Current Location")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Latitude",
                f"{lat:.6f}"
            )

        with col2:
            st.metric(
                "Longitude",
                f"{lon:.6f}"
            )

st.markdown("---")

# =====================================================
# MODEL CONFIG
# =====================================================

MODEL_PATH = "models/vit_emotion_model.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# MODEL LOADER
# =====================================================

@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if "head.weight" in checkpoint:
        num_classes = checkpoint[
            "head.weight"
        ].shape[0]

    elif "classifier.weight" in checkpoint:
        num_classes = checkpoint[
            "classifier.weight"
        ].shape[0]

    else:
        raise RuntimeError(
            "Classifier head not found"
        )

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes
    )

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =====================================================
# CLASSES
# =====================================================

CLASS_NAMES = [
    "Ahegao", "Angry", "Happy", "Neutral", "Sad", "Surprise"]

# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])

# =====================================================
# FACE DETECTION
# =====================================================

def detect_face(image):

    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(
        open_cv_image,
        cv2.COLOR_RGB2BGR
    )

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        +
        "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(
        open_cv_image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = open_cv_image[
            y:y+h,
            x:x+w
        ]
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )
        return Image.fromarray(
            face
        )

    return image

# =====================================================
# INPUT METHOD
# =====================================================

# input_option = st.radio(
#     "Select Image Input Method",
#     [
#         "Upload Image",
#         "Capture From Camera"
#     ]
# )
tab1, tab2 = st.tabs(
    [
        "📁 Upload Image",
        "📷 Capture From Camera"
    ]
)

image = None

with tab1:

    uploaded_file = st.file_uploader(
        "Choose Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

with tab2:

    camera_image = st.camera_input(
        "Capture Image"
    )

    if camera_image:

        image = Image.open(
            camera_image
        ).convert("RGB")





# image = None

# if input_option == "Upload Image":

#     uploaded_file = st.file_uploader(
#         "Upload Image",
#         type=[
#             "jpg",
#             "jpeg",
#             "png"
#         ]
#     )

#     if uploaded_file:
#         image = Image.open(
#             uploaded_file
#         ).convert("RGB")

# else:
#     camera_image = st.camera_input(
#         "Capture Image"
#     )
#     if camera_image:
#         image = Image.open(
#             camera_image
#         ).convert("RGB")


with st.spinner(
    "🤖 AI is analyzing your facial expression..."
):
    progress = st.progress(0)

    for i in range(100):

        progress.progress(i + 1)

    progress.empty()
    
# =====================================================
# PREDICTION
# =====================================================

    if image is not None:

        image = detect_face(image)
        left, right = st.columns([1.2,1])
        with left:
            st.markdown("### 🖼 Uploaded Image")
            st.image(
                image,
                width=350
            )

        img_tensor = transform(
            image
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(
                img_tensor
            )

        probs = torch.softmax(
            outputs,
            dim=1
        )

        conf, pred = torch.max(
            probs,
            1
        )

        detected_emotion = CLASS_NAMES[
            pred.item()
        ]

        # =====================================
        # AHEGAO -> FEAR
        # =====================================

        if detected_emotion == "Ahegao":

            detected_emotion = "Fear"

        confidence = float(
            conf.item()
        )

        with right:

            st.markdown("### 🤖 AI Prediction")

            kpi_card(
                "😊",
                "Detected Emotion",
                detected_emotion,
                "success"
            )

            kpi_card(
                "🎯",
                "Confidence",
                f"{confidence*100:.2f}%",
                "warning"
            )
            
            st.progress(
                confidence
            )

        # =====================================
        # SAVE TO SESSION
        # =====================================

        st.session_state.emotion = (
            detected_emotion
        )

        st.session_state.confidence = (
            confidence
        )

        # =====================================
        # SAVE BUTTON
        # =====================================
        left, right = st.columns(2)
        with left:
            if st.button(
                "💾 Save Emotion"
            ):
                emotion_id = execute_insert(
                """
                INSERT INTO Emotion_History
                (
                    user_id,
                    emotion,
                    confidence
                )
                VALUES
                (%s,%s,%s)
                """,
                (
                st.session_state.user_id,
                detected_emotion,
                confidence
                )
                )

                st.session_state.emotion_id = emotion_id

                st.session_state.emotion = detected_emotion

                st.session_state.confidence = confidence

                st.success(
                    f"✅ Emotion saved successfully."
                )

                st.info(
                    f"""
                    Emotion: {detected_emotion}

                    Confidence: {confidence*100:.2f}%
                    """ 
                )
    
        # =====================================
        # CONTINUE
        # =====================================
        with right:
            if st.button(
                "➡ Continue To Food Recommendation"
            ):
                if "emotion_id" not in st.session_state:
                    st.error(
                        "Please save the emotion first."
                    )

                else:
                    st.switch_page(
                        "pages/4_Food_Recommendation.py"
                    )
