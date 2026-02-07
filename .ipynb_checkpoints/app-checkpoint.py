import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "vit_emotion_model.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# LOAD EMOTION MODEL
# =====================================================
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if "head.weight" in checkpoint:
        num_classes = checkpoint["head.weight"].shape[0]
    elif "classifier.weight" in checkpoint:
        num_classes = checkpoint["classifier.weight"].shape[0]
    else:
        raise RuntimeError("Classifier not found")

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
# EMOTION LABELS
# =====================================================
CLASS_NAMES = ["Ahegao", "Angry", "Happy", "Neutral", "Sad", "Surprise"]

# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# FOOD DATABASE (PANDAS)
# =====================================================
food_db = pd.DataFrame([
    [1, "Vegetable Soup", "Sad", 120, 3, 3, "Asian", 150],
    [2, "Chocolate", "Sad", 300, 25, 20, "Western", 250],
    [3, "Yogurt", "Angry", 90, 4, 2, "Any", 100],
    [4, "Biryani", "Happy", 450, 5, 25, "Asian", 350],
    [5, "Fruit Salad", "Neutral", 80, 10, 1, "Any", 120],
    [6, "Grilled Fish", "Neutral", 200, 2, 8, "Asian", 300],
    [7, "Burger", "Happy", 500, 6, 30, "Western", 400],
    [8, "Smoothie", "Angry", 180, 12, 2, "Any", 180],
    [9, "Rice & Dal", "Neutral", 220, 4, 6, "Asian", 200],
], columns=["food_id","food","emotion","calories","sugar","fat","cuisine","price"])

# =====================================================
# HEALTH RULES
# =====================================================
health_rules = {
    "Diabetes": {"sugar_max": 10},
    "Heart": {"fat_max": 10},
    "Obesity": {"calories_max": 300}
}

# =====================================================
# RULE-BASED FILTER
# =====================================================
def rule_based_filter(food_db, user, health_rules):
    # Emotion prioritization
    preferred = food_db[food_db["emotion"] == user["emotion"]]
    others = food_db[food_db["emotion"] != user["emotion"]]
    df = pd.concat([preferred, others])

    # Health filters
    for cond in user["health_conditions"]:
        rule = health_rules.get(cond, {})
        if "sugar_max" in rule:
            df = df[df["sugar"] <= rule["sugar_max"]]
        if "fat_max" in rule:
            df = df[df["fat"] <= rule["fat_max"]]
        if "calories_max" in rule:
            df = df[df["calories"] <= rule["calories_max"]]

    # Budget filter
    df = df[df["price"] <= user["budget"]]

    return df

# =====================================================
# PREFERENCE SCORING
# =====================================================
def preference_score(row, user):
    score = 0
    if row["cuisine"] in user["preferred_cuisine"] or row["cuisine"] == "Any":
        score += 1
    if row["emotion"] == user["emotion"]:
        score += 2
    return score

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Emotion-Aware Food Recommendation", layout="centered")
st.title("😊 Emotion-Aware Food Recommendation System")

# ---------------- IMAGE INPUT ----------------
uploaded_file = st.file_uploader("Upload Facial Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    detected_emotion = CLASS_NAMES[pred.item()]
    st.markdown(f"## 🎭 Detected Emotion: **{detected_emotion}**")
    st.markdown(f"Confidence: **{conf.item()*100:.2f}%**")

    # ---------------- USER INPUT ----------------
    st.markdown("### 👤 User Preferences")
    budget = st.slider("Budget (PKR)", 100, 500, 200)
    preferred_cuisine = st.multiselect(
        "Preferred Cuisine",
        ["Asian", "Western", "Any"],
        default=["Asian"]
    )

    st.markdown("### ❤️ Health Conditions")
    health_conditions = []
    if st.checkbox("Diabetes"):
        health_conditions.append("Diabetes")
    if st.checkbox("Heart Problem"):
        health_conditions.append("Heart")
    if st.checkbox("Obesity"):
        health_conditions.append("Obesity")

    user = {
        "emotion": detected_emotion,
        "health_conditions": health_conditions,
        "budget": budget,
        "preferred_cuisine": preferred_cuisine
    }

    # ---------------- RECOMMENDATION ----------------
    st.markdown("### 🍽 Recommended Foods")

    filtered_foods = rule_based_filter(food_db, user, health_rules)

    if filtered_foods.empty:
        st.warning("No food matches your constraints.")
    else:
        filtered_foods["score"] = filtered_foods.apply(
            lambda row: preference_score(row, user), axis=1
        )
        filtered_foods = filtered_foods.sort_values(
            by="score", ascending=False
        )

        for _, row in filtered_foods.iterrows():
            st.write(
                f"✅ **{row['food']}** | "
                f"Cuisine: {row['cuisine']} | "
                f"Price: PKR {row['price']} | "
                f"Calories: {row['calories']}"
            )
