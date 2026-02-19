import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Emotion Food Recommender",
                   layout="centered")

st.title("😊 Emotion-Based Food Recommendation System")

# =====================================================
# MODEL LOAD
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    checkpoint = torch.load("vit_emotion_model.pth",
                            map_location=DEVICE)
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=6
    )
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

CLASS_NAMES = ["Ahegao","Angry","Happy",
               "Neutral","Sad","Surprise"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# =====================================================
# DUMMY USER DB
# =====================================================
users_db = pd.DataFrame([
    [1,"Sohaib",["Diabetes"],300],
    [2,"Ali",[],500],
    [3,"Ahmed",["Heart"],250],
],columns=["user_id","name","health","budget"])

user_name = st.selectbox(
    "Select User Profile",
    users_db["name"]
)

user_profile = users_db[
    users_db["name"]==user_name
].iloc[0]

# =====================================================
# FOOD DB
# =====================================================
food_db = pd.DataFrame([
    [1,"Vegetable Soup","Sad",120,3,3,"Asian",150],
    [2,"Chocolate","Sad",300,25,20,"Western",250],
    [3,"Yogurt","Angry",90,4,2,"Any",100],
    [4,"Biryani","Happy",450,5,25,"Asian",350],
    [5,"Fruit Salad","Neutral",80,10,1,"Any",120],
    [6,"Grilled Fish","Neutral",200,2,8,"Asian",300],
    [7,"Burger","Happy",500,6,30,"Western",400],
    [8,"Smoothie","Angry",180,12,2,"Any",180],
],columns=["food_id","food","emotion",
           "calories","sugar","fat",
           "cuisine","price"])

# =====================================================
# RATINGS
# =====================================================
ratings = pd.DataFrame([
    [1,1,4],[1,3,5],[1,4,4],[1,5,3],
    [2,2,5],[2,7,4],[2,8,5],[2,6,3],
    [3,3,4],[3,5,5],[3,4,3],[3,1,4]
],columns=["user_id","food_id","rating"])

user_item = ratings.pivot_table(
    index="user_id",
    columns="food_id",
    values="rating"
).fillna(0)

health_rules = {
    "Diabetes":{"sugar_max":10},
    "Heart":{"fat_max":10}
}

def rule_filter(df,user):
    for cond in user["health"]:
        rule=health_rules.get(cond,{})
        if "sugar_max" in rule:
            df=df[df["sugar"]<=rule["sugar_max"]]
        if "fat_max" in rule:
            df=df[df["fat"]<=rule["fat_max"]]
    df=df[df["price"]<=user["budget"]]
    return df

def collaborative_group(user_id,candidate_ids):
    valid=[i for i in candidate_ids
           if i in user_item.columns]
    if not valid:
        return candidate_ids
    matrix=user_item[valid]
    sim=cosine_similarity(matrix)
    sim_df=pd.DataFrame(sim,
        index=matrix.index,
        columns=matrix.index)
    similar=sim_df[user_id]\
        .sort_values(ascending=False)[1:]
    scores=pd.Series(dtype=float)
    for u,s in similar.items():
        scores=scores.add(matrix.loc[u]*s,
                          fill_value=0)
    scores=scores.sort_values(ascending=False)
    return scores.index.tolist()

# =====================================================
# CAMERA SNAPSHOT
# =====================================================
img = st.camera_input("📸 Capture Your Emotion")

if img:

    image = Image.open(img).convert("RGB")
    st.image(image,caption="Captured Image")

    tensor = transform(image)\
        .unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out=model(tensor)
        pred=torch.argmax(out,1)

    emotion = CLASS_NAMES[pred.item()]
    st.success(f"Detected Emotion: {emotion}")

    emo_group = food_db[
        food_db["emotion"]==emotion
    ]

    filtered = rule_filter(
        emo_group,
        user_profile
    )

    candidate_ids = filtered["food_id"].tolist()

    rec_ids = collaborative_group(
        user_profile["user_id"],
        candidate_ids
    )

    final = filtered[
        filtered["food_id"].isin(rec_ids)
    ]

    st.subheader("🍽 Recommended Foods")

    if final.empty:
        st.write("No food found")
    else:
        for _,row in final.iterrows():
            st.write(f"✅ {row['food']} (PKR {row['price']})")
