import streamlit as st
import pandas as pd
import pickle
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Medical Insurance Prediction",
    page_icon="🏥",
    layout="wide"
)

# ================= LOAD MODEL =================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ================= DATABASE =================
conn = sqlite3.connect("insurance_app.db", check_same_thread=False)
cursor = conn.cursor()

# ---------- ENSURE TABLE EXISTS ----------
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    sex TEXT,
    bmi REAL,
    children INTEGER,
    smoker TEXT,
    region TEXT,
    predicted_cost REAL,
    date_time TEXT
)
""")
conn.commit()

# ---------- AUTO REMOVE username COLUMN ----------
cursor.execute("PRAGMA table_info(predictions)")
cols = [c[1] for c in cursor.fetchall()]

if "username" in cols:
    cursor.execute("""
    CREATE TABLE predictions_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age INTEGER,
        sex TEXT,
        bmi REAL,
        children INTEGER,
        smoker TEXT,
        region TEXT,
        predicted_cost REAL,
        date_time TEXT
    )
    """)
    cursor.execute("""
    INSERT INTO predictions_new (id, age, sex, bmi, children, smoker, region, predicted_cost, date_time)
    SELECT id, age, sex, bmi, children, smoker, region, predicted_cost, date_time
    FROM predictions
    """)
    cursor.execute("DROP TABLE predictions")
    cursor.execute("ALTER TABLE predictions_new RENAME TO predictions")
    conn.commit()

# ================= SIDEBAR INPUT =================
st.sidebar.title("🧮 User Input")

age = st.sidebar.number_input("Age", 18, 100, 25)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
children = st.sidebar.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

predict_btn = st.sidebar.button("🔮 Predict")

# ================= ENCODING =================
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0

region_map = {
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
}
region_encoded = region_map[region]

# ================= MAIN UI =================
st.title("🏥 Medical Insurance Cost Prediction System")
st.markdown("---")

tabs = st.tabs(["🧮 Prediction", "📊 Dashboard", "🗂 Records", "ℹ About"])

# ================= TAB 1 : PREDICTION =================
with tabs[0]:
    if predict_btn:
        input_df = pd.DataFrame(
            [[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]],
            columns=["age", "sex", "bmi", "children", "smoker", "region"]
        )

        prediction = model.predict(input_df)[0]

        cursor.execute("""
        INSERT INTO predictions
        (age, sex, bmi, children, smoker, region, predicted_cost, date_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            age, sex, bmi, children, smoker, region,
            round(prediction, 2),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()

        st.success("Prediction Successful 🎉")
        st.markdown(
            f"""
            <div style="padding:30px;background:#f0f2f6;border-radius:15px;text-align:center;">
                <h2>₹ {prediction:,.2f}</h2>
                <p>Your Estimated Insurance Cost</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ================= TAB 2 : DASHBOARD =================
with tabs[1]:
    st.subheader("📊 Complete Dashboard Analytics")

    df = pd.read_sql("SELECT * FROM predictions", conn)

    if df.empty:
        st.info("No data available.")
    else:
        # ---- METRICS ----
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", len(df))
        c2.metric("Average Cost", f"₹ {df['predicted_cost'].mean():,.0f}")
        c3.metric("Maximum Cost", f"₹ {df['predicted_cost'].max():,.0f}")

        st.markdown("---")

        # ---- PIE CHARTS ----
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🥧 Smoker vs Non-Smoker")
            fig1, ax1 = plt.subplots()
            df["smoker"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1)
            ax1.set_ylabel("")
            st.pyplot(fig1)

        with col2:
            st.markdown("### 🥧 Region-wise Distribution")
            fig2, ax2 = plt.subplots()
            df["region"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax2)
            ax2.set_ylabel("")
            st.pyplot(fig2)

        st.markdown("---")

        # ---- LINE & SCATTER ----
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### 📈 Age vs Cost")
            fig3, ax3 = plt.subplots()
            ax3.plot(df["age"], df["predicted_cost"], marker="o")
            ax3.set_xlabel("Age")
            ax3.set_ylabel("Cost")
            st.pyplot(fig3)

        with col4:
            st.markdown("### 🔵 BMI vs Cost")
            fig4, ax4 = plt.subplots()
            ax4.scatter(df["bmi"], df["predicted_cost"])
            ax4.set_xlabel("BMI")
            ax4.set_ylabel("Cost")
            plt.xticks(rotation=90)  
            st.pyplot(fig4)

        st.markdown("---")

        st.download_button(
            "⬇ Download Dashboard CSV",
            df.to_csv(index=False),
            "dashboard_data.csv",
            "text/csv"
        )

        if st.button("🗑 Delete All Records"):
            cursor.execute("DELETE FROM predictions")
            conn.commit()
            st.rerun()

# ================= TAB 3 : RECORDS =================
with tabs[2]:
    df = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)
    st.dataframe(df, use_container_width=True)

# ================= TAB 4 : ABOUT =================
with tabs[3]:
    st.markdown("""
    ### About This Application
    - Medical Insurance Cost Prediction using Machine Learning  
    - Interactive Analytics Dashboard  
    - SQLite Database Integration  

    **MCA Major Project – Streamlit ML Application**
    """)
