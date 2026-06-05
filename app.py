import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page configuration with a premium look
st.set_page_config(
    page_title="Customer Segmenter Pro",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply a high-end, responsive custom backdrop and typography styling (Streamlit background theme)
st.markdown("""
<style>
    /* Styling for premium modern Streamlit workspace background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 23, 42) 0%, rgb(30, 41, 59) 90.1%);
        color: #f1f5f9;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Header/Subheader and text tweaks */
    h1, h2, h3, p, span, label {
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    .stHeader {
        background-color: transparent !important;
    }
    
    /* Elegant containers (glassmorphism look) */
    div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    /* Highlight banners */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data():
    csv_path = os.path.join("data", "datasets.csv")
    if not os.path.exists(csv_path):
        # Fallback to local script relative
        csv_path = os.path.join(os.path.dirname(__file__), "data", "datasets.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # Generate backup structure mock data if not exist (ensures resilience)
    return pd.DataFrame({
        'CustomerID': range(1, 201),
        'Gender': np.random.choice(['Male', 'Female'], 200),
        'Age': np.random.randint(18, 70, 200),
        'Annual Income (k$)': np.random.randint(15, 137, 200),
        'Spending Score (1-100)': np.random.randint(1, 100, 200)
    })

df = load_data()

# Helper function to load model or train inline
def get_model(k_clusters, features, use_scaling=True):
    X = df[features].values
    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
        
    kmeans = KMeans(n_clusters=k_clusters, init='k-means++', max_iter=300, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    return kmeans, scaler, labels

# SIDEBAR: Streamlit controls
st.sidebar.image("https://img.icons8.com/color/96/streamlit.png", width=64)
st.sidebar.title("K-Means Controls")
st.sidebar.markdown("Configure your clustering parameters below:")

selected_k = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=12, value=5, step=1)
selected_features = st.sidebar.multiselect(
    "Features for Clustering",
    options=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
    default=['Annual Income (k$)', 'Spending Score (1-100)']
)

use_scaling = st.sidebar.checkbox("Normalize Features (StandardScaler)", value=True)

# Run modeling
if len(selected_features) < 2:
    st.warning("⚠️ Please select at least 2 features to segment your audience.")
    st.stop()

kmeans, scaler, labels = get_model(selected_k, selected_features, use_scaling)
df_clustered = df.copy()
df_clustered['Cluster'] = labels

# MAIN PAGE HEADER
st.title("🧩 Customer Segmentation & K-Means Analytics Workstation")
st.markdown("---")

# Layout metrics
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">{}</div>
        <div class="metric-label">Total Customers</div>
    </div>
    """.format(len(df)), unsafe_allow_html=True)
with col_m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">{}</div>
        <div class="metric-label">Adjusted Clusters (k)</div>
    </div>
    """.format(selected_k), unsafe_allow_html=True)
with col_m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">{:.1f}</div>
        <div class="metric-label">Avg Annual Income</div>
    </div>
    """.format(df['Annual Income (k$)'].mean()), unsafe_allow_html=True)
with col_m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">{:.1f}</div>
        <div class="metric-label">Model Inertia (WCSS)</div>
    </div>
    """.format(kmeans.inertia_), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Layout Tabs
tab_charts, tab_predict, tab_dataObj = st.tabs(["📊 Cluster Plotting", "🔮 Client Prediction Portal", "🗄️ Dataset View"])

with tab_charts:
    st.subheader("High-Dimensional Spatial Slicing")
    col_plot, col_info = st.columns([2, 1])
    
    with col_plot:
        # Create a beautiful clusters matplotlib figure
        fig, ax = plt.subplots(figsize=(9, 5), facecolor='#1e293b')
        ax.set_facecolor('#0f172a')
        
        # Determine plot axes
        x_col = selected_features[0]
        y_col = selected_features[1]
        
        # Color mapping palette
        palette = sns.color_palette("husl", selected_k)
        
        sns.scatterplot(
            data=df_clustered,
            x=x_col,
            y=y_col,
            hue='Cluster',
            palette=palette,
            style='Gender',
            s=120,
            alpha=0.85,
            edgecolor='white',
            linewidth=0.6,
            ax=ax
        )
        
        # Add labels and style axes
        ax.set_title(f"Visualizing Clusters: {x_col} vs {y_col}", fontsize=14, color='white', pad=15)
        ax.set_xlabel(x_col, color='#94a3b8', fontsize=11, labelpad=10)
        ax.set_ylabel(y_col, color='#94a3b8', fontsize=11, labelpad=10)
        ax.tick_params(colors='#94a3b8', labelsize=10)
        
        # Legend custom styling
        legend = ax.legend(facecolor='#1e293b', edgecolor='rgba(255,255,255,0.1)', fontproperties={'size': 9})
        for text in legend.get_texts():
            text.set_color('white')
            
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_info:
        st.subheader("Segment Archetypes")
        st.markdown("Statistical centers and behavioral patterns for your custom cohorts:")
        
        # Show segment metrics
        for i in range(selected_k):
            sub_df = df_clustered[df_clustered['Cluster'] == i]
            score_avg = sub_df['Spending Score (1-100)'].mean()
            income_avg = sub_df['Annual Income (k$)'].mean()
            age_avg = sub_df['Age'].mean()
            
            with st.expander(f"🟢 Cohort #{i} Cluster Profile ({len(sub_df)} users)"):
                st.markdown(f"""
                - **Average Age**: {age_avg:.1f} years old
                - **Average Annual Revenue**: ${income_avg:.1f}k/year
                - **Average Spending Propensity Index (1-100)**: {score_avg:.1f}
                """)

with tab_predict:
    st.subheader("Real-Time Smart Consumer Classifier")
    st.markdown("Deploy your pipeline on theoretical live customer criteria to classify them into the calculated clusters:")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        inp_age = st.slider("Customer Age", min_value=18, max_value=75, value=35)
    with col_p2:
        inp_income = st.slider("Customer Annual Revenue ($k)", min_value=15, max_value=140, value=70)
    with col_p3:
        inp_score = st.slider("Spending Score Affinity (1-100)", min_value=1, max_value=100, value=50)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔮 Run Segmentation Model Classifier", use_container_width=True):
        # Build the vector based on selected features
        sample = {}
        for feat in selected_features:
            if 'Age' in feat:
                sample[feat] = inp_age
            elif 'Income' in feat:
                sample[feat] = inp_income
            elif 'Spending' in feat or 'Score' in feat:
                sample[feat] = inp_score
                
        input_vector = np.array([sample[feat] for feat in selected_features]).reshape(1, -1)
        
        # Standardize vector if scaler was trained
        if use_scaling and scaler is not None:
            processed_vector = scaler.transform(input_vector)
        else:
            processed_vector = input_vector
            
        pred_label = kmeans.predict(processed_vector)[0]
        
        st.success(f"🎉 **Classification Finalized!** This consumer is grouped in **Cluster #{pred_label}**.")
        st.info(f"The structural coordinates of the matched segment cluster centroid inside the model feature-space are: {kmeans.cluster_centers_[pred_label]}")

with tab_dataObj:
    st.subheader("Raw Datasets Explorer")
    st.markdown("Detailed table representation of original and aggregated feature values:")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        gender_q = st.radio("Gender Criteria Filter", ["All", "Male", "Female"], horizontal=True)
    with col_f2:
        search_q = st.text_input("Look up Customer by ID / Custom query")
        
    filtered = df_clustered.copy()
    if gender_q != "All":
        filtered = filtered[filtered['Gender'] == gender_q]
        
    if search_q.strip().isdigit():
        filtered = filtered[filtered['CustomerID'] == int(search_q.strip())]
        
    st.dataframe(filtered, use_container_width=True)
