"""
app.py

Main Streamlit application file that serves as the user interface for the diabetes prediction app.

Functionality:
- Loads and displays the dataset with basic information
- Preprocesses data and trains a PyTorch (utils.py) based deep learning model in memory
- Accepts user input through a sidebar (interactive form)
- Uses the trained model to predict the probability of diabetes
- Displays prediction result to the user

This app is intended for educational/demo purposes and uses a simple, locally trained model.
"""




import streamlit as st
import pandas as pd # for contribution to the project
import numpy as np # for contribution to the project
import torch # for contribution to the project
import plotly.express as px
import plotly.graph_objects as go
from model import DiabetesModel, DiabetesTrainer
from utils import (
    load_diabetes_data, preprocess_data, save_scaler, load_scaler,
    prepare_input_data, get_dataset_summary, validate_input_data,
    format_prediction_result, check_model_files, get_feature_descriptions
)



# page config
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        padding-top: 1rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1e293b;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Input section */
    .input-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff; /* putih */
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ffffff; /* garis putih */
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }


    
    /* Prediction results */
    .prediction-result {
        font-size: 1.1rem;
        font-weight: 500;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .diabetes-positive {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 50%, #fecaca 100%);
        color: #7f1d1d;
        border: 2px solid #f87171;
        box-shadow: 0 8px 25px rgba(248, 113, 113, 0.15);
    }
    
    .diabetes-positive::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ef4444, #dc2626);
    }
    
    .diabetes-negative {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 50%, #bbf7d0 100%);
        color: #14532d;
        border: 2px solid #4ade80;
        box-shadow: 0 8px 25px rgba(74, 222, 128, 0.15);
    }
    
    .diabetes-negative::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #22c55e, #16a34a);
    }
    
    .result-header {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .result-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        background: rgba(255, 255, 255, 0.7);
        padding: 0.75rem;
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    
    .stat-label {
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.8;
        margin-bottom: 0.25rem;
    }
    
    .stat-value {
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    /* Results container */
    .results-container {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    /* Gauge container */
    .gauge-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
        margin: 1rem 0;
    }
    
    /* Metrics styling */
    .stMetric {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #f1f5f9;
    }
    
    .stMetric > div {
        font-weight: 600;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Form styling */
    .stForm {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        transform: translateY(-1px);
        box-shadow: 0 6px 12px -1px rgba(0, 0, 0, 0.15);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8fafc;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 2rem 0;
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        color: #1e40af;
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
        color: #92400e;
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
        color: #065f46;
    }
</style>
""", unsafe_allow_html=True)







# ==================================LOGIC===========================================

@st.cache_data
def load_and_process_data():
    """
    Loads the diabetes dataset and generates basic summary statistics.

    This function is cached using streamlit @st.cache_data decorator,
    which means it wont re run unless the underlying data changes. 
    This improves app performance by avoiding unnecessary reloading and 
    processing during interactions.

    Returns:
        tuple:
            - df (pd.DataFrame): The full diabetes dataset as a DataFrame.
            - summary (dict): A dictionary containing key statistics, such as 
              total samples, diabetes case counts, and basic descriptive stats.
    """
    df = load_diabetes_data()  # Attempt to load the dataset from CSV
    if df is not None:
        summary = get_dataset_summary(df)  # Generate a summary dictionary from the data
        return df, summary

    # If loading failed, return None values to handle downstream gracefully
    return None, None




def train_model_if_needed():
    """
    Trains the model if it hasn't been trained yet.

    This function checks whether the trained model file and scaler file exist.
    If not, it performs the full training pipeline:
    - Loads and preprocesses the data
    - Initializes the model and trainer
    - Trains the model with visual feedback via streamlit
    - Evaluates and saves the trained model and scaler for future use

    Returns:
        bool: True if the model is ready (trained or already exists), False otherwise
    """
    model_exists, scaler_exists = check_model_files()

    if not model_exists or not scaler_exists:
        # Notify the user that model training is starting
        st.markdown(
            '<div class="info-box">Model not found. Starting training process...</div>',
            unsafe_allow_html=True
        )

        # Attempt to load the dataset
        df = load_diabetes_data()
        if df is None:
            st.markdown(
                "<div class='warning-box'>❌ Dataset not found! Make sure 'diabetes.csv' is located in the same directory.</div>",
                unsafe_allow_html=True
            )
            return False

        # preprocess the dataset (split + scale)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

        # Save the fitted scaler for future use
        save_scaler(scaler)

        # Init the model and training handler
        model = DiabetesModel(input_size=8)
        trainer = DiabetesTrainer(model)

        # Visual feedback during training
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Train the model========================================================================================================
        trainer.train_model(X_train, y_train, epochs=50, batch_size=32)

##################################################################################################################################


        # Complete the progress bar and notify the user
        progress_bar.progress(1.0)
        status_text.text('Training complete!')

        # Evaluate model performance on the test set
        accuracy = trainer.evaluate_model(X_test, y_test)

        # save the trained model to file
        trainer.save_model('diabetes_model.pth')

        # sshow succes message with final accuracy
        st.markdown(
            f'<div class="success-box">✅ Model trained successfully with accuracy: {accuracy:.2%}</div>',
            unsafe_allow_html=True
        )
        st.balloons()

        return True

    # Model and scaler already exist
    return True





def load_trained_model():
    """
    Loads a previously trained model along with its associated scaler.

    This function:
    - initializes a fresh model and trainer instnce
    - Loads the saved model weights and optimizer state from file
    - Loads the fitted scaler used during training

    Returns:
        tuple: (trainer, scaler)
            - trainer (DiabetesTrainer): The loaded trainer with model state restored
            - scaler (StandardScaler): The loaded scaler used for input normalizatio
    """
    # Init the model and trainer
    model = DiabetesModel(input_size=8)
    trainer = DiabetesTrainer(model)

    # Load model weights and optimizer state from disk
    trainer.load_model('diabetes_model.pth')

    # Load the scaler used during traning
    scaler = load_scaler()

    return trainer, scaler










###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
# MAIN FUNCTION ===================================================

def main():
    """
    Streamlit app entry point for the Diabetes Prediction System.

    This function handles:
    - Displaying the app header and description
    - Loading and validating the dataset
    - Triggering model training if needed
    - Loading the trained model and scaler
    - Handling any errors during the setup process
    """
    
    # ===== HEADER UI =====
    st.markdown(
        '<h1 class="main-header">Diabetes Prediction System</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Machine Learning Solution for Diabetes Risk Assessment</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ===== LOAD DATASET =====
    df, summary = load_and_process_data()

    # Handle missing or invalid dataset
    if df is None:
        st.markdown(
            "<div class='warning-box'>❌ Dataset not found! Please ensure 'diabetes.csv' is located in the same directory.</div>",
            unsafe_allow_html=True
        )

        # Show expected CSV format for user reference
        with st.expander("Dataset Format Requirements", expanded=True):
            st.markdown("**Expected CSV format:**")
            st.code("""
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
...
            """)
        return  #stop execution if dataset is missing

    # ===== TRAIN MODEL IF NOT YET TRAINED =====
    if not train_model_if_needed():
        return

    # ===== LOAD TRAINED MODEL & SCALER =====
    try:
        trainer, scaler = load_trained_model()
    except Exception as e:
        st.markdown(
            f"<div class='warning-box'>❌ Error loading model: {e}</div>",
            unsafe_allow_html=True
        )
        return

    
    
    # ===== MAIN LAYOUT =====
    col1, col2 = st.columns([2.2, 1.8], gap="large")

    with col1:
        # ===== DATASET OVERVIEW SECTION =====
        st.markdown(
            '<div class="section-header">Dataset Overview & Analytics</div>',
            unsafe_allow_html=True
        )

        # display key dataset metrics in card style column
        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric(
                label="Total Samples",
                value=f"{summary['total_samples']:,}",
                delta=None
            )

        with metric_cols[1]:
            st.metric(
                label="Features",
                value=summary['total_features'],
                delta=None
            )

        with metric_cols[2]:
            st.metric(
                label="Diabetes Cases",
                value=f"{summary['diabetes_cases']:,}",
                delta=f"{summary['diabetes_percentage']:.1f}%"
            )

        with metric_cols[3]:
            st.metric(
                label="Healthy Cases",
                value=f"{summary['non_diabetes_cases']:,}",
                delta=f"{100 - summary['diabetes_percentage']:.1f}%"
            )

        # Add spacing below metrics
        st.markdown("<br>", unsafe_allow_html=True)

        
        # ===== PIE CHART VISUALIZATION =====
        # Create a styled pie chart showing the rqtio of diabetes vs nondiabetes cases
        fig_dist = px.pie(
            values=[summary['diabetes_cases'], summary['non_diabetes_cases']],
            names=['Diabetes Positive', 'Diabetes Negative'],
            title="<b></b>",
            color_discrete_map={
                'Diabetes Positive': '#ef4444',
                'Diabetes Negative': '#22c55e'
            }
        )

        # Customize layout appearance
        fig_dist.update_layout(
            title_font_size=18,
            title_font_color="#ffffff",
            title_x=0.5,
            font=dict(family="Inter", size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # trace styling for better readability
        fig_dist.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=12,
            marker=dict(line=dict(color='#ffffff', width=2))
        )

        # Display pie chart in the app
        st.plotly_chart(fig_dist, use_container_width=True)

        # ===== FEATURE STATISTICS TABLE =====
        # EXSpandable section showing detailed statistics for each feature
        with st.expander("Detailed Feature Statistics", expanded=False):
            st.markdown("**Statistical Summary of Dataset Features:**")
            st.dataframe(
                summary['feature_stats'].style.format("{:.2f}"),
                use_container_width=True
            )

    

    with col2:
        # ===== PATIENT DATA INPUT SECTION =====
        st.markdown(
            '<div class="section-header">Patient Data Input</div>',
            unsafe_allow_html=True
        )

        # tooltips for each input feature
        descriptions = get_feature_descriptions()

        # ===== INPUT FORM =====
        with st.form("prediction_form"):
            st.markdown("**Please enter patient information below:**")

            input_data = {}

            # Numeric input for each feature, with min/max and tooltips
            input_data['Pregnancies'] = st.number_input(
                "**Pregnancies**",
                min_value=0, max_value=20, value=1, step=1,
                help=descriptions['Pregnancies']
            )

            input_data['Glucose'] = st.number_input(
                "**Glucose Level**",
                min_value=0, max_value=300, value=120, step=1,
                help=descriptions['Glucose']
            )

            input_data['BloodPressure'] = st.number_input(
                "**Blood Pressure**",
                min_value=0, max_value=200, value=80, step=1,
                help=descriptions['BloodPressure']
            )

            input_data['SkinThickness'] = st.number_input(
                "**Skin Thickness**",
                min_value=0, max_value=100, value=20, step=1,
                help=descriptions['SkinThickness']
            )

            input_data['Insulin'] = st.number_input(
                "**Insulin Level**",
                min_value=0, max_value=1000, value=85, step=1,
                help=descriptions['Insulin']
            )

            input_data['BMI'] = st.number_input(
                "**Body Mass Index (BMI)**",
                min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                help=descriptions['BMI']
            )

            input_data['DiabetesPedigreeFunction'] = st.number_input(
                "**Diabetes Pedigree Function**",
                min_value=0.0, max_value=3.0, value=0.5, step=0.001,
                help=descriptions['DiabetesPedigreeFunction']
            )

            input_data['Age'] = st.number_input(
                "**Age**",
                min_value=18, max_value=120, value=30, step=1,
                help=descriptions['Age']
            )

            # Form submit button
            submitted = st.form_submit_button(
                "Analyze Diabetes Risk",
                use_container_width=True
            )

    # variable to hold prediction result, evaluated later outside the layout block
    prediction_data = None

    

    if submitted:
        # ===== VALIDATE USER INPUT =====
        is_valid, message = validate_input_data(input_data)

        if not is_valid:
            # SHowvalidation warning if input is invalid
            st.markdown(
                f'<div class="warning-box">❌ {message}</div>',
                unsafe_allow_html=True
            )
        else:
            try:
                # ===== PREPARE INPUT FOR MODEL ====
                processed_input = prepare_input_data(input_data, scaler)

                # ===== MAKE PREDICTION =====
                probability = trainer.predict_probability(processed_input)

                # Format prediction output
                result = format_prediction_result(probability)

                # Store prediction result for rendering outside the form
                prediction_data = {
                    'result': result,
                    'input_data': input_data
                }

                # Display success messege
                st.markdown(
                    '<div class="success-box">✅ Prediction completed! Results displayed below.</div>',
                    unsafe_allow_html=True
                )

            except Exception as e:
                # show any error that occurs during prediction
                st.markdown(
                    f'<div class="warning-box">❌ Prediction error: {e}</div>',
                    unsafe_allow_html=True
                )



    # ===== PREDICTION RESULTS SECTION =====
    if prediction_data is not None:
        st.markdown("<br><br>", unsafe_allow_html=True)

        # section title
        st.markdown(
            '<div class="section-header">Prediction Results</div>',
            unsafe_allow_html=True
        )

        # Retrieve predicton result and user input
        result = prediction_data['result']
        input_data = prediction_data['input_data']

        # Create 3 column layout for balance ui
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1], gap="large")

        with result_col1:
            # ===== PREDICTION RESULT CARD =====
            if result['prediction'] == 1:
                # User is predicted to be at risk 0f diabetes
                st.markdown(f"""
                <div class="prediction-result diabetes-positive">
                    <div class="result-header">
                        <span style="font-size: 3rem;">⚠️</span>
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 700; margin: 1rem 0;">
                        {result['status']}
                    </div>
                    <div class="result-stats">
                        <div class="stat-item">
                            <div class="stat-label">Risk Probability</div>
                            <div class="stat-value" style="font-size: 1.4rem;">{result['probability']:.1%}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Confidence Level</div>
                            <div class="stat-value" style="font-size: 1.4rem;">{result['confidence']:.1%}</div>
                        </div>
                    </div>
                    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8; line-height: 1.4;">
                        🏥 <strong>Recommendation:</strong><br>
                        Consult with a healthcare provider for further evaluation and a personalized treatment plan.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # user is predicted to be healthy / not at risk
                st.markdown(f"""
                <div class="prediction-result diabetes-negative">
                    <div class="result-header">
                        <span style="font-size: 3rem;">✅</span>
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 700; margin: 1rem 0;">
                        {result['status']}
                    </div>
                    <div class="result-stats">
                        <div class="stat-item">
                            <div class="stat-label">Risk Probability</div>
                            <div class="stat-value" style="font-size: 1.4rem;">{result['probability']:.1%}</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Confidence Level</div>
                            <div class="stat-value" style="font-size: 1.4rem;">{result['confidence']:.1%}</div>
                        </div>
                    </div>
                    <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8; line-height: 1.4;">
                        💚 <strong>Recommendation:</strong><br>
                        Continue maintaining healthy lifestyle habits and have regular health checkups.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        
        with result_col2:
            # ===== RISK GAUGE CHART =====
            # This visual gauge represents the predicted diabetes probability.
            # Color-coded regions indicate risk severity based on thresholds.

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['probability'] * 100,  # Convert to percentage
                domain={'x': [0, 1], 'y': [0, 1]},
                title={
                    'text': "<b style='font-size: 20px; color: #1e293b;'>Diabetes Risk Assessment</b>",
                    'font': {'size': 18}
                },
                number={
                    'font': {'size': 36, 'color': '#1e293b'},
                    'suffix': '%'
                },
                gauge={
                    'axis': {
                        'range': [None, 100],
                        'tickwidth': 2,
                        'tickcolor': "#64748b",
                        'tickfont': {'size': 14, 'color': '#64748b'}
                    },
                    'bar': {
                        'color': "#3b82f6" if result['probability'] < 0.5 else "#ef4444",
                        'thickness': 0.25,
                        'line': {'color': "#ffffff", 'width': 3}
                    },
                    'steps': [
                        {'range': [0, 25], 'color': "#22c55e"},   # Low risk
                        {'range': [25, 50], 'color': "#84cc16"}, # Mild risk
                        {'range': [50, 75], 'color': "#f59e0b"}, # High risk
                        {'range': [75, 100], 'color': "#ef4444"} # Critical risk
                    ],
                    'borderwidth': 3,
                    'bordercolor': "#e2e8f0",
                    'threshold': {
                        'line': {'color': "#7c3aed", 'width': 5},
                        'thickness': 0.8,
                        'value': 50  # Midpoint threshold marker
                    }
                }
            ))

            # Style the gauge chart for clean integration
            fig_gauge.update_layout(
                height=400,
                font=dict(family="Inter", color="#1e293b"),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=80, b=20)
            )

            # Display the gauge chart inside the result column
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


        

        with result_col3:
            # ===== RISK LEVEL CARD & INPUT SUMMARY =====


            # Determine qualitative risk level based on the predicted probability

            if result['probability'] < 0.25:
                risk_level = "LOW RISK"
                risk_color = "#22c55e"
                risk_icon = "🟢"
                risk_desc = "Your diabetes risk is minimal. Maintain current healthy habits."
            elif result['probability'] < 0.50:
                risk_level = "MODERATE RISK"
                risk_color = "#84cc16"
                risk_icon = "🟡"
                risk_desc = "Some risk factors present. Consider lifestyle improvements."
            elif result['probability'] < 0.75:
                risk_level = "HIGH RISK"
                risk_color = "#f59e0b"
                risk_icon = "🟠"
                risk_desc = "Significant risk detected. Medical consultation recommended."
            else:
                risk_level = "VERY HIGH RISK"
                risk_color = "#ef4444"
                risk_icon = "🔴"
                risk_desc = "Critical risk level. Immediate medical attention advised."

            #display styled risk level card
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="background: linear-gradient(135deg, {risk_color}15 0%, {risk_color}25 100%);
                            color: {risk_color};
                            font-weight: 700;
                            font-size: 1.2rem;
                            padding: 1rem;
                            border-radius: 15px;
                            border: 2px solid {risk_color}40;
                            margin-bottom: 1rem;">
                    {risk_icon}<br>
                    <strong>{risk_level}</strong>
                </div>
                <div style="font-size: 0.9rem; color: #64748b; line-height: 1.5; text-align: left;">
                    {risk_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ===== iNPUT SUMMARY =====
            # display selected patient input values to give quick context
            st.markdown("**Input Summary:**")
            st.markdown(f"""
            <div style="background: #f8fafc; color: #1e293b; padding: 1rem; border-radius: 8px; font-size: 0.85rem;">
                <strong>Age:</strong> {input_data['Age']} years<br>
                <strong>BMI:</strong> {input_data['BMI']}<br>
                <strong>Glucose:</strong> {input_data['Glucose']} mg/dL<br>
                <strong>Blood Pressure:</strong> {input_data['BloodPressure']} mmHg<br>
                <strong>Pregnancies:</strong> {input_data['Pregnancies']}<br>
            </div>
            """, unsafe_allow_html=True)

        # Close any result wrapper divs and provide spacing for additional swctions
        st.markdown('</div>', unsafe_allow_html=True)  # end results container
        st.markdown("<br>", unsafe_allow_html=True)    # Add extra vertical space


        
        # ===== EXPANDED SECTION: DETAILED SUMMARY ====
        # This expandable panel provides both an analysis of potential risk factors
        # based on patient input and a complete recap of all input values submitted.
        with st.expander("Analysis & Complete Input Data", expanded=False):
            analysis_col1, analysis_col2 = st.columns(2)

            # --- Risk FActor Analysis ---
            with analysis_col1:
                st.markdown("**Risk Factor Analysis:**")

                # Simple rule-based evaluation to highlight elevated risk inputs
                risk_factors = []
                if input_data['Glucose'] > 140:
                    risk_factors.append("• Elevated glucose levels")
                if input_data['BMI'] > 30:
                    risk_factors.append("• High BMI (obesity)")
                if input_data['BloodPressure'] > 90:
                    risk_factors.append("• Elevated blood pressure")
                if input_data['Age'] > 45:
                    risk_factors.append("• Advanced age")

                # Display detected risk factors with emphasis
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"<span style='color: #f59e0b;'>{factor}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<span style='color: #22c55e;'>• No major risk factors detected</span>",
                        unsafe_allow_html=True
                    )

            # --- Complete Input Data Summary ---
            with analysis_col2:
                st.markdown("**Complete Patient Information:**")

                # Display each field from the user input
                for key, value in input_data.items():
                    #adjust label formatting if needed for better display
                    formatted_key = key.replace('DiabetesPedigreeFunction', 'Diabetes Pedigree')
                    st.markdown(f"• **{formatted_key}:** {value}")




    # === Footer =====
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="footer" style="text-align: center;">
        <h4 style="color: #1e293b; margin-bottom: 0.5rem;">Diabetes Prediction System</h4>
        <p style="color: #64748b; margin-bottom: 0.3rem;">
            <strong>Powered by:</strong> PyTorch • Streamlit • Machine Learning
        </p>
        <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">
            ⚠️ <strong>Disclaimer:</strong> This tool is for educational and research purposes only. 
            Always consult a healthcare professional for medical advice.
        </p>
        <hr style="border-top: 1px solid #e2e8f0; margin: 1.5rem auto; width: 80%;">
        <p style="color: #94a3b8; font-size: 0.85rem;">
            Created with ❤️ by <strong>rexzea</strong><br>
            <a href="https://github.com/rexzea" target="_blank" style="color: #3b82f6;">GitHub</a> • 
            <a href="https://www.instagram.com/alzennora?igsh=Ym8wZHFjcWRxaWhx" target="_blank" style="color: #3b82f6;">Instagram</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
