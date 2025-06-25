import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')





# page config
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)










st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .predict-button {
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .predict-button:hover {
        background-color: #1B5E7D;
    }
    .high-risk {
        color: #E63946;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .low-risk {
        color: #2A9D8F;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)









# neural network model
class DiabetesPredictor(nn.Module):
    def __init__(self, input_size=8):
        super(DiabetesPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x










#cache the data loading and model training
@st.cache_data
def load_and_prepare_data():
    try:
        df = pd.read_csv('dataset/diabetes.csv')
        
        # data cleaning ---- replace 0s with median for certain columns
        columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in columns_to_replace:
            df[col] = df[col].replace(0, df[col].median())
        
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file 'diabetes.csv' not found. Please ensure it's in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.stop()








@st.cache_resource
def train_model(X_train, y_train, X_test, y_test):
    # init model
    model = DiabetesPredictor()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # convert to pytorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # training loop
    model.train()
    losses = []
    
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # evaluate the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs.numpy() > 0.5).astype(int)
        accuracy = accuracy_score(y_test, test_predictions)
    
    return model, accuracy, losses





def main():
    #header
    st.markdown('<h1 class="main-header">🩺 Diabetes Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict diabetes risk using advanced deep learning</p>', unsafe_allow_html=True)
    
    # load & prepare data
    df = load_and_prepare_data()
    # prepare features and target
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df[feature_columns]
    y = df['Outcome']
    # aplit and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # train the model
    model, accuracy, losses = train_model(X_train_scaled, y_train.values, X_test_scaled, y_test.values)
    # sdiebar user input
    st.sidebar.markdown('<h2 class="sub-header">Enter Your Information</h2>', unsafe_allow_html=True)
    # input fields
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 1, help='Number of times pregnant')
    glucose = st.sidebar.slider('Glucose Level', 0, 200, 120, help='Plasma glucose concentration')
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70, help='Diastolic blood pressure (mm Hg)')
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20, help='Triceps skin fold thickness (mm)')
    insulin = st.sidebar.slider('Insulin', 0, 846, 79, help='2-Hour serum insulin (mu U/ml)')
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 25.0, step=0.1, help='Body mass index (weight in kg/(height in m)^2)')
    diabetes_pedigree = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5, step=0.01, help='Diabetes pedigree function')
    age = st.sidebar.slider('Age', 21, 81, 30, help='Age in years')



  
    # prediction buttdon
    if st.sidebar.button('Predict Diabetes Risk', key='predict_btn'):
        # input data
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                               insulin, bmi, diabetes_pedigree, age]])
        input_scaled = scaler.transform(input_data)
      
        model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_scaled)
            prediction = model(input_tensor).item()
            
        #result
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Prediction Result")
        
        if prediction > 0.5:
            st.sidebar.markdown(f'<p class="high-risk">⚠High Risk of Diabetes</p>', unsafe_allow_html=True)
            st.sidebar.markdown(f"**Confidence:** {prediction:.2%}")
        else:
            st.sidebar.markdown(f'<p class="low-risk">Low Risk of Diabetes</p>', unsafe_allow_html=True)
            st.sidebar.markdown(f"**Confidence:** {(1-prediction):.2%}")
    
    #content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
        st.markdown("### Dataset Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        st.markdown("### Diabetes Distribution")
        outcome_counts = df['Outcome'].value_counts()
        fig_pie = px.pie(values=outcome_counts.values, 
                        names=['Non-Diabetic', 'Diabetic'],
                        title='Distribution of Diabetes Cases',
                        color_discrete_sequence=['#2A9D8F', '#E63946'])
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("### Feature Correlation Heatmap")
        correlation_matrix = df[feature_columns + ['Outcome']].corr()
        fig_heatmap = px.imshow(correlation_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               color_continuous_scale='RdBu',
                               title='Feature Correlation Matrix')
        st.plotly_chart(fig_heatmap, use_container_width=True)


  
    with col2:
        st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
        st.metric(
            label="Model Accuracy",
            value=f"{accuracy:.2%}",
            delta="High Performance"
        )
        st.markdown("### Training Loss")
        loss_df = pd.DataFrame({'Epoch': range(len(losses)), 'Loss': losses})
        fig_loss = px.line(loss_df, x='Epoch', y='Loss', 
                          title='Model Training Loss Over Time',
                          color_discrete_sequence=['#2E86AB'])
        st.plotly_chart(fig_loss, use_container_width=True)
        st.markdown("### Dataset Information")
        st.info(f"""
        **Total Samples:** {len(df)}
        
        **Features:** {len(feature_columns)}
        
        **Diabetic Cases:** {df['Outcome'].sum()} ({df['Outcome'].mean():.1%})
        
        **Non-Diabetic Cases:** {len(df) - df['Outcome'].sum()} ({(1-df['Outcome'].mean()):.1%})
        """)
        st.markdown("### Feature Importance")
        feature_importance = abs(df[feature_columns].corrwith(df['Outcome'])).sort_values(ascending=False)
        fig_importance = px.bar(x=feature_importance.values, 
                               y=feature_importance.index,
                               orientation='h',
                               title='Feature Importance (Correlation with Outcome)',
                               color_discrete_sequence=['#A23B72'])
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
  
    st.markdown("---")
    st.markdown('<h2 class="sub-header">Key Insights</h2>', unsafe_allow_html=True)





  

  
    col3, col4, col5 = st.columns(3)
    
    with col3:
        avg_glucose_diabetic = df[df['Outcome'] == 1]['Glucose'].mean()
        avg_glucose_non_diabetic = df[df['Outcome'] == 0]['Glucose'].mean()
        st.metric(
            label="Avg Glucose (Diabetic)",
            value=f"{avg_glucose_diabetic:.0f}",
            delta=f"{avg_glucose_diabetic - avg_glucose_non_diabetic:.0f} vs Non-Diabetic"
        )
    
    with col4:
        avg_bmi_diabetic = df[df['Outcome'] == 1]['BMI'].mean()
        avg_bmi_non_diabetic = df[df['Outcome'] == 0]['BMI'].mean()
        st.metric(
            label="Avg BMI (Diabetic)",
            value=f"{avg_bmi_diabetic:.1f}",
            delta=f"{avg_bmi_diabetic - avg_bmi_non_diabetic:.1f} vs Non-Diabetic"
        )
    
    with col5:
        avg_age_diabetic = df[df['Outcome'] == 1]['Age'].mean()
        avg_age_non_diabetic = df[df['Outcome'] == 0]['Age'].mean()
        st.metric(
            label="Avg Age (Diabetic)",
            value=f"{avg_age_diabetic:.0f}",
            delta=f"{avg_age_diabetic - avg_age_non_diabetic:.0f} vs Non-Diabetic"
        )
    



  
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #555; padding: 2rem 0 1rem 0; font-family: "Segoe UI", sans-serif; font-size: 15px;'>
        <p>Rexzea - <strong>Diabetes Predictor</strong> - Powered by PyTorch & Streamlit</p>
        <p><em>Note: This tool is for educational purposes only. Please consult healthcare professionals for medical advice.</em></p>
        <p>
            <a href="https://github.com/rexzea" target="_blank" style="text-decoration: none; color: #007acc; margin-right: 15px;">
                GitHub
            </a>
            |
            <a href="https://www.instagram.com/alzennora?igsh=Ym8wZHFjcWRxaWhx" target="_blank" style="text-decoration: none; color: #007acc; margin-left: 15px;">
                Instagram
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)






if __name__ == "__main__":
    main()
