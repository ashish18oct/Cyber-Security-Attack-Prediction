import streamlit as st
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
import plotly.express as px
import os

# Page configuration
st.set_page_config(page_title="XGBoost Prediction Tool", layout="wide")
st.title('🔮 XGBoost Model Prediction Dashboard')

# Function to load the model with proper error handling
@st.cache_resource
def load_model():
    try:
        with open('xgboosttrained.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model and get feature information
model = load_model()

if model:
    try:
        # Get feature names - handle different XGBoost versions
        if hasattr(model, 'get_booster'):
            feature_names = model.get_booster().feature_names
            st.success(f"✅ Model loaded successfully with {len(feature_names)} features")
        else:
            # For older XGBoost versions
            feature_names = model.feature_names
            st.success(f"✅ Model loaded successfully with {len(feature_names)} features")
    except Exception as e:
        st.warning(f"Could not extract feature names: {e}")
        feature_names = []
        st.info("Will attempt to use input features as provided")
else:
    st.error("❌ Failed to load model. Please check if 'xgboosttrained.pkl' exists and is valid.")
    st.stop()

# Create tabs for different prediction methods
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction from CSV"])

# Single Prediction Tab
with tab1:
    st.header("📝 Single Prediction")
    st.subheader("Enter Input Parameters")
    
    # Create input form with columns for better organization
    input_data = {}
    
    # Create multiple columns for inputs to save space
    num_cols = 3
    col_list = st.columns(num_cols)
    
    # If we have feature names, use them to create the form
    if feature_names:
        for i, feature in enumerate(feature_names):
            # Distribute inputs across columns
            col_idx = i % num_cols
            
            # Make reasonable guesses about input type based on feature name
            if any(keyword in feature.lower() for keyword in ['is_', '_is_', 'flag', 'binary', 'bool']):
                input_data[feature] = col_list[col_idx].selectbox(
                    f"{feature}", [0, 1], key=f"select_{feature}"
                )
            else:
                input_data[feature] = col_list[col_idx].number_input(
                    f"{feature}", value=0, key=f"input_{feature}"
                )
    else:
        # Create some default inputs if features are unknown
        st.warning("Feature names could not be extracted from model. Using default inputs.")
        default_features = ["Alerts/Warnings", "Proxy_Used", "Source_IP_Is_Private", "Packet Length"]
        
        for i, feature in enumerate(default_features):
            col_idx = i % num_cols
            input_data[feature] = col_list[col_idx].number_input(f"{feature}", value=0)
    
    # Prediction button
    if st.button("Make Prediction", key="single_predict"):
        with st.spinner("Processing prediction..."):
            try:
                # Create a DataFrame for prediction
                input_df = pd.DataFrame([input_data])
                
                # Different approaches for prediction (try multiple methods if needed)
                prediction_methods = [
                    lambda: model.predict(input_df, validate_features=False),
                    lambda: model.predict(input_df[feature_names] if feature_names else input_df),
                    lambda: model.predict(input_df.values)
                ]
                
                # Try different prediction methods until one works
                for pred_method in prediction_methods:
                    try:
                        prediction = pred_method()
                        break
                    except Exception:
                        continue
                else:
                    raise Exception("All prediction methods failed")
                
                # Display results
                st.success("✅ Prediction completed!")
                
                # Show prediction with nice formatting
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction Result", f"{prediction[0]:.4f}" if isinstance(prediction[0], (float, np.float32, np.float64)) else prediction[0])
                    
                with col2:
                    # If model can produce probability scores
                    try:
                        probas = model.predict_proba(input_df)
                        highest_proba = np.max(probas[0]) * 100
                        st.metric("Confidence", f"{highest_proba:.2f}%")
                    except:
                        pass
                
                # Visualize result if it's a classification with probabilities
                try:
                    probas = model.predict_proba(input_df)
                    classes = model.classes_ if hasattr(model, 'classes_') else [f"Class {i}" for i in range(len(probas[0]))]
                    
                    proba_df = pd.DataFrame({
                        'Class': classes,
                        'Probability': probas[0]
                    })
                    
                    fig = px.bar(
                        proba_df, 
                        x='Class', 
                        y='Probability',
                        title='Prediction Probabilities by Class',
                        color='Probability',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig)
                except:
                    pass
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.info("Try adjusting the input values or check if the model expects different features.")

# Batch Prediction Tab
with tab2:
    st.header("📊 Batch Prediction")
    st.subheader("Upload CSV File for Prediction")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Display file info and preview
        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
        st.write(file_details)
        
        # Read CSV and show preview
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.write(df.head())
            
            # Show statistics about the data
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isna().sum().sum())
            
            # Choose prediction approach
            st.subheader("Prediction Settings")
            prediction_approach = st.radio(
                "Select prediction approach:",
                ["Auto-detect features", "Map columns to model features", "Use raw data (bypass feature validation)"]
            )
            
            # Run prediction
            if st.button("Run Batch Prediction", key="batch_predict"):
                with st.spinner("Processing batch prediction..."):
                    try:
                        # Different approaches based on user selection
                        if prediction_approach == "Auto-detect features":
                            # Check for feature match and adjust automatically
                            if feature_names:
                                missing_features = set(feature_names) - set(df.columns)
                                extra_features = set(df.columns) - set(feature_names)
                                
                                if missing_features:
                                    st.warning(f"Adding missing features: {', '.join(missing_features)}")
                                    for feature in missing_features:
                                        df[feature] = 0
                                
                                if extra_features:
                                    st.info(f"Ignoring extra features: {', '.join(extra_features)}")
                                
                                # Use only the needed features in the right order
                                prediction_df = df[feature_names]
                                predictions = model.predict(prediction_df)
                            else:
                                # No feature names available, try direct prediction
                                predictions = model.predict(df, validate_features=False)
                        
                        elif prediction_approach == "Map columns to model features":
                            if feature_names:
                                # Create mapping of available columns to model features
                                mapping = {}
                                for i, col in enumerate(df.columns):
                                    if i < len(feature_names):
                                        mapping[col] = feature_names[i]
                                
                                # Create mapped dataframe
                                mapped_df = df.rename(columns=mapping)
                                
                                # Add missing columns with zeros
                                for feature in feature_names:
                                    if feature not in mapped_df.columns:
                                        mapped_df[feature] = 0
                                
                                # Predict using mapped features
                                prediction_df = mapped_df[feature_names]
                                predictions = model.predict(prediction_df)
                            else:
                                st.error("Cannot map columns - feature names unknown")
                                st.stop()
                        
                        else:  # Use raw data
                            # Try different methods for prediction
                            try:
                                predictions = model.predict(df, validate_features=False)
                            except:
                                try:
                                    predictions = model.predict(df.values)
                                except Exception as e:
                                    st.error(f"Raw prediction failed: {e}")
                                    st.stop()
                        
                        # Add predictions to results
                        results_df = df.copy()
                        results_df['Prediction'] = predictions
                        
                        # Show results
                        st.success("✅ Batch prediction completed!")
                        st.subheader("Results")
                        st.write(results_df.head(10))
                        
                        # Generate summary statistics on predictions
                        st.subheader("Prediction Summary")
                        
                        # For classification problems
                        try:
                            # Count prediction values
                            prediction_counts = results_df['Prediction'].value_counts().reset_index()
                            prediction_counts.columns = ['Class', 'Count']
                            
                            # Create visualization
                            fig = px.pie(
                                prediction_counts, 
                                values='Count', 
                                names='Class',
                                title='Distribution of Predictions',
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            st.plotly_chart(fig)
                        except:
                            # For regression problems
                            try:
                                fig = px.histogram(
                                    results_df,
                                    x='Prediction',
                                    title='Distribution of Predictions',
                                    color_discrete_sequence=['blue']
                                )
                                st.plotly_chart(fig)
                            except:
                                st.info("Could not create prediction visualization")
                        
                        # Add download button for results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"prediction_results_{uploaded_file.name}",
                            mime="text/csv"
                        )
                        
                        # Calculate accuracy if target column exists
                        if 'Actual' in results_df.columns or 'Target' in results_df.columns or 'Label' in results_df.columns:
                            st.subheader("Accuracy Metrics")
                            
                            # Find target column
                            target_col = None
                            for col_name in ['Actual', 'Target', 'Label']:
                                if col_name in results_df.columns:
                                    target_col = col_name
                                    break
                            
                            if target_col:
                                # For classification
                                try:
                                    acc = accuracy_score(results_df[target_col], results_df['Prediction'])
                                    st.metric("Accuracy", f"{acc:.4f}")
                                except:
                                    # For regression
                                    try:
                                        rmse = np.sqrt(mean_squared_error(results_df[target_col], results_df['Prediction']))
                                        st.metric("RMSE", f"{rmse:.4f}")
                                    except:
                                        st.info("Could not calculate accuracy metrics")
                        
                    except Exception as e:
                        st.error(f"❌ Batch prediction failed: {e}")
                        st.info("Check that your CSV has the correct format and contains the required features.")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

# Add a note about model and features at the bottom
if model and feature_names:
    with st.expander("Model Information"):
        st.write("Model type:", type(model).__name__)
        st.write("Expected features:")
        # Format feature list in columns for better visibility
        feature_cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            feature_cols[i % 3].write(f"- {feature}")
