import streamlit as st
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, mean_squared_error, confusion_matrix, 
                             precision_score, recall_score, f1_score, r2_score,
                             mean_absolute_error, classification_report)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import gc
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier


# Page configuration
st.set_page_config(page_title="XGBoost Prediction Tool", layout="wide")
st.title(' CyberShield Prediction Dashboard')

# Enable garbage collection to help with memory management
gc.enable()

def release_memory():
    """Force garbage collection to release memory"""
    gc.collect()

# Helper function to convert numpy types to Python native types
def convert_to_native_types(obj):
    """Convert numpy/pandas types to native Python types"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Enhanced datetime detection and handling
def is_datetime_column(series):
    """Check if a column contains datetime values"""
    if series.dtype == 'datetime64[ns]':
        return True
    
    # Check first non-null value
    if len(series.dropna()) > 0:
        first_val = series.dropna().iloc[0]
        if isinstance(first_val, str):
            try:
                # Check if it can be parsed as datetime
                pd.to_datetime(first_val)
                return True
            except:
                pass
    return False

# Handle IP addresses and network data
def process_ip_addresses(df, ip_columns):
    """Extract features from IP addresses"""
    for col in ip_columns:
        if col in df.columns:
            try:
                # Extract the first octet as a feature
                df[f"{col}_first_octet"] = df[col].apply(
                    lambda x: int(str(x).split('.')[0]) if isinstance(x, str) and '.' in str(x) else 0
                )
                
                # Count the total dots as a feature
                df[f"{col}_dot_count"] = df[col].apply(
                    lambda x: str(x).count('.') if isinstance(x, str) else 0
                )
            except Exception as e:
                st.warning(f"Error processing  {col}: {e}")
    
    return df

# Handle categorical values
def encode_categorical_columns(df, cat_columns):
    """
    Encode categorical columns using label encoding
    """
    result_maps = {}
    
    for col in cat_columns:
        if col in df.columns and df[col].dtype == 'object':
            # Create encoder
            le = LabelEncoder()
            # Fit and transform
            non_null_mask = df[col].notna()
            if non_null_mask.any():
                values = df.loc[non_null_mask, col]
                df.loc[non_null_mask, col] = le.fit_transform(values)
                # Store mapping
                result_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            # Fill NaN values with -1
            df[col] = df[col].fillna(-1)
    
    return df, result_maps

# Session state for tracking accuracy history
if 'accuracy_history' not in st.session_state:
    st.session_state.accuracy_history = []
    
if 'cyber_plots' not in st.session_state:
    st.session_state.cyber_plots = {}

if 'optimized_model' not in st.session_state:
    st.session_state.optimized_model = None

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

# Function to prepare dataframe for prediction
def prepare_dataframe_for_prediction(df, feature_names):
    """Ensures dataframe has all required features with proper types"""
    prediction_df = pd.DataFrame()
    
    # Ensure all expected features exist with proper types
    for feature in feature_names:
        if feature in df.columns:
            # Copy existing feature
            prediction_df[feature] = df[feature]
        else:
            # Add missing feature with default value
            prediction_df[feature] = 0
    
    # Handle IP addresses and categorical data
    for col in prediction_df.columns:
        if prediction_df[col].dtype == 'object':
            # Check if the column might be an IP address
            if any(str(x).count('.') == 3 for x in prediction_df[col].dropna().head()):
                # Handle IP addresses - convert to categorical
                prediction_df[col] = prediction_df[col].astype('category').cat.codes
            # Check if column appears to be categorical
            elif prediction_df[col].nunique() < 20:
                # Convert categorical strings to category codes
                prediction_df[col] = prediction_df[col].astype('category').cat.codes
            else:
                try:
                    # Try numeric conversion with coercion
                    prediction_df[col] = pd.to_numeric(prediction_df[col], errors='coerce')
                    # Fill NaN values with 0
                    prediction_df[col] = prediction_df[col].fillna(0)
                except:
                    # Fallback - use categorical encoding
                    prediction_df[col] = prediction_df[col].astype('category').cat.codes
    
    # Ensure correct column order to match model expectations
    prediction_df = prediction_df[feature_names]
    
    return prediction_df

# Memory-optimized function for cybersecurity plots
def generate_cybersecurity_plots(df, predictions, max_size=1000):
    """Generate AI-enhanced cybersecurity visualizations with memory limits"""
    plots = {}
    
    # If dataframe is too large, sample it to avoid memory issues
    if len(df) > max_size:
        st.warning(f"Dataset is large ({len(df)} rows). Sampling {max_size} rows for visualizations.")
        sample_idx = np.random.choice(len(df), size=max_size, replace=False)
        df_sample = df.iloc[sample_idx].copy()
        predictions_sample = predictions[sample_idx] if isinstance(predictions, np.ndarray) else predictions.iloc[sample_idx]
    else:
        df_sample = df.copy()
        predictions_sample = predictions
    
    # 1. Anomaly Detection Plot - With memory optimization
    try:
        # Create a copy of the dataframe with numeric columns only
        numeric_df = df_sample.select_dtypes(include=['number']).copy()
        
        if len(numeric_df.columns) >= 2:
            # Limit to 2 columns to reduce memory usage
            cols_to_use = numeric_df.columns[:2]
            numeric_df = numeric_df[cols_to_use].copy()
            
            # Use Isolation Forest for anomaly detection
            isolation_model = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = isolation_model.fit_predict(numeric_df)
            
            # Create minimal dataframe for plot
            plot_df = pd.DataFrame({
                'x': numeric_df.iloc[:, 0],
                'y': numeric_df.iloc[:, 1],
                'is_anomaly': anomaly_scores == -1
            })
            
            # Create plot with minimal data
            fig = px.scatter(
                plot_df, 
                x='x', 
                y='y',
                color='is_anomaly',
                color_discrete_map={True: 'red', False: 'blue'},
                title='AI Anomaly Detection (Sample)',
                labels={
                    'is_anomaly': 'Anomaly',
                    'x': numeric_df.columns[0],
                    'y': numeric_df.columns[1]
                }
            )
            plots['anomaly_detection'] = fig
            
            # Free memory
            del plot_df, numeric_df, anomaly_scores
            gc.collect()
    except Exception as e:
        st.error(f"Anomaly detection error: {e}")
    
    # 2. Threat Classification Distribution - Memory optimized
    try:
        # Create a simplified dataframe just for the counts
        counts = pd.Series(predictions_sample).value_counts().reset_index()
        counts.columns = ['Threat', 'Count']
        
        # Create pie chart with explicit column names
        fig = px.pie(
            counts,
            values='Count',
            names='Threat',
            title='Threat Classification (Sample)',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        plots['threat_classification'] = fig
        
        # Free memory
        del counts
        gc.collect()
    except Exception as e:
        pass  # Silently ignore errors here
    
    # Only try to create network heatmap if the dataset is small enough
    if len(df_sample) < 500:
        try:
            # Check if source/destination columns exist
            source_cols = [col for col in df_sample.columns if any(src_kw in col.lower() 
                                                            for src_kw in ['source', 'src', 'from'])]
            dest_cols = [col for col in df_sample.columns if any(dst_kw in col.lower() 
                                                           for dst_kw in ['dest', 'dst', 'to', 'target'])]
            
            if source_cols and dest_cols:
                src_col = source_cols[0]
                dst_col = dest_cols[0]
                
                # Ensure categorical values to reduce memory usage
                df_sample[src_col] = df_sample[src_col].astype('category')
                df_sample[dst_col] = df_sample[dst_col].astype('category')
                
                # Limit unique values to prevent memory explosion
                src_values = df_sample[src_col].value_counts().nlargest(10).index
                dst_values = df_sample[dst_col].value_counts().nlargest(10).index
                
                filtered_df = df_sample[
                    df_sample[src_col].isin(src_values) & 
                    df_sample[dst_col].isin(dst_values)
                ].copy()
                
                # Only create heatmap if we have sufficient data
                if len(filtered_df) > 10:
                    # Create connection matrix with limited values
                    connections = filtered_df.groupby([src_col, dst_col]).size().reset_index(name='count')
                    
                    # Only create pivot if data is small enough
                    pivot_table = connections.pivot_table(
                        values='count', 
                        index=src_col, 
                        columns=dst_col, 
                        fill_value=0
                    )
                    
                    # Convert to dense array to reduce memory usage
                    z_values = pivot_table.values
                    
                    # Create heatmap with minimal data
                    fig = px.imshow(
                        z_values,
                        labels=dict(x="Destination", y="Source", color="Count"),
                        x=list(pivot_table.columns),
                        y=list(pivot_table.index),
                        title='Network Connections (Top 10 Sources/Destinations)'
                    )
                    plots['network_heatmap'] = fig
                    
                    # Free memory
                    del filtered_df, connections, pivot_table, z_values
                    gc.collect()
        except Exception as e:
            # Skip silently - network visualizations are optional
            pass
    
    # Final garbage collection
    gc.collect()
    return plots

# Function to optimize XGBoost model
def optimize_xgboost_model(X, y, task_type='regression'):
    """Train and optimize an XGBoost model for better performance"""
    
    with st.spinner("Optimizing model... This may take a while."):
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Feature engineering
            status_text.text("Feature preprocessing...")
            
            # Scale features for better performance
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            progress_bar.progress(10)
            
            # Feature selection
            status_text.text("Performing feature selection...")
            feature_selector = SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=42) if task_type == 'regression' 
                else RandomForestClassifier(n_estimators=100, random_state=42)
            )
            feature_selector.fit(X_train_scaled, y_train)
            
            # Get selected features
            selected_features_mask = feature_selector.get_support()
            selected_features = X.columns[selected_features_mask].tolist()
            
            # Apply feature selection
            X_train_selected = feature_selector.transform(X_train_scaled)
            X_test_selected = feature_selector.transform(X_test_scaled)
            
            progress_bar.progress(30)
            status_text.text("Running hyperparameter optimization...")
            
            # Hyperparameter optimization
            if task_type == 'regression':
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_jobs=-1,
                    random_state=42
                )
                
                # Define parameter grid for regression
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2]
                }
            else:
                model = xgb.XGBClassifier(
                    n_jobs=-1,
                    random_state=42
                )
                
                # Define parameter grid for classification
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            
            # Use RandomizedSearchCV to speed up the process
            grid_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=10,  # Try 10 parameter combinations
                scoring='neg_mean_absolute_error' if task_type == 'regression' else 'accuracy',
                cv=3,
                n_jobs=-1,
                verbose=0,
                random_state=42
            )
            
            # Fit grid search
            grid_search.fit(X_train_selected, y_train)
            
            progress_bar.progress(70)
            status_text.text("Training final model with best parameters...")
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Retrain on full dataset with best parameters
            if task_type == 'regression':
             model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='mae',  # Add here instead
                n_jobs=-1,
                random_state=42)
            else:
                model = xgb.XGBClassifier(
                    eval_metric='error',  # Add here instead
                    n_jobs=-1,
                    random_state=42)

        
            # Make predictions
            y_pred = best_model.predict(X_test_selected)
            
            # Calculate metrics
            metrics = {}
            if task_type == 'regression':
                metrics['mae'] = mean_absolute_error(y_test, y_pred)
                metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                metrics['r2'] = r2_score(y_test, y_pred)
                
                # Calculate MAPE if no zeros in y_test
                if not np.any(y_test == 0):
                    metrics['mape'] = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            else:
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
            
            progress_bar.progress(100)
            status_text.text("Optimization complete!")
            
            # Create feature importance plot
            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': [f"Feature_{i}" for i in range(X_train_selected.shape[1])],
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance of Optimized Model'
                )
            else:
                fig = None
            
            # Return the optimized model, scaler, feature selector, and metrics
            model_package = {
                'model': best_model,
                'scaler': scaler,
                'feature_selector': feature_selector,
                'selected_features': selected_features,
                'best_params': grid_search.best_params_,
                'metrics': metrics,
                'importance_plot': fig
            }
            
            return model_package
            
        except Exception as e:
            st.error(f"Error in model optimization: {e}")
            return None

# Load model and get feature information
model = load_model()

if model:
    try:
        # Get feature names - handle different XGBoost versions
        if hasattr(model, 'get_booster'):
            feature_names = model.get_booster().feature_names
            st.success(f" Model loaded successfully with {len(feature_names)} features")
        else:
            # For older XGBoost versions
            feature_names = model.feature_names
            st.success(f" Model loaded successfully with {len(feature_names)} features")
    except Exception as e:
        st.warning(f"Could not extract feature names: {e}")
        feature_names = []
        st.info("Will attempt to use input features as provided")
else:
    st.error(" Failed to load model. Please check if 'xgboosttrained.pkl' exists and is valid.")
    st.stop()

# Create tabs for different prediction methods
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Single Prediction", 
    "Batch Prediction with Accuracy", 
    "Performance History", 
    "Cybersecurity AI Insights",
    "Model Optimization"
])

# Single Prediction Tab
with tab1:
    st.header("Single Prediction")
    st.subheader("Enter Input Parameters")
    
    # Create input form with columns for better organization
    input_data = {}
    
    # Create multiple columns for inputs to save space
    num_cols = 3
    col_list = st.columns(num_cols)
    
    # If we have feature names, use them to create the form
    if feature_names:
        for i, feature in enumerate(feature_names):
            # Distribute inputs across columns - Convert to Python int
            col_idx = int(i % num_cols)
            
            # Make reasonable guesses about input type based on feature name
            if any(keyword in feature.lower() for keyword in ['is_', '_is_', 'flag', 'binary', 'bool']):
                input_data[feature] = col_list[col_idx].selectbox(
                    f"{feature}", [0, 1], 
                    key=f"select_{feature}"
                )
            else:
                input_data[feature] = col_list[col_idx].number_input(
                    f"{feature}", value=0, 
                    key=f"input_{feature}"
                )
    else:
        # Create some default inputs if features are unknown
        st.warning("Feature names could not be extracted from model. Using default inputs.")
        default_features = ["Alerts/Warnings", "Proxy_Used", "Source_IP_Is_Private", "Packet Length"]
        
        for i, feature in enumerate(default_features):
            col_idx = int(i % num_cols)
            input_data[feature] = col_list[col_idx].number_input(f"{feature}", value=0)
    
    # Option to use optimized model if available
    use_optimized = False
    if 'optimized_model' in st.session_state and st.session_state.optimized_model:
        use_optimized = st.checkbox("Use optimized model for prediction", value=True)
    
    # Prediction button
    if st.button("Make Prediction", key="single_predict"):
        with st.spinner("Processing prediction..."):
            try:
                # Create a DataFrame for prediction
                input_df = pd.DataFrame([input_data])
                
                if use_optimized and st.session_state.optimized_model:
                    # Use the optimized model pipeline
                    opt_model = st.session_state.optimized_model
                    
                    # Apply the same preprocessing steps used during training
                    X_scaled = opt_model['scaler'].transform(input_df)
                    X_selected = opt_model['feature_selector'].transform(X_scaled)
                    
                    # Make prediction with optimized model
                    prediction = opt_model['model'].predict(X_selected)
                    
                    # Display model choice
                    st.info("Prediction made with the optimized model.")
                elif feature_names:
                    prediction_df = prepare_dataframe_for_prediction(input_df, feature_names)
                    prediction = model.predict(prediction_df)
                else:
                    # Different approaches for prediction
                    prediction_methods = [
                        lambda: model.predict(input_df, validate_features=False),
                        lambda: model.predict(input_df.values)
                    ]
                    
                    # Try different methods until one works
                    for pred_method in prediction_methods:
                        try:
                            prediction = pred_method()
                            break
                        except Exception:
                            continue
                    else:
                        raise Exception("All prediction methods failed")
                
                # Display results
                st.success("Prediction completed!")
                
                # Show prediction with nice formatting
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction Result", f"{prediction[0]:.4f}" if isinstance(prediction[0], (float, np.float32, np.float64)) else prediction[0])
                    
                with col2:
                    # If model can produce probability scores
                    try:
                        if use_optimized:
                            probas = opt_model['model'].predict_proba(X_selected)
                        else:
                            probas = model.predict_proba(prediction_df if feature_names else input_df)
                        highest_proba = np.max(probas[0]) * 100
                        st.metric("Confidence", f"{highest_proba:.2f}%")
                    except:
                        pass
                
                # Visualize result if it's a classification with probabilities
                try:
                    if use_optimized:
                        probas = opt_model['model'].predict_proba(X_selected)
                        classes = opt_model['model'].classes_ if hasattr(opt_model['model'], 'classes_') else [f"Class {i}" for i in range(len(probas[0]))]
                    else:
                        probas = model.predict_proba(prediction_df if feature_names else input_df)
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
                st.error(f"Prediction failed: {e}")
                st.info("Try adjusting the input values or check if the model expects different features.")
    
    # Free up memory
    release_memory()

# Batch Prediction Tab with Enhanced Accuracy Metrics
with tab2:
    st.header("Batch Prediction with Accuracy Metrics")
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
                st.metric("Rows", convert_to_native_types(df.shape[0]))
            with col2:
                st.metric("Columns", convert_to_native_types(df.shape[1]))
            with col3:
                st.metric("Missing Values", convert_to_native_types(df.isna().sum().sum()))
            
            # Feature diagnostics
            if feature_names:
                missing_features = set(feature_names) - set(df.columns)
                extra_features = set(df.columns) - set(feature_names)
                
                st.subheader("Feature Diagnostics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Expected Features", len(feature_names))
                    st.metric("Missing Features", len(missing_features))
                with col2:
                    st.metric("Provided Features", len(df.columns))
                    st.metric("Extra Features", len(extra_features))
                
                if missing_features:
                    with st.expander(f"Missing Features ({len(missing_features)})"):
                        st.write(", ".join(sorted(missing_features)))
                
                if extra_features:
                    with st.expander(f"Extra Features ({len(extra_features)})"):
                        st.write(", ".join(sorted(extra_features)))
            
            # Enhanced Target Column Selection - Fix int64 issue
            st.subheader("Target Column for Accuracy Calculation")
            potential_target_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                                   ['actual', 'target', 'label', 'ground_truth', 'truth', 'real', 'expected'])]
            
            # Convert index to Python int to fix int64 error
            default_target_idx = 0 
            if potential_target_cols:
                try:
                    idx = df.columns.get_indexer([potential_target_cols[0]])[0]
                    default_target_idx = int(idx) if idx >= 0 else 0
                except:
                    default_target_idx = 0
                    
            target_col = st.selectbox("Select target column for accuracy evaluation:", 
                                     options=df.columns, 
                                     index=default_target_idx,
                                     help="Choose the column containing actual/ground truth values to compare with predictions")
            
            # Check if target column contains datetime values
            if target_col and is_datetime_column(df[target_col]):
                st.warning(f"!'{target_col}' appears to contain datetime values. For accuracy metrics, consider using a non-datetime column.")
            
            # Determine prediction task type
            if target_col:
                unique_vals = df[target_col].nunique()
                if unique_vals <= 10:  # Heuristic for classification
                    task_type = st.radio("Task type:", ["Classification", "Regression"], 
                                       help="Classification for categorical outcomes, regression for continuous values")
                else:
                    task_type = st.radio("Task type:", ["Regression", "Classification"],
                                       help="Classification for categorical outcomes, regression for continuous values")
            
            # Choose prediction approach
            st.subheader("Prediction Settings")
            prediction_approach = st.radio(
                "Select prediction approach:",
                ["Auto-detect and fix features", "Use raw data (bypass feature validation)"]
            )
            
            # Option to use optimized model if available
            use_optimized = False
            if 'optimized_model' in st.session_state and st.session_state.optimized_model:
                use_optimized = st.checkbox("Use optimized model for prediction", value=True)
            
            # Run prediction
            if st.button("Run Batch Prediction with Accuracy Analysis", key="batch_predict"):
                with st.spinner("Processing batch prediction..."):
                    try:
                        # Apply feature engineering to IP addresses if present
                        ip_columns = [col for col in df.columns if 'ip' in col.lower() or 'address' in col.lower()]
                        if ip_columns:
                            df = process_ip_addresses(df, ip_columns)
                        
                        # Handle categorical data
                        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
                        if categorical_columns:
                            df, _ = encode_categorical_columns(df, categorical_columns)
                        
                        # Use the optimized model if selected
                        if use_optimized and st.session_state.optimized_model:
                            opt_model = st.session_state.optimized_model
                            
                            # Apply the optimized model pipeline
                            try:
                                # Make sure we have all required columns
                                required_cols = df.columns
                                missing_cols = set(required_cols) - set(df.columns)
                                
                                if missing_cols:
                                    for col in missing_cols:
                                        df[col] = 0
                                
                                # Apply preprocessing steps
                                X_scaled = opt_model['scaler'].transform(df)
                                X_selected = opt_model['feature_selector'].transform(X_scaled)
                                
                                # Make prediction
                                predictions = opt_model['model'].predict(X_selected)
                                
                                st.info("Using optimized model for prediction.")
                            except Exception as e:
                                st.error(f"Error using optimized model: {e}")
                                st.info("Falling back to original model...")
                                use_optimized = False
                        
                        # Use original model if not using optimized or optimized failed
                        if not use_optimized:
                            # Different approaches based on user selection
                            if prediction_approach == "Auto-detect and fix features":
                                # Ensure DataFrame has all required features
                                if feature_names:
                                    st.info("Preparing dataframe with all required features...")
                                    prediction_df = prepare_dataframe_for_prediction(df, feature_names)
                                    
                                    # Make prediction with properly prepared data
                                    predictions = model.predict(prediction_df)
                                else:
                                    # No feature names available, try direct prediction
                                    predictions = model.predict(df, validate_features=False)
                            
                            else:  # Use raw data
                                # Try different methods for prediction
                                st.warning("Using raw prediction without feature validation. This may fail if features don't match.")
                                try:
                                    predictions = model.predict(df, validate_features=False)
                                except Exception as e:
                                    st.error(f"First attempt failed: {e}")
                                    try:
                                        # Second attempt: try numpy array approach
                                        predictions = model.predict(df.values)
                                    except Exception as e:
                                        # Third attempt: try to fix by adding missing features
                                        if feature_names:
                                            st.info("Attempting to recover by adding missing features...")
                                            prediction_df = prepare_dataframe_for_prediction(df, feature_names)
                                            predictions = model.predict(prediction_df)
                                        else:
                                            raise Exception(f"Raw prediction failed: {e}")
                        
                        # Add predictions to results
                        results_df = df.copy()
                        results_df['Prediction'] = predictions
                        
                        # Show results
                        st.success("Batch prediction completed!")
                        st.subheader("Results")
                        st.write(results_df.head(10))
                        
                        # Generate cybersecurity plots
                        cyber_plots = generate_cybersecurity_plots(df, predictions)
                        
                        # Store plots for the Cybersecurity tab
                        st.session_state.cyber_plots = cyber_plots
                        
                        # Calculate comprehensive accuracy metrics if target column exists
                        if target_col and target_col in results_df.columns:
                            st.header("📈 Accuracy Metrics Dashboard")
                            
                            # Create expanders for detailed metrics sections
                            accuracy_expander = st.expander("Detailed Accuracy Metrics", expanded=True)
                            with accuracy_expander:
                                # Check if target column contains datetime values (FIXED)
                                if is_datetime_column(results_df[target_col]):
                                    st.warning(f"Column '{target_col}' contains datetime values and cannot be directly used for numeric comparison.")
                                    st.info("Converting to string for comparison.")
                                    
                                    # Convert both to strings for comparison
                                    y_true = results_df[target_col].astype(str)
                                    y_pred = results_df['Prediction'].astype(str)
                                    
                                    # Set task type to classification since we're comparing strings
                                    task_type = "Classification"
                                else:
                                    # For classification
                                    if task_type == "Classification":
                                        # Standard conversion attempt
                                        try:
                                            # First try converting everything to numeric
                                            y_true = pd.to_numeric(results_df[target_col], errors='coerce')
                                            y_pred = pd.to_numeric(results_df['Prediction'], errors='coerce')
                                            
                                            # If conversion failed (NaN values), convert both to strings instead
                                            if y_true.isna().any() or y_pred.isna().any():
                                                st.warning("Numeric conversion failed, using string comparison instead")
                                                y_true = results_df[target_col].astype(str)
                                                y_pred = results_df['Prediction'].astype(str)
                                        except:
                                            # Fallback to string comparison
                                            y_true = results_df[target_col].astype(str)
                                            y_pred = results_df['Prediction'].astype(str)
                                    
                                        # Calculate classification metrics
                                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                        
                                        with metrics_col1:
                                            acc = accuracy_score(y_true, y_pred)
                                            st.metric("Accuracy", f"{acc:.4f}")
                                        
                                        with metrics_col2:
                                            try:
                                                prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                                                st.metric("Precision", f"{prec:.4f}")
                                            except Exception as e:
                                                st.metric("Precision", "N/A")
                                                st.write(f"Error: {e}")
                                        
                                        with metrics_col3:
                                            try:
                                                rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                                                st.metric("Recall", f"{rec:.4f}")
                                            except Exception as e:
                                                st.metric("Recall", "N/A")
                                                st.write(f"Error: {e}")
                                        
                                        # F1 Score and support
                                        metrics_col1, metrics_col2 = st.columns(2)
                                        with metrics_col1:
                                            try:
                                                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                                                st.metric("F1 Score", f"{f1:.4f}")
                                            except Exception as e:
                                                st.metric("F1 Score", "N/A")
                                                st.write(f"Error: {e}")
                                        
                                        with metrics_col2:
                                            st.metric("Sample Count", f"{len(y_true)}")
                                        
                                        # Confusion Matrix (Fixed for mixed datatypes)
                                        try:
                                            st.subheader("Confusion Matrix")
                                            
                                            # Convert to same type for comparison (strings work well for both numeric and datetime)
                                            y_true_str = y_true.astype(str)
                                            y_pred_str = y_pred.astype(str)
                                            
                                            # Get unique classes (using string representation)
                                            classes = sorted(list(set(y_true_str.unique()) | set(y_pred_str.unique())))
                                            
                                            # Create class mapping for matrix
                                            class_to_idx = {cls: i for i, cls in enumerate(classes)}
                                            
                                            # Create confusion matrix using the mapping
                                            cm = np.zeros((len(classes), len(classes)), dtype=int)
                                            
                                            for true_val, pred_val in zip(y_true_str, y_pred_str):
                                                cm[class_to_idx[true_val]][class_to_idx[pred_val]] += 1
                                            
                                            # Create visualization
                                            fig = ff.create_annotated_heatmap(
                                                z=cm, 
                                                x=classes, 
                                                y=classes, 
                                                colorscale='Viridis',
                                                annotation_text=cm.astype(int)
                                            )
                                            fig.update_layout(
                                                title_text='Confusion Matrix', 
                                                xaxis_title="Predicted Label",
                                                yaxis_title="True Label"
                                            )
                                            st.plotly_chart(fig)
                                            
                                        except Exception as e:
                                            st.error(f"Could not generate confusion matrix: {e}")
                                            st.info("Try converting your labels to a consistent format before prediction")
                                        
                                        # Error Analysis
                                        st.subheader("Error Analysis")
                                        # Find misclassified samples
                                        results_df['Correct'] = y_true == y_pred
                                        error_df = results_df[~results_df['Correct']].copy()
                                        
                                        if not error_df.empty:
                                            misclassified_count = len(error_df)
                                            total_count = len(results_df)
                                            error_percentage = 100 * misclassified_count / total_count
                                            st.write(f"Found {misclassified_count} misclassified samples out of {total_count} ({error_percentage:.2f}%)")
                                            st.dataframe(error_df.head(10))
                                        else:
                                            st.success("Perfect classification! All samples correctly classified.")
                                    
                                    # For regression
                                    else:
                                        # Convert to numeric
                                        try:
                                            y_true = pd.to_numeric(results_df[target_col], errors='raise')
                                            y_pred = pd.to_numeric(results_df['Prediction'], errors='raise')
                                        except Exception as e:
                                            st.error(f"Cannot convert to numeric: {e}")
                                            st.info("For regression, both target and prediction must be numeric. Try using classification instead.")
                                            st.stop()
                                        
                                        # Calculate regression metrics
                                        metrics_col1, metrics_col2 = st.columns(2)
                                        
                                        with metrics_col1:
                                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                                            st.metric("RMSE", f"{rmse:.4f}")
                                            
                                            mae = mean_absolute_error(y_true, y_pred)
                                            st.metric("MAE", f"{mae:.4f}")
                                        
                                        with metrics_col2:
                                            r2 = r2_score(y_true, y_pred)
                                            st.metric("R² Score", f"{r2:.4f}")
                                            
                                            # Calculate MAPE if no zeros in y_true
                                            if not np.any(y_true == 0):
                                                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                                                st.metric("MAPE", f"{mape:.2f}%")
                                        
                                        # Residual Analysis
                                        st.subheader("Residual Analysis")
                                        results_df['Residual'] = y_true - y_pred
                                        results_df['Abs_Residual'] = np.abs(results_df['Residual'])
                                        
                                        # Residual plots
                                        fig1 = px.scatter(
                                            results_df, x="Prediction", y="Residual",
                                            title="Residual Plot (Actual - Predicted)",
                                            color="Abs_Residual",
                                            color_continuous_scale="RdYlGn_r"
                                        )
                                        st.plotly_chart(fig1)
                                        
                                        # Distribution of residuals
                                        fig2 = px.histogram(
                                            results_df, x="Residual",
                                            title="Distribution of Residuals",
                                            color_discrete_sequence=["blue"]
                                        )
                                        fig2.add_vline(x=0, line_dash="dash", line_color="red")
                                        st.plotly_chart(fig2)
                                        
                                        # Worst predictions
                                        st.subheader("Largest Prediction Errors")
                                        worst_predictions = results_df.sort_values('Abs_Residual', ascending=False).head(10)
                                        st.dataframe(worst_predictions)
                                        
                                        # Show error improvement if using optimized model
                                        if use_optimized and 'optimized_model' in st.session_state and st.session_state.optimized_model:
                                            st.subheader("Optimization Impact")
                                            opt_metrics = st.session_state.optimized_model['metrics']
                                            
                                            # Compare current performance with optimization metrics
                                            comparison_cols = st.columns(3)
                                            
                                            with comparison_cols[0]:
                                                mae_diff = opt_metrics.get('mae', 0) - mae
                                                mae_pct = (mae_diff / mae) * 100 if mae != 0 else 0
                                                st.metric("MAE Improvement", 
                                                         f"{abs(mae_diff):.4f}", 
                                                         f"{mae_pct:.1f}%" if mae_diff < 0 else f"-{mae_pct:.1f}%")
                                            
                                            with comparison_cols[1]:
                                                rmse_diff = opt_metrics.get('rmse', 0) - rmse
                                                rmse_pct = (rmse_diff / rmse) * 100 if rmse != 0 else 0
                                                st.metric("RMSE Improvement", 
                                                         f"{abs(rmse_diff):.4f}", 
                                                         f"{rmse_pct:.1f}%" if rmse_diff < 0 else f"-{rmse_pct:.1f}%")
                                            
                                            with comparison_cols[2]:
                                                r2_diff = r2 - opt_metrics.get('r2', 0)
                                                st.metric("R² Improvement", 
                                                         f"{abs(r2_diff):.4f}", 
                                                         f"{r2_diff:.1f}" if r2_diff > 0 else f"-{abs(r2_diff):.1f}")
                                                
                                        # Model improvement suggestions
                                        st.subheader("Model Improvement Suggestions")
                                        
                                        # Different suggestions based on metric values
                                        if r2 < 0:
                                            st.error("Your model has serious issues (negative R²). Consider:")
                                            st.write("- Complete retraining with feature engineering")
                                            st.write("- Using the 'Model Optimization' tab to create an improved model")
                                            st.write("- Checking for data quality issues or outliers")
                                            st.write("- Transforming target variable (log transform, etc.)")
                                        elif r2 < 0.3:
                                            st.warning("Your model has poor accuracy (low R²). Consider:")
                                            st.write("- Adding more relevant features")
                                            st.write("- Trying different algorithms (Random Forest, Gradient Boosting)")
                                            st.write("- Increasing XGBoost's complexity (depth, estimators)")
                                        elif r2 < 0.7:
                                            st.info("Your model has moderate accuracy. To improve:")
                                            st.write("- Fine-tune hyperparameters")
                                            st.write("- Create interaction features")
                                            st.write("- Try ensemble methods")
                                        else:
                                            st.success("Your model has good accuracy, but can still improve:")
                                            st.write("- Focus on reducing outlier errors")
                                            st.write("- Consider stacked models for final improvements")
                            
                            # Add current accuracy to history
                            run_metrics = {
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'filename': uploaded_file.name,
                                'samples': convert_to_native_types(len(results_df)),
                                'task_type': task_type,
                                'model_type': 'Optimized' if use_optimized else 'Original'
                            }
                            
                            if task_type == "Classification":
                                if 'acc' in locals():
                                    run_metrics['accuracy'] = acc
                                if 'prec' in locals():
                                    run_metrics['precision'] = prec
                                if 'rec' in locals():
                                    run_metrics['recall'] = rec
                                if 'f1' in locals():
                                    run_metrics['f1'] = f1
                            else:
                                if 'rmse' in locals():
                                    run_metrics['rmse'] = rmse
                                if 'mae' in locals():
                                    run_metrics['mae'] = mae
                                if 'r2' in locals():
                                    run_metrics['r2'] = r2
                                if 'mape' in locals():
                                    run_metrics['mape'] = mape
                            
                            st.session_state.accuracy_history.append(run_metrics)
                            
                            # Save run with a name
                            with st.expander("Save this accuracy run"):
                                run_name = st.text_input("Run name", value=f"Run {len(st.session_state.accuracy_history)}")
                                if st.button("Save Run Metrics"):
                                    run_metrics['name'] = run_name
                                    st.success(f"Saved run as '{run_name}'")
                        
                        # Generate prediction summary visualization
                        st.subheader("Prediction Distribution")
                        
                        # For classification
                        if task_type == "Classification":
                            # Count prediction values - convert to Python native types
                            prediction_counts = results_df['Prediction'].value_counts().reset_index()
                            
                            # Create visualization with correct column names
                            fig = px.pie(
                                prediction_counts, 
                                values=prediction_counts.columns[1],  # Second column (values/counts)
                                names=prediction_counts.columns[0],   # First column (categories/index)
                                title='Distribution of Predictions',
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )
                            st.plotly_chart(fig)
                        else:
                            # For regression - convert to Python native types
                            fig = px.histogram(
                                results_df,
                                x='Prediction',
                                title='Distribution of Predictions',
                                color_discrete_sequence=['blue']
                            )
                            st.plotly_chart(fig)
                        
                        # Add download button for results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"prediction_results_{uploaded_file.name}",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Batch prediction failed: {e}")
                        st.info("The model requires specific features. Try using the 'Auto-detect and fix features' option.")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    
    # Release memory
    release_memory()

# Performance History Tab
with tab3:
    st.header("📈 Performance History")
    
    if st.session_state.accuracy_history:
        # Convert history to dataframe
        history_df = pd.DataFrame(st.session_state.accuracy_history)
        st.write(f"Performance history from {len(history_df)} prediction runs:")
        st.dataframe(history_df)
        
        # Plot history over time
        if len(history_df) > 1:
            st.subheader("Performance Trends")
            
            # Create separate views for classification and regression
            class_runs = history_df[history_df['task_type'] == 'Classification']
            reg_runs = history_df[history_df['task_type'] == 'Regression']
            
            # Plot classification metrics
            if not class_runs.empty and len(class_runs) > 1:
                st.write("Classification Performance Trend")
                # Convert metrics columns to numeric
                for col in ['accuracy', 'precision', 'recall', 'f1']:
                    if col in class_runs.columns:
                        class_runs[col] = pd.to_numeric(class_runs[col], errors='coerce')
                
                # Get only columns that actually exist
                y_cols = [col for col in ['accuracy', 'precision', 'recall', 'f1'] if col in class_runs.columns]
                
                if y_cols:  # Only plot if we have metrics
                    fig = px.line(
                        class_runs,
                        x=range(len(class_runs)),
                        y=y_cols,
                        title="Classification Metrics Over Time",
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Run Number", yaxis_title="Score")
                    st.plotly_chart(fig)
            
            # Plot regression metrics
            if not reg_runs.empty and len(reg_runs) > 1:
                st.write("Regression Performance Trend")
                # Convert metrics columns to numeric
                for col in ['rmse', 'mae', 'r2', 'mape']:
                    if col in reg_runs.columns:
                        reg_runs[col] = pd.to_numeric(reg_runs[col], errors='coerce')
                
                # Compare original vs optimized model if both exist
                if 'model_type' in reg_runs.columns and reg_runs['model_type'].nunique() > 1:
                    st.subheader("Original vs Optimized Model Performance")
                    
                    # Create comparison visualizations
                    for metric in ['rmse', 'mae', 'r2']:
                        if metric in reg_runs.columns:
                            fig = px.box(
                                reg_runs,
                                x='model_type',
                                y=metric,
                                title=f"{metric.upper()} by Model Type",
                                color='model_type'
                            )
                            st.plotly_chart(fig)
                
                # Plot R2 separately (higher is better)
                if 'r2' in reg_runs.columns:
                    fig1 = px.line(
                        reg_runs,
                        x=range(len(reg_runs)),
                        y=['r2'],
                        title="R² Score Over Time",
                        markers=True
                    )
                    fig1.update_layout(xaxis_title="Run Number", yaxis_title="Score")
                    st.plotly_chart(fig1)
                
                # Plot error metrics (lower is better)
                error_cols = [col for col in ['rmse', 'mae', 'mape'] if col in reg_runs.columns]
                if error_cols:
                    fig2 = px.line(
                        reg_runs,
                        x=range(len(reg_runs)),
                        y=error_cols,
                        title="Error Metrics Over Time",
                        markers=True
                    )
                    fig2.update_layout(xaxis_title="Run Number", yaxis_title="Error")
                    st.plotly_chart(fig2)
        
        # Option to clear history
        if st.button("Clear Performance History"):
            st.session_state.accuracy_history = []
            st.experimental_rerun()
    else:
        st.info("No performance history available yet. Run some batch predictions with accuracy analysis to populate this section.")
    
    # Release memory
    release_memory()

# Cybersecurity AI Insights Tab
with tab4:
    st.header("🔒 Cybersecurity AI Insights")
    
    try:
        if 'cyber_plots' in st.session_state and st.session_state.cyber_plots:
            plots = st.session_state.cyber_plots
            
            # Display plots individually to manage memory better
            if 'anomaly_detection' in plots:
                st.subheader("AI Anomaly Detection")
                st.plotly_chart(plots['anomaly_detection'], use_container_width=True)
                st.markdown("""
                **What this shows:** Points marked in red are potential security anomalies detected by our AI model.
                These represent unusual patterns that deviate from normal behavior and may indicate security threats.
                
                **Actions to take:**
                - Investigate red points (anomalies) for potential security breaches
                - Review security logs for these specific data points
                - Consider adding these patterns to your security monitoring
                """)
                # Release memory after each heavy visualization
                release_memory()
            
            if 'threat_classification' in plots:
                st.subheader("Threat Classification Distribution")
                st.plotly_chart(plots['threat_classification'], use_container_width=True)
                st.markdown("""
                **What this shows:** The distribution of different security threats detected in your dataset.
                
                **Actions to take:**
                - Focus security resources on the most common threat types
                - Develop mitigation strategies for the prevalent threats
                - Track changes in this distribution over time to identify emerging threats
                """)
                # Release memory
                release_memory()
            
            # Only attempt to display the network heatmap if it exists
            if 'network_heatmap' in plots:
                try:
                    st.subheader("Network Connection Patterns")
                    st.plotly_chart(plots['network_heatmap'], use_container_width=True)
                    st.markdown("""
                    **What this shows:** The intensity of connections between different network sources and destinations.
                    Brighter colors indicate more frequent connections.
                    
                    **Actions to take:**
                    - Identify unusual connection patterns between sources and destinations
                    - Look for unexpected hot spots that might indicate data exfiltration
                    - Monitor high-traffic routes for potential bandwidth abuse or DDoS vectors
                    """)
                except MemoryError:
                    st.error("Insufficient memory to display network heatmap.")
                except Exception as e:
                    st.error(f"Error displaying network heatmap: {e}")
                # Release memory
                release_memory()
        else:
            st.info("No cybersecurity insights available yet. Run a batch prediction first to generate AI-powered visualizations.")
            
            # Show sample visualizations
            st.subheader("Sample Cybersecurity AI Visualizations")
            st.write("Here are examples of the insights that will be generated when you upload data:")
            
            # Sample tabs for different visualization types
            sample_tabs = st.tabs(["Anomaly Detection", "Threat Classification", "Network Analysis"])
            
            with sample_tabs[0]:
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*-dTk5Wy5dAr9NwYr8GBY1Q.png", 
                        caption="AI-powered anomaly detection helps identify unusual patterns that may indicate security threats")
            
            with sample_tabs[1]:
                st.image("https://cdn-blog.seculore.com/wp-content/uploads/2018/11/Image-8-600x288.png",
                       caption="Threat classification helps understand the distribution of different attack types")
            
            with sample_tabs[2]:
                st.image("https://miro.medium.com/v2/resize:fit:1158/1*aTiiBWsKYuCUQgmTS5qJWg.png",
                       caption="Network traffic analysis reveals connection patterns and potential data exfiltration paths")
    except Exception as e:
        st.error(f"Error in cybersecurity insights tab: {e}")
    
    # Release memory
    release_memory()

# Model Optimization Tab
with tab5:
    st.header("Model Optimization (Improve Accuracy)")
    
    st.write("This tab helps you create an optimized model to improve prediction accuracy.")
    
    st.info("""
    ### How to use this tab:
    1. Upload your training data (CSV file with features and target)
    2. Select your target column
    3. Choose the optimization options
    4. Click "Train Optimized Model"
    5. Use the optimized model in prediction tabs
    """)
    
    # Upload training data
    train_file = st.file_uploader("Upload training data (CSV)", type="csv", key="train_data")
    
    if train_file is not None:
        try:
            # Load training data
            train_df = pd.read_csv(train_file)
            
            st.write("Training Data Preview:")
            st.write(train_df.head())
            
            # Select target column
            target_col = st.selectbox(
                "Select target column",
                options=train_df.columns,
                help="This is the column you want to predict"
            )
            
            # Task type
            task_type = st.radio(
                "Task type",
                ["Regression", "Classification"],
                help="Regression for predicting numbers, Classification for categories"
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                # Feature selection
                feature_selection = st.checkbox("Use feature selection", value=True,
                                              help="Automatically select the most important features")
                
                # Hyperparameter tuning
                hyperparam_tuning = st.checkbox("Perform hyperparameter tuning", value=True,
                                              help="Find optimal model parameters")
                
                # Feature engineering options
                handle_categorical = st.checkbox("Auto-encode categorical features", value=True,
                                               help="Convert text categories to numbers")
                
                handle_ip = st.checkbox("Extract features from IP addresses", value=True,
                                      help="Get useful features from IP address columns")
            
            # Button to train optimized model
            if st.button("Train Optimized Model"):
                with st.spinner("Training optimized model... This may take several minutes"):
                    try:
                        # Prepare data
                        # 1. Handle categorical columns
                        categorical_columns = [col for col in train_df.columns if train_df[col].dtype == 'object']
                        if categorical_columns and handle_categorical:
                            train_df, _ = encode_categorical_columns(train_df, categorical_columns)
                        
                        # 2. Handle IP addresses
                        ip_columns = [col for col in train_df.columns if 'ip' in col.lower() or 'address' in col.lower()]
                        if ip_columns and handle_ip:
                            train_df = process_ip_addresses(train_df, ip_columns)
                        
                        # 3. Drop rows with missing values in target
                        train_df = train_df.dropna(subset=[target_col])
                        
                        # 4. Fill remaining missing values
                        train_df = train_df.fillna(0)
                        
                        # Prepare X and y
                        y = train_df[target_col]
                        X = train_df.drop(columns=[target_col])
                        
                        # Run model optimization
                        optimized_model = optimize_xgboost_model(X, y, task_type=task_type.lower())
                        
                        if optimized_model:
                            # Store optimized model
                            st.session_state.optimized_model = optimized_model
                            
                            # Display results
                            st.success("Model optimization complete!")
                            
                            # Show metrics
                            metrics = optimized_model['metrics']
                            st.subheader("Optimized Model Performance")
                            
                            col1, col2 = st.columns(2)
                            
                            if task_type == "Regression":
                                with col1:
                                    st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                                    st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                                
                                with col2:
                                    st.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
                                    if 'mape' in metrics:
                                        st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
                            else:
                                with col1:
                                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                                    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                                
                                with col2:
                                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                                    st.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
                            
                            # Show feature importance
                            if 'importance_plot' in optimized_model:
                                st.subheader("Feature Importance")
                                st.plotly_chart(optimized_model['importance_plot'])
                            
                            # Show best parameters
                            st.subheader("Best Parameters")
                            st.json(optimized_model['best_params'])
                            
                            # Show selected features
                            st.subheader("Selected Features")
                            st.write(f"Selected {len(optimized_model['selected_features'])} out of {X.shape[1]} features")
                            st.write(optimized_model['selected_features'])
                            
                            # Instructions for using optimized model
                            st.info("""
                            ### How to use the optimized model:
                            1. Go to the 'Single Prediction' or 'Batch Prediction' tab
                            2. Check the "Use optimized model" checkbox
                            3. Run your predictions
                            
                            The optimized model will be used instead of the original model.
                            """)
                    except Exception as e:
                        st.error(f"Error during model optimization: {e}")
                        st.info("Try with a different dataset or adjust the advanced options.")
        except Exception as e:
            st.error(f"Error loading training data: {e}")
    
    # Release memory
    release_memory()

# Add a note about model and features at the bottom
if model and feature_names:
    with st.expander("Model Information"):
        st.write("Model type:", type(model).__name__)
        st.write(f"Expected feature count: {len(feature_names)}")
        
        # Show only a subset of features to avoid memory issues
        st.write("Sample of features (first 20):")
        for feature in sorted(feature_names)[:20]:
            st.write(f"- {feature}")
        
        if len(feature_names) > 20:
            remaining = len(feature_names) - 20
            st.write(f"... and {remaining} more features")

# Force final memory cleanup
release_memory()
