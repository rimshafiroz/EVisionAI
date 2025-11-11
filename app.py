import streamlit as st
from src.data_loader import load_data, preprocess_data
from src.eda import plot_correlation, plot_sales_by_brand, plot_price_distribution
from src.model import train_price_model, forecast_sales
from src.chatbot import chatbot
import pandas as pd
import os

st.set_page_config(page_title="EVisionAI Dashboard", page_icon="🚗", layout="wide")

st.title("🚗 EVisionAI Dashboard")
st.markdown("### Electric Vehicle Sales & Adoption Analytics Platform")

# Load and preprocess data
try:
    df = load_data()
    df = preprocess_data(df)
    st.sidebar.success(f"✅ Data loaded: {len(df)} records")
except FileNotFoundError as e:
    st.error(f"❌ {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Sidebar options
option = st.sidebar.selectbox("Choose Module", ["Sales Forecast", "Price Prediction", "EDA", "Chatbot"])

# Display data info in sidebar
if st.sidebar.checkbox("Show Data Info"):
    st.sidebar.write(f"**Rows:** {len(df)}")
    st.sidebar.write(f"**Columns:** {len(df.columns)}")
    st.sidebar.write(f"**Column names:**")
    for col in df.columns.tolist()[:15]:
        st.sidebar.write(f"- {col}")
    if len(df.columns) > 15:
        st.sidebar.write(f"... and {len(df.columns) - 15} more")

if option == "Sales Forecast":
    st.subheader("📈 Sales Forecasting")
    try:
        sales_yearly, future_years, forecast = forecast_sales(df)
        
        st.write("### Yearly EV Sales Trend")
        st.line_chart(sales_yearly.set_index('year')['sales'])
        
        st.write("### Forecast for Next Years")
        forecast_df = pd.DataFrame({
            "Year": future_years,
            "Forecasted Sales": forecast
        })
        forecast_df['Forecasted Sales'] = forecast_df['Forecasted Sales'].round(0).astype(int)
        st.dataframe(forecast_df, use_container_width=True)
        
        # Display forecast in a more visual way
        if len(forecast) > 0:
            cols = st.columns(min(len(forecast), 2))
            for idx, (col, year, forecast_value) in enumerate(zip(cols, future_years, forecast)):
                with col:
                    st.metric(f"Forecast {int(year)}", f"{forecast_value:,.0f} units")
    except Exception as e:
        st.error(f"Error in sales forecasting: {str(e)}")

elif option == "Price Prediction":
    st.subheader("💰 Price Prediction")
    try:
        # Check if model can be trained (store model in session state to avoid retraining)
        if 'price_model' not in st.session_state or 'price_rmse' not in st.session_state:
            with st.spinner("Training price prediction model..."):
                model, rmse = train_price_model(df)
                st.session_state.price_model = model
                st.session_state.price_rmse = rmse
        else:
            model = st.session_state.price_model
            rmse = st.session_state.price_rmse
        
        st.success(f"✅ Model trained successfully! RMSE: ${rmse:,.2f}")
        
        st.write("### Predict EV Price")
        
        col1, col2 = st.columns(2)
        with col1:
            battery = st.number_input("Battery (kWh)", min_value=10, max_value=200, value=50, step=5)
            range_km = st.number_input("Range (km)", min_value=100, max_value=800, value=300, step=50)
        with col2:
            year = st.number_input("Year", min_value=2015, max_value=2025, value=2023, step=1)
            acceleration = st.number_input("0-100 km/h Acceleration (s)", min_value=2.0, max_value=20.0, value=7.5, step=0.5)
        
        # Find brand column
        brand_col = None
        for col in df.columns:
            col_lower = col.lower()
            if brand_col is None and ('brand' in col_lower or 'manufacturer' in col_lower):
                brand_col = col
                break
        
        if brand_col:
            brands = sorted([str(b) for b in df[brand_col].dropna().unique() if str(b) != 'nan'])
            if brands:
                brand = st.selectbox("Brand", brands)
            else:
                st.warning("No valid brands found in dataset")
                brand = "Unknown"
        else:
            st.warning("Brand column not found in dataset")
            brand = "Unknown"
        
        if st.button("Predict Price", type="primary"):
            try:
                # Prepare input for model - match training format
                input_data = {
                    "battery_kwh": [battery],
                    "range_km": [range_km],
                    "year": [year],
                    "acceleration": [acceleration],
                    "brand": [brand]
                }
                input_df = pd.DataFrame(input_data)
                
                # Encode categorical variables same way as training
                input_df = pd.get_dummies(input_df, columns=['brand'], drop_first=True, dtype=int)
                
                # Ensure same columns as training data
                if hasattr(model, 'feature_names_in_'):
                    model_features = model.feature_names_in_
                    # Add missing columns with 0
                    for col in model_features:
                        if col not in input_df.columns:
                            input_df[col] = 0
                    # Reorder columns to match model
                    input_df = input_df[model_features]
                else:
                    st.warning("Model does not have feature names. Prediction may be inaccurate.")
                
                price_pred = model.predict(input_df)[0]
                
                # Display prediction
                st.success(f"### Predicted EV Price: ${price_pred:,.2f}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.exception(e)
    
    except Exception as e:
        st.error(f"Error training price prediction model: {str(e)}")
        st.exception(e)

elif option == "EDA":
    st.subheader("📊 Exploratory Data Analysis")
    
    # Ensure assets directory exists
    os.makedirs("assets", exist_ok=True)
    
    try:
        st.write("### Correlation Heatmap")
        plot_correlation(df)
        if os.path.exists("assets/corr_heatmap.png"):
            st.image("assets/corr_heatmap.png", use_container_width=True)
        else:
            st.warning("Correlation plot could not be generated")
    except Exception as e:
        st.error(f"Error generating correlation plot: {str(e)}")
    
    try:
        st.write("### Sales by Brand")
        plot_sales_by_brand(df)
        if os.path.exists("assets/sales_by_brand.png"):
            st.image("assets/sales_by_brand.png", use_container_width=True)
        else:
            st.warning("Sales by brand plot could not be generated")
    except Exception as e:
        st.error(f"Error generating sales by brand plot: {str(e)}")
    
    try:
        st.write("### Price Distribution")
        plot_price_distribution(df)
        if os.path.exists("assets/price_dist.png"):
            st.image("assets/price_dist.png", use_container_width=True)
        else:
            st.warning("Price distribution plot could not be generated")
    except Exception as e:
        st.error(f"Error generating price distribution plot: {str(e)}")

elif option == "Chatbot":
    st.subheader("🤖 EV Chatbot")
    st.write("Ask questions about EV sales, prices, models, and more!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    query = st.chat_input("Ask a question about EVs...")
    
    if query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Get chatbot response
        try:
            answer = chatbot(df, query)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Rerun to update chat display
        st.rerun()
    
    # Example questions
    st.sidebar.write("### Example Questions:")
    example_questions = [
        "What is the average price of EVs?",
        "Which model has the highest sales?",
        "What are the sales forecasts?",
        "What brands are available?"
    ]
    for example in example_questions:
        if st.sidebar.button(example, key=f"example_{example}"):
            try:
                answer = chatbot(df, example)
                st.session_state.messages.append({"role": "user", "content": example})
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
