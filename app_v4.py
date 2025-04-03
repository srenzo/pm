import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import boto3
import json
import io
import re
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from pathlib import Path
from botocore.config import Config
from tabulate import tabulate
import os

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")

# Set up page config
st.set_page_config(
    page_title="Parquet Data Chat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure matplotlib for Streamlit
plt.switch_backend('Agg')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.dpi'] = 100

# Title
st.title("📊 Parquet Data Chat")
st.markdown("Upload your parquet files and chat with your data using Claude")

# Initialize session state
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "aws_configured" not in st.session_state:
    st.session_state.aws_configured = False
if "show_code" not in st.session_state:
    st.session_state.show_code = True

# Sidebar configuration
with st.sidebar:
    st.header("AWS Configuration")
    
    aws_region = st.text_input("AWS Region", value="us-east-1")
    aws_profile = st.text_input("AWS Profile (optional)")
    
    st.header("Application Settings")
    
    st.session_state.show_code = st.toggle(
        "Show Code Details", 
        value=st.session_state.show_code,
        help="When enabled, shows the generated code and explanation. When disabled, only shows results."
    )
    
    st.header("Data Upload")
    
    # Create a .streamlit/config.toml file with maxUploadSize = 300
    st.info("Maximum file size: 300MB")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Parquet Files", 
        type=["parquet"], 
        accept_multiple_files=True
    )
    
    # Load button
    if st.button("Load Data"):
        if uploaded_files:
            with st.spinner("Loading data..."):
                for file in uploaded_files:
                    try:
                        file_name = Path(file.name).stem
                        # Read parquet file
                        parquet_table = pq.read_table(file)
                        df = parquet_table.to_pandas()
                        # Store in session state
                        st.session_state.dataframes[file_name] = df
                        st.success(f"✅ Loaded {file_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {e}")
    
    # Data explorer toggle
    if st.session_state.dataframes:
        st.header("🔍 Data Explorer")
        show_explorer = st.checkbox("Show Data Explorer", value=True)
        
        # Test connection button
        if st.button("Test AWS Connection"):
            try:
                if aws_profile:
                    session = boto3.Session(profile_name=aws_profile)
                    client = session.client("bedrock-runtime", region_name=aws_region)
                else:
                    client = boto3.client("bedrock-runtime", region_name=aws_region)
                
                # Test with a simple prompt
                test_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 100,
                    "temperature": 0.5,
                    "system": "You are a helpful assistant.",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "Hello, this is a test."}]}
                    ]
                }
                
                response = client.invoke_model(
                    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(test_body)
                )
                
                # Check if the response contains content
                response_body = json.loads(response["body"].read())
                if "content" in response_body:
                    st.success("✅ AWS connection successful")
                    st.session_state.aws_configured = True
                else:
                    st.error("❌ AWS connection failed: Invalid response")
            except Exception as e:
                st.error(f"❌ AWS connection failed: {e}")

# Main area
if not st.session_state.dataframes:
    st.info("📊 Please upload your parquet files to begin.")
else:
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat", "🔍 Data Explorer"])
    
    # Chat tab
    with tab1:
        if not st.session_state.aws_configured:
            st.warning("⚠️ Please test your AWS connection in the sidebar before using the chat")
        else:
            # Function to call Claude
            def call_claude(prompt, system_prompt):
                try:
                    if aws_profile:
                        session = boto3.Session(profile_name=aws_profile)
                        client = session.client("bedrock-runtime", region_name=aws_region)
                    else:
                        client = boto3.client("bedrock-runtime", region_name=aws_region)
                    
                    # Prepare messages - FIXED: Ensure messages alternate correctly
                    messages = []
                    
                    # Only include complete user-assistant pairs from history
                    # plus the current user message
                    if len(st.session_state.messages) > 0:
                        # If odd number of messages, we need to handle it differently
                        msg_count = len(st.session_state.messages)
                        
                        # Process complete pairs
                        for i in range(0, msg_count - (msg_count % 2), 2):
                            if i+1 < msg_count:  # Ensure we have a pair
                                user_msg = st.session_state.messages[i]
                                asst_msg = st.session_state.messages[i+1]
                                
                                if user_msg["role"] == "user" and asst_msg["role"] == "assistant":
                                    messages.append({"role": "user", "content": [{"type": "text", "text": user_msg["content"]}]})
                                    messages.append({"role": "assistant", "content": [{"type": "text", "text": asst_msg["content"]}]})
                    
                    # Always add the current prompt as a user message
                    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
                    
                    # Create request body
                    body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 4000,
                        "temperature": 0.5,
                        "system": system_prompt,
                        "messages": messages
                    }
                    
                    # For debugging
                    logger.info(f"Message payload structure: {json.dumps([msg['role'] for msg in messages])}")
                    
                    # Call the model
                    response = client.invoke_model(
                        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(body)
                    )
                    
                    # Parse response
                    response_body = json.loads(response["body"].read())
                    return response_body["content"][0]["text"]
                except Exception as e:
                    logger.error(f"Claude API error: {str(e)}")
                    logger.error(traceback.format_exc())
                    return f"Error calling Claude: {str(e)}"
            
            # Function to create data summary
            def create_data_summary():
                summary = "# Data Summary\n\n"
                for name, df in st.session_state.dataframes.items():
                    summary += f"## {name}\n"
                    summary += f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
                    summary += f"- Columns: {', '.join(df.columns.tolist())}\n"
                    
                    # Sample data
                    summary += "- Sample data (first 3 rows):\n"
                    sample_df = df.head(3)
                    
                    # Use to_markdown if available, otherwise fallback
                    try:
                        summary += sample_df.to_markdown() + "\n\n"
                    except:
                        try:
                            summary += tabulate(sample_df, headers='keys', tablefmt='pipe') + "\n\n"
                        except:
                            summary += str(sample_df) + "\n\n"
                    
                    # Column types
                    summary += "- Column types:\n"
                    for col, dtype in df.dtypes.items():
                        summary += f"  - {col}: {dtype}\n"
                    
                    # Basic statistics for numeric columns
                    num_cols = df.select_dtypes(include=['number']).columns
                    if len(num_cols) > 0:
                        summary += "- Numeric column statistics:\n"
                        stats_df = df[num_cols].describe().T
                        try:
                            summary += stats_df.to_markdown() + "\n\n"
                        except:
                            try:
                                summary += tabulate(stats_df, headers='keys', tablefmt='pipe') + "\n\n"
                            except:
                                summary += str(stats_df) + "\n\n"
                    
                    # Missing values
                    missing = df.isnull().sum()
                    if missing.sum() > 0:
                        summary += "- Missing values:\n"
                        for col, count in missing.items():
                            if count > 0:
                                pct = 100 * count / len(df)
                                summary += f"  - {col}: {count} ({pct:.1f}%)\n"
                    
                    summary += "\n"
                
                return summary
            
            # Function to execute code
            def execute_code(code):
                try:
                    # Create local namespace
                    local_namespace = {
                        'pd': pd,
                        'np': np,
                        'plt': plt,
                        'sns': sns,
                        'dataframes': st.session_state.dataframes,
                        'st': st  # Pass streamlit directly
                    }
                    
                    # Execute with redirect
                    buffer = io.StringIO()
                    with st.spinner("Executing code..."):
                        # Add matplotlib configuration
                        setup_code = """
# Configure matplotlib
import matplotlib
matplotlib.use('Agg')  # Force Agg backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.autolayout'] = True
plt.rcParams['figure.dpi'] = 100
"""
                        # Add streamlit utilities
                        streamlit_utils = """
# Helper function for consistent plotting
def show_plot(plot_func):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_func(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
"""
                        # Combine code
                        full_code = setup_code + streamlit_utils + code
                        
                        # Execute the code
                        exec(full_code, local_namespace)
                        
                        # Check for matplotlib figures
                        if plt.get_fignums():
                            for fig_num in plt.get_fignums():
                                try:
                                    fig = plt.figure(fig_num)
                                    st.pyplot(fig)
                                except Exception as fig_error:
                                    st.error(f"Error displaying figure: {fig_error}")
                            plt.close('all')
                        
                        # Display any DataFrames that might have been created
                        for var_name, var_val in local_namespace.items():
                            if var_name not in ['pd', 'np', 'plt', 'sns', 'dataframes', 'st', 'matplotlib', 'show_plot'] and isinstance(var_val, pd.DataFrame):
                                if not var_name.startswith('_'):  # Skip internal variables
                                    st.write(f"DataFrame: {var_name}")
                                    st.dataframe(var_val)
                        
                        # Return from execution
                        return True
                except Exception as e:
                    st.error(f"Error executing code: {str(e)}")
                    st.error("Traceback: " + traceback.format_exc())
                    return False
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Check for code in assistant messages and execute
                    if message["role"] == "assistant":
                        code_blocks = re.findall(r"```python(.*?)```", message["content"], re.DOTALL)
                        if code_blocks and len(code_blocks) > 0:
                            # Take the last code block (usually the most complete one)
                            with st.expander("Execution Results", expanded=True):
                                # Show code if enabled
                                if st.session_state.show_code:
                                    st.code(code_blocks[-1], language="python")
                                # Execute the code
                                execute_code(code_blocks[-1])
            
            # System prompt
            system_prompt = f"""
            You are a data analysis assistant helping to analyze parquet data files.

            {create_data_summary()}

            When generating answers:
            1. Be clear and concise
            2. Refer to specific column names from the data
            3. Provide complete explanations of your approach
            4. Include executable Python code for all analyses and visualizations

            When creating visualizations:
            - Always use matplotlib or seaborn with proper configuration
            - Create figures using fig, ax = plt.subplots()
            - Always include plt.tight_layout()
            - Display plots using st.pyplot(fig)
            - Close figures with plt.close(fig) after plotting

            You can use Streamlit (st) commands directly in your code for better displays:
            - st.dataframe() to show tables
            - st.pyplot() for figures 
            - st.write() for text output
            - st.metric() for key figures

            Always write clean, complete code that includes all necessary imports.

            Example visualization pattern:
            ```python
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Get data
            df = dataframes['dataset_name']

            # Create figure and axes
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create the visualization
            ax.plot(df['x_column'], df['y_column'])
            # OR for seaborn: sns.histplot(data=df, x='column', ax=ax)

            # Add labels
            ax.set_title('Descriptive Title')
            ax.set_xlabel('X-Axis Label')
            ax.set_ylabel('Y-Axis Label')

            # Ensure layout is optimized
            plt.tight_layout()

            # Display with Streamlit
            st.pyplot(fig)

            # Clean up
            plt.close(fig)
            ```
            """
            
            # Chat input
            if prompt := st.chat_input("Ask about your data..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Claude is thinking..."):
                        response = call_claude(prompt, system_prompt)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Check for code and execute
                    code_blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
                    if code_blocks and len(code_blocks) > 0:
                        with st.expander("Execution Results", expanded=True):
                            # Show code if enabled
                            if st.session_state.show_code:
                                st.code(code_blocks[-1], language="python")
                            # Execute the code
                            execute_code(code_blocks[-1])
                
                # Add assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            # First-time help
            if not st.session_state.messages:
                st.info("👋 Start by asking questions about your data! For example:")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    - What tables are available in my data?
                    - Summarize the customer data
                    - Show me the distribution of sales by region
                    """)
                with col2:
                    st.markdown("""
                    - Find correlations between price and quantity
                    - Plot a time series of monthly revenue
                    - What are the top 10 products by sales?
                    """)
    
    # Data Explorer tab
    with tab2:
        if not show_explorer:
            st.info("Enable the Data Explorer in the sidebar to view your data.")
        else:
            st.header("Data Explorer")
            
            # Dataset selector
            dataset_names = list(st.session_state.dataframes.keys())
            selected_dataset = st.selectbox("Select Dataset", dataset_names)
            
            if selected_dataset:
                df = st.session_state.dataframes[selected_dataset]
                
                # Display info
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", f"{df.shape[0]:,}")
                col2.metric("Columns", f"{df.shape[1]}")
                
                # Calculate memory usage safely
                try:
                    memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
                    col3.metric("Memory Usage", f"{memory_usage:.2f} MB")
                except:
                    col3.metric("Memory Usage", "Unknown")
                
                # Column selector
                all_columns = list(df.columns)
                selected_columns = st.multiselect(
                    "Select columns to display",
                    options=all_columns,
                    default=all_columns[:min(5, len(all_columns))]
                )
                
                # Row limit
                row_limit = st.slider("Number of rows", 5, 100, 10)
                
                # Display data
                if selected_columns:
                    # Fixed parameter name from use_column_width to use_container_width
                    st.dataframe(df[selected_columns].head(row_limit), use_container_width=True)
                    
                    # Data statistics
                    st.subheader("Column Statistics")
                    
                    # Check if there are numeric columns
                    num_cols = df[selected_columns].select_dtypes(include=['number']).columns
                    if not num_cols.empty:
                        st.write("Numeric Columns:")
                        st.dataframe(df[num_cols].describe(), use_container_width=True)
                    
                    # Check if there are categorical columns
                    cat_cols = df[selected_columns].select_dtypes(include=['object', 'category', 'bool']).columns
                    if not cat_cols.empty:
                        st.write("Categorical Columns:")
                        for col in cat_cols:
                            with st.expander(f"{col} - Value Counts"):
                                counts = df[col].value_counts().reset_index()
                                counts.columns = [col, 'Count']
                                st.dataframe(counts, use_container_width=True)
                    
                    # Simple visualizations
                    st.subheader("Quick Visualizations")
                    
                    viz_tab1, viz_tab2 = st.tabs(["Numeric Analysis", "Categorical Analysis"])
                    
                    with viz_tab1:
                        # Numeric columns
                        if not num_cols.empty:
                            viz_col1, viz_col2 = st.columns(2)
                            
                            with viz_col1:
                                num_col = st.selectbox("Select numeric column", num_cols)
                                
                                if st.button("Show Histogram"):
                                    try:
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        ax.hist(df[num_col].dropna(), bins=20)
                                        ax.set_title(f'Histogram of {num_col}')
                                        ax.set_xlabel(num_col)
                                        ax.set_ylabel('Frequency')
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    except Exception as e:
                                        st.error(f"Error creating histogram: {e}")
                            
                            with viz_col2:
                                if len(num_cols) >= 2:
                                    num_col2 = st.selectbox("Select second numeric column", [c for c in num_cols if c != num_col])
                                    
                                    if st.button("Show Scatter Plot"):
                                        try:
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            ax.scatter(df[num_col], df[num_col2], alpha=0.5)
                                            ax.set_title(f'Scatter Plot: {num_col} vs {num_col2}')
                                            ax.set_xlabel(num_col)
                                            ax.set_ylabel(num_col2)
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.close(fig)
                                        except Exception as e:
                                            st.error(f"Error creating scatter plot: {e}")
                                else:
                                    st.info("Need at least 2 numeric columns for scatter plot")
                        else:
                            st.info("No numeric columns selected")
                    
                    with viz_tab2:
                        # Categorical columns
                        if not cat_cols.empty:
                            cat_col = st.selectbox("Select categorical column", cat_cols)
                            
                            if st.button("Show Bar Chart"):
                                try:
                                    # Get value counts and limit to top 10
                                    value_counts = df[cat_col].value_counts().nlargest(10)
                                    
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.bar(value_counts.index.astype(str), value_counts.values)
                                    ax.set_title(f'Top 10 values in {cat_col}')
                                    ax.set_xlabel(cat_col)
                                    ax.set_ylabel('Count')
                                    plt.xticks(rotation=45, ha='right')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                                except Exception as e:
                                    st.error(f"Error creating bar chart: {e}")
                            
                            # If we have both categorical and numeric columns
                            if not num_cols.empty:
                                st.write("Relationship between categorical and numeric data:")
                                num_col_for_cat = st.selectbox("Select numeric column for analysis", num_cols)
                                
                                if st.button("Show Box Plot"):
                                    try:
                                        # Limit to top 8 categories to avoid cluttered plot
                                        top_cats = df[cat_col].value_counts().nlargest(8).index
                                        filtered_df = df[df[cat_col].isin(top_cats)]
                                        
                                        fig, ax = plt.subplots(figsize=(12, 6))
                                        sns.boxplot(x=cat_col, y=num_col_for_cat, data=filtered_df, ax=ax)
                                        ax.set_title(f'Box Plot of {num_col_for_cat} by {cat_col}')
                                        plt.xticks(rotation=45, ha='right')
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)
                                    except Exception as e:
                                        st.error(f"Error creating box plot: {e}")
                        else:
                            st.info("No categorical columns selected")
                else:
                    st.info("Please select at least one column")

if __name__ == "__main__":
    # Create .streamlit directory and config.toml if they don't exist
    import os
    if not os.path.exists(".streamlit"):
        os.makedirs(".streamlit")
    
    config_path = ".streamlit/config.toml"
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write("[server]\n")
            f.write("maxUploadSize = 300\n")
