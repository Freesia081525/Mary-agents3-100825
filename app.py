import os
import streamlit as st
import google.generativeai as genai
import yaml
from pathlib import Path
import pandas as pd
import io

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Gemini Multi-Agent System", page_icon="🏎️", layout="wide")

# Enhanced Ferrari-style CSS (No changes here)
st.markdown("""
    <style>
        .main {background-color: #0d0d0d; color: #ffffff;}
        h1, h2, h3 {color: #ff2800; text-align: center; font-family: 'Arial Black', sans-serif;}
        .stButton>button {
            background-color: #ff2800;
            color: white;
            border: 2px solid #ff2800;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #ffffff;
            color: #ff2800;
            border: 2px solid #ff2800;
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
            background-color: #1a1a1a;
            color: #fff;
            border-radius: 8px;
        }
        .stFileUploader>div>div>button {
            border: 2px dashed #ff2800;
            background-color: #1a1a1a;
            color: #ff2800;
        }
        .stMultiSelect>div>div>div>div {
            background-color: #1a1a1a;
        }
        .stExpander {
            background-color: #1e1e1e;
            border-radius: 10px;
        }
        .keyword {
            color: coral;
            font-weight: bold;
        }
        .results-box {
            background-color: #1f1f1f;
            border: 1px solid #444;
            padding: 15px;
            border-radius: 10px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- AGENT & API CONFIGURATION --------------------

@st.cache_data
def load_agents_config():
    """Load agents configuration from agents.yaml using a robust path."""
    try:
        config_path = Path(__file__).parent / "agents.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        st.warning("agents.yaml not found. Using default agents.")
    except Exception as e:
        st.error(f"Error loading agents.yaml: {e}")

    # Fallback default configuration
    return {
        'agents': {
            'Data Transformer': {'description': 'Transforms raw data (CSV, JSON) into Markdown tables.', 'default_prompt': 'Transform the following data into a clean Markdown table. Infer the headers if they are missing.', 'temperature': 0.1, 'max_tokens': 4096},
            'Data Summarizer': {'description': 'Summarizes data and extracts keywords.', 'default_prompt': 'Analyze the provided data and generate a comprehensive summary. After the summary, on a new line, write "Keywords:" followed by a comma-separated list of the 5-7 most important keywords.', 'temperature': 0.4, 'max_tokens': 2048},
            'Insight Extractor': {'description': 'Extracts key insights and patterns from data.', 'default_prompt': 'From the provided data, identify and extract the most significant insights, trends, or anomalies.', 'temperature': 0.5, 'max_tokens': 4096},
            'Follow-up Question Generator': {'description': 'Generates relevant follow-up questions.', 'default_prompt': 'Based on the final analysis, generate 3-5 insightful follow-up questions that could guide further investigation.', 'temperature': 0.6, 'max_tokens': 2048}
        }
    }

# API Key Management
if 'GEMINI_API_KEY' not in st.session_state:
    st.session_state['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")

with st.sidebar:
    st.header("🔑 API Configuration")
    api_key_input = st.text_input("Enter your Gemini API Key:", type="password", value=st.session_state.get('GEMINI_API_KEY', ''))
    if st.button("Set API Key"):
        st.session_state['GEMINI_API_KEY'] = api_key_input
        st.success("API Key set!")
        st.rerun()

if st.session_state.get('GEMINI_API_KEY'):
    try:
        genai.configure(api_key=st.session_state['GEMINI_API_KEY'])
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
else:
    st.warning("Please provide your Gemini API Key in the sidebar to begin.")
    st.stop()

# -------------------- HELPER FUNCTIONS --------------------

@st.cache_resource
def get_gemini_model():
    """Returns a cached instance of the Gemini model."""
    return genai.GenerativeModel("gemini-2.0-flash")

# BUG FIX: Upgraded function to return a (success, content) tuple for robust error handling.
def execute_gemini_agent(prompt, data_context, temp, max_tok):
    """
    Executes a Gemini agent and returns a tuple (success, content).
    On failure, success is False and content is the error message.
    """
    if not st.session_state.get('GEMINI_API_KEY'):
        return False, "Error: Gemini API key is not configured."
    try:
        model = get_gemini_model()
        generation_config = genai.GenerationConfig(temperature=temp, max_output_tokens=max_tok)
        full_prompt = f"CONTEXT:\n{data_context}\n\n---\n\nTASK:\n{prompt}"
        response = model.generate_content(full_prompt, generation_config=generation_config)
        
        if not response.text:
             return False, "API returned an empty response. The model may have generated no content or been blocked. Check safety settings in your Google AI Studio."
        return True, response.text
    except Exception as e:
        return False, f"An unexpected error occurred during the API call: {str(e)}"

def parse_and_prepare_data(files, text):
    """Reads uploaded files or pasted text and returns content as a string."""
    # ... (No changes to this function)
    content = ""
    if files:
        for file in files:
            content += f"--- START OF FILE: {file.name} ---\n"
            try:
                if file.type == "text/csv":
                    df = pd.read_csv(file)
                    content += df.to_string() + "\n"
                elif file.type == "application/json":
                    df = pd.read_json(io.StringIO(file.getvalue().decode("utf-8")))
                    content += df.to_string() + "\n"
                else:
                    content += file.getvalue().decode("utf-8") + "\n"
            except Exception as e:
                st.error(f"Error processing file {file.name}: {e}")
            content += f"--- END OF FILE: {file.name} ---\n\n"
    elif text:
        content = text
    return content

# -------------------- SESSION STATE INITIALIZATION --------------------
if 'workflow_data' not in st.session_state:
    st.session_state.workflow_data = {}

# -------------------- MAIN UI --------------------
st.title("🏎️ Gemini Multi-Agent Analysis Hub")
st.header("Step 1: Upload or Paste Your Datasets")

agents_config = load_agents_config()
agents = agents_config.get('agents', {})

uploaded_files = st.file_uploader(
    "📂 Upload Datasets (CSV, JSON, TXT)",
    type=["csv", "json", "txt"],
    accept_multiple_files=True
)
pasted_text = st.text_area("📋 Or Paste Raw Data Here:", height=200, placeholder="Paste your raw data here...")

if st.button("1. Process and Summarize Data", type="primary"):
    # BUG FIX: Selectively clear downstream results, not the entire state.
    keys_to_clear = ['step_1_markdown', 'step_1_markdown_edited', 'step_2_summary_html', 'final_result', 'multi_agent_steps', 'follow_up_questions']
    for key in keys_to_clear:
        if key in st.session_state.workflow_data:
            del st.session_state.workflow_data[key]
        if key in st.session_state:
             if key in st.session_state:
                 del st.session_state[key]


    raw_data = parse_and_prepare_data(uploaded_files, pasted_text)
    if raw_data:
        # Step 1: Transform Data
        with st.spinner("Agent 1: Transforming data into Markdown..."):
            transformer_agent = agents.get('Data Transformer', {})
            success, markdown_tables = execute_gemini_agent(
                transformer_agent.get('default_prompt', ''),
                raw_data,
                transformer_agent.get('temperature', 0.1),
                transformer_agent.get('max_tokens', 4096)
            )
        
        if success:
            st.session_state.workflow_data['step_1_markdown'] = markdown_tables
            
            # Step 2: Summarize Data
            with st.spinner("Agent 2: Generating comprehensive summary..."):
                summarizer_agent = agents.get('Data Summarizer', {})
                success, summary_and_keywords = execute_gemini_agent(
                    summarizer_agent.get('default_prompt', ''),
                    markdown_tables,
                    summarizer_agent.get('temperature', 0.4),
                    summarizer_agent.get('max_tokens', 2048)
                )

            if success:
                # BUG FIX: Robust keyword parsing
                summary = summary_and_keywords
                keywords = []
                if "keywords:" in summary_and_keywords.lower():
                    parts = summary_and_keywords.lower().split("keywords:")
                    summary = parts[0].strip()
                    keywords_str = parts[1].strip() if len(parts) > 1 else ""
                    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                else:
                    st.warning("Could not automatically extract keywords from the summary.")

                summary_html = summary
                for keyword in keywords:
                    summary_html = summary_html.replace(keyword, f'<span class="keyword">{keyword}</span>')
                
                st.session_state.workflow_data['step_2_summary_html'] = summary_html
            else:
                st.error(f"Data Summarization Failed: {summary_and_keywords}")
        else:
            st.error(f"Data Transformation Failed: {markdown_tables}")
    else:
        st.warning("Please upload or paste data to process.")
    
    # BUG FIX: Force a rerun to ensure the UI updates with the new state.
    st.rerun()

# --- Data Display and Modification ---
if 'step_1_markdown' in st.session_state.workflow_data:
    st.header("Step 2: Review and Modify Data")
    st.subheader("📝 Editable Markdown Tables")
    st.session_state.workflow_data['step_1_markdown_edited'] = st.text_area(
        "You can modify the Markdown tables below before the next step.",
        value=st.session_state.workflow_data['step_1_markdown'],
        height=300
    )

    if 'step_2_summary_html' in st.session_state.workflow_data:
        st.subheader("📊 Comprehensive Summary")
        st.markdown(f"<div class='results-box'>{st.session_state.workflow_data['step_2_summary_html']}</div>", unsafe_allow_html=True)

# ... The rest of the script remains the same as it was already robust ...
# -------------------- AGENT EXECUTION WORKFLOW --------------------
if 'step_1_markdown_edited' in st.session_state.workflow_data:
    st.header("Step 3: Configure and Execute Agent Workflow")
    workflow_type = st.radio("Select Workflow Type:", ["Single Agent", "Multi-Agent Sequence"], horizontal=True)
    agent_names = [name for name in agents.keys() if name not in ['Data Transformer', 'Data Summarizer']]
    
    # ... (No changes needed in the Single Agent and Multi-Agent UI logic) ...
    if workflow_type == "Single Agent":
        selected_agent = st.selectbox("Select Agent:", agent_names)
        if selected_agent:
            agent_config = agents.get(selected_agent, {})
            with st.expander(f"Configure '{selected_agent}'", expanded=True):
                prompt = st.text_area("Prompt:", value=agent_config.get('default_prompt', ''), height=100, key=f"single_prompt")
                temp = st.slider("Temperature:", 0.0, 1.0, float(agent_config.get('temperature', 0.5)), 0.05, key=f"single_temp")
                max_tok = st.number_input("Max Tokens:", 512, 8192, int(agent_config.get('max_tokens', 4096)), key=f"single_tokens")

            if st.button(f"🚀 Execute {selected_agent}"):
                with st.spinner(f"Executing {selected_agent}..."):
                    initial_data = st.session_state.workflow_data['step_1_markdown_edited']
                    success, result = execute_gemini_agent(prompt, initial_data, temp, max_tok)
                    if success:
                        st.session_state.workflow_data['final_result'] = result
                    else:
                        st.error(f"Agent Execution Failed: {result}")

    else: # Multi-Agent Sequence
        selected_agents = st.multiselect("Select agents in execution order:", agent_names)
        st.session_state.multi_agent_configs = {}
        for agent_name in selected_agents:
            agent_config = agents.get(agent_name, {})
            with st.expander(f"Configure Agent: {agent_name}", expanded=True):
                prompt = st.text_area("Prompt:", value=agent_config.get('default_prompt', ''), height=100, key=f"multi_prompt_{agent_name}")
                temp = st.slider("Temperature:", 0.0, 1.0, float(agent_config.get('temperature', 0.5)), 0.05, key=f"multi_temp_{agent_name}")
                max_tok = st.number_input("Max Tokens:", 512, 8192, int(agent_config.get('max_tokens', 4096)), key=f"multi_tokens_{agent_name}")
                st.session_state.multi_agent_configs[agent_name] = {'prompt': prompt, 'temp': temp, 'max_tok': max_tok}

        if st.button("🚀 Execute Multi-Agent Workflow"):
            st.session_state.workflow_data.pop('final_result', None)
            st.session_state.workflow_data.pop('multi_agent_steps', None)
            
            if selected_agents:
                with st.spinner("Executing multi-agent workflow..."):
                    current_data = st.session_state.workflow_data['step_1_markdown_edited']
                    st.session_state.multi_agent_steps = []
                    workflow_failed = False
                    for i, agent_name in enumerate(selected_agents):
                        st.info(f"Running Step {i+1}: {agent_name}...")
                        config = st.session_state.multi_agent_configs[agent_name]
                        success, result = execute_gemini_agent(config['prompt'], current_data, config['temp'], config['max_tok'])
                        if success:
                            st.session_state.multi_agent_steps.append({'agent': agent_name, 'output': result})
                            current_data = result
                        else:
                            st.error(f"Workflow failed at Step {i+1} ({agent_name}): {result}")
                            workflow_failed = True
                            break
                    
                    if not workflow_failed:
                        st.session_state.workflow_data['final_result'] = current_data
                        st.success("Multi-agent workflow completed!")
            else:
                st.warning("Please select at least one agent for the multi-agent workflow.")

# --- Display Multi-Agent Intermediate and Final Results ---
if 'multi_agent_steps' in st.session_state and st.session_state.multi_agent_steps:
    st.header("Step 4: Review Workflow Results")
    current_input_for_next_step = ""
    for i, step in enumerate(st.session_state.multi_agent_steps):
        st.subheader(f"Output from Step {i+1}: {step['agent']}")
        edited_output = st.text_area("", value=step['output'], height=250, key=f"edited_step_{i}", label_visibility="collapsed")
        st.session_state.multi_agent_steps[i]['output_edited'] = edited_output
        current_input_for_next_step = edited_output

    if st.button("Update Final Result from Modified Steps"):
        st.session_state.workflow_data['final_result'] = current_input_for_next_step
        st.success("Final result updated!")

# --- Final Result and Follow-up ---
if 'final_result' in st.session_state.workflow_data:
    st.header("🏁 Final Analysis Result")
    st.markdown(f"<div class='results-box'>{st.session_state.workflow_data['final_result']}</div>", unsafe_allow_html=True)
    if st.button("🤔 Generate Follow-up Questions"):
        with st.spinner("Generating follow-up questions..."):
            question_agent = agents.get('Follow-up Question Generator', {})
            success, questions = execute_gemini_agent(
                question_agent.get('default_prompt', ''),
                st.session_state.workflow_data['final_result'],
                question_agent.get('temperature', 0.6),
                question_agent.get('max_tokens', 2048)
            )
            if success:
                st.session_state.follow_up_questions = questions
            else:
                st.error(f"Could not generate questions: {questions}")

if 'follow_up_questions' in st.session_state:
    st.header("❓ Suggested Follow-up Questions")
    st.markdown(f"<div class='results-box'>{st.session_state.follow_up_questions}</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.write("Built with **Streamlit + Gemini API** | Multi-Agent System with YAML Configuration")
