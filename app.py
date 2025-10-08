import os
import streamlit as st
import google.generativeai as genai
import yaml
from pathlib import Path
import pandas as pd
import io

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Gemini Multi-Agent System", page_icon="🏎️", layout="wide")

# Enhanced Ferrari-style CSS
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
    """Load agents configuration from agents.yaml"""
    try:
        config_path = Path("agents.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        st.warning("agents.yaml not found. Using default agents.")
        return {
            'agents': {
                'Data Transformer': {'description': 'Transforms raw data (CSV, JSON) into Markdown tables.', 'default_prompt': 'Transform the following data into a clean Markdown table. Infer the headers if they are missing.', 'temperature': 0.1, 'max_tokens': 4096},
                'Data Summarizer': {'description': 'Summarizes data and extracts keywords.', 'default_prompt': 'Analyze the provided data and generate a comprehensive summary. After the summary, list the top 5-7 most important keywords, separated by commas.', 'temperature': 0.4, 'max_tokens': 2048},
                'Insight Extractor': {'description': 'Extracts key insights and patterns from data.', 'default_prompt': 'From the provided data, identify and extract the most significant insights, trends, or anomalies.', 'temperature': 0.5, 'max_tokens': 4096},
                'Follow-up Question Generator': {'description': 'Generates relevant follow-up questions.', 'default_prompt': 'Based on the final analysis, generate 3-5 insightful follow-up questions that could guide further investigation.', 'temperature': 0.6, 'max_tokens': 2048}
            }
        }
    except Exception as e:
        st.error(f"Error loading agents.yaml: {e}")
        return {'agents': {}}

# Initialize Gemini API
if 'GEMINI_API_KEY' not in st.session_state:
    st.session_state['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")

if not st.session_state['GEMINI_API_KEY']:
    with st.sidebar:
        st.header("API Key Configuration")
        api_key_input = st.text_input("🔑 Enter your Gemini API Key:", type="password", key="api_key_input")
        if st.button("Set API Key"):
            st.session_state['GEMINI_API_KEY'] = api_key_input
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

def execute_gemini_agent(prompt, data_context, temp, max_tok):
    """Generic function to run a Gemini agent."""
    if not st.session_state.get('GEMINI_API_KEY'):
        return "Error: Gemini API key is not configured."
    try:
        model = genai.GenerativeModel("gemini-2.0-flash", generation_config={"temperature": temp, "max_output_tokens": max_tok})
        full_prompt = f"{data_context}\n\n---\n\nTask:\n{prompt}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error during API call: {e}"

def parse_and_prepare_data(files, text):
    """Reads uploaded files or pasted text and returns content."""
    content = ""
    if files:
        for file in files:
            content += f"--- START OF FILE: {file.name} ---\n"
            if file.type == "text/csv":
                df = pd.read_csv(file)
                content += df.to_string() + "\n"
            elif file.type == "application/json":
                df = pd.read_json(io.StringIO(file.getvalue().decode("utf-8")))
                content += df.to_string() + "\n"
            else:
                content += file.getvalue().decode("utf-8") + "\n"
            content += f"--- END OF FILE: {file.name} ---\n\n"
    elif text:
        content = text
    return content

# -------------------- SESSION STATE INITIALIZATION --------------------
if 'workflow_data' not in st.session_state:
    st.session_state.workflow_data = {} # Stores data for each step

# -------------------- MAIN UI --------------------
st.title("🏎️ Gemini Multi-Agent Analysis Hub")
st.header("Step 1: Upload or Paste Your Datasets")

agents_config = load_agents_config()
agents = agents_config.get('agents', {})

# --- Data Input ---
uploaded_files = st.file_uploader(
    "📂 Upload Datasets (CSV, JSON, TXT)",
    type=["csv", "json", "txt"],
    accept_multiple_files=True
)
pasted_text = st.text_area("📋 Or Paste Raw Data Here:", height=200)

if st.button("1. Process and Summarize Data", type="primary"):
    raw_data = parse_and_prepare_data(uploaded_files, pasted_text)
    if raw_data:
        with st.spinner("Engines starting... Transforming data into Markdown..."):
            transformer_agent = agents.get('Data Transformer', {})
            markdown_tables = execute_gemini_agent(
                transformer_agent.get('default_prompt', ''),
                raw_data,
                transformer_agent.get('temperature', 0.1),
                transformer_agent.get('max_tokens', 4096)
            )
            st.session_state.workflow_data['step_1_markdown'] = markdown_tables

        with st.spinner("Generating comprehensive summary..."):
            summarizer_agent = agents.get('Data Summarizer', {})
            summary_and_keywords = execute_gemini_agent(
                summarizer_agent.get('default_prompt', ''),
                markdown_tables,
                summarizer_agent.get('temperature', 0.4),
                summarizer_agent.get('max_tokens', 2048)
            )
            # Simple split to separate summary from keywords
            parts = summary_and_keywords.split("Keywords:")
            summary = parts[0].strip()
            keywords_str = parts[1].strip() if len(parts) > 1 else ""
            keywords = [k.strip() for k in keywords_str.split(',')]

            # Highlight keywords in summary
            for keyword in keywords:
                if keyword:
                    summary = summary.replace(keyword, f'<span class="keyword">{keyword}</span>')

            st.session_state.workflow_data['step_2_summary'] = summary
    else:
        st.warning("Please upload or paste data to process.")

# --- Data Display and Modification ---
if 'step_1_markdown' in st.session_state:
    st.header("Step 2: Review and Modify Data")
    st.subheader("📝 Editable Markdown Tables")
    st.session_state.workflow_data['step_1_markdown_edited'] = st.text_area(
        "You can modify the Markdown tables below before the next step.",
        value=st.session_state.workflow_data['step_1_markdown'],
        height=300
    )

    st.subheader("📊 Comprehensive Summary")
    st.markdown(f"<div class='results-box'>{st.session_state.workflow_data['step_2_summary']}</div>", unsafe_allow_html=True)


# -------------------- AGENT EXECUTION WORKFLOW --------------------
if 'step_1_markdown_edited' in st.session_state:
    st.header("Step 3: Configure and Execute Agent Workflow")

    workflow_type = st.radio("Select Workflow Type:", ["Single Agent", "Multi-Agent Sequence"], horizontal=True)

    agent_names = list(agents.keys())
    # Exclude helper agents from direct selection if desired
    agent_names = [name for name in agent_names if name not in ['Data Transformer', 'Data Summarizer']]

    if workflow_type == "Single Agent":
        selected_agent = st.selectbox("Select Agent:", agent_names)
        agent_config = agents.get(selected_agent, {})

        with st.expander(f"Configure '{selected_agent}'", expanded=True):
            prompt = st.text_area("Prompt:", value=agent_config.get('default_prompt', ''), height=100, key=f"single_prompt")
            temp = st.slider("Temperature:", 0.0, 1.0, float(agent_config.get('temperature', 0.5)), 0.05, key=f"single_temp")
            max_tok = st.number_input("Max Tokens:", 512, 8192, int(agent_config.get('max_tokens', 4096)), key=f"single_tokens")

        if st.button(f"🚀 Execute {selected_agent}"):
            with st.spinner(f"Executing {selected_agent}..."):
                initial_data = st.session_state.workflow_data['step_1_markdown_edited']
                result = execute_gemini_agent(prompt, initial_data, temp, max_tok)
                st.session_state.workflow_data['final_result'] = result

    else: # Multi-Agent Sequence
        selected_agents = st.multiselect("Select agents in execution order:", agent_names)

        st.session_state.multi_agent_configs = {}
        for agent_name in selected_agents:
            st.subheader(f"Configure Agent: {agent_name}")
            agent_config = agents.get(agent_name, {})
            with st.expander(f"Settings for '{agent_name}'", expanded=True):
                prompt = st.text_area("Prompt:", value=agent_config.get('default_prompt', ''), height=100, key=f"multi_prompt_{agent_name}")
                temp = st.slider("Temperature:", 0.0, 1.0, float(agent_config.get('temperature', 0.5)), 0.05, key=f"multi_temp_{agent_name}")
                max_tok = st.number_input("Max Tokens:", 512, 8192, int(agent_config.get('max_tokens', 4096)), key=f"multi_tokens_{agent_name}")
                st.session_state.multi_agent_configs[agent_name] = {'prompt': prompt, 'temp': temp, 'max_tok': max_tok}

        if st.button("🚀 Execute Multi-Agent Workflow"):
            st.session_state.workflow_data.pop('final_result', None) # Clear previous final result
            st.session_state.workflow_data.pop('multi_agent_steps', None) # Clear previous steps
            
            with st.spinner("Executing multi-agent workflow..."):
                current_data = st.session_state.workflow_data['step_1_markdown_edited']
                st.session_state.multi_agent_steps = []

                for i, agent_name in enumerate(selected_agents):
                    st.info(f"Running Step {i+1}: {agent_name}...")
                    config = st.session_state.multi_agent_configs[agent_name]
                    result = execute_gemini_agent(config['prompt'], current_data, config['temp'], config['max_tok'])
                    st.session_state.multi_agent_steps.append({'agent': agent_name, 'output': result})
                    current_data = result # Output of one agent becomes input for the next

                st.session_state.workflow_data['final_result'] = current_data
            st.success("Multi-agent workflow completed!")

# --- Display Multi-Agent Intermediate and Final Results ---
if 'multi_agent_steps' in st.session_state:
    st.header("Step 4: Review Workflow Results")
    for i, step in enumerate(st.session_state.multi_agent_steps):
        st.subheader(f"Output from Step {i+1}: {step['agent']}")
        
        # Allow editing of each step's output
        edited_output = st.text_area(f"Modify output from {step['agent']}:", value=step['output'], height=250, key=f"edited_step_{i}")
        st.session_state.multi_agent_steps[i]['output_edited'] = edited_output

    if st.button("Regenerate Final Output from Modified Steps"):
         with st.spinner("Rerunning final steps..."):
            # This logic demonstrates how to chain edited outputs.
            # For simplicity, this example just uses the last edited output as the final one.
            # A more complex app could re-run subsequent agents.
            st.session_state.workflow_data['final_result'] = st.session_state.multi_agent_steps[-1]['output_edited']


# --- Final Result and Follow-up ---
if 'final_result' in st.session_state:
    st.header("🏁 Final Analysis Result")
    st.markdown(f"<div class='results-box'>{st.session_state.workflow_data['final_result']}</div>", unsafe_allow_html=True)

    if st.button("🤔 Generate Follow-up Questions"):
        with st.spinner("Generating follow-up questions..."):
            question_agent = agents.get('Follow-up Question Generator', {})
            follow_up_prompt = question_agent.get('default_prompt', '')
            final_data_context = st.session_state.workflow_data['final_result']
            questions = execute_gemini_agent(
                follow_up_prompt,
                final_data_context,
                question_agent.get('temperature', 0.6),
                question_agent.get('max_tokens', 2048)
            )
            st.session_state.follow_up_questions = questions

if 'follow_up_questions' in st.session_state:
    st.header("❓ Suggested Follow-up Questions")
    st.markdown(f"<div class='results-box'>{st.session_state.follow_up_questions}</div>", unsafe_allow_html=True)


# -------------------- FOOTER & SAMPLE YAML --------------------
st.markdown("---")
st.write("Built with **Streamlit + Gemini API** | Multi-Agent System with YAML Configuration")

with st.expander("📋 Sample agents.yaml Configuration"):
    st.code("""
agents:
  Data Transformer:
    description: Transforms raw data (CSV, JSON) into clean Markdown tables, inferring headers if missing.
    default_prompt: Review the following raw data. Your task is to accurately convert it into a well-structured Markdown table. If headers are not explicitly provided, infer appropriate headers based on the data content.
    temperature: 0.1
    max_tokens: 4096

  Data Summarizer:
    description: Analyzes data to provide a summary and extracts key terms.
    default_prompt: "Analyze the provided data in the Markdown table(s). Generate a comprehensive summary that covers the main points and overall trends. After the summary, on a new line, write 'Keywords:' followed by a comma-separated list of the 5-7 most important keywords from your summary."
    temperature: 0.4
    max_tokens: 2048

  Insight Extractor:
    description: Extracts deep insights, patterns, and anomalies from structured data.
    default_prompt: From the provided data, identify and extract the most significant insights, trends, or anomalies. Present your findings as a bulleted list. Focus on actionable information that is not immediately obvious.
    temperature: 0.5
    max_tokens: 4096

  Compliance Checker:
    description: Checks data or text against a specific set of compliance rules.
    default_prompt: Review the submission against FDA 510(k) compliance requirements. Identify any potential gaps, inconsistencies, or areas of concern. List each finding clearly.
    temperature: 0.2
    max_tokens: 4096

  Follow-up Question Generator:
    description: Generates relevant follow-up questions based on a final analysis.
    default_prompt: Based on the preceding analysis and data, generate a list of 3 to 5 insightful follow-up questions. These questions should aim to uncover deeper insights, address potential gaps, or guide the next steps of the investigation.
    temperature: 0.6
    max_tokens: 2048
""", language="yaml")
