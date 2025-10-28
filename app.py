import os
import streamlit as st
import google.generativeai as genai
import yaml
from pathlib import Path
import pandas as pd
import io

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Multi-Agent Analysis Hub", page_icon="🌍", layout="wide")

# -------------------- THEME DEFINITIONS --------------------
THEMES = {
    "Iceland": {
        "primary": "#00D9FF",
        "secondary": "#B8E6F0",
        "bg_main": "#0A1929",
        "bg_secondary": "#132F4C",
        "text": "#E3F2FD",
        "accent": "#82E9FF",
        "icon": "🇮🇸",
        "gradient": "linear-gradient(135deg, #0A1929 0%, #1A3A52 100%)"
    },
    "Canada": {
        "primary": "#FF0000",
        "secondary": "#FFB3B3",
        "bg_main": "#1C0000",
        "bg_secondary": "#330000",
        "text": "#FFFFFF",
        "accent": "#FF6B6B",
        "icon": "🇨🇦",
        "gradient": "linear-gradient(135deg, #1C0000 0%, #4D0000 100%)"
    },
    "Paris": {
        "primary": "#FFD700",
        "secondary": "#FFF4D6",
        "bg_main": "#1A1A2E",
        "bg_secondary": "#16213E",
        "text": "#F0E68C",
        "accent": "#DAA520",
        "icon": "🇫🇷",
        "gradient": "linear-gradient(135deg, #1A1A2E 0%, #2D3561 100%)"
    },
    "Rome": {
        "primary": "#C19A6B",
        "secondary": "#E8D5B7",
        "bg_main": "#2C1810",
        "bg_secondary": "#3D2414",
        "text": "#F5E6D3",
        "accent": "#D4AF7A",
        "icon": "🇮🇹",
        "gradient": "linear-gradient(135deg, #2C1810 0%, #5C3A24 100%)"
    },
    "Venice": {
        "primary": "#4A90E2",
        "secondary": "#A7D8FF",
        "bg_main": "#0F2027",
        "bg_secondary": "#203A43",
        "text": "#E0F7FF",
        "accent": "#6BB6FF",
        "icon": "🛶",
        "gradient": "linear-gradient(135deg, #0F2027 0%, #2C5364 100%)"
    },
    "Copenhagen": {
        "primary": "#FF6B9D",
        "secondary": "#FFD1DC",
        "bg_main": "#1A0B1A",
        "bg_secondary": "#2D1F2D",
        "text": "#FFE4E9",
        "accent": "#FF8FB3",
        "icon": "🇩🇰",
        "gradient": "linear-gradient(135deg, #1A0B1A 0%, #3D2E3D 100%)"
    },
    "Munich": {
        "primary": "#00A8E8",
        "secondary": "#B3E5FC",
        "bg_main": "#001F3F",
        "bg_secondary": "#003B5C",
        "text": "#E1F5FE",
        "accent": "#4FC3F7",
        "icon": "🇩🇪",
        "gradient": "linear-gradient(135deg, #001F3F 0%, #005073 100%)"
    },
    "Swiss": {
        "primary": "#E74C3C",
        "secondary": "#FADBD8",
        "bg_main": "#0E0E0E",
        "bg_secondary": "#1C1C1C",
        "text": "#FFFFFF",
        "accent": "#EC7063",
        "icon": "🇨🇭",
        "gradient": "linear-gradient(135deg, #0E0E0E 0%, #2D2D2D 100%)"
    },
    "Space": {
        "primary": "#9D00FF",
        "secondary": "#E0B3FF",
        "bg_main": "#000000",
        "bg_secondary": "#0D0221",
        "text": "#E0E0E0",
        "accent": "#BD00FF",
        "icon": "🚀",
        "gradient": "linear-gradient(135deg, #000000 0%, #1a0033 50%, #0D0221 100%)"
    }
}

# -------------------- SESSION STATE INITIALIZATION --------------------
if 'theme' not in st.session_state:
    st.session_state.theme = "Space"
if 'workflow_data' not in st.session_state:
    st.session_state.workflow_data = {}
if 'GEMINI_API_KEY' not in st.session_state:
    st.session_state['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY", "")

# -------------------- DYNAMIC CSS --------------------
def apply_theme(theme_name):
    theme = THEMES[theme_name]
    st.markdown(f"""
        <style>
            .main {{
                background: {theme['gradient']};
                color: {theme['text']};
            }}
            h1, h2, h3 {{
                color: {theme['primary']};
                text-align: center;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            }}
            .stButton>button {{
                background: linear-gradient(135deg, {theme['primary']} 0%, {theme['accent']} 100%);
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 24px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }}
            .stButton>button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            }}
            .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {{
                background-color: {theme['bg_secondary']};
                color: {theme['text']};
                border: 1px solid {theme['primary']};
                border-radius: 8px;
            }}
            .stFileUploader>div {{
                border: 2px dashed {theme['primary']};
                background-color: {theme['bg_secondary']};
                border-radius: 10px;
            }}
            .stExpander {{
                background-color: {theme['bg_secondary']};
                border: 1px solid {theme['primary']};
                border-radius: 10px;
            }}
            .keyword {{
                color: {theme['accent']};
                font-weight: bold;
                text-shadow: 0 0 5px {theme['accent']};
            }}
            .results-box, .markdown-preview {{
                background-color: {theme['bg_secondary']};
                border: 1px solid {theme['primary']};
                padding: 20px;
                border-radius: 12px;
                margin-top: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .theme-selector {{
                text-align: center;
                padding: 10px;
                margin-bottom: 20px;
            }}
            [data-testid="stSidebar"] {{
                background: {theme['gradient']};
            }}
            .stRadio > div {{
                background-color: {theme['bg_secondary']};
                padding: 10px;
                border-radius: 8px;
            }}
            div[data-baseweb="select"] > div {{
                background-color: {theme['bg_secondary']};
                border-color: {theme['primary']};
            }}
        </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state.theme)

# -------------------- AGENT & API CONFIGURATION --------------------
@st.cache_data
def load_agents_config():
    """Load agents configuration from agents.yaml."""
    try:
        # Try multiple possible paths for Hugging Face deployment
        possible_paths = [
            Path(__file__).parent / "agents.yaml",
            Path("agents.yaml"),
            Path("/app/agents.yaml"),
            Path("./agents.yaml")
        ]
        
        for config_path in possible_paths:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
    except Exception as e:
        st.warning(f"Could not load agents.yaml: {e}. Using defaults.")

    # Fallback default configuration
    return {
        'agents': {
            'Data Transformer': {
                'description': 'Transforms raw data into Markdown tables.',
                'default_prompt': 'Transform the following data into a clean Markdown table. Infer headers if missing.',
                'temperature': 0.1,
                'max_tokens': 4096
            },
            'Data Summarizer': {
                'description': 'Summarizes data and extracts keywords.',
                'default_prompt': 'Analyze the data and generate a summary. After the summary, write "Keywords:" followed by 5-7 important keywords.',
                'temperature': 0.4,
                'max_tokens': 2048
            },
            'Insight Extractor': {
                'description': 'Extracts key insights and patterns.',
                'default_prompt': 'From the data, identify significant insights, trends, or anomalies.',
                'temperature': 0.5,
                'max_tokens': 4096
            },
            'Follow-up Question Generator': {
                'description': 'Generates relevant follow-up questions.',
                'default_prompt': 'Based on the analysis, generate 3-5 insightful follow-up questions for further investigation.',
                'temperature': 0.6,
                'max_tokens': 2048
            }
        }
    }

# -------------------- HELPER FUNCTIONS --------------------
@st.cache_resource
def get_gemini_model():
    """Returns a cached instance of the Gemini model."""
    return genai.GenerativeModel("gemini-2.5-flash")

def execute_gemini_agent(prompt, data_context, temp, max_tok):
    """Executes a Gemini agent and returns (success, content)."""
    if not st.session_state.get('GEMINI_API_KEY'):
        return False, "Error: Gemini API key not configured."
    try:
        model = get_gemini_model()
        generation_config = genai.GenerationConfig(
            temperature=temp,
            max_output_tokens=max_tok
        )
        full_prompt = f"CONTEXT:\n{data_context}\n\n---\n\nTASK:\n{prompt}"
        response = model.generate_content(full_prompt, generation_config=generation_config)
        if not response.parts:
            return False, "API returned empty response. Check safety settings."
        return True, response.text
    except Exception as e:
        return False, f"API error: {str(e)}"

def parse_and_prepare_data(files, text):
    """Reads uploaded files or pasted text."""
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
                st.error(f"Error processing {file.name}: {e}")
            content += f"--- END OF FILE: {file.name} ---\n\n"
    elif text:
        content = text
    return content

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Theme Selector
    st.subheader("🎨 Select Theme")
    theme_cols = st.columns(3)
    theme_names = list(THEMES.keys())
    
    for idx, theme_name in enumerate(theme_names):
        col = theme_cols[idx % 3]
        with col:
            if st.button(f"{THEMES[theme_name]['icon']} {theme_name}", key=f"theme_{theme_name}"):
                st.session_state.theme = theme_name
                st.rerun()
    
    st.markdown("---")
    
    # API Key Configuration
    st.subheader("🔑 API Key")
    api_key_input = st.text_input(
        "Gemini API Key:",
        type="password",
        value=st.session_state.get('GEMINI_API_KEY', ''),
        help="Enter your Google Gemini API key"
    )
    if st.button("Set API Key"):
        st.session_state['GEMINI_API_KEY'] = api_key_input
        try:
            genai.configure(api_key=api_key_input)
            st.success("✅ API Key configured!")
        except Exception as e:
            st.error(f"❌ Configuration failed: {e}")

# Configure API if key exists
if st.session_state.get('GEMINI_API_KEY'):
    try:
        genai.configure(api_key=st.session_state['GEMINI_API_KEY'])
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
else:
    st.warning("⚠️ Please provide your Gemini API Key in the sidebar.")
    st.info("Get your API key at: https://makersuite.google.com/app/apikey")
    st.stop()

# -------------------- MAIN UI --------------------
theme = THEMES[st.session_state.theme]
st.title(f"{theme['icon']} Multi-Agent Analysis Hub")
st.markdown(f"<p style='text-align: center; color: {theme['secondary']}; font-size: 18px;'>Current Theme: <strong>{st.session_state.theme}</strong></p>", unsafe_allow_html=True)

st.header("📊 Step 1: Upload or Paste Your Data")

agents_config = load_agents_config()
agents = agents_config.get('agents', {})

uploaded_files = st.file_uploader(
    "📂 Upload Datasets (CSV, JSON, TXT)",
    type=["csv", "json", "txt"],
    accept_multiple_files=True
)
pasted_text = st.text_area(
    "📋 Or Paste Raw Data:",
    height=200,
    placeholder="Paste your data here..."
)

if st.button("🚀 Process and Summarize Data", type="primary"):
    # Clear previous results
    keys_to_clear = ['step_1_markdown', 'step_1_markdown_edited', 'step_2_summary_html', 
                     'final_result', 'multi_agent_steps', 'follow_up_questions']
    for key in keys_to_clear:
        st.session_state.workflow_data.pop(key, None)

    raw_data = parse_and_prepare_data(uploaded_files, pasted_text)
    if raw_data:
        with st.spinner("🔄 Agent 1: Transforming data into Markdown..."):
            transformer = agents.get('Data Transformer', {})
            success, markdown_tables = execute_gemini_agent(
                transformer.get('default_prompt', ''),
                raw_data,
                transformer.get('temperature', 0.1),
                transformer.get('max_tokens', 4096)
            )
        
        if success:
            st.session_state.workflow_data['step_1_markdown'] = markdown_tables
            with st.spinner("🔄 Agent 2: Generating summary..."):
                summarizer = agents.get('Data Summarizer', {})
                success_summary, summary_and_keywords = execute_gemini_agent(
                    summarizer.get('default_prompt', ''),
                    markdown_tables,
                    summarizer.get('temperature', 0.4),
                    summarizer.get('max_tokens', 2048)
                )

            if success_summary:
                summary = summary_and_keywords
                keywords = []
                if "keywords:" in summary_and_keywords.lower():
                    parts = summary_and_keywords.split("Keywords:")
                    if len(parts) > 1:
                        summary = parts[0].strip()
                        keywords_str = parts[1].strip()
                        keywords = [k.strip() for k in keywords_str.split(',')]
                
                summary_html = summary
                for keyword in keywords:
                    summary_html = summary_html.replace(keyword, f'<span class="keyword">{keyword}</span>')
                
                st.session_state.workflow_data['step_2_summary_html'] = summary_html
                st.success("✅ Data processed successfully!")
            else:
                st.error(f"❌ Summarization failed: {summary_and_keywords}")
        else:
            st.error(f"❌ Transformation failed: {markdown_tables}")
    else:
        st.warning("⚠️ Please upload or paste data first.")
    
    st.rerun()

# -------------------- DATA DISPLAY AND MODIFICATION --------------------
if 'step_1_markdown' in st.session_state.workflow_data:
    st.header("✏️ Step 2: Review and Modify Data")
    
    st.subheader("📝 Edit Markdown Source")
    if 'step_1_markdown_edited' not in st.session_state.workflow_data:
        st.session_state.workflow_data['step_1_markdown_edited'] = st.session_state.workflow_data['step_1_markdown']
    
    st.session_state.workflow_data['step_1_markdown_edited'] = st.text_area(
        "Modify the Markdown tables below:",
        value=st.session_state.workflow_data['step_1_markdown_edited'],
        height=250,
        key="markdown_editor"
    )
    
    st.subheader("👁️ Live Preview")
    st.markdown(
        f"<div class='markdown-preview'>{st.session_state.workflow_data['step_1_markdown_edited']}</div>",
        unsafe_allow_html=True
    )

    if 'step_2_summary_html' in st.session_state.workflow_data:
        st.subheader("📊 Comprehensive Summary")
        st.markdown(
            f"<div class='results-box'>{st.session_state.workflow_data['step_2_summary_html']}</div>",
            unsafe_allow_html=True
        )

# -------------------- AGENT EXECUTION WORKFLOW --------------------
if 'step_1_markdown_edited' in st.session_state.workflow_data:
    st.header("🤖 Step 3: Configure Agent Workflow")
    
    workflow_type = st.radio(
        "Select Workflow Type:",
        ["Single Agent", "Multi-Agent Sequence"],
        horizontal=True
    )
    
    agent_names = [name for name in agents.keys() 
                   if name not in ['Data Transformer', 'Data Summarizer']]
    
    if workflow_type == "Single Agent":
        selected_agent = st.selectbox("Select Agent:", agent_names)
        if selected_agent:
            agent_config = agents.get(selected_agent, {})
            with st.expander(f"⚙️ Configure '{selected_agent}'", expanded=True):
                prompt = st.text_area(
                    "Prompt:",
                    value=agent_config.get('default_prompt', ''),
                    height=100,
                    key="single_prompt"
                )
                col1, col2 = st.columns(2)
                with col1:
                    temp = st.slider(
                        "Temperature:",
                        0.0, 1.0,
                        float(agent_config.get('temperature', 0.5)),
                        0.05,
                        key="single_temp"
                    )
                with col2:
                    max_tok = st.number_input(
                        "Max Tokens:",
                        512, 8192,
                        int(agent_config.get('max_tokens', 4096)),
                        key="single_tokens"
                    )

            if st.button(f"🚀 Execute {selected_agent}", type="primary"):
                with st.spinner(f"Running {selected_agent}..."):
                    initial_data = st.session_state.workflow_data['step_1_markdown_edited']
                    success, result = execute_gemini_agent(prompt, initial_data, temp, max_tok)
                    if success:
                        st.session_state.workflow_data['final_result'] = result
                        st.success("✅ Agent execution completed!")
                        st.rerun()
                    else:
                        st.error(f"❌ Execution failed: {result}")

    else:  # Multi-Agent Sequence
        selected_agents = st.multiselect("Select agents in order:", agent_names)
        
        if 'multi_agent_configs' not in st.session_state:
            st.session_state.multi_agent_configs = {}
        
        for agent_name in selected_agents:
            agent_config = agents.get(agent_name, {})
            with st.expander(f"⚙️ Configure: {agent_name}", expanded=False):
                prompt = st.text_area(
                    "Prompt:",
                    value=agent_config.get('default_prompt', ''),
                    height=100,
                    key=f"multi_prompt_{agent_name}"
                )
                col1, col2 = st.columns(2)
                with col1:
                    temp = st.slider(
                        "Temperature:",
                        0.0, 1.0,
                        float(agent_config.get('temperature', 0.5)),
                        0.05,
                        key=f"multi_temp_{agent_name}"
                    )
                with col2:
                    max_tok = st.number_input(
                        "Max Tokens:",
                        512, 8192,
                        int(agent_config.get('max_tokens', 4096)),
                        key=f"multi_tokens_{agent_name}"
                    )
                st.session_state.multi_agent_configs[agent_name] = {
                    'prompt': prompt,
                    'temp': temp,
                    'max_tok': max_tok
                }

        if st.button("🚀 Execute Multi-Agent Workflow", type="primary"):
            if selected_agents:
                st.session_state.workflow_data.pop('final_result', None)
                st.session_state.workflow_data.pop('multi_agent_steps', None)
                
                with st.spinner("Running multi-agent workflow..."):
                    current_data = st.session_state.workflow_data['step_1_markdown_edited']
                    st.session_state.multi_agent_steps = []
                    workflow_failed = False
                    
                    for i, agent_name in enumerate(selected_agents):
                        st.info(f"▶️ Step {i+1}: {agent_name}")
                        config = st.session_state.multi_agent_configs[agent_name]
                        success, result = execute_gemini_agent(
                            config['prompt'],
                            current_data,
                            config['temp'],
                            config['max_tok']
                        )
                        if success:
                            st.session_state.multi_agent_steps.append({
                                'agent': agent_name,
                                'output': result
                            })
                            current_data = result
                        else:
                            st.error(f"❌ Failed at Step {i+1} ({agent_name}): {result}")
                            workflow_failed = True
                            break
                    
                    if not workflow_failed:
                        st.session_state.workflow_data['final_result'] = current_data
                        st.success("✅ Multi-agent workflow completed!")
                        st.rerun()
            else:
                st.warning("⚠️ Please select at least one agent.")

# -------------------- MULTI-AGENT RESULTS --------------------
if 'multi_agent_steps' in st.session_state and st.session_state.multi_agent_steps:
    st.header("📋 Step 4: Review Workflow Results")
    
    for i, step in enumerate(st.session_state.multi_agent_steps):
        st.subheader(f"🔹 Step {i+1}: {step['agent']}")
        edited_output = st.text_area(
            f"Output from {step['agent']}:",
            value=step.get('output_edited', step['output']),
            height=200,
            key=f"edited_step_{i}"
        )
        st.session_state.multi_agent_steps[i]['output_edited'] = edited_output
    
    if st.button("💾 Update Final Result from Edits"):
        final_edited = st.session_state.multi_agent_steps[-1].get('output_edited', 
                                                                    st.session_state.multi_agent_steps[-1]['output'])
        st.session_state.workflow_data['final_result'] = final_edited
        st.success("✅ Final result updated!")
        st.rerun()

# -------------------- FINAL RESULT --------------------
if 'final_result' in st.session_state.workflow_data:
    st.header("🏆 Final Analysis Result")
    st.markdown(
        f"<div class='results-box'>{st.session_state.workflow_data['final_result']}</div>",
        unsafe_allow_html=True
    )
    
    if st.button("❓ Generate Follow-up Questions", type="primary"):
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
                st.rerun()
            else:
                st.error(f"❌ Could not generate questions: {questions}")

# -------------------- FOLLOW-UP QUESTIONS --------------------
if 'follow_up_questions' in st.session_state:
    st.header("💡 Suggested Follow-up Questions")
    st.markdown(
        f"<div class='results-box'>{st.session_state.follow_up_questions}</div>",
        unsafe_allow_html=True
    )

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    f"<p style='text-align: center; color: {theme['secondary']};'>"
    f"Built with ❤️ using <strong>Streamlit</strong> + <strong>Gemini API</strong> | "
    f"Multi-Agent System with Dynamic Themes</p>",
    unsafe_allow_html=True
)
