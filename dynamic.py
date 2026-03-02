from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import prompt, prompttemplate
import streamlit as st # TO do this install streamit by run this command pip install streamlit
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
st.header('Research Tool')
paper_input = st.selectbox(
    "Select Research Paper Name",
[
"Select...",
"Attention Is All You Need",
"BERT: Pre-training of Deep Bidirectional Transformers",
"GPT-3: Language Models are Few-Shot Learners",
"Diffusion Models Beat GANs on Image Synthesis"
]
)
style_input = st.selectbox(
"Select Explanation Style",
["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)
length_input = st.selectbox(
"Select Explanation Length",
["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# fill the placeholder
prompt = template.invoke({'paper_input': paper_input,
'style_input': style_input,
'length_input': length_input 
})

if st.button('Summarize'):
result = model.invoke(prompt)
st.write(result.content)