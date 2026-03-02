# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from streamlit as st 
# load_dotenv()
# model=ChatGroq(model="")
# st.header('Research tool')
# user=st.text_input('enter your prompt')
# if st.button('summarize')

#import the chat model
from langchain_groq import ChatGroq
# import load_dotenv
from dotenv import load_dotenv
#UI-streamlit
import streamlit as st # TO do this install streamit by run this command pip install streamlit
#activate the dotenv
load_dotenv()
# Initialize the model
model = ChatGroq(model='llama-3.1-8b-instant')


st.header('Research Tool')
user_input = st.text_input('Enter your prompt')
if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)