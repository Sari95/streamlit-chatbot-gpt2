# Importing the libraries
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Loading the GPT-2 model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

st.title("Einfacher Chatbot mit GPT-2")

# Initializing the chat history if it doesn't already exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Showing the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Responding to the user's input
if prompt := st.chat_input("Schreibe deine Nachricht:"):
    # Showing the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tokenizing the input and generating the response with GPT-2
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=100)
    
    # Decoding the model's output to text
    bot_message = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Showing the response of the chatbot
    st.session_state.messages.append({"role": "assistant", "content": bot_message})
    with st.chat_message("assistant"):
        st.markdown(bot_message)

