import streamlit as st
from huggingface_hub import InferenceClient
import os
import sys

st.title("CODEFUSSION ☄")  

base_url = "https://api-inference.huggingface.co/models/"

API_KEY = os.environ.get('HUGGINGFACE_API_KEY')
# print(API_KEY)
# headers = {"Authorization":"Bearer "+API_KEY}

model_links = {
    "LegacyLift🚀": base_url + "mistralai/Mistral-7B-Instruct-v0.2",  
    "ModernMigrate⭐": base_url + "mistralai/Mixtral-8x7B-Instruct-v0.1",  
    "RetroRecode🔄": base_url + "microsoft/Phi-3-mini-4k-instruct" 
}

# Pull info about the model to display
model_info = {
    "LegacyLift🚀": {
        'description': """The LegacyLift model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
            \nThis model is best for minimal problem-solving, content writing, and daily tips.\n""",
        'logo': './11.jpg'
    },

    "ModernMigrate⭐": {
        'description': """The ModernMigrate model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
            \nThis model excels in coding, logical reasoning, and high-speed inference. \n""",
        'logo': './2.jpg'
    },

    "RetroRecode🔄": {
        'description': """The RetroRecode model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
          \nThis model is best suited for critical development, practical knowledge, and serverless inference.\n""",
        'logo': './3.jpg'
    },
}

def format_promt(message, custom_instructions=None):
    prompt = ""
    if custom_instructions:
        prompt += f"[INST] {custom_instructions} [/INST]"
    prompt += f"[INST] {message} [/INST]"
    return prompt

def reset_conversation():
    '''
    Resets Conversation
    '''
    st.session_state.conversation = []
    st.session_state.messages = []
    return None

models = [key for key in model_links.keys()]

selected_model = st.sidebar.selectbox("Select Model", models)

temp_values = st.sidebar.slider('Select a temperature value', 0.0, 1.0, (0.5))

st.sidebar.button('Reset Chat', on_click=reset_conversation)  # Reset button

st.sidebar.write(f"You're now chatting with **{selected_model}**")
st.sidebar.markdown(model_info[selected_model]['description'])
st.sidebar.image(model_info[selected_model]['logo'])
st.sidebar.markdown("*Generating the code might go slow if you are using low power resources *")


if "prev_option" not in st.session_state:
    st.session_state.prev_option = selected_model

if st.session_state.prev_option != selected_model:
    st.session_state.messages = []
    # st.write(f"Changed to {selected_model}")
    st.session_state.prev_option = selected_model
    reset_conversation()

repo_id = model_links[selected_model]

st.subheader(f'{selected_model}')
# st.title(f'ChatBot Using {selected_model}')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"Hi I'm {selected_model}, How can I help you today?"):
    custom_instruction = "Act like a Human in conversation"

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    formated_text = format_promt(prompt, custom_instruction)

    with st.chat_message("assistant"):
        client = InferenceClient(
            model=model_links[selected_model], )

        output = client.text_generation(
            formated_text,
            temperature=temp_values,  # 0.5
            max_new_tokens=3000,
            stream=True
        )

        response = st.write_stream(output)
    st.session_state.messages.append({"role": "assistant", "content": response})