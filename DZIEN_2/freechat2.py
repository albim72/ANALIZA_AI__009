import os
import ipywidgets as widgets
from transformers import pipeline
from IPython.display import display

# Instalacja brakujących pakietów
def install_dependencies():
    try:
        import transformers
        import ipywidgets
    except ImportError:
        os.system("pip install transformers ipywidgets")

install_dependencies()

# Ładowanie modelu NLP
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")

def chat_response(message):
    response = chatbot(message, max_length=100, pad_token_id=50256)
    return response[0]['generated_text']

# Interfejs użytkownika w Jupyter Notebook
input_box = widgets.Text(placeholder="Wpisz wiadomość...")
output_area = widgets.Output()

def on_send(_):
    user_input = input_box.value
    if user_input.strip():
        input_box.value = ""
        with output_area:
            print(f"Ty: {user_input}")
            bot_response = chat_response(user_input)
            print(f"Bot: {bot_response}")

input_box.on_submit(on_send)

display(input_box, output_area)
