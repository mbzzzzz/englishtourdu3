pip install streamlit transformers
pip install torch



import streamlit as st
import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# Load the fine-tuned model
model_name = "abdulwaheed1/english-to-urdu-translation-mbart"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="en_XX", tgt_lang="ur_PK")
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Streamlit app layout
st.title("English to Urdu Translation")
st.write("Enter English text below and click the button to translate it to Urdu.")

# Text input
english_text = st.text_area("Input English Text")

# Button for translation
if st.button("Translate"):
    if english_text:
        # Tokenization
        inputs = tokenizer(english_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate translation
        with torch.no_grad():
            translated_ids = model.generate(**inputs)
        
        # Decode translation
        urdu_translation = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        
        # Display result
        st.subheader("Translated Text:")
        st.write(urdu_translation)
    else:
        st.error("Please enter some English text to translate.")

streamlit run app.py
