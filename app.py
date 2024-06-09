import streamlit as st
from PIL import Image
from gtts import gTTS
import pyttsx3

# from transformers import pipeline
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration


def text_to_speech(text):
    # Initialize the pyttsx3 engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty("rate", 150)  # Speed percent (can go over 100)
    engine.setProperty("volume", 0.9)  # Volume (0.0 to 1.0)

    # Convert text to speech
    engine.say(text)

    # Wait for speech to finish
    engine.runAndWait()


st.title("Blind Assistance System")
st.write("---")
# Take inputs from the streamlit
model_name = st.sidebar.selectbox("Select Pre-trained Model:", ("BLIP-Long", "BLIP-Large", "VIT-GPT-2", "BLIP-Base"))
st.write("The purpose of this project is to build a blind assistance system using custom or existing pre-trained models. I have implemented it using 3 different models.")
st.write("1]VIT-GPT2: The Vision Encoder Decoder Model has been used to initialize an image-to-text model using VIT as the encoder and the pre-trained language model GPT2 as the decoder.")
st.write("2]BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation: a)Large and b) Base. BLIP is a new Vision Language Processing framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones.")

uploaded_file = st.camera_input(label="Capture an image", label_visibility="visible")

if uploaded_file is not None:

    if model_name == "VIT-GPT-2":
        st.write("VIT-GPT-2 model is selected.")
        # image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

        model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        feature_extractor = ViTImageProcessor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image.", use_column_width=True)
        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
        pixel_values = feature_extractor(
            images=[image1], return_tensors="pt"
        ).pixel_values
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        st.write(preds[0])
        text_to_speech(preds[0])
    elif model_name == "BLIP-Large":
        st.write("BLIP-Large model is selected.")
        # image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

        processor_blip_large = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        model_blip_large = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        inputs = processor_blip_large(image, return_tensors="pt")
        out = model_blip_large.generate(**inputs)
        # st.write(image_to_text(image)[0]["generated_text"])
        # text_to_speech(image_to_text(image)[0]["generated_text"])
        st.write(processor_blip_large.decode(out[0], skip_special_tokens=True))
        text_to_speech(processor_blip_large.decode(out[0], skip_special_tokens=True))
    elif model_name == "BLIP-Long":
        st.write("BLIP-Long model is selected.")
        # image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

        processor_blip_large = BlipProcessor.from_pretrained(
            "unography/blip-large-long-cap"
        )
        model_blip_large = BlipForConditionalGeneration.from_pretrained(
            "unography/blip-large-long-cap"
        )
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        inputs = processor_blip_large(image, return_tensors="pt")
        pixel_values = inputs.pixel_values
        out = model_blip_large.generate(pixel_values=pixel_values, max_length=250)
        st.write(processor_blip_large.decode(out[0], skip_special_tokens=True))
        text_to_speech(processor_blip_large.decode(out[0], skip_special_tokens=True))
    else:
        st.write("BLIP-Base model is selected.")
        # image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

        processor_blip = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model_blip = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        inputs = processor_blip(image, return_tensors="pt")
        out = model_blip.generate(**inputs)
        st.write(processor_blip.decode(out[0], skip_special_tokens=True))
        text_to_speech(processor_blip.decode(out[0], skip_special_tokens=True))
