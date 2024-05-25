from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import streamlit as st
# Load pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("model")
feature_extractor = ViTImageProcessor.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to generate captions
def generate_caption(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# Streamlit app
def main():
    st.title("Image Captioning App")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display uploaded image
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
        st.write("")

        # Generate caption on button click
        if st.button("Generate Caption"):
            image_paths = [uploaded_image]
            captions = generate_caption(image_paths)
            st.write("Generated Caption:", captions[0])

if __name__ == "__main__":
    main()
