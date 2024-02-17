import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model('C:/Users/Dell/PycharmProjects/plant_disease_prediction/plant_disease.h5')

CLASS_NAMES = ['Tomato-Bacterial_spot','Potato-Early_blight','Corn-Common_rust']

st.title('Plant Disease Detection')
st.markdown("Upload an image of the plant leaf")

plant_image = st.file_uploader("Chose an image....", type = "jpg")
submit = st.button('Predict')

if submit:

    if plant_image is not None:
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR")
        opencv_image = cv2.resize(opencv_image, (256,256))
        opencv_image.shape = (1, 256, 256, 3)
        y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(y_pred)]
        result = result.split('-')
        disease = result[1].split('_')
        st.title(str("This is a "+ result[0] + " leaf with " + disease[0] + " " + disease[1]))

        if result[0] == 'Potato':
            st.write("""Potato Early Blight Overview
Potato early blight, caused by the fungal pathogen Alternaria solani, is characterized by dark, concentric
lesions on potato leaves, impacting plant health and yield. The term \"early bird\" is often associated with
the disease due to its tendency to appear early in the growing season.
Symptoms:
1. Leaf Lesions: Dark, concentric lesions with a target-like appearance develop on the leaves,
typically starting as small spots and enlarging over time.
2. Leaf Yellowing: Surrounding the lesions, the affected areas of the leaves often exhibit yellowing,
leading to a reduction in chlorophyll and overall plant vigor.
3. Stem and Tuber Infection: In severe cases, stems and even tubers may become infected,
resulting in cankers or lesions that impact the quality and yield of the potato crop.
4. Foliage Dieback: Advanced stages of early blight can lead to premature defoliation, reducing the
plant\'s ability to photosynthesize and affecting the overall health of the potato plant. \n
Prevention Strategies:
1. Crop Rotation: Rotate potato crops to different fields to disrupt the disease cycle.
2. Varietal Selection: Choose potato varieties that demonstrate resistance to early blight.
3. Field Sanitation: Practice good hygiene by removing and destroying infected plant material to
limit disease spread.
4. Fungicide Application: Use fungicides judiciously and according to recommended schedules for
effective disease control.
5. Cultural Practices: Optimize plant spacing, promote air circulation, and avoid overhead irrigation
to create less favorable conditions for the fungus.
Implementing these prevention measures collectively can help manage and mitigate the impact of
potato early blight in agricultural settings.""")
        elif result[0] == 'Tomato':
            st.write("""Tomato Bacterial Spot Overview:\n
Tomato bacterial spot is a plant disease caused by the bacterium Xanthomonas campestris pv.
vesicatoria. It primarily affects tomatoes but can also impact other related crops. The disease manifests
as small, dark lesions on leaves, fruit, and stems, leading to reduced plant vigor and yield.

Symptoms:

1. Lesions: Characterized by small, dark spots with a water-soaked appearance on leaves, fruit, and
stems.
2. Leaf Yellowing: Infected leaves may exhibit yellowing around the lesions, affecting overall plant
health.
3. Fruit Damage: Bacterial spot can cause blemishes and discoloration on tomatoes, rendering
them unmarketable. \n
Prevention and Management:
1. Resistant Varieties: Choose tomato varieties with resistance to bacterial spot.
2. Crop Rotation: Rotate tomatoes with non-host crops to break the disease cycle.
3. Sanitation: Remove and destroy infected plant material to reduce bacterial inoculum.
4. Copper-based Sprays: Apply copper-based fungicides early in the growing season as a preventive
measure.
5. Avoid Overhead Irrigation: Minimize moisture on foliage, as the bacterium thrives in wet
conditions.""")
        else:
            st.write("""Corn Common Rust Overview:
Corn common rust is a fungal disease caused by the pathogen Puccinia sorghi. It affects corn plants and
can lead to yield reduction. The disease is characterized by the development of small, raised, reddish-
brown pustules on both sides of the corn leaves, which contain masses of rust-colored spores. While
common rust typically does not cause severe damage, severe infections can weaken the plant and
impact grain quality.\n
Symptoms:

1. Reddish-Brown Pustules: Small, raised lesions with a reddish-brown appearance develop on the
upper and lower surfaces of corn leaves.
2. Linear Arrangement: Pustules often align in linear patterns along the leaf veins.
3. Leaf Discoloration: As the disease progresses, affected leaves may turn yellow, leading to
premature senescence.\n
Management:

1. Fungicides: Application of fungicides, especially during the early stages of infection, can help
manage common rust.
2. Resistant Varieties: Planting corn varieties with resistance to common rust can reduce the
impact of the disease.
3. Crop Rotation: Rotate corn crops with non-host plants to break the disease cycle.
4. Sanitation: Remove and destroy infected crop residues to minimize overwintering spores.

Understanding and promptly addressing common rust symptoms are essential for implementing
effective management strategies and preserving corn yield.""")
