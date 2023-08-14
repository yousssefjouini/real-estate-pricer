from __future__ import annotations

from PIL import Image
import streamlit as st

import os

st.write('Welcome to the model interpretability page')

for root, folders, files in os.walk('.'):
    if 'feature_imp.png' in files:
        path = root
        break
image1 = Image.open(f'{path}/feature_imp.png')
image2 = Image.open(f'{path}/subpop.png')
image3 = Image.open(f'{path}/shap.png')
st.write('Feature importance graph')
st.image(image1, caption='Feature importance')
st.write('Graph on Subpopulation metrics depending on surface')
st.image(image2, caption='Subpopulation metrics')
st.write('Shapley Additive Explanations graph')
st.image(image3, caption='Shap plot')
