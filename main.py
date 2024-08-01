import streamlit as st
import pandas as pd
import time
from datetime import datetime
from models import leaf_disease_detection

st.set_page_config(page_title='NeubAltics - Dileepkumar Assignment')
st.title('NeubAltics - Dileepkumar Assignment Submission')

upload_file = st.file_uploader('Upload the Image (JPG) File', type={'JPG'})

path = upload_file

if upload_file is not None:
    #text = upload_file.getvalue() #.decode("utf-8")
    st.success("Successfully Uploaded")
    
    st.image(path)
    
    leaf_disease_detection.testing(path)

else:
    st.warning('Please upload the JPG file')
