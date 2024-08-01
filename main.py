import streamlit as st
import pandas as pd
import time
from datetime import datetime
from models import leaf_disease_detection

st.set_page_config(page_title='NeubAltics - Assignment')
st.title('NeubAltics - Assignment Submission')

upload_file = st.file_uploader('Upload the Image (JPG) File', type={'JPG'})

if upload_file is not None:
    #text = upload_file.getvalue() #.decode("utf-8")
    st.success("Successfully Uploaded")
    
    st.image(upload_file)
    
    leaf_disease_detection.testing(upload_file)

else:
    st.warning('Please upload the JPG file')
