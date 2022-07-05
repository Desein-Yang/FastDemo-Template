
from unittest.util import _MAX_LENGTH
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import streamlit as st
import pandas as pd
import torch
import time
def read_data(name):
    root_dir = "/cognitive_comp/yangqi/data"
    curdir_list = os.listdir(os.path.join(root_dir, name))
    display,df_name = [],[]
    for f in curdir_list:
        f = root_dir + "/" + name + "/" + f
        if f.split(".")[-1] == "txt" and 'dial' in f:
            df = pd.read_csv(f, sep="\t")
            display.append(df)
            df_name.append(f)
    return display,df_name

def show_data(df,name,rows=3):
    df = df.sample(n=rows, axis=0)
    
    st.write("Data source : " + name)
    st.write("Data rows   : " + str(df.shape[0]))
    st.table(df)


# set up
def setup():
    st.set_page_config(
        page_title="知识辅助对话数据", 
        page_icon=":shark:",
        layout="wide",
        initial_sidebar_state="expanded", 
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",            'About': "# This is a header. This is an *extremely* cool app!"}
    )

# dataset path list (csv, txt)
datasets = ["DuSinc"]

setup()
# dataset
st.header("Dataset")
#st.write("This is an overview of datasets.")

for name in datasets:
    display,df_name = read_data(name)

for df,n in zip(display,df_name):
    show_data(df,n)
