from doctest import Example
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
import streamlit as st
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup():
    st.set_page_config(
        page_title="知识辅助对话", 
        page_icon=":shark:",
        layout="wide",
        initial_sidebar_state="expanded", 
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",            'About': "# This is a header. This is an *extremely* cool app!"}
    )

#@st.cache(suppress_st_warning=True)
def load_model(hf_model_path, hf_token_path):
    if hf_model_path is None:
        hf_model_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"
    tokenizer = GPT2Tokenizer.from_pretrained(hf_token_path)
    model = GPT2LMHeadModel.from_pretrained(hf_model_path)

    model.to(device)

    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # [PAD]
    # tokenizer.add_special_tokens({"bos_token": "<s>"})  # <s>
    # tokenizer.add_special_tokens({"eos_token": "</s>"})  # </s>
    # tokenizer.add_special_tokens({"unk_token": "<unk>"})  # <unk>]
    return tokenizer, model


setup()

st.header("Demo for knowledge-based Dialogue")

# side bar
sbform = st.sidebar.form("参数设置")
n_sample = sbform.slider("设置返回条数",min_value=1,max_value=10,value=3)
text_length = sbform.slider('生成长度:',min_value=32,max_value=512,value=256,step=32)
# text_level = sbform.slider('文本多样性:',min_value=0.1,max_value=1.0,value=0.9,step=0.1)
max_new_tokens = sbform.number_input('最大新词数:',min_value=0,max_value=20,value=10,step=1)
model_id = sbform.selectbox('选择模型',['Wenzhong-110M','Wenzhong-Finetune-110M','Wenzhong-3.5B'])
example = sbform.selectbox('选择样例',['None','1'])
sbform.form_submit_button("提交")

# model id config
if model_id == 'Wenzhong-110M':
    #model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/ckpt/hf_pretrained_epoch3_step3906"
    model_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"
    token_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"
elif model_id == 'Wenzhong-Finetune-110M':
    model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/ckpt_2/hf_pretrained_model"
    token_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"
elif model_id == "Wenzhong-3.5B":
    model_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-3.5B"
    token_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-3.5B"
else: 
    model_path = None
    token_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"
tokenizer, model = load_model(model_path, token_path)
tokenizer.bos_token = '[SEP]'
tokenizer.eos_token = '[SEP]'
tokenizer.pad_token = '[PAD]'

if example == '1':
    kno = "三星堆遗址是公元前16世纪至公元前14世纪世界青铜文明的重要代表，对研究早期国家的进程及宗教意识的发展有重要价值，在人类文明发展史上占有重要地位。 它是中国西南地区一处具有区域中心地位的最大的都城遗址。 它的发现，为已消逝的 古蜀国 提供了独特的物证，把四川地区的文明史向前推进了2000多年。 [2] 三星堆遗址的发现，始于当地农民燕道诚于1929年淘沟时偶然发现的一坑玉石器。"
    src = "三星堆的历史有多少年"
    
# main page
form = st.form("文本输入")

if example is not "None":
    input_text_kno = form.text_input('知识',value='',placeholder=kno)
    input_text_dia = form.text_input('上文',value='',placeholder=src)
else:
    input_text_kno = form.text_input('知识',value='',placeholder='')
    input_text_dia = form.text_input('上文',value='',placeholder='')

form.form_submit_button("提交")
with st.spinner('生成对话中'):
    if input_text_kno and input_text_dia:
        
        input_text = f'knowledge: {input_text_kno} question: {input_text_dia} answer:'
        input_text = tokenizer(input_text,return_tensors='pt')
        input_text.to(device)
        outputs = model.generate(
            **input_text,
            return_dict_in_generate=True,
            output_scores=True,
            max_length=text_length+len(input_text_kno+input_text_dia),
            do_sample=True,
            top_k=10,
            top_p=0.6,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=0,
            num_return_sequences=n_sample,
            max_new_tokens=max_new_tokens,
        )
        # GreedySearchDecoderOnlyOutput(sequences=tensor([[seq1],[seq2]],device='cuda:0'), scores=None, attentions=None, hidden_states=None)

        answers = []
        
        for idx, sent in enumerate(outputs.sequences):
            result = tokenizer.decode(sent,skip_special_tokens=True)
            print(result)

            result = result.split(tokenizer.eos_token)[0]
            print(result)
            answer = result.split(sep="answer:",maxsplit=1)[1]
            answers.append(answer)

        #st.markdown(f"""**知识:**{input_text_kno}\n""")
        st.markdown(f"""**上文:** {input_text_dia}\n""")
        #st.markdown(f"""**回答:** {answers}\n""")
        for idx, ans in enumerate(answers):
            st.markdown(f"""**回答{idx}:** {ans}\n""")
        
