from doctest import Example
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os, random
import streamlit as st
import torch
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)

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

def truncate_input_sequence(document:str, max_num_tokens:int):
    total_length = len(document)
    if total_length <= max_num_tokens:
        return document
    else: 
        return document[:max_num_tokens]

def read_json(file):
    examples = []
    with open(file,'r',encoding='utf8')as fp:
        while True:
            line = fp.readline()
            if not line: #EOF
                break
            s = json.loads(line)
            examples.append(s)
    return examples

def get_example(examples,idx, context=2):
    #s = examples[random.randint(0,len(examples))]
    s = examples[idx]
    s["knowledge"] = truncate_input_sequence(s["knowledge"],256-2)
    if "[SEP]" in s["src"]:
        src = " ".join(s["src"].split("[SEP]")[-1*context:]) #上下文处理不太好，所以只暴露最后几条上下文，相当于滑动 windows
    else:
        src = s["src"]
    src = truncate_input_sequence(src, 224-2)
    return s["knowledge"], src, s["tgt"]




setup()

st.header("Demo for knowledge-based Dialogue")

# side bar
sbform = st.sidebar.form("参数设置")
n_sample = sbform.slider("设置返回条数",min_value=1,max_value=10,value=3)
text_length = sbform.slider('生成长度:',min_value=32,max_value=512,value=256,step=32)
#text_level = sbform.slider('文本多样性:',min_value=0.1,max_value=1.0,value=0.9,step=0.1)
top_k =      sbform.slider('Top-K:',min_value=0,max_value=20,value=10,step=1)
top_p =     sbform.slider('Top-P:',min_value=0.0,max_value=1.0,value=0.6,step=0.1)
rep_pen =      sbform.slider('Repeat Penalty:',min_value=0.0,max_value=1.0,value=0.6,step=0.1)
context = sbform.number_input('上下文句子数:',min_value=0,max_value=5,value=1,step=1)
model_id = sbform.selectbox('选择模型',
    ['Wenzhong-Finetune-110M-Loss0.39',
     'Wenzhong-Finetune-110M-Loss0.7',
     'Wenzhong-Finetune-110M-Loss0.2',
     'Wenzhong-3.5B'])
example_id = sbform.selectbox('选择样例',['None','1','2','3','4','5','6','7','8'])
sbform.form_submit_button("提交")

# model id config
if model_id == 'Wenzhong-Finetune-110M-Loss0.7':
    #model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/ckpt/hf_pretrained_epoch3_step3906"
    model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/Wenzhong-GPT2-110M/ckpt/hf_pretrained_model"
elif model_id == 'Wenzhong-Finetune-110M-Loss0.39':
    model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/ckpt_2/hf_pretrained_model"
elif model_id == "Wenzhong-Finetune-110M-Loss0.2":
    model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/Wenzhong-GPT2-110M/ckpt/hf_pretrained_epoch16_step20000"
elif model_id == "Wenzhong-3.5B":
    model_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-3.5B"
    token_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-3.5B"
else: 
    model_path = None
token_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"
tokenizer, model = load_model(model_path, token_path)
tokenizer.eos_token = '[SEP]'
tokenizer.pad_token = '[PAD]'

 
# main page
form = st.form("文本输入")

data_file = "/cognitive_comp/yangqi/data/DuSinc/dev_dial.json"
examples = read_json(data_file)
prepared_examples = [2,12,55,67,140,151,161,186]

if example_id is not "None":
    kno, src, tgt = get_example(examples, prepared_examples[int(example_id)-1], context)
    
    input_text_kno = form.text_input('知识',value=kno,placeholder=kno)
    input_text_dia = form.text_input('上文',value=src,placeholder=src)
else:
    input_text_kno = form.text_input('知识',value='',placeholder='')
    input_text_dia = form.text_input('上文',value='',placeholder='')

form.form_submit_button("提交")

with st.spinner('生成对话中'):
    if input_text_dia:
        if input_text_kno is None:
            input_text_kno = ""
        
        input_text = f'knowledge: {input_text_kno} question: {input_text_dia} answer:'
        input_text = tokenizer(input_text,return_tensors='pt')
        input_text.to(device)

        model.eval()

        with torch.no_grad():
            outputs = model.generate(
                **input_text,
                return_dict_in_generate=True,
                output_scores=True,
                #max_length=text_length+512,
                do_sample=True,
                #temperature = temp,
                top_k= top_k,
                top_p=top_p,
                repetition_penalty = rep_pen,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=0,
                num_return_sequences=n_sample,
                max_new_tokens=text_length,
            )
        # GreedySearchDecoderOnlyOutput(sequences=tensor([[seq1],[seq2]],device='cuda:0'), scores=None, attentions=None, hidden_states=None)

        answers = []
        
        for idx, sent in enumerate(outputs.sequences):
            result = tokenizer.decode(sent,skip_special_tokens=True)
            result = result.split(tokenizer.eos_token)[0]
            answer = result.split(sep="answer:",maxsplit=1)[1]
            answers.append(answer)

        #st.markdown(f"""**知识:**{input_text_kno}\n""")
        st.markdown(f"""**知识:** {input_text_kno}\n""")
        st.markdown(f"""**上文:** {input_text_dia}\n""")
        #st.markdown(f"""**回答:** {answers}\n""")
        for idx, ans in enumerate(answers):
            st.markdown(f"""**回答{idx}:** {ans}\n""")
        st.markdown(f"""**参考答案:** {tgt}\n""")
        
