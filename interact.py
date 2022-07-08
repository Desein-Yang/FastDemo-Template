
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os, random, re
import streamlit as st
import torch
import json
import jieba
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
def load_model(model_id):
    if model_id == 'Wenzhong-Finetune-110M-Loss0.7':
        #model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/ckpt/hf_pretrained_epoch3_step3906"
        model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/Wenzhong-GPT2-110M/ckpt/hf_pretrained_model"
    elif model_id == 'Wenzhong-Finetune-Loss0.39' or "Wenzhong-Finetune-Loss0.2" or "Wenzhong-Finetune-Loss0.1":
        model_path = f"/cognitive_comp/yangqi/model/{model_id}"
    elif model_id == "Wenzhong-GPT2-3.5B" or "Wenzhong-GPT2-110M":
        model_path = f"/cognitive_comp/yangqi/model/{model_id}"
        token_path = f"/cognitive_comp/yangqi/model/{model_id}"
    else: 
        model_path = None
    token_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"

    if model_path is None:
        model_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"

    tokenizer = GPT2Tokenizer.from_pretrained(token_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    model.to(device)

    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # [PAD]
    # tokenizer.add_special_tokens({"bos_token": "<s>"})  # <s>
    # tokenizer.add_special_tokens({"eos_token": "</s>"})  # </s>
    # tokenizer.add_special_tokens({"unk_token": "<unk>"})  # <unk>]
    tokenizer.eos_token = '[SEP]'
    tokenizer.pad_token = '[PAD]'

    return tokenizer, model

def load_from_json(file):
    examples = []
    with open(file,'r',encoding='utf8')as fp:
        while True:
            line = fp.readline()
            if not line: #EOF
                break
            s = json.loads(line)
            examples.append(s)
    return examples

def preprocess(examples,idx, context=2):

    def truncate(document:str, max_num_tokens:int):
        total_length = len(document)
        if total_length <= max_num_tokens:
            return document
        else: 
            return document[:max_num_tokens]

    #s = examples[random.randint(0,len(examples))]
    s = examples[idx]
    s["knowledge"] = truncate(s["knowledge"],256-2)
    if "[SEP]" in s["src"]:
        src = " ".join(s["src"].split("[SEP]")[-1*context:]) #上下文处理不太好，所以只暴露最后几条上下文，相当于滑动 windows
    else:
        src = s["src"]
    src = truncate(src, 128-2)
    return s["knowledge"], src, s["tgt"]

def generate(input_text_dia, input_text_kno, input_text_tgt):
    """ Input raw text -> candidates answers (return n_samples)
    """
    if input_text_kno is None:
        input_text_kno = ""
    
    if input_text_tgt is None:
        input_text_tgt = "手动输入，无"
        
    input_text = f'knowledge: {input_text_kno} context: {input_text_dia} response:'
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
            #temperature = 0.7,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty = rep_pen,
            eos_token_id=tokenizer.eos_token_id ,
            pad_token_id=0,
            num_return_sequences=n_sample,
            max_new_tokens=text_length,
        )
    # GreedySearchDecoderOnlyOutput(sequences=tensor([[seq1],[seq2]],device='cuda:0'), scores=None, attentions=None, hidden_states=None)

    answers = []
    
    for idx, sent in enumerate(outputs.sequences):
        result = tokenizer.decode(sent,skip_special_tokens=True)
        result = result.split(tokenizer.eos_token)[0]
        answer = result.split(sep="response:",maxsplit=1)[1]
        answers.append(answer)

    return {
        "answers": answers,
        "kno" : input_text_kno,
        "src" : input_text_dia,
        "tgt" : input_text_tgt
    }

def bleu_fn(references, candidates):
    score_list = []
    for ref, can in zip(references, candidates):
        can = normalize_answer(can)
        reference = [" ".join(jieba.cut(ref)).split()]  # may have multiple ref, need [[ref1]]
        candidate = " ".join(jieba.cut(can)).split()

        chencherry = SmoothingFunction()
        score = sentence_bleu(reference,candidate,weights=(0.5,0.5),smoothing_function=chencherry.method1)
        score_list.append(score)

    score = sum(score_list) / len(score_list)
    return score

def f1_fn(references, candidates):
    def pre_recall_f1(reference,candidate):
        from collections import Counter
        print(reference)
        print(candidate)
        common = Counter(reference) & Counter(candidate)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(candidate)
        recall = 1.0 * num_same / len(reference)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    pre, re, f1 = [],[],[]
    for ref, can in zip(references, candidates):
        can = normalize_answer(can)
        reference = [" ".join(jieba.cut(ref)).split()]
        candidate = " ".join(jieba.cut(can)).split()
        
        (_pre, _re, _f1)  = [pre_recall_f1(r, candidate) for r in reference][0]
        pre.append(_pre)
        re.append(_re)
        f1.append(_f1)
    return sum(pre)/len(pre), sum(re)/len(re), sum(f1)/len(f1)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    re_art = re.compile(r"\b(是|的|啊)\b")
    re_punc = re.compile(r"[!\"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\'，。？！]")
    def remove_articles(text):
        return re_art.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return re_punc.sub(" ", text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def inference(example_id):            
    data_file = "/cognitive_comp/yangqi/data/DuSinc/dev_dial.json"
    examples = load_from_json(data_file)
    if example_id is not "ALL": 
        form = st.form("文本输入")

        if example_id is not "None":    #load from dev
            prepared_examples = [2,12,55,67,136,151,161,120]
            kno, src, tgt = preprocess(examples, prepared_examples[int(example_id)-1], context)
        else:
            kno, src, tgt = "","",""   # from input
        
        input_text_kno = form.text_input('知识',value=kno,placeholder=kno)
        input_text_dia = form.text_input('上文',value=src,placeholder=src)
        input_text_tgt = tgt

        form.form_submit_button("提交")

        with st.spinner('生成对话中'):
            if input_text_dia:
                output = generate(input_text_dia, input_text_kno, input_text_tgt)

                #st.markdown(f"""**知识:**{input_text_kno}\n""")
                st.markdown(f"""**知识:** {output["kno"]}\n""")
                st.markdown(f"""**上文:** {output["src"]}\n""")
                #st.markdown(f"""**回答:** {answers}\n""")
                for idx, ans in enumerate(output["answers"]):
                    st.markdown(f"""**回答{idx}:** {ans}\n""")
                st.markdown(f"""**参考答案:** {output["tgt"]}\n""")
    else: # eval with dev
        candidates, references = [],[]
        with st.spinner("正在评估中"):
            #for idx in tqdm(range(len(examples))):
            for idx in tqdm(range(30)):
                kno, src, tgt = preprocess(examples, idx, context=1)
                output = generate(src, kno, tgt)

                candidates.append(output["answers"][0])
                references.append(output["tgt"])

            st.write("candidate")
            st.write(candidates)
            st.write("reference")
            st.write(references)
            bleu_score = bleu_fn(references, candidates)
            f1_score = f1_fn(references, candidates)
            st.write(f"Bleu score on dev : {bleu_score:.4f}")
            st.write(f"Prec score on dev : {f1_score[0]:.4f}")
            st.write(f"Re   score on dev : {f1_score[1]:.4f}")
            st.write(f"F1   score on dev : {f1_score[2]:.4f}")






setup()

st.header("Demo for knowledge-based Dialogue")

# side bar
sbform = st.sidebar.form("参数设置")
n_sample = sbform.slider("设置返回条数",min_value=1,max_value=10,value=3)
text_length = sbform.slider('生成长度:',min_value=32,max_value=256,value=80,step=16)
#text_level = sbform.slider('文本多样性:',min_value=0.1,max_value=1.0,value=0.9,step=0.1)
top_k =      sbform.slider('Top-K:',min_value=0,max_value=20,value=0,step=1)
top_p =     sbform.slider('Top-P:',min_value=0.0,max_value=1.0,value=0.6,step=0.1)
rep_pen =      sbform.slider('Repeat Penalty:',min_value=1.0,max_value=2.0,value=1.1,step=0.1)
context = sbform.number_input('上下文句子数:',min_value=0,max_value=5,value=1,step=1)
model_id = sbform.selectbox('选择模型',
    ['Wenzhong-Finetune-Loss0.2',
     'Wenzhong-Finetune-Loss0.1',
     'Wenzhong-GPT2-110M',
     'Wenzhong-GPT2-3.5B'])
example_id = sbform.selectbox('选择样例',['None','ALL','1','2','3','4','5','6','7','8'])
sbform.form_submit_button("提交")

# model id config
tokenizer, model = load_model(model_id)

# get data
inference(example_id)

    




        
