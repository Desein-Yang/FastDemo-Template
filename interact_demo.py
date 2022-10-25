
import streamlit as st
#from fengshen.examples.wenzhong_dialo.eval import DialogueTest, QueryTest

import argparse
from collections import Counter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os, random, re
import torch
import json
import jieba
from torchmetrics.functional import bleu_score
#from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestModule:
    def __init__(self, **args):
        self.args = argparse.Namespace(**args)
        self.model_id = self.args.model_id
        self.tokenizer, self.model = self.load_model(self.args.model_id)

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def load_model(self, model_id):
        if model_id == 'Wenzhong-Finetune-CKPT':
            #model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/ckpt/hf_pretrained_epoch3_step3906"
            # model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/Wenzhong-GPT2-110M/ckpt/hf_pretrained_model"
            model_path = "/cognitive_comp/yangqi/logs/wenzhong_dusinc/Wenzhong-GPT2-110M/ckpt_dial/hf_pretrained_epoch5_step5860/"
        elif model_id == 'Wenzhong-Finetune-Loss0.39' or "Wenzhong-Finetune-Loss0.2" or "Wenzhong-Finetune-Loss0.1" or "Wenzhong-Finetune-Query-Loss0.1":
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
    
    def load_from_json(self, data_file):
        examples = []
        with open(data_file,'r',encoding='utf8')as fp:
            while True:
                line = fp.readline()
                if not line: #EOF
                    break
                s = json.loads(line)
                examples.append(s)
        return examples

    @staticmethod
    def truncate(document:str, max_num_tokens:int):
        total_length = len(document)
        if total_length <= max_num_tokens:
            return document
        else: 
            return document[:max_num_tokens]

    def preprocess(self): # need rewrite with each task
        raise NotImplementedError

    def generate(self,input_dict,prompt):
        with torch.no_grad():
            outputs = self.model.generate(
                **input_dict["input_text"],
                return_dict_in_generate=True,
                output_scores=True,
                #max_length=text_length+512,
                do_sample=True,
                num_beams=4,
                temperature = 0.9,
                top_k = self.args.top_k,
                top_p = self.args.top_p,
                repetition_penalty = self.args.rep_pen,
                eos_token_id = self.tokenizer.eos_token_id ,
                pad_token_id = 0,
                num_return_sequences = self.args.n_sample,
                max_new_tokens       = self.args.text_length,
            )
        # GreedySearchDecoderOnlyOutput(sequences=tensor([[seq1],[seq2]],device='cuda:0'), scores=None, attentions=None, hidden_states=None)

        answers = []
        
        for idx, sent in enumerate(outputs.sequences):
            result = self.tokenizer.decode(sent,skip_special_tokens=True)
            result = result.split(self.tokenizer.eos_token)[0]
            answer = result.split(sep=prompt,maxsplit=1)[1]
            #else:
            #    answer = result.split(sep="query:",maxsplit=1)[1]
            answers.append(answer)

        input_dict["answers"] = answers
        return input_dict        

    @staticmethod
    def bleu_fn(references, candidates, n=2):
        unigram, bigram = [],[]
        for ref, can in zip(references, candidates):
            reference = [[ref]]
            candidate = [can]
            #reference = [[ref[i] for i in range(len(ref))]]
            #candidate = [can[i] for i in range(len(can))]
            #can = normalize_answer(can)
            #reference = [" ".join(jieba.cut(ref)).split()]  # may have multiple ref, need [[ref1]]
            #candidate = " ".join(jieba.cut(can)).split()

            #chencherry = SmoothingFunction()
            #score1, score2= sentence_bleu(reference,candidate,weights=[(1.0,),(0.5,0.5)]) #methods 7 ref prophetnet
            score1 = bleu_score(candidate,reference,1,False)
            score2 = bleu_score(candidate,reference,2,False)
            unigram.append(score1)
            bigram.append(score2)

        return sum(unigram) / len(unigram), sum(bigram) / len(bigram)

    @staticmethod
    def f1_fn(references, candidates, n=1):
        def pre_recall_f1(reference,candidate):
            from collections import Counter
            common = Counter(reference) & Counter(candidate)
            num_same = sum(common.values()) #TP 
            if num_same == 0:
                return 0, 0, 0
            precision = 1.0 * num_same / len(candidate)
            recall = 1.0 * num_same / len(reference)
            f1 = (2 * precision * recall) / (precision + recall)
            return precision, recall, f1

        pre, re, f1 = [],[],[]
        for ref, can in zip(references, candidates):
            #can = normalize_answer(can)
            if n == 1: # token live
                reference = [[ref[i] for i in range(len(ref))]]
                candidate = [can[i] for i in range(len(can))]
            else: # word level
                reference = [" ".join(jieba.cut(ref)).split()]
                candidate = " ".join(jieba.cut(can)).split()
            
            (_pre, _re, _f1)  = [pre_recall_f1(r, candidate) for r in reference][0]
            pre.append(_pre)
            re.append(_re)
            f1.append(_f1)
        return sum(pre)/len(pre), sum(re)/len(re), sum(f1)/len(f1)

    def acc_fn(references, candidates):
        num_same = 0 # 2-classification
        for ref, can in zip(references, candidates):
            ref = 0 if ref == "不检索" else 1
            can = 0 if can == "不检索" or "不" in can else 1
            if ref == can:
                num_same = num_same + 1
        
        return num_same / len(candidates)            


    # Ref: https://github.com/microsoft/ProphetNet/blob/master/GLGE_baselines/script/script/evaluate/personachat/eval.py
    @staticmethod
    def dist_fn(candidates):
        batch_size = len(candidates)
        intra_dist1, intra_dist2 = [],[]
        unigrams_all, bigrams_all = Counter(), Counter()

        for can in candidates:
            unigrams = Counter(can)
            bigrams = Counter(zip(can,can[1:]))
            intra_dist1.append((len(unigrams)+1e-12) / (len(can)+1e-5))
            intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(can)-1)+1e-5))

            unigrams_all.update(unigrams)
            bigrams_all.update(bigrams)

        inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
        inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
        intra_dist1 = sum(intra_dist1) / len(intra_dist1)
        intra_dist2 = sum(intra_dist2) / len(intra_dist2)
        return inter_dist1, inter_dist2, intra_dist1, intra_dist2 
                

    # The four following functions are adapted from Meta ParlAI:
    # https://github.com/facebookresearch/ParlAI/blob/main/parlai/core/metrics.py
    @staticmethod
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

    def evaluate(self):
        raise NotImplementedError
class DialogueTest(TestModule):
    def __init__(self, **args):
        super().__init__(**args)
        self.task = 'dial'

        #data_file = f"/cognitive_comp/yangqi/data/DuSinc/dev_{self.task}.json"
        data_file = f"/cognitive_comp/common_data/dialogue/DuSinc/test.json"
        #data_file = f"/cognitive_comp/yangqi/data/DuSinc/test_{self.task}_a.json"
        
        self.examples = self.load_from_json(data_file)

    def preprocess(self, item, context=1, max_kno_len=256, max_src_len=128, max_tgt_len=128):
        if "knowledge" in item:
            item["kno"] = item["knowledge"]  #convert name

        if "[SEP]" in item["src"]:
            src = " ".join(item["src"].split("[SEP]")[-1*context:]) #上下文处理不太好，所以只暴露最后几条上下文，相当于滑动 windows
        else:
            src = item["src"]
        src = self.truncate(src, max_src_len-2)
        kno = self.truncate(item["kno"],max_kno_len-2)
        tgt = self.truncate(item["tgt"], max_tgt_len-2)
        
        input_text = f'knowledge: {kno} context: {src} response:'
        input_text = self.tokenizer(input_text,return_tensors='pt')
        input_text.to(device)

        return {
            "input_text":input_text,
            "kno"       :kno,
            "src"       :src,
            "tgt"       :tgt
        }

    def generate(self, input_dict, prompt="response:"):
        return super().generate(input_dict, prompt)

    def evaluate(self, nums, metrics):
        nums = min(nums,len(self.examples))
        candidates, references = [],[]

        
        for idx in tqdm(range(nums)):
        #for idx in tqdm(range(100)):
            #kno, src, tgt = data[idx]
            item = self.examples[idx]
            input_dict = self.preprocess(item)
            output = self.generate(input_dict)

            candidates.append(output["answers"][0])
            references.append(output["tgt"])

        scores = {}
        if "bleu" in metrics:
            bleu_score = self.bleu_fn(references, candidates, 2)
            scores["bleu"] = bleu_score

        if "f1" in metrics:
            f1_score = self.f1_fn(references, candidates, 1)
            scores["f1"] = f1_score

        if "dist" in metrics:
            dist_score = self.dist_fn(candidates)
            scores["dist"] = dist_score

        self.output_file(candidates,bleu_score,f1_score,dist_score)
        return scores

    @staticmethod
    def output_file(candidates, bleu_score,f1_score,dist_score):
        f = open("./output_dial.txt",'w')
        for can in candidates:
            f.write(can)
        f.write(f"bleu1/2_score{bleu_score}")
        f.write(f"pre/re/f1_score{f1_score}")
        f.write(f"dist1/2_score{dist_score}")
        

# test demo pipeline

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

setup()

st.header("Demo for knowledge-based Dialogue")

# side bar
config_dict = {}
sbform = st.sidebar.form("参数设置")

config_dict["n_sample"]    = sbform.slider("设置返回条数",min_value=1,max_value=10,value=3)
config_dict["text_length"] = sbform.slider('生成长度:',min_value=32,max_value=256,value=80,step=16)
#text_level = sbform.slider('文本多样性:',min_value=0.1,max_value=1.0,value=0.9,step=0.1)
config_dict["top_k"]       = sbform.slider('Top-K:',min_value=0,max_value=20,value=0,step=1)
config_dict["top_p"]       = sbform.slider('Top-P:',min_value=0.0,max_value=1.0,value=0.6,step=0.1)
config_dict["rep_pen"]     = sbform.slider('Repeat Penalty:',min_value=1.0,max_value=2.0,value=1.1,step=0.1)
config_dict["context"]     = sbform.number_input('上下文句子数:',min_value=0,max_value=5,value=1,step=1)
config_dict["model_id"]    = sbform.selectbox('选择模型',
                                    ['Wenzhong-Finetune-CKPT',
                                    'Wenzhong-Finetune-Loss0.2',
                                    'Wenzhong-Finetune-Loss0.1',
                                    "Wenzhong-Finetune-Query-Loss0.1",
                                    'Wenzhong-GPT2-110M',
                                    'Wenzhong-GPT2-3.5B'])
# config_dict["model_id"]    = sbform.selectbox('选择模型',
#                                     ['Wenzhong-GPT2-epoch1',
#                                     'Wenzhong-GPT2-epoch10',
#                                     'Wenzhong-GPT2-epoch20',
#                                     'Wenzhong-GPT2-110M',
#                                     'Wenzhong-GPT2-3.5B'])
config_dict["example_id"] = sbform.selectbox('选择样例',['None','ALL','1','2','3','4','5','6','7','8'])
sbform.form_submit_button("提交")

def dialpage(config_dict, dialoguetest):
    example_id = config_dict["example_id"]
    if example_id is "ALL": # eval
        with st.spinner("正在评估中"):
            scores = dialoguetest.evaluate(1041,["bleu","f1","dist"])

            bleu_score,f1_score,dist_score = scores["bleu"], scores["f1"], scores["dist"]
            st.write(f"Bleu1/2 score on dev : {bleu_score[0]:.4f}/{bleu_score[1]:.4f}")
            st.write(f"Prec score on dev : {f1_score[0]:.4f}")
            st.write(f"Re   score on dev : {f1_score[1]:.4f}")
            st.write(f"F1   score on dev : {f1_score[2]:.4f}")
            st.write(f"Dist1/2 score on dev : {dist_score[0]:.4f}/{dist_score[1]:.4f}")

    else:
        form = st.form("文本输入")

        if example_id is not "None":
            
            prepared_examples = [2,12,39,67,136,151,161,120]
            idx = prepared_examples[int(example_id)-1]
            item = dialoguetest.examples[idx]

            item = dialoguetest.preprocess(
                item = item,
                context=config_dict["context"]
            ) # to display preprocess text on screen
            input_text_kno = form.text_input('知识',value=item["kno"],placeholder=item["kno"])
            input_text_dia = form.text_input('上文',value=item["src"],placeholder=item["src"]) 
            input_text_tgt = item["tgt"]
        else:
            input_text_kno = form.text_input('知识',value="")
            input_text_dia = form.text_input('上文',value="")
            input_text_tgt = " "
        
        form.form_submit_button("提交")
        input_dict = dialoguetest.preprocess(
            item = {
                "src": input_text_dia,
                "kno": input_text_kno,
                "tgt": input_text_tgt
            },
            context=config_dict["context"]
        )
        

        with st.spinner('生成对话中'):
            if input_text_dia:
                output = dialoguetest.generate(input_dict, prompt="response:")

                st.markdown(f"""**知识:** {output["kno"]}\n""")
                st.markdown(f"""**上文:** {output["src"]}\n""")
                for idx, ans in enumerate(output["answers"]):
                    st.markdown(f"""**回答{idx}:** {ans}\n""")
                st.markdown(f"""**参考答案:** {output["tgt"]}\n""")

def querypage(config_dict, querytest):
    example_id = config_dict["example_id"]
    if example_id is "ALL": # eval
        with st.spinner("正在评估中"):
            scores = dialoguetest.evaluate(1052,["acc","f1","dist"])

            acc_score,f1_score,dist_score = scores["acc"], scores["f1"], scores["dist"]
            st.write(f"ACC score on dev : {acc_score[0]:.4f}")
            st.write(f"F1   score on dev : {f1_score[2]:.4f}")
            st.write(f"Dist1/2 score on dev : {dist_score[0]:.4f}/{dist_score[1]:.4f}")
    else:
        form = st.form("文本输入")

        if example_id is not "None":
            prepared_examples = [2,12,55,67,136,151,161,120]
            idx = prepared_examples[int(example_id)-1]
            item = querytest.examples[idx]

            item = dialoguetest.preprocess(
                item = item,
                context=config_dict["context"]
            ) # to display preprocess text on screen
            input_text_dia = form.text_input('上文',value=item["src"],placeholder=item["src"]) 
            input_text_tgt = item["tgt"]
        else:
            input_text_dia = form.text_input('上文',value="")
            input_text_tgt = " "
        
        form.form_submit_button("提交")
        input_dict = querytest.preprocess(
            item = {
                "src": input_text_dia,
                "tgt": input_text_tgt
            },
            context=config_dict["context"]
        )
        
        with st.spinner('生成对话中'):
            if input_text_dia:
                output = querytest.generate(input_dict, prompt="response:")
                st.markdown(f"""**上文:** {output["src"]}\n""")
                for idx, ans in enumerate(output["answers"]):
                    st.markdown(f"""**回答{idx}:** {ans}\n""")
                st.markdown(f"""**参考答案:** {output["tgt"]}\n""")

#config_args = argparse.Namespace(**config_dict)
dialoguetest = DialogueTest(**config_dict)
dialpage(config_dict, dialoguetest)
