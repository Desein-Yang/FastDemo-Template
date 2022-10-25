
import streamlit as st
from inference import DialogueTest, TestModule


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
