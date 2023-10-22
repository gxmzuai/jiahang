import os, sys
import shutil
from pydub import AudioSegment
import gradio as gr
import mdtex2html

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed
)

from arguments import ModelArguments, DataTrainingArguments

model = None
tokenizer = None
audio_text = ""
response = ""

"""Override Chatbot.postprocess"""

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
    global audio_text, response
    print(f'in predict:{audio_text}')
    if audio_text != "":
        input = audio_text
    chatbot.append((parse_text(input), ""))
    response = None
    for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
                                                                return_past_key_values=True,
                                                                max_length=max_length, top_p=top_p,
                                                                temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))
        # print(f'in predict:{response}')
        yield chatbot, history, past_key_values

    # 这里已经生成完毕回答
    print(response)
    audio_text = ""
    


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None


from aip import AipSpeech

APP_ID = '41498694'
API_KEY = 'HG3sdTF1ildW78Cd8Oh77pcW'
SECRET_KEY = 'n8LmPVvt3r65wgI3I4G8MXe60ojWQSWx'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

def baidu_audio_to_text(audio_path, format='wav', rate=16000):
    """
        音频文件需要先保存成wav文件，然后再把路径给到audio_path
        这个函数返回由识别出的文字
    """
    with open(audio_path, 'rb') as fp:
        data = fp.read()

    result = client.asr(data, format, rate, {
        'dev_pid': 1537,
    })
    return result['result'][0]

# 保存音频的函数
def save_audio(audio_input):

    destination = "./saved_audio.wav"
    shutil.copy(audio_input, destination)

        # 使用pydub加载音频
    audio = AudioSegment.from_wav(destination)
    
    # 转换为单通道
    mono_audio = audio.set_channels(1).set_frame_rate(16000)
    
    # 保存为单通道音频文件
    mono_audio.export('channel1.wav', format="wav")
    global audio_text
    audio_text = baidu_audio_to_text("./channel1.wav")
    print(f'in save_audio:{audio_text}')

def play_audio():
    pass

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ZhiPei Bot</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
                submitBtn = gr.Button("Submit_text", variant="primary")
                
            with gr.Column(min_width=32, scale=1):
                audio_input = gr.Audio(source="microphone", type="filepath", label="Your Voice", format='wav')
                with gr.Column(min_width=16, scale=1):
                    Submit_audio = gr.Button("Submit_audio", variant="primary")
                with gr.Column(min_width=16, scale=1):
                    audio_playBtn = gr.Button("Play response audio", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                [chatbot, history, past_key_values], show_progress=True)
    
    # submitBtn.click(pr, [user_input], [], show_progress=True)

    Submit_audio.click(save_audio, [audio_input],[], show_progress=True)
    
    submitBtn.click(reset_user_input, [], [user_input])

    Submit_audio.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                [chatbot, history, past_key_values], show_progress=True)

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

    audio_playBtn.click(play_audio, [], [], show_progress=True)


def main():
    global model, tokenizer

    parser = HfArgumentParser((
        ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        model_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)

    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    if model_args.ptuning_checkpoint is not None:
        print(f"Loading prefix_encoder weight from {model_args.ptuning_checkpoint}")
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    model = model.cuda()
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model.transformer.prefix_encoder.float()
    
    model = model.eval()
    demo.queue().launch(share=True, inbrowser=True)



if __name__ == "__main__":
    main()

# 为什么第一次运行的时候是先运行predict再运行text_to_audio？