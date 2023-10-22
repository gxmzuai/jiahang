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
    """
    这个函数用于对聊天机器人的输出进行后处理。
    
    参数:
    y: 包含(消息, 响应)元组的列表
    
    返回值:
    返回处理后的(消息, 响应)元组的列表，其中的文本可能已经被转换为HTML。
    """
    
    # 如果输入为空，则返回空列表
    if y is None:
        return []
    
    # 遍历列表中的每一个(消息, 响应)元组
    for i, (message, response) in enumerate(y):
        
        # 使用mdtex2html.convert方法转换消息和响应中的Markdown和LaTeX为HTML，
        # 如果消息或响应为None，则保留为None。
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
        
    # 返回处理后的列表
    return y

# 为Chatbot类添加postprocess方法
gr.Chatbot.postprocess = postprocess

def parse_text(text):
    """
    这个函数用于解析输入的文本，并将其转换为适用于HTML显示的格式。
    该函数是从https://github.com/GaiZhenbiao/ChuanhuChatGPT/ 复制过来的。
    
    参数:
    text: 输入的原始文本
    
    返回值:
    返回处理后，适用于HTML显示的文本。
    """
    
    # 按换行符分割文本成为多行
    lines = text.split("\n")
    
    # 移除空行
    lines = [line for line in lines if line != ""]
    
    # 初始化计数器，用于追踪代码块(```)出现的次数
    count = 0
    
    # 遍历所有行
    for i, line in enumerate(lines):
        
        # 如果行中包含代码块标识符("```")
        if "```" in line:
            count += 1
            items = line.split('`')
            
            # 如果是代码块的开始
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
                
            # 如果是代码块的结束
            else:
                lines[i] = f'<br></code></pre>'
        
        # 如果不是代码块标识符行
        else:
            
            # 为其他行（除了第一行）添加HTML换行标签
            if i > 0:
                
                # 如果在代码块内，进行特殊字符转义
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
    
    # 将所有行连接为一个字符串
    text = "".join(lines)
    
    return text


def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
    """
    这个函数用于预测聊天机器人的响应。

    参数:
    input: 用户输入的文本
    chatbot: 聊天记录列表，包含(消息, 响应)元组
    max_length: 预测响应的最大长度
    top_p: 排序概率
    temperature: 温度参数，控制输出多样性
    history: 聊天历史
    past_key_values: 用于transformer模型的past_key_values参数

    返回值:
    返回预测的聊天记录、聊天历史和past_key_values
    """
    global audio_text, response  # 声明全局变量

    print(f'in predict:{audio_text}')  # 打印当前的音频转文本内容

    # 如果有音频转文本的内容，用它替代输入
    if audio_text != "":
        input = audio_text

    # 添加用户的输入到聊天记录，并进行格式处理
    chatbot.append((parse_text(input), ""))
    response = None  # 初始化响应变量

    # 调用模型进行预测
    for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
                                                                return_past_key_values=True,
                                                                max_length=max_length, top_p=top_p,
                                                                temperature=temperature):
        # 更新聊天记录，包括模型的响应
        chatbot[-1] = (parse_text(input), parse_text(response))

    # 打印生成的响应
    print(response)

    # 清空全局的音频转文本内容
    audio_text = ""

    # 返回预测结果
    yield chatbot, history, past_key_values

def reset_user_input():
    """
    这个函数用于重置用户的输入。
    
    返回值:
    返回调用gr.update函数的结果，通常用于清空输入框。
    """
    return gr.update(value='')


def reset_state():
    """
    这个函数用于重置聊天状态。
    
    返回值:
    返回一个空的聊天记录列表，空的历史记录列表和None（用于past_key_values）。
    """
    return [], [], None

from aip import AipSpeech

APP_ID = '41498694'
API_KEY = 'HG3sdTF1ildW78Cd8Oh77pcW'
SECRET_KEY = 'n8LmPVvt3r65wgI3I4G8MXe60ojWQSWx'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

def baidu_audio_to_text(audio_path, format='wav', rate=16000):
    """
    这个函数用于将音频文件转换成文本。
    首先，确保音频文件是以wav格式保存的，然后提供该文件的路径到audio_path参数。
    
    参数:
    audio_path: 音频文件的路径
    format: 音频格式，默认是'wav'
    rate: 采样率，默认是16000
    
    返回值:
    返回由百度API识别出的文字。
    """
    
    # 以二进制方式打开音频文件
    with open(audio_path, 'rb') as fp:
        data = fp.read()

    # 使用百度的asr接口进行语音识别
    result = client.asr(data, format, rate, {
        'dev_pid': 1537,  # 识别模型，1537为普通话模型
    })
    
    # 返回识别结果中的第一项，即识别出的文字
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

# 定义一个空函数，用于后续的音频播放功能
def play_audio():
    pass

# 创建一个gr.Blocks界面
with gr.Blocks() as demo:
    # 添加一个标题
    gr.HTML("""<h1 align="center">ZhiPei Bot</h1>""")

    # 初始化一个聊天机器人界面
    chatbot = gr.Chatbot()

    # 创建一个行容器
    with gr.Row():
        
        # 创建一个占4份比例的列容器
        with gr.Column(scale=4):
            
            # 创建一个全宽度的列容器，用于文本输入和提交按钮
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
                submitBtn = gr.Button("Submit_text", variant="primary")
                
            # 创建一个窄列容器，用于音频输入
            with gr.Column(min_width=32, scale=1):
                audio_input = gr.Audio(source="microphone", type="filepath", label="Your Voice", format='wav')
                
                # 创建两个半宽度的列容器，用于音频相关按钮
                with gr.Column(min_width=16, scale=1):
                    Submit_audio = gr.Button("Submit_audio", variant="primary")
                with gr.Column(min_width=16, scale=1):
                    audio_playBtn = gr.Button("Play response audio", variant="primary")
        
        # 创建一个占1份比例的列容器，用于其他按钮和滑块
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    # 初始化状态变量
    history = gr.State([])
    past_key_values = gr.State(None)

    # 定义按钮的点击事件
    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)
    
    Submit_audio.click(save_audio, [audio_input],[], show_progress=True)
    
    submitBtn.click(reset_user_input, [], [user_input])

    Submit_audio.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                       [chatbot, history, past_key_values], show_progress=True)

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

    audio_playBtn.click(play_audio, [], [], show_progress=True)

def main():
    global model, tokenizer  # 定义全局变量 model 和 tokenizer

    # 使用 HfArgumentParser 解析 ModelArguments
    parser = HfArgumentParser((ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果脚本只有一个参数并且是一个 json 文件的路径，那么从该 json 文件中解析参数
        model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        model_args = parser.parse_args_into_dataclasses()[0]

    # 加载预训练模型和对应的配置
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    # 设置配置的一些属性
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    # 如果存在 P-tuning 的检查点，从该检查点中加载模型权重
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

    # 如果指定了量化位数，进行模型量化
    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    # 将模型移到 GPU 上
    model = model.cuda()

    # 如果设置了 pre_seq_len，执行相应的操作（可能与 P-tuning v2 相关）
    if model_args.pre_seq_len is not None:
        model.transformer.prefix_encoder.float()
    
    # 设置模型为评估（evaluation）模式
    model = model.eval()

    # 启动 Gradio 界面
    demo.queue().launch(share=True, inbrowser=True)


# 当该脚本作为主程序运行时，执行 main 函数
if __name__ == "__main__":
    main()

# 为什么第一次运行的时候是先运行predict再运行text_to_audio？
