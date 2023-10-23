import os, sys
import shutil
from pydub import AudioSegment
import gradio as gr
import mdtex2html
from aip import AipSpeech
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
from io import BytesIO
from arguments import ModelArguments, DataTrainingArguments
import base64
from gtts import gTTS

model = None
tokenizer = None
audio_text = ""
response = ""
text_to_audio = None
audio_ongoing = False

# 百度api相关
APP_ID = '41498694'
API_KEY = 'HG3sdTF1ildW78Cd8Oh77pcW'
SECRET_KEY = 'n8LmPVvt3r65wgI3I4G8MXe60ojWQSWx'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

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

gr.Chatbot.postprocess = postprocess  # 为Chatbot类添加postprocess方法

def parse_text(text):
    """
    这个函数用于解析输入的文本,并将其转换为适用于HTML显示的格式。
    该函数是从https://github.com/GaiZhenbiao/ChuanhuChatGPT/ 复制过来的。
    
    参数:
    text: 输入的原始文本
    
    返回值:
    返回处理后,适用于HTML显示的文本。
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

def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
    """
    这个函数用于预测聊天机器人的响应。

    参数:
    input: 用户输入的文本或音频
    chatbot: 聊天记录列表，包含(消息, 响应)元组
    max_length: 预测响应的最大长度
    top_p: 排序概率
    temperature: 温度参数，控制输出多样性
    history: 聊天历史
    past_key_values: 用于transformer模型的past_key_values参数

    返回值:
    返回预测的聊天记录、聊天历史和past_key_values
    """
    if input is None or input == "":
        return None
    print(f'checkpoint:{input}')

    if os.path.exists(input):
        # 当前输入是一个音频文件，input内容为其临时路径，例：/tmp/gradio/6a829d93c68a3d1aa98f7868da61a79ee1040130/audio-0-100.wav
        audio = AudioSegment.from_wav(input)  
        audio = audio.set_channels(1).set_frame_rate(16000).export('audio_input.wav', format="wav")   # 转换为单通道并保存

        input = baidu_audio_to_text("./audio_input.wav")
        if input == "":
            # 防止录音过短，什么都没识别出来
            chatbot.append((parse_text("你好像什么都没有说哦。"), ""))
            return chatbot, None, None

    # 添加用户的输入到聊天记录，并进行格式处理
    chatbot.append((parse_text(input), ""))

    global response, text_to_audio
    # 即将生成新的回答，将原来的回答删除
    text_to_audio = None
    response = ""
    # 调用模型进行预测
    for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
                                                                return_past_key_values=True,
                                                                max_length=max_length, top_p=top_p,
                                          
                                                                temperature=temperature):
        chatbot[-1] = (parse_text(input), parse_text(response))  # 更新聊天记录，包括模型的响应
        
        yield chatbot, history, past_key_values # 返回预测结果

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

    if user_input._id == 9:
        print("select audio")
    if user_input._id == 6:
        print('select text')
    # audio_input其实是一个临时文件的路径，例：/tmp/gradio/6a829d93c68a3d1aa98f7868da61a79ee1040130/audio-0-100.wav
    audio = AudioSegment.from_wav(audio_input)  
    audio = audio.set_channels(1).set_frame_rate(16000).export('audio_input.wav', format="wav")   # 转换为单通道并保存
    
    # global audio_text, response
    audio_text = baidu_audio_to_text("./audio_input.wav")
    print(audio_text)





    # 如果有音频转文本的内容，用它替代输入
    # if audio_text != "":
    #     input = audio_text

    # # 添加用户的输入到聊天记录，并进行格式处理
    # chatbot.append((parse_text(input), ""))
    # response = None  # 初始化响应变量

    # # 调用模型进行预测
    # for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
    #                                                             return_past_key_values=True,
    #                                                             max_length=max_length, top_p=top_p,
    #                                                             temperature=temperature):
    #     # 更新聊天记录，包括模型的响应
    #     chatbot[-1] = (parse_text(input), parse_text(response))
        
    #     yield chatbot, history, past_key_values # 返回预测结果


    # print(response)    # 打印生成的响应


    # audio_text = ""  # 清空全局的音频转文本内容

def play_audio():
    global response, text_to_audio, audio_ongoing
    if response != "":
        if text_to_audio is None:
            text_to_audio = client.synthesis(response, 'zh', 2, {
                                                    'vol': 4,
                                                    })
        print('go across')
        audio = base64.b64encode(text_to_audio).decode("utf-8")
        audio_player = f'<audio src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'

        return audio_player
    return None

def clear_audio():
    # 清除刚才创建的HTML元素
    return None

# 创建一个gr.Blocks界面
with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Emolink</h1>""")  # 添加一个标题
    chatbot = gr.Chatbot()  # 初始化一个聊天机器人界面

    # 创建一个行容器
    with gr.Row():
        with gr.Column(scale=4):  # 创建一个占4份比例的列容器
            text_input = gr.Textbox(show_label=False, placeholder="Input...", lines=9, container=False)
            submit_text_btn = gr.Button("Submit_text", variant="primary")

        with gr.Column(scale=1):
                audio_input = gr.Audio(source="microphone", type="filepath", label="Your Voice", scale=2)
                submit_audio_btn = gr.Button("Submit_audio", variant="primary", scale=1)
                play_audio_btn = gr.Button("播放", variant="primary", scale=1)
        with gr.Column(scale=1):  # 创建一个占1份比例的列容器，用于其他按钮和滑块
            emptyBtn = gr.Button("Clear History", scale=1)
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True, scale=1)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True, scale=1)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True, scale=1)

    audio_play_html = gr.HTML(visible=False)
              

    history = gr.State([])  # 初始化状态变量
    past_key_values = gr.State(None)


    # 定义按钮的点击事件
    submit_text_btn.click(predict, [text_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)
    
    submit_audio_btn.click(predict, [audio_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)

    play_audio_btn.click(play_audio, [], outputs=[audio_play_html])
    # play_audio_btn.click(clear_audio, [], outputs=[audio_play_html])

    
    submit_text_btn.click(reset_user_input, [], [text_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

    # play_audio_btn.click(play_audio, [], [], show_progress=True)

def main():
    global model, tokenizer  # 定义全局变量 model 和 tokenizer

    parser = HfArgumentParser((ModelArguments))  # 使用 HfArgumentParser 解析 ModelArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        
        model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]  # 如果脚本只有一个参数并且是一个 json 文件的路径，那么从该 json 文件中解析参数
    else:
        model_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)  # 加载预训练模型和对应的配置
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

    model = model.cuda()  # 将模型移到 GPU 上

    # 如果设置了 pre_seq_len，执行相应的操作（可能与 P-tuning v2 相关）
    if model_args.pre_seq_len is not None:
        model.transformer.prefix_encoder.float()
    
    model = model.eval()  # 设置模型为评估（evaluation）模式
    demo.queue().launch(share=True, inbrowser=True)  # 启动 Gradio 界面


if __name__ == "__main__":  # 当该脚本作为主程序运行时，执行 main 函数
    main()
