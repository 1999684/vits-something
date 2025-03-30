import os
import json
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
from flask_cors import CORS
from inference import synthesize_text, synthesize_text_with_speaker  # 添加导入 synthesize_text_with_speaker 和 synthesize_text 函数

HISTORY_FILE = "./audio_history.txt"

# 设置 espeak 路径
espeak_path = r"D:\000myself\eSpeak\command_line"  # 根据 espeak 安装路径修改
os.environ['PATH'] += os.pathsep + espeak_path
app = Flask(__name__)
CORS(app)
# 配置 VITS 的路径和模型
MODEL_PATH = "./pretrained_models/pretrained_ljs.pth"
CONFIG_PATH = "./configs/ljs_base.json"
# 添加多说话人模型配置
MULTI_SPEAKER_MODEL_PATH = "./pretrained_models/pretrained_vctk.pth"
MULTI_SPEAKER_CONFIG_PATH = "./configs/vctk_base.json"

def save_to_history(file_name, text):
    record = {
        "file_name": file_name,
        "text": text,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    # 如果文件不存在，则创建新文件
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            f.write("[]")  # 初始化为空列表
    # 读取现有记录
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    # 添加新记录
    history.append(record)
    # 写回文件
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    # 定义输出文件路径
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")  # 精确到微秒
    output_wav = f"output_{timestamp}.wav"
    try:
        # 调用 synthesize_text 函数生成音频
        synthesize_text(text, MODEL_PATH, CONFIG_PATH, output_wav)
        save_to_history(output_wav, text)
        # 返回信号，告知前端音频已准备好
        return jsonify({'message': 'Audio ready', 'audio_file': output_wav}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# 提供静态文件支持，允许前端访问生成的音频文件
# 将现有的 @app.route('/synthesize_multi_speaker', methods=['POST']) 改为:
@app.route('/synthesize_with_speaker', methods=['POST'])
def synthesize_with_speaker():
    try:
        data = request.json
        if data is None:
            print("Error: No JSON data in request")
            return jsonify({'error': 'No JSON data provided'}), 400
            
        print(f"Received data: {data}")  # 打印接收到的数据
        
        text = data.get('text', '')
        speaker_id = data.get('speaker_id', 0)  # 默认使用说话人ID 0
        
        print(f"Text: '{text}', Speaker ID: {speaker_id}")  # 打印提取的参数
        
        if not text:
            print("Error: Empty text parameter")
            return jsonify({'error': 'No text provided'}), 400
        
        # 尝试将speaker_id转换为整数
        try:
            speaker_id = int(speaker_id)
        except (ValueError, TypeError):
            print(f"Error: Invalid speaker_id format: {speaker_id}")
            return jsonify({'error': 'Invalid speaker_id format'}), 400
        
        # 定义输出文件路径
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")  # 精确到微秒
        output_wav = f"output_speaker{speaker_id}_{timestamp}.wav"
        
        # 调用多说话人合成函数
        synthesize_text_with_speaker(
            text, 
            MULTI_SPEAKER_MODEL_PATH, 
            MULTI_SPEAKER_CONFIG_PATH, 
            output_wav, 
            speaker_id
        )
        
        # 保存到历史记录
        save_to_history(output_wav, f"{text} (Speaker {speaker_id})")
        
        # 返回信号，告知前端音频已准备好
        return jsonify({
            'message': 'Audio ready', 
            'audio_file': output_wav,
            'speaker_id': speaker_id
        }), 200
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        print(f"Error in synthesize_with_speaker: {error_msg}")
        print(stack_trace)
        return jsonify({'error': error_msg}), 500
@app.route('/get_speakers', methods=['GET'])
def get_speakers():
    # 这里可以根据模型的实际情况返回可用的说话人列表
    # VCTK数据集通常有约100个说话人
    speakers = [{"id": i, "name": f"Speaker {i}"} for i in range(100)]
    return jsonify({'speakers': speakers})

@app.route('/get_history', methods=['GET'])
def get_history():
    if not os.path.exists(HISTORY_FILE):
        return jsonify({'history': []})
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    return jsonify({'history': history})

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('.', filename)

@app.route('/delete_history', methods=['POST'])
def delete_history():
    """
    删除历史记录和对应的音频文件
    """
    try:
        data = request.json
        file_name = data.get('file_name')
        
        if not file_name:
            return jsonify({'error': '缺少文件名参数'}), 400
        
        # 构建音频文件路径 - 直接使用根目录
        audio_path = file_name  # 文件直接存储在根目录
        
        # 检查文件是否存在
        if os.path.exists(audio_path):
            # 删除音频文件
            os.remove(audio_path)
            print(f"已删除音频文件: {audio_path}")
        else:
            print(f"音频文件不存在: {audio_path}")
        
        # 读取历史记录
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f) if os.path.getsize(HISTORY_FILE) > 0 else []
            
            # 过滤掉要删除的记录
            updated_history = [item for item in history if item.get('file_name') != file_name]
            
            # 保存更新后的历史记录
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(updated_history, f, ensure_ascii=False, indent=4)
            
            return jsonify({'message': '删除成功'})
        else:
            return jsonify({'error': '历史记录文件不存在'}), 404
            
    except Exception as e:
        print(f"删除历史记录时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)