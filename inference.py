#!/usr/bin/env python
# coding: utf-8
import os
import torch
from scipy.io.wavfile import write
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
def synthesize_text(text, model_path, config_path, output_wav):
    # 加载配置文件
    hps = utils.get_hparams_from_file(config_path)
    # 初始化模型（在 CPU 上运行）
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    
    _ = net_g.eval()
    _ = utils.load_checkpoint(model_path, net_g, None)
    # 获取文本序列
    stn_tst = get_text(text, hps)
    # 推理生成音频（在 CPU 上运行）
    try:
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)  # 不再使用 .cuda()
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])  # 不再使用 .cuda()
            
            # 添加日志以便调试
            print(f"Input shape: {x_tst.shape}, Length shape: {x_tst_lengths.shape}")
            
            audio = net_g.infer(
                x_tst, 
                x_tst_lengths, 
                noise_scale=.667, 
                noise_scale_w=0.8, 
                length_scale=1
            )[0][0,0].data.cpu().float().numpy()
            
        # 保存音频文件
        write(output_wav, hps.data.sampling_rate, audio)
        return output_wav
    except Exception as e:
        # 打印详细错误信息
        import traceback
        print(f"Error in synthesize_text: {str(e)}")
        print(traceback.format_exc())
        raise
def synthesize_text_with_speaker(text, model_path, config_path, output_wav, speaker_id=0):
    """
    使用多说话人模型合成语音
    
    参数:
        text (str): 要合成的文本
        model_path (str): 模型文件路径
        config_path (str): 配置文件路径
        output_wav (str): 输出音频文件路径
        speaker_id (int): 说话人ID，默认为0
    """
    try:
        hps = utils.get_hparams_from_file(config_path)
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model)
        _ = net_g.eval()
        
        # 加载模型
        utils.load_checkpoint(model_path, net_g, None)
        
        # 处理文本
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)  # 在CPU上运行
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])  # 在CPU上运行
            sid = torch.LongTensor([speaker_id])  # 在CPU上运行
            
            # 添加日志以便调试
            print(f"Multi-speaker input shape: {x_tst.shape}, Length shape: {x_tst_lengths.shape}, Speaker ID: {speaker_id}")
            
            audio = net_g.infer(
                x_tst, 
                x_tst_lengths, 
                sid=sid, 
                noise_scale=.667, 
                noise_scale_w=0.8, 
                length_scale=1
            )[0][0,0].data.cpu().float().numpy()
        
        # 保存音频
        write(output_wav, hps.data.sampling_rate, audio)
        return output_wav
    except Exception as e:
        # 打印详细错误信息
        import traceback
        print(f"Error in synthesize_text_with_speaker: {str(e)}")
        print(traceback.format_exc())
        raise