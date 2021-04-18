# PyTorch WoBERT
原版Tensorflow权重(https://github.com/ZhuiyiTechnology/WoBERT)
- WoBERT:[chinese_wobert_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1BrdFSx9_n1q2uWBiQrpalw) (提取码：kim2)
- WoBERT+:[chinese_wobert_plus_L-12_H-768_A-12.zip](https://pan.baidu.com/s/1Ltq3ltQsyBCj56zoOOvI9A) (提取码：aedw)

## 安装
```bash
pip install git+https://github.com/JunnYu/WoBERT_pytorch.git
```
## MLM测试
```python
import torch
from transformers import BertForMaskedLM as WoBertForMaskedLM
from wobert import WoBertTokenizer
pretrained_model_or_path_list = [
    "junnyu/wobert_chinese_plus_base", "junnyu/wobert_chinese_base"
]
for path in pretrained_model_or_path_list:
    text = "今天[MASK]很好，我[MASK]去公园玩。"
    tokenizer = WoBertTokenizer.from_pretrained(path)
    model = WoBertForMaskedLM.from_pretrained(path)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs).logits[0]
    outputs_sentence = ""
    for i, id in enumerate(tokenizer.encode(text)):
        if id == tokenizer.mask_token_id:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i].topk(k=5)[1])
            outputs_sentence += "[" + "||".join(tokens) + "]"
        else:
            outputs_sentence += "".join(
                tokenizer.convert_ids_to_tokens([id],
                                                skip_special_tokens=True))
    print(outputs_sentence)
# PLUS WoBERT 今天[天气||阳光||天||心情||空气]很好，我[想||要||打算||准备||就]去公园玩。
# WoBERT 今天[天气||阳光||天||心情||空气]很好，我[想||要||就||准备||也]去公园玩。
```
 
## 手动权重转换
```python
#修改path路径
python convert_tf_to_pytorch.py
```