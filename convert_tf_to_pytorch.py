from transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
# chinese_wobert_plus
path = "E:/BaiduNetdiskDownload/chinese_wobert_plus_L-12_H-768_A-12"
tf_checkpoint_path = path + "/bert_model.ckpt"
bert_config_file = path + "/bert_config.json"
pytorch_dump_path = "wobert_chinese_plus_base/pytorch_model.bin"

convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file,
                                 pytorch_dump_path)

# chinese_wobert
path = "E:/BaiduNetdiskDownload/chinese_wobert_L-12_H-768_A-12"
tf_checkpoint_path = path + "/bert_model.ckpt"
bert_config_file = path + "/bert_config.json"
pytorch_dump_path = "wobert_chinese_base/pytorch_model.bin"

convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file,
                                 pytorch_dump_path)