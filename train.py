# coding=utf-8
from PMI  import PMI_train
from Tokenizer import Tokenizer
import os
import pickle
# from extract import *
# word frequence file
word_freq_path = './data/word_freq.txt'
# char set file
common_char_path = './data/common_char_set.txt'
# same pinyin char file
same_pinyin_path = './data/same_pinyin.txt'
# custom confusion set
custom_confusion_path = './data/custom_confusion.txt'
# custom word for segment
custom_word_path = './data/custom_word.txt'
# 停用词
stop_word_file_path = "./data/stopwords.txt"

source_path="./data/sentence.txt" # 训练的句子


# 自定义切词词表，及其工具
tokenizer = Tokenizer(word_freq_path=word_freq_path,
                      common_char_path=common_char_path,
                      same_pinyin_path=same_pinyin_path,
                      custom_confusion_path=custom_confusion_path,
                      custom_word_path=custom_word_path)

pm = PMI_train(
         tokenizer=tokenizer,
         stop_word_file_path=stop_word_file_path, source_path=source_path)




mapping_file = './PKL.pkl'
with open(mapping_file, 'wb') as f:  # 将参数写入pik
    mappings = {
        "stopwords":pm.stopwords,
        "document":pm.document,
        "miniprobability":pm.miniprobability,
        "minitogether":pm.minitogether,
        "set_word":pm.set_word,
        "dict_frq_word":pm.dict_frq_word,
    }
    pickle.dump(mappings, f)
print("documents read done")
print('pm.calculate_lis("小提琴","朗姆酒")', pm.calculate_lis("小提琴", "朗姆酒"))
print('pm.calculate_lis("小提琴","大钢琴")', pm.calculate_lis("小提琴", "大钢琴"))