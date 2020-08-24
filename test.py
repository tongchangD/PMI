import pickle
import datetime
from PMI import PMI_test
from Tokenizer import Tokenizer
mapping_file = './PKL.pkl'
with open(mapping_file, 'rb') as f:
    mapping = pickle.load(f)
    stopwords=mapping["stopwords"]
    document=mapping["document"]
    miniprobability=mapping["miniprobability"]
    minitogether=mapping["minitogether"]
    set_word=mapping["set_word"]
    dict_frq_word=mapping["dict_frq_word"]

word_freq_path = './data/word_freq.txt'
# char set file
common_char_path = './data/common_char_set.txt'
# same pinyin char file
same_pinyin_path = './data/same_pinyin.txt'
# custom confusion set
custom_confusion_path = './data/custom_confusion.txt'
# custom word for segment
custom_word_path = './data/custom_word.txt'
# 特定拼音词汇表
custom_pinyin_word_path="./data/custom_pinyin_word.txt"
tokenizer = Tokenizer(word_freq_path=word_freq_path,
                          common_char_path=common_char_path,
                          same_pinyin_path=same_pinyin_path,
                          custom_confusion_path=custom_confusion_path,
                          custom_word_path=custom_word_path)

pm = PMI_test(tokenizer=tokenizer, stopwords=stopwords, document=document, miniprobability=miniprobability,
              minitogether=minitogether, set_word=set_word, dict_frq_word=dict_frq_word)


print("documents read done")
print('pm.calculate_lis("小提琴","朗姆酒")', pm.calculate_lis("小提琴", "朗姆酒"))
print('pm.calculate_lis("小提琴","大钢琴")', pm.calculate_lis("小提琴", "大钢琴"))