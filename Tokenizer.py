import os
import jieba
import codecs
import re

import operator
# 加载char集
def load_char_set(path):
    words = set()
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for w in f:
            words.add(w.strip())
    return words

# 加载同音字
def load_same_pinyin(path, sep='\t'):
    """
    加载同音字
    :param path:
    :param sep:
    :return:
    """
    result = dict()
    if not os.path.exists(path):
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split(sep)
            if parts and len(parts) > 2:
                key_char = parts[0]
                same_pron_same_tone = set(list(parts[1]))
                same_pron_diff_tone = set(list(parts[2]))
                value = same_pron_same_tone.union(same_pron_diff_tone)
                if len(key_char) > 1 or not value:
                    continue
                result[key_char] = value
    return result




class Tokenizer(object):
    def __init__(self, word_freq_path="",common_char_path="",
                          same_pinyin_path="",
                          custom_confusion_path="",
                          custom_word_path=""):
        self.model = jieba
        if os.path.exists(word_freq_path):
            self.model.set_dictionary(word_freq_path)

        self.word_freq_path = word_freq_path
        self.common_char_path=common_char_path
        self.same_pinyin_path=same_pinyin_path
        self.custom_confusion_path = custom_confusion_path
        self.custom_word_path = custom_word_path

        # 词、频数dict
        self.word_freq = self.load_word_freq_dict(self.word_freq_path)
        # 加载字符表
        self.cn_char_set = load_char_set(self.common_char_path)
        # same pinyin
        self.same_pinyin = load_same_pinyin(self.same_pinyin_path)


        # 自定义混淆集
        self.custom_confusion = self._get_custom_confusion_dict(self.custom_confusion_path)
        # 自定义切词词典
        self.custom_word_dict = self.load_word_freq_dict(self.custom_word_path)
        # 合并切词词典及自定义词典
        self.word_freq.update(self.custom_word_dict)
        self.initialized_corrector=False
        self.check_corrector_initialized()



    def load_word_freq_dict(self,path):
        """
        加载切词词典
        :param path:
        :return:
        """
        word_freq = {}
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                info = line.split()
                if len(info) < 1:
                    continue
                word = info[0]
                # 取词频，默认1
                freq = int(info[1]) if len(info) > 1 else 1
                word_freq[word] = freq
        return word_freq

        # tcd 这是添加 词表的

    # 检测是否进行初始化
    def check_corrector_initialized(self):
        if not self.initialized_corrector:
            self.initialize_corrector()

    # 进行初始化
    def initialize_corrector(self):
        # 加载用户自定义词典
        for w, f in self.custom_word_dict.items():
            self.model.add_word(w, freq=f)

        # 加载混淆集词典
        for k, word in self.custom_confusion.items():
            # 添加到分词器的自定义词典中
            self.model.add_word(k)
            # self.model.add_word(word)
            for x in word:
                self.model.add_word(x)
        self.initialized_corrector = True


    def _get_custom_confusion_dict(self, path):
        """
        取自定义困惑集
        :param path:
        :return: dict, {variant: origin}, eg: {"交通先行": "交通限行"}
        """
        confusion = {}
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                info = line.split()
                if len(info) < 2:
                    continue
                variant = info[0]
                origin = info[1:]
                freq = 1
                if isinstance(origin, list):
                    for x in origin:
                        self.word_freq[x] = freq
                else:
                    self.word_freq[origin] = freq
                confusion[variant] = origin
        return confusion


    def tokenize(self, sentence):
        """
        直接切词并返回切词位置
        :param sentence:
        :return: (word, start_index, end_index) model='default'
        """
        return list(self.model.tokenize(sentence, HMM=False))
        # return list(self.model.tokenize(sentence))

    def tokenize_lis(self, sentence):
        """
        直接返回切词列表
        :param sentence:
        :return: [word] model='default'
        """
        return list(word[0] for word in self.model.tokenize(sentence, HMM=False))

    def tokenize_list(self, sentence, jiebacut):
        """ 需根据传入的jiabacut 修改最终切词 并返回切词位置
        :param sentence:
        :return: (word, start_index, end_index) model='default'
        """
        if jiebacut == []:
            return list(self.model.tokenize(sentence, HMM=False))
        else:

            reslis = []
            res=[]
            for cut_word in jiebacut:  # 根据错点 切割句子 最终句子可能还有尾,需要切词
                res+=list(self.model.tokenize(sentence[:sentence.index(cut_word)], HMM=False))
                res.append((cut_word,sentence.index(cut_word),sentence.index(cut_word)+len(cut_word)))
                sentence=sentence[sentence.index(cut_word)+len(cut_word):]
            if sentence!="":
                res+=list(self.model.tokenize(sentence, HMM=False))
            res=[words[0] for words in res]
            begin=0
            for word in res:
                reslis.append((word,begin,begin+len(word)))
                begin+=len(word)
            return reslis

    def tokenize_err(self,sentence,maybe_error):
        if len(maybe_error)==0:
            return list(self.model.tokenize(sentence, HMM=False))
        else:
            """直接根据错点位置进行切词"""
            # 倒序排列
            maybe_error= sorted(maybe_error, key=operator.itemgetter(2), reverse=True)
            lis=[]
            reslis=[]
            for error in maybe_error:
                temp=[word[0] for word in self.model.tokenize(sentence[error[2]:])]
                temp.reverse()
                lis += temp
                lis.append(error[0])
                sentence = sentence[:error[1]]
            if sentence != "":
                temp=[word[0] for word in self.model.tokenize(sentence)]
                temp.reverse()
                lis+=temp
            begin=0
            end=0
            lis.reverse() # 正序
            for word in lis:
                reslis.append((word,begin,begin+len(word)))
                begin = end+len(word)
                end = begin
            return reslis


if __name__ == '__main__':
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

    tokenizer = Tokenizer(word_freq_path=word_freq_path,
                          common_char_path=common_char_path,
                          same_pinyin_path=same_pinyin_path,
                          custom_confusion_path=custom_confusion_path,
                          custom_word_path=custom_word_path)

    print(tokenizer.tokenize_list("请问你在哪",["你在哪"]))
    print("done")
