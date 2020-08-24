# coding=utf-8
import jieba
import jieba.analyse
import jieba.posseg as pseg
import os
import re
import codecs


class PMI_train:
    def __init__(self, tokenizer,
                 stop_word_file_path, source_path):
        self.tokenizer=tokenizer
        self.stopwords = self.stopwordslist(stop_word_file_path)
        self.source_path = source_path
        self.document = self.make_document(self.source_path)
        self.miniprobability = float(1.0) / self.document.__len__()
        self.minitogether = float(0.0) / self.document.__len__()
        self.set_word = self.getset_word()
        self.dict_frq_word = self.get_dict_frq_word()
    def __getstate__(self):
        return self.__dict__

    def make_document(self, path):
        documents = []
        f = open(path, 'r', encoding="utf-8")
        data = f.readlines()
        if data is not None:
            for i, sentences in enumerate(data):
                extractwords = []
                wordlist = self.CutWithPartOfSpeech(sentences)
                # print("wordlist",wordlist)  #
                words = self.ExtractWord(wordlist)
                # print("wordswords", words)
                for word in words:
                    extractwords.append(word)
                documents.append(set(extractwords))
                if i % 10000 == 0:
                    print("运行至 %d 行" % i)
        print("数据加载完毕")
        return documents

    def calcularprobability(self, document, wordlist):

        """
        :param document:
        :param wordlist:
        :function : 计算单词的document frequency
        :return: document frequency
        """

        total = document.__len__()
        number = 0
        for doc in document:
            if set(wordlist).issubset(doc):
                number += 1
        percent = float(number) / total
        return percent

    def togetherprobablity(self, document, wordlist1, wordlist2):

        """
        :param document:
        :param wordlist1:
        :param wordlist2:
        :function: 计算单词的共现概率
        :return:共现概率
        """

        joinwordlist = wordlist1 + wordlist2
        percent = self.calcularprobability(document, joinwordlist)
        return percent

    def getset_word(self):
        """
        :function: 得到document中的词语词典
        :return: 词语词典
        """
        list_word = []
        for doc in self.document:
            list_word = list_word + list(doc)
        set_word = []
        for w in list_word:

            if set_word.count(w) == 0:
                set_word.append(w)
        # print("set_word",set_word)
        return set_word

    def get_dict_frq_word(self):
        """
        :function: 对词典进行剪枝,剪去出现频率较少的单词
        :return: 剪枝后的词典
        """
        dict_frq_word = {}
        for i in range(0, self.set_word.__len__(), 1):
            list_word = []
            # print("self.set_word[i]",self.set_word[i])
            list_word.append(self.set_word[i])
            probability = self.calcularprobability(self.document, list_word)
            # dict_frq_word[self.set_word[i]] = probability
            if probability > self.miniprobability:
                # print("self.miniprobability",self.miniprobability)
                dict_frq_word[self.set_word[i]] = probability
        # print("dict_frq_word",dict_frq_word)
        return dict_frq_word

    def calculate_nmi(self, joinpercent, wordpercent1, wordpercent2):
        """
        function: 计算词语共现的nmi值
        :param joinpercent:
        :param wordpercent1:
        :param wordpercent2:
        :return:nmi
        """
        return (joinpercent) / (wordpercent1 * wordpercent2)

    # def get_pmi(self):
    #     """
    #     function:返回符合阈值的pmi列表
    #     :return:pmi列表
    #     """
    #     dict_pmi = {}
    #     # print ("dict_frq_word",dict_frq_word)
    #     for word1 in self.dict_frq_word:
    #         wordpercent1 = self.dict_frq_word[word1]
    #         for word2 in self.dict_frq_word:
    #             if word1 == word2:
    #                 continue
    #             wordpercent2 = self.dict_frq_word[word2]
    #             list_together = []
    #             list_together.append(word1)
    #             list_together.append(word2)
    #             together_probability = self.calcularprobability(self.document, list_together)
    #             if together_probability > self.minitogether:
    #                 string = word1 + ',' + word2
    #                 dict_pmi[string] = self.calculate_nmi(together_probability, wordpercent1, wordpercent2)
    #     return dict_pmi

    def calculate_lis(self, word1, word2):
        if word1 != word2:
            wordpercent1 = 1
            wordpercent2 = 1
            if word1 in self.dict_frq_word:
                wordpercent1 = self.dict_frq_word[word1]
            if word2 in self.dict_frq_word:
                wordpercent2 = self.dict_frq_word[word2]
            list_together = []
            list_together.append(word1)
            list_together.append(word2)
            together_probability = self.calcularprobability(self.document, list_together)
            return self.calculate_nmi(together_probability, wordpercent1, wordpercent2)
        else:
            return 0

    def calculate_word_lis(self, word1, list):
        calculate_score = []
        for word2 in list:
            if word1 != word2:
                wordpercent1 = 1
                wordpercent2 = 1
                if word1 in self.dict_frq_word:
                    wordpercent1 = self.dict_frq_word[word1]
                if word2 in self.dict_frq_word:
                    wordpercent2 = self.dict_frq_word[word2]
                list_together = []
                list_together.append(word1)
                list_together.append(word2)
                together_probability = self.calcularprobability(self.document, list_together)
                calculate_score.append(self.calculate_nmi(together_probability, wordpercent1, wordpercent2))
        calculate_score = sum(calculate_score) / len(calculate_score)  # 取平均值
        return calculate_score

    # 下面全是数据获取
    def removeEmoji(self, sentence):
        return re.sub('\[.*?\]', '', sentence)

    def CutWithPartOfSpeech(self, sentence):
        sentence = self.removeEmoji(sentence)
        words = self.tokenizer.tokenize_lis(sentence.strip())
        outlis = []
        for word in words:
            if word not in outlis: # word not in self.stopwords:
                outlis.append(word)
        return list(set(outlis))

    def segment(self, list):
        querylist = []
        outstr = ''
        for word in list:
            if word == ' ':
                continue
            # 加停用词
            if word not in self.stopwords:
                outstr += word
                outstr += "$"
        querylist.append(outstr.split('$'))
        return querylist

    def stopwordslist(self, filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords

    def ExtractWord(self, wordlist):
        sentence = ','.join(wordlist)
        # words = jieba.analyse.extract_tags(sentence,4)
        words = self.tokenizer.model.analyse.extract_tags(sentence)  # 默认前20个词
        wordlist = []
        for w in words:
            wordlist.append(w)
        return wordlist

    def RemoveStopWord(self, wordlist):
        stopWords = self.GetStopWords()
        keywords = []
        for word in wordlist:
            if word not in stopWords:
                keywords.append(word)
        return keywords


class PMI_test:
    def __init__(self,tokenizer,stopwords, document, miniprobability, minitogether, set_word,dict_frq_word):
        self.tokenizer=tokenizer
        self.stopwords =stopwords
        self.document =document
        self.miniprobability =miniprobability
        self.minitogether =minitogether
        self.set_word =set_word
        self.dict_frq_word = dict_frq_word

    def calcularprobability(self, document, wordlists):
        """
        :param document:
        :param wordlist:
        :function : 计算单词的document frequency
        :return: document frequency
        """
        total = document.__len__()
        number = 0
        for doc in document:
            if set(wordlists).issubset(doc):
                number += 1
        percent = float(number) / total
        return percent

    def togetherprobablity(self, document, wordlist1, wordlist2):

        """
        :param document:
        :param wordlist1:
        :param wordlist2:
        :function: 计算单词的共现概率
        :return:共现概率
        """
        joinwordlist = wordlist1 + wordlist2
        percent = self.calcularprobability(document, joinwordlist)
        return percent

    def calculate_nmi(self, joinpercent, wordpercent1, wordpercent2):
        """
        function: 计算词语共现的nmi值
        :param joinpercent:
        :param wordpercent1:
        :param wordpercent2:
        :return:nmi
        """
        return (joinpercent) / (wordpercent1 * wordpercent2)

    def calculate_lis(self, word1, word2):

        if word1 != word2:
            wordpercent1 = 1
            wordpercent2 = 1
            if word1 in self.dict_frq_word:
                wordpercent1 = self.dict_frq_word[word1]
            # print(wordpercent1)
            if word2 in self.dict_frq_word:
                wordpercent2 = self.dict_frq_word[word2]
            # print(wordpercent2)
            list_together = []
            list_together.append(word1)
            list_together.append(word2)
            together_probability = self.calcularprobability(self.document, list_together)
            return self.calculate_nmi(together_probability, wordpercent1, wordpercent2)
        else:
            return 0

    def calculate_word_lis(self, word,wordpercent, Key):
        if len(Key)==0 or wordpercent == 0.0 or len(word.strip()) == 0 :
            return 0
        calculate_score = []
        for word2 in Key.keys():
            wordpercent2 = Key[word2]
            list_together = []
            list_together.append(word)
            list_together.append(word2)
            together_probability = self.calcularprobability(self.document, list_together)  # 同时出现的概率
            # print("word", word, together_probability)
            if together_probability > (self.miniprobability):
                # print(together_probability, wordpercent, wordpercent2)
                calculate_score.append(self.calculate_nmi(together_probability, wordpercent, wordpercent2))
                # print("self.calculate_nmi",list_together,together_probability,wordpercent, wordpercent2,self.calculate_nmi(together_probability, wordpercent, wordpercent2))
            # else:
            #     print("no",list_together,together_probability)
        calculate_score = sum(calculate_score) / (len(calculate_score) if len(calculate_score) != 0 else 1)  # 取平均值
        return calculate_score

    # 下面全是数据获取
    def removeEmoji(self, sentence):
        return re.sub('\[.*?\]', '', sentence)

    def CutWithPartOfSpeech(self, sentence):
        sentence = self.removeEmoji(sentence)
        words = self.tokenizer.tokenize_lis(sentence.strip())
        outstr = ""
        for word in words:
            if word not in self.stopwords:
                outstr += word
                outstr += "$"
        return list(set(outstr.split('$')))

    def segment_with_stopwords(self, list):
        outwords = []
        for word in list:
            if word == ' ':
                continue
            # 加停用词
            if word not in self.stopwords and word not in outwords:
                outwords.append(word)
        return outwords

    def segment(self, list):
        outwords = []
        for word in list:
            if word == ' ' or word not in self.dict_frq_word:
                continue
            # 加停用词
            if word not in outwords:
                outwords.append(word)
        return outwords

    def stopwordslist(self, filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return stopwords


    def RemoveStopWord(self, wordlist):
        stopWords = self.GetStopWords()
        keywords = []
        for word in wordlist:
            if word not in stopWords:
                keywords.append(word)
        return keywords

