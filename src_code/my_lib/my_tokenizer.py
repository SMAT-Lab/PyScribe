#coding=utf-8
import nltk
import re

def tokenize_english(text,vocabs=None,keep_punc=True,keep_stopword=True,lemmatize=True,lower=True):
    '''
    使用nltk分词
    :param text: 待分词的英文文本
    :param keep_punc: 是否保留标点符号
    :param keep_punc: 是否做词干提取，文本匹配时很有用
    :return: 分词后的词语列表
    '''
    # if lower:
    #     text=text.lower()

    text=text.replace("``",'"').replace("''",'"').replace('`',"'")
    text=re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*",r'<url>',text,re.S)

    text=' - '.join(text.split('-'))
    # text=text.replace('- -','--')
    words=nltk.word_tokenize(text)
    if lemmatize:
        lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
        if vocabs is not None:
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
                     if word not in vocabs else word for word in words]
        else:
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
                     for word in words]

    if not keep_punc:
        stop_puncs_0 = ['|', '{}', '()', '[]', '&', '*',
                        '/', '//', '#', '\\', '~', '""', '‖', '§']
        stop_puncs_1 = ['、', '\'', '"', '.', ':', ',', '...', '{', '}', '(', ')', '[', ']',
                        ';', '?', '!', '-', '--']
        # stop_puncs_2 = ["``", "''",'`']
        stop_puncs = stop_puncs_0 + stop_puncs_1
        words=[word for word in words if word not in stop_puncs]
    if not keep_stopword:
        stop_words = nltk.corpus.stopwords.words('english')
        words=[word for word in words if word not in stop_words]
    if lower:
        if vocabs is not None:
            words=[word.lower() if word not in vocabs else word for word in words]
        else:
            words=[word.lower() for word in words]
    return words

def tokenize_glove(text,vocabs=None,keep_punc=True,keep_stopword=True,lemmatize=True,lower=False):
    '''
    针对glove词库做分词
    :param text:
    :param vocabs: glove字典,glove字典中词语区分大小写
    :param keep_punc:
    :param keep_stopword:
    :param lemmatize:
    :param lower:
    :return:
    '''
    # stop_puncs=['"','.',':',',','...','|','{','}','{}','(',')','()','[',']','[]','&','*','`',
    #             '/','//','#','\\','~','、',';','?','!','\'','-','--','""','‖','§']

    # stemmer=nltk.stem.SnowballStemmer(language='english')
    # lemmatizer = nltk.stem.WordNetLemmatizer()  #词干提取
    # text = (' ' + text + ' ').lower()
    # if text.startwith('\''):
    #     text=text.lstrip('\'')
    #     text='\' '+text
    # eyes = "[8:=;]"
    # nose = "['`\-]?"
    text=text.replace("``",'"').replace("''",'"').replace('`',"'")
    text=re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*",r'<url>',text,re.S)

    text=re.sub(r"#{[8:=;]}#{['`\-]?}[)d]+|[)d]+#{['`\-]?}#{[8:=;]}",'<smile>',text,re.S)
    text = re.sub(r"#{[8:=;]}#{['`\-]?}p+", '<lolface>', text,re.S)
    text = re.sub(r"#{[8:=;]}#{['`\-]?}\(+|\)+#{['`\-]?}#{[8:=;]}", '<sadface>', text, re.S)
    text = re.sub(r"#{[8:=;]}#{['`\-]?}[\/|l*]", '<neutralface>', text, re.S)
    text = re.sub(r"<3", '<heart>', text,re.S)
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", '<number>', text, re.S)

    text = (' ' + text).replace(' \'',' \' ').strip()
    # text = (' ' + text + ' ').lower().lstrip(' \'').rstrip('\' ').strip()
    text=' - '.join(text.split('-'))
    # text=text.replace('- -','--')
    words=nltk.word_tokenize(text)
    if lemmatize:
        lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取

        if vocabs is not None:
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
                     if word not in vocabs else word for word in words]

        else:
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
                     for word in words]
    words = ' '.join(words).replace('< url >', '<url>').replace('< smile >', '<smile>') \
        .replace('< lolface >', '<lolface>').replace('< sadface >', '<sadface>') \
        .replace('< neutralface >', '<neutralface>').replace('< heart >', '<v>') \
        .replace('< number >', '<number>').split()
    # words = text.split()
    if not keep_punc:
        stop_puncs_0 = ['|', '{}', '()', '[]', '&', '*',
                        '/', '//', '#', '\\', '~', '""', '‖', '§']
        stop_puncs_1 = ['、', '\'', '"', '.', ':', ',', '...', '{', '}', '(', ')', '[', ']',
                        ';', '?', '!', '-', '--']
        # stop_puncs_2 = ["``", "''",'`']
        stop_puncs = stop_puncs_0 + stop_puncs_1
        words=[word for word in words if word not in stop_puncs]
    if not keep_stopword:
        stop_words = nltk.corpus.stopwords.words('english')
        words=[word for word in words if word not in stop_words]
    if lower:
        if vocabs is not None:
            words=[word.lower() if word not in vocabs else word for word in words]
        else:
            words=[word.lower() for word in words]
    return words

from nltk import WordPunctTokenizer
import string
punc_str2="""‖`§！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
punc_str=string.punctuation+punc_str2
def tokenize_python(code,keep_puc=False,punc_str=punc_str):
    code = code.replace('_', ' _ ')
    tokens = WordPunctTokenizer().tokenize(code)
    # tokens = nltk.word_tokenize(code)
    # for punc in punc_str:
    #     code=code.replace(punc,' '+punc+' ')
    # code= ' '.join(tokens)
    if not keep_puc:
        code=re.sub(r'[{}]'.format(punc_str),' ',' '.join(tokens),re.S)
        tokens=code.split()
    # if keep_puc:
    #     tokens=list(filter(lambda x: x not in pucs,tokens))
    return tokens
# def tokenize_text_en(text,
#                      vocabs=None,
#                      keep_mark=True,
#                      lemmatize=False,
#                      lower=False,
#                      dash_sep=False):
#     '''
#     使用nltk分词
#     :param text: 待分词的英文文本
#     :param vocabs: 是否在词根提取时添加参考词库，在词库中的不做词根提取
#     :param keep_mark: 是否保留标点符号
#     :param lemmatize: 是否做词根提取
#     :param lower: 是否全部转换成小写
#     :param dash_sep: 是否将dash连接的词语分离开
#     :return:
#     '''
#     '''
#
#     :param text: 待分词的英文文本
#     :param keep_mark: 是否保留标点符号
#     :param keep_mark: 是否做词干提取，文本匹配时很有用
#     :return: 分词后的词语列表
#     '''
#     # stop_marks=['"','.',':',',','...','|','{','}','{}','(',')','()','[',']','[]','&','*','`',
#     #             '/','//','#','\\','~','、',';','?','!','\'','-','--','""','‖','§']
#     stop_marks_0 = ['|', '{}', '()', '[]', '&', '*', '`',
#                     '/', '//', '#', '\\', '~', '""', '‖', '§']
#     stop_marks_1 = ['、', '\'','"', '.', ':', ',', '...', '{', '}', '(', ')', '[', ']',
#                     ';', '?', '!', '-', '--']
#     # stemmer=nltk.stem.SnowballStemmer(language='english')
#     # lemmatizer = nltk.stem.WordNetLemmatizer()  #词干提取
#     # text = (' ' + text + ' ').lower()
#     # if text.startwith('\''):
#     #     text=text.lstrip('\'')
#     #     text='\' '+text
#     # eyes = "[8:=;]"
#     # nose = "['`\-]?"
#
#     text=re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*",'<url>',text,re.S)
#     text=re.sub(r"#{[8:=;]}#{['`\-]?}[)d]+|[)d]+#{['`\-]?}#{[8:=;]}",'<smile>',text,re.S)
#     text = re.sub(r"#{[8:=;]}#{['`\-]?}p+", '<lolface>', text,re.S)
#     text = re.sub(r"#{[8:=;]}#{['`\-]?}\(+|\)+#{['`\-]?}#{[8:=;]}", '<sadface>', text, re.S)
#     text = re.sub(r"#{[8:=;]}#{['`\-]?}[\/|l*]", '<neutralface>', text, re.S)
#     text = re.sub(r"<3", '<heart>', text,re.S)
#     text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", '<number>', text, re.S)
#
#     # text=re.sub("<3", '<HEART>', text)
#     if lower:
#         text=text.lower()
#     text = (' ' + text).replace(' \'',' \' ').strip()
#     # text = (' ' + text + ' ').lower().lstrip(' \'').rstrip('\' ').strip()
#
#     if dash_sep:
#         text=' - '.join(text.split('-'))
#
#     # text=text.replace('- -','--')
#     words=nltk.word_tokenize(text)
#
#     if lemmatize:   #是否根据具体情况做词干提取
#         lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
#
#     if isinstance(vocabs,list):
#         for i,word in enumerate(words):
#             if word not in vocabs:
#                 word=word[:1]+word[1:].lower()  #除首字母外都小写
#                 if word not in vocabs:
#                     word=word.lower()   #全部小写
#                     if word not in vocabs and lemmatizer:
#                         word=lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='n'), pos='v'),pos='a')
#             words[i]=word
#     elif vocabs is None:
#         if lemmatize:
#             words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos='n'), pos='v'), pos='a')
#                      for word in words]
#
#     # if lemmatize:
#     #     lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
#     #
#     #     if vocabs is not None:
#     #         words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
#     #                  if word not in vocabs else word for word in words]
#     #
#     #     else:
#     #         words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
#     #                  for word in words]
#
#     text=' '.join(words).replace("``",'"').replace("''",'"')
#
#     if lower:
#         text=text.replace('< url >','<url>').replace('< smile >','<smile>')\
#             .replace('< lolface >','<lolface>').replace('< sadface >','<sadface>')\
#             .replace('< neutralface >','<neutralface>').replace('< heart >','<v>')\
#             .replace('< number >','<number>')
#
#     words=text.split()
#     words = [word for word in words if word not in stop_marks_0]
#     if not keep_mark:
#         words=[word for word in words if word not in stop_marks_1]
#     return words


# def parse_chinese(text,model_path,visual=False):
#     '''
#     使用斯坦福解析工具对中文文本进行语法解析
#     https://www.jianshu.com/p/002157665bfd
#     @InProceedings{manning-EtAl:2014:P14-5,
#       author    = {Manning, Christopher D. and  Surdeanu, Mihai  and  Bauer, John  and  Finkel, Jenny  and  Bethard, Steven J. and  McClosky, David},
#       title     = {The {Stanford} {CoreNLP} Natural Language Processing Toolkit},
#       booktitle = {Association for Computational Linguistics (ACL) System Demonstrations},
#       year      = {2014},
#       pages     = {55--60},
#       url       = {http://www.aclweb.org/anthology/P/P14/P14-5010}
#     }
#     :param text:
#     :param model_path: path of stanford-corenlp-4.0.0-models-chinese.jar
#     :return:
#     '''
#     nlp=StanfordCoreNLP(model_path)




if __name__=='__main__':
    # s='she does not want to did-it inter-mediate is went fully <number> <LOLFACE>.'
    # print(tokenize_text_en(s))

    text='https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb '
    text=re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", '<URL>', text,re.S)
    print(text)

    text = '😊#{eyes}#{nose})d'
    text = re.sub(r"#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}",'<SMILE>',text,re.S)
    print(text)

    text='a3asdfasd32 234 adj'
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", '<NUMBER>', text, re.S)
    print(text)

    code='''
    def tokenize_python(code,keep_puc=False,punc_str=punc_str):
    code = code.replace('_', ' _ ')
    tokens = WordPunctTokenizer().tokenize(code)'''
    print(tokenize_python(code))