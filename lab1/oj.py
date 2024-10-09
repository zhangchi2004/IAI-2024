import json
import math

pinyins = {}
words = {}
bi_words = {}

lmd = 0.999999

class Node:
    def __init__(
        self,
        py,
        word,
    ):
        self.py = py
        self.word = word
        self.prev = None
        self.acum_log_p = -math.inf

all_counts_for_py = {}
def get_all_counts_for_py(py):
    global words
    if py in all_counts_for_py:
        return all_counts_for_py[py]
    all_counts_for_py[py] = sum(words[py]['counts'])
    return all_counts_for_py[py]
    
all_counts_for_bi_py = {}
def get_all_counts_for_bi_py(bi_py):
    global bi_words
    if bi_py in all_counts_for_bi_py:
        return all_counts_for_bi_py[bi_py]
    all_counts_for_bi_py[bi_py] = sum(bi_words[bi_py]['counts'])
    return all_counts_for_bi_py[bi_py]

py_dict = {}
bi_py_dict = {}
def get_dict_of_py(py):
    global py_dict, words
    if py in py_dict:
        return py_dict[py]
    py_dict[py] = {}
    if py in words:
        for word,count in zip(words[py]['words'],words[py]['counts']):
            py_dict[py][word] = count
    return py_dict[py]

def get_dict_of_bi_py(bi_py):
    global bi_py_dict, bi_words
    if bi_py in bi_py_dict:
        return bi_py_dict[bi_py]
    bi_py_dict[bi_py] = {}
    if bi_py in bi_words:
        for bi_word,count in zip(bi_words[bi_py]['words'],bi_words[bi_py]['counts']):
            bi_py_dict[bi_py][bi_word] = count
    return bi_py_dict[bi_py]

def get_log_p(word,py):
    c1 = get_dict_of_py(py)[word] if word in get_dict_of_py(py) else 0
    return math.log(c1/get_all_counts_for_py(py))

def get_log_p_cond(word1,word2,py1,py2): # log(p(word2|word1))
    global pinyins
    bi_py = py1+' '+py2
    bi_word = word1+' '+word2
    c2 = get_dict_of_bi_py(bi_py)[bi_word] if bi_word in get_dict_of_bi_py(bi_py) else 0
    if word1 not in get_dict_of_py(py1):
        p_cond = 0
    else:
        p_cond = c2/get_dict_of_py(py1)[word1]
    
    c1 = get_dict_of_py(py2)[word2] if word2 in get_dict_of_py(py2) else 0
    p2 = c1 / get_all_counts_for_py(py2)    
    
    return math.log(lmd*p_cond + (1-lmd)*p2)
    
class Net:
    def __init__(self,query):
        global words
        self.query = query
        self.layers = []
        for py in query:
            self.layers.append([Node(py,word) for word in words[py]['words']])
        
        #layers[0]
        layer0_len = len(self.layers[0])
        for node in self.layers[0]:
            node.acum_log_p = get_log_p(node.word,node.py)
        
        #layers[1:]
        for layer_num in range(1,len(self.layers)):
            for node in self.layers[layer_num]:
                for prev_node in self.layers[layer_num-1]:
                    log_p = get_log_p_cond(prev_node.word,node.word,prev_node.py,node.py)
                    if prev_node.acum_log_p + log_p > node.acum_log_p:
                        node.acum_log_p = prev_node.acum_log_p + log_p
                        node.prev = prev_node
    
    def get_max_sentence(self):
        max_node = max(self.layers[-1],key=lambda x:x.acum_log_p)
        reversed_sentence = [max_node.word]
        while max_node.prev:
            reversed_sentence.append(max_node.prev.word)
            max_node = max_node.prev
        sentence = ""
        while reversed_sentence:
            sentence += reversed_sentence.pop()
        return sentence

import sys
def main():
    
    # sys.stdin = open("./测试语料/my_input.txt",'r')
    # sys.stdout = open("./测试语料/my_output.txt",'w')
    global pinyins, words, bi_words
    with open("./word2pinyin.txt",'r',encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] not in pinyins:
                pinyins[parts[0]] = [parts[1]]
            else:
                pinyins[parts[0]].append(parts[1])
            
    with open('./1_word.txt','r',encoding='utf-8') as f:
        words = json.load(f)
    with open('./2_word.txt','r',encoding='utf-8') as f:
        bi_words = json.load(f)
        
    while True:
        try:
            inp = input()
            if inp:
                query = inp.strip().split()
                net = Net(query)
                print(net.get_max_sentence())
            else:
                break
        except EOFError:
            break

if __name__ == '__main__':
    main()
