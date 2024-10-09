# Implement the pinyin input method.

# Package `ujson` need to be installed for fast json loading, 
# or we can replace all 'ujson' in the following code to `json`.

import ujson
import math
import time
import argparse
import sys

words = {}
bi_words = {}
tri_words = {}

lmd = 0.99999999 # weight for bi-word conditional probability.
mu = 0.9 # weight for tri-word conditional probability.

all_counts_for_py = {}
def get_all_counts_for_py(py):
    global words,all_counts_for_py
    if py in all_counts_for_py:
        return all_counts_for_py[py]
    else:
        all_counts_for_py[py] = sum(words[py].values())
        return all_counts_for_py[py]

all_counts_for_bi_py = {}
def get_all_counts_for_bi_py(bi_py):
    global bi_words,all_counts_for_bi_py
    if bi_py in all_counts_for_bi_py:
        return all_counts_for_bi_py[bi_py]
    else:
        all_counts_for_bi_py[bi_py] = sum(bi_words[bi_py].values())
        return all_counts_for_bi_py[bi_py]

all_counts_for_tri_py = {}
def get_all_counts_for_tri_py(tri_py):
    global tri_words,all_counts_for_tri_py
    if tri_py in all_counts_for_tri_py:
        return all_counts_for_tri_py[tri_py]
    else:
        all_counts_for_tri_py[tri_py] = sum(tri_words[tri_py].values())
        return all_counts_for_tri_py[tri_py]

# def get_all_counts_for_py(py):
#     global words
#     return words[py]["count"] if py in words else 0

# def get_all_counts_for_bi_py(bi_py):
#     global bi_words
#     return bi_words[bi_py]["count"] if bi_py in bi_words else 0

# def get_all_counts_for_tri_py(tri_py):
#     global tri_words
#     return tri_words[tri_py]["count"] if tri_py in tri_words else 0

def get_p_1(word,py):
    global words
    c1 = words[py][word] if word in words[py] else 0
    return c1/get_all_counts_for_py(py)

def get_log_p_1(word,py):
    return math.log(get_p_1(word,py))

def get_p_2(word1,word2,py1,py2): # p(word2|word1)
    global bi_words, words
    bi_py = py1+' '+py2
    bi_word = word1+' '+word2
    c2 = 0
    if bi_py in bi_words and bi_word in bi_words[bi_py]:
        c2 = bi_words[bi_py][bi_word]
    if word1 not in words[py1]:
        p_cond = 0
    else:
        p_cond = c2/words[py1][word1]
    
    p2 = get_p_1(word2,py2)
    return lmd*p_cond + (1-lmd)*p2

def get_log_p_2(word1,word2,py1,py2): # log(p(word2|word1))
    return math.log(get_p_2(word1,word2,py1,py2))

    
def get_log_p_3(word1,word2,word3,py1,py2,py3): # log(p(word3|word1,word2))
    global tri_words, bi_words,words
    tri_py = py1+' '+py2+' '+py3
    tri_word = word1+' '+word2+' '+word3
    c3 = 0
    if tri_py in tri_words and tri_word in tri_words[tri_py]:
        c3 = tri_words[tri_py][tri_word]
    if word1+' '+word2 not in bi_words[py1+' '+py2]:
        p_cond = 0
    else:
        p_cond = c3/bi_words[py1+' '+py2][word1+' '+word2]
    
    p2 = get_p_2(word2,word3,py2,py3)
    return math.log(mu*p_cond + (1-mu)*p2)



class Bi_Node:
    def __init__(
        self,
        py,
        word,
    ):
        self.py = py
        self.word = word
        self.prev = None
        self.acum_log_p = -math.inf

class Bi_Net: # the hidden Markov Net
    def __init__(self,query):
        global words
        self.query = query
        self.layers = []
        for py in query:
            self.layers.append([Bi_Node(py,word) for word in words[py] if word!="count"])
        
        #layers[0]
        layer0_len = len(self.layers[0])
        for node in self.layers[0]:
            node.acum_log_p = get_log_p_1(node.word,node.py)
        
        #layers[1:]
        for layer_num in range(1,len(self.layers)):
            for node in self.layers[layer_num]:
                for prev_node in self.layers[layer_num-1]:
                    log_p = get_log_p_2(prev_node.word,node.word,prev_node.py,node.py)
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



class Tri_Node:
    def __init__(
        self,
        py,
        word,
    ):
        self.py = py
        self.word = word
        self.prev = None
        self.pprev = None
        self.prev_p_list = [] #[(acum_p, pprev_best),(),(),...] as order of prev.
        # The above list is used to record all bi-word probability of adjacent words for 3-gram model.
        
class Tri_Net:
    def __init__(self,query):
        global words
        self.query = query
        self.layers = []
        for py in query:
            self.layers.append([Tri_Node(py,word) for word in words[py] if word!="count"])
        
        #layers[0]
        layer0_len = len(self.layers[0])
        for node in self.layers[0]:
            node.prev_p_list = [get_log_p_1(node.word,node.py)]
        
        #layers[1]
        if(len(self.layers)>1):
            for i,node in enumerate(self.layers[1]):
                for j,prev_node in enumerate(self.layers[0]):
                    log_p = get_log_p_2(prev_node.word,node.word,prev_node.py,node.py)
                    node.prev_p_list.append((log_p + prev_node.prev_p_list[0], -1))
                        
        #layers[2:]
        for layer_num in range(2,len(self.layers)):
            for node in self.layers[layer_num]:
                for prev_i,prev_node in enumerate(self.layers[layer_num-1]):
                    node.prev_p_list.append((-math.inf,-1))
                    for pprev_i,pprev_node in enumerate(self.layers[layer_num-2]):
                        log_p_cond = get_log_p_3(pprev_node.word,prev_node.word,node.word,pprev_node.py,prev_node.py,node.py)
                        if log_p_cond + prev_node.prev_p_list[pprev_i][0] > node.prev_p_list[prev_i][0]:
                            node.prev_p_list[prev_i] = (log_p_cond + prev_node.prev_p_list[pprev_i][0],pprev_i)
                            
    
    def get_max_sentence(self):
        max_of_last_layer = []
        for node in self.layers[-1]:
            max = -math.inf
            max_j = -1
            for j,tup in enumerate(node.prev_p_list):
                if tup[0]>max:
                    max = tup[0]
                    max_j = j
            max_of_last_layer.append((max,max_j))
                    
                    
        max = -math.inf
        max_i = -1
        i_prev = -1
        for i,tup in enumerate(max_of_last_layer):
            if tup[0]>max:
                max = tup[0]
                max_i = i
                i_prev = tup[1]
        
        reversed_sentence=[]
        layer_num = -1
        while True:
            node = self.layers[layer_num][max_i]
            reversed_sentence.append(node.word)
            i_pprev = node.prev_p_list[i_prev][1]
            
            if i_pprev == -1:
                reversed_sentence.append(self.layers[layer_num-1][i_prev].word)
                break
            max_i = i_prev
            i_prev = i_pprev
            
            layer_num -= 1
            
        sentence = ""
        while reversed_sentence:
            sentence += reversed_sentence.pop()
        return sentence

def bi_model():
    global words, bi_words
    with open('../data/mid/words.json','r',encoding='utf-8') as f:
        words = ujson.load(f)
    with open('../data/mid/bi_words.json','r',encoding='utf-8') as f:
        bi_words = ujson.load(f)
        
    start = time.time()
        
    while True:
        try:
            inp = input()
            if inp:
                query = inp.strip().split()
                net = Bi_Net(query)
                print(net.get_max_sentence())
            else:
                break
        except EOFError:
            break
    end = time.time()
    # print(end-start)

def tri_model(shortened = False):
    global words, bi_words, tri_words
    with open('../data/mid/words.json','r',encoding='utf-8') as f:
        words = ujson.load(f)
    with open('../data/mid/bi_words.json','r',encoding='utf-8') as f:
        bi_words = ujson.load(f)
    if shortened:
        with open('../data/mid/tri_words_2.json','r',encoding='utf-8') as f:
            tri_words = ujson.load(f)
    else:
        with open('../data/mid/tri_words.json','r',encoding='utf-8') as f:
            tri_words = ujson.load(f)
        
    start = time.time()
    while True:
        try:
            inp = input()
            if inp:
                query = inp.strip().split()
                net = Tri_Net(query)
                print(net.get_max_sentence())
            else:
                break
        except EOFError:
            break
    end = time.time()
    # print(end-start)


def main():
    # sys.stdin = open('../data/std_input.txt','r',encoding='utf-8')
    # sys.stdout = open('../data/my_output.txt','w',encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bi', help='bi, tri, tri_s') # tri_s for 3-gram model with shortened middle data.
    args = parser.parse_args()
    if(args.model=="bi"):bi_model()
    if(args.model=="tri"):tri_model()
    if(args.model=="tri_s"):tri_model(shortened=True)
    
if __name__ == '__main__':
    main()