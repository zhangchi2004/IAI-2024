# Create middle files used by the input algorithm.

# Package `ujson` need to be installed for fast json loading and dumping, 
# or we can replace all 'ujson' in the following code to `json`.

# To run this code, we need to place the "语料库" folder in `../data/text`. 

# ../
#  |- src
#  |- data
#      |- mid
#      |- text
#          |- 语料库    (To be placed!)
#              |- sina_news_gbk
#                  |- 2016-04.txt
#                  |- ...
#      |- std_input.txt
#      |- std_output.txt

import ujson
import copy
import os
pinyins = {}
os.makedirs("../data/mid",exist_ok=True) # Place to store the middle files.
with open("../data/text/拼音汉字表.txt",'r',encoding='gb2312') as f:
    for line in f:
        pyandword = line.strip().split()
        py = pyandword[0]
        for i in range(1,len(pyandword)):
            word = pyandword[i]
            if word not in pinyins: pinyins[word]=[py]
            else: pinyins[word].append(py)

words = {}
bi_words = {} 
tri_words = {}
for i in range(4,12): # use sina_news to train the model only.
    filename = "../data/text/语料库/sina_news_gbk/2016-0"+str(i)+".txt" if i<10 else "../data/text/语料库/sina_news_gbk/2016-"+str(i)+".txt"
    with open(filename,'r',encoding='gbk') as f:
        k=0
        for line in f:
            k+=1
            if k%100==0: print(k)
            jsonobj = ujson.loads(line)
            text = jsonobj['html']
            for word in text: # make word list for single word
                if word not in pinyins: continue
                for py in pinyins[word]:
                    if py not in words: words[py] = {"count":0}
                    words[py]["count"]+=1
                    if word in words[py]:
                        words[py][word] += 1
                    else:
                        words[py][word] = 1
            for i in range(len(text)-1): # make bi-word list for bigram language model
                bi_word = text[i]+' '+text[i+1]
                if text[i] not in pinyins or text[i+1] not in pinyins: continue
                for py1 in pinyins[text[i]]:
                    for py2 in pinyins[text[i+1]]:
                        bi_py = py1+' '+py2
                        if bi_py not in bi_words: bi_words[bi_py] = {"count":0}
                        bi_words[bi_py]["count"] += 1
                        if bi_word in bi_words[bi_py]:
                            bi_words[bi_py][bi_word] += 1
                        else:
                            bi_words[bi_py][bi_word] = 1
                            
            for i in range(len(text)-2): # make tri-word list for 3-gram language model
                tri_word = text[i]+' '+text[i+1]+' '+text[i+2]
                if text[i] not in pinyins or text[i+1] not in pinyins or text[i+2] not in pinyins: continue
                for py1 in pinyins[text[i]]:
                    for py2 in pinyins[text[i+1]]:
                        for py3 in pinyins[text[i+2]]:
                            tri_py = py1+' '+py2+' '+py3
                            if tri_py not in tri_words: tri_words[tri_py] = {"count":0}
                            tri_words[tri_py]["count"] += 1
                            if tri_word in tri_words[tri_py]:
                                tri_words[tri_py][tri_word] += 1
                            else:
                                tri_words[tri_py][tri_word] = 1
                                
with open("../data/mid/words.json",'w',encoding='utf-8') as f:
    ujson.dump(words,f,ensure_ascii=False,indent=4)
with open("../data/mid/bi_words.json",'w',encoding='utf-8') as f:
    ujson.dump(bi_words,f,ensure_ascii=False,indent=4)
with open("../data/mid/tri_words.json",'w',encoding='utf-8') as f:
    ujson.dump(tri_words,f,ensure_ascii=False,indent=4)


# import ujson
# import copy
# with open("./tri_words.json",'r',encoding='utf-8') as f:
#     tri_words = ujson.load(f)

tri_words_2 = copy.deepcopy(tri_words)
for tri_py in tri_words: # create a copy of the tri-word list, deleting the unfrequent words to shorten the file length.
    for tri_word in tri_words[tri_py]:
        if tri_words[tri_py][tri_word] < 16:
            tri_words[tri_py]["count"] -= tri_words[tri_py][tri_word]
            del tri_words_2[tri_py][tri_word]
    if tri_words[tri_py]["count"] < 16:
        del tri_words_2[tri_py]
with open("../data/mid/tri_words_2.json",'w',encoding='utf-8') as f:
    ujson.dump(tri_words_2,f,ensure_ascii=False,indent=4)