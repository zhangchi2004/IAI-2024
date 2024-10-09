with open("../data/my_output.txt",'r',encoding='utf-8') as f:
    my_output = f.readlines()
    
with open("../data/std_output.txt",'r',encoding='utf-8') as f:
    std_output = f.readlines()

correct_lines = 0
correct_words = 0
all_words = 0
for i,(my,std) in enumerate(zip(my_output,std_output)):
    if my == std: correct_lines+=1
    for my_word,std_word in zip(my,std):
        if my_word == std_word: correct_words+=1
        all_words+=1
print("sentence accuracy: ",correct_lines/len(my_output))
print("word accuracy: ",correct_words/all_words)
        