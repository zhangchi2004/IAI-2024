# 拼音输入法

> 张驰 2022010754 <zhang-ch22@mails.tsinghua.edu.cn>

当前文件结构：

- data/
  - mid/
  - text/
    - 拼音汉字表.txt
- src/
  - makedata.py
  - pinyin.py
  - eval.py

运行输入法需要在文件中添加语料库。将语料库文件夹添加在 `data/text/` 路径下。使得新的文件结构如下：

- data/
  - mid/
  - text/
    - 拼音汉字表.txt
    - 语料库/
      - sina_news_gbk/
        - 2016.04.txt
        - ...
- src/
  - makedata.py
  - pinyin.py
  - eval.py
  
将语料库正确放在上述路径下后，运行 `makedata.py` 会根据语料库，在 `data/mid/` 路径下生成中间文件。

运行 `pinyin.py` 会根据中间文件，执行拼音输入法。

目前的 `pinyin.py` 支持基于二元和三元语言模型的拼音输入法。可以在运行 `pinyin.py` 时增加参数指定使用的语言模型。

`python pinyin.py` 默认情况下使用基于二元模型的拼音输入法。

`python pinyin.py --model=bi` 指定基于二元模型的拼音输入法。

`python pinyin.py --model=tri` 指定基于三元模型的拼音输入法。

`python pinyin.py --model=tri_s` 指定使用缩小三元词中间文件规模（为了减少文件大小，将三元词中间文件中出现次数小于15次的三元词删除）的基于三元模型的拼音输入法。

`eval.py` 为用于检测标准测例的准确率。