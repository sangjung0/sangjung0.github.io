---
title: 토큰화
date: 2025-02-20 21:28:00 +0900
categories: [AI, Data Engineering]
tags: [AI, Data Engineering, Data Preprocessing, Text Preprocessing]
pin: false
math: true
mermaid: true
---


# 토큰화 (Tokenization)
토큰화는 데이터 전처리 기법 중 하나로, 코퍼스(corpus)에서 토큰이라 불리는 단위로 나누는 작업을 말한다.
토큰의 단위는 다양하지만, 의미있는 단위로 토큰을 정의한다

# 단어 토큰화 (Word Tokenization)
토큰의 기준을단어로 하는 경우 단어 토큰화라고 한다.
여기서 단어는 단어, 단어구, 문자열로도 간주 된다.

토큰화는 단순하게 특수문자를 제거하는 정제(cleaning) 작업과 띄어쓰기를 기준으로 나누는 것이 아니다.
특수문자에와 띄어쓰기에도 숨겨진 의미가 있을 수 있다.

토큰을 나누는 기준은 여러 도구에 따라 다르다.

```python

from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# 테스트 문자열 이다.
test_string = "In Python, you can use lists, tuples, and dictionaries! But don’t forget about set as well. If you want to print \"Hello, world!\", just type print(\"Hello, world!\")—but be sure to check the quotation marks! Here's a math example: a^2 + b^2 = c^2, or even ∑(n=1 to ∞) 1/n^2 = π^2/6. Fascinating, isn’t it? Also, an email address can look like user@example.com, or even something like user_name-123@domain.co.kr—both formats are valid."

print('word_tokenize:', " ".join(word_tokenize(test_string)))
print('WordPunctTokenizer:', " ".join(WordPunctTokenizer().tokenize(test_string)))
print('text_to_word_sequence:', " ".join(text_to_word_sequence(test_string)))
```

```plain
// 출력
word_tokenize: In Python , you can use lists , tuples , and dictionaries ! But don ’ t forget about set as well . If you want to print `` Hello , world ! `` , just type print ( `` Hello , world ! `` ) —but be sure to check the quotation marks ! Here 's a math example : a^2 + b^2 = c^2 , or even ∑ ( n=1 to ∞ ) 1/n^2 = π^2/6 . Fascinating , isn ’ t it ? Also , an email address can look like user @ example.com , or even something like user_name-123 @ domain.co.kr—both formats are valid .

WordPunctTokenizer: In Python , you can use lists , tuples , and dictionaries ! But don ’ t forget about set as well . If you want to print " Hello , world !", just type print (" Hello , world !")— but be sure to check the quotation marks ! Here ' s a math example : a ^ 2 + b ^ 2 = c ^ 2 , or even ∑( n = 1 to ∞) 1 / n ^ 2 = π ^ 2 / 6 . Fascinating , isn ’ t it ? Also , an email address can look like user @ example . com , or even something like user_name - 123 @ domain . co . kr — both formats are valid .

text_to_word_sequence: in python you can use lists tuples and dictionaries but don’t forget about set as well if you want to print hello world just type print hello world —but be sure to check the quotation marks here's a math example a 2 b 2 c 2 or even ∑ n 1 to ∞ 1 n 2 π 2 6 fascinating isn’t it also an email address can look like user example com or even something like user name 123 domain co kr—both formats are valid
```

word_tokenize는 '"'를 ''''로 바꾸는 것을 알 수 있다. 
그리고 ''s'와 같이 '''는 유의미하게 나누는 것 처럼 보인다.
이메일 주소를 의미있게 효과적으로 나누었다.
WordPunctTokenizer 같은 경우, 의미 없이 기호를 그룹화 하는 느낌이 있다.
그리고 띄어쓰기를 기준으로 단순히 나눈 것 같다.
text_to_word_sequence는 'here's'와 같이 의미있는 기호를 제외하고는 모든 기호를 제거했다. 그리고 띄어쓰기를 기준으로 나누었으며, 'don't'와 같이 기호를 하나의 토큰으로 묶었다.

이 처럼, 토큰화 도구마다 다른 결과를 가지며, 작업에 따라 어떤 방식으로 토큰화를 할지 선택할 필요성이 있다.

```python

from nltk.tokenize import TreebankWordTokenizerfrom nltk.to

print('TreebankWordTokenizer:', " ".join(TreebankWordTokenizer().tokenize(test_string)))
```

```plain
//출력
TreebankWordTokenizer: In Python , you can use lists , tuples , and dictionaries ! But don’t forget about set as well. If you want to print `` Hello , world ! '' , just type print ( `` Hello , world ! '' ) —but be sure to check the quotation marks ! Here 's a math example : a^2 + b^2 = c^2 , or even ∑ ( n=1 to ∞ ) 1/n^2 = π^2/6. Fascinating , isn’t it ? Also , an email address can look like user @ example.com , or even something like user_name-123 @ domain.co.kr—both formats are valid .
```

이는 표준으로 쓰이고 있는 토큰화 방법인 Peen Treebank Tokenization이다.
'isn't'와 같은 경우는 모두 하나의 토큰으로 표기하며, '``'와 ''''을 이용하여 '"a"'을 표현한다.
이메일 또한 효과적으로 토큰화 한 것을 볼 수 있다.


# 문장 토큰화(Sentence Tokenization)
토큰 단위가 문장인 경우이다. 문장 분류라고도 부른다.
문장 토큰화 또한 언어 오타 여부 등에 따라 대응해야 하기에 쉽지 않다.

```python
from nltk.tokenize import sent_tokenize
print('-- sent_tokenize -- \n', "\n".join(sent_tokenize(test_string)))
```

```plain
-- sent_tokenize -- 
 In Python, you can use lists, tuples, and dictionaries!
But don’t forget about set as well.
If you want to print "Hello, world!
", just type print("Hello, world!
")—but be sure to check the quotation marks!
Here's a math example: a^2 + b^2 = c^2, or even ∑(n=1 to ∞) 1/n^2 = π^2/6.
Fascinating, isn’t it?
Also, an email address can look like user@example.com, or even something like user_name-123@domain.co.kr—both formats are valid.
```

위의 sent_tokenize는 효과적으로 문장을 구분 했지만, 부족한 점이 보인다.
'If you want to print "Hello, world!' 부분에서 문장이 끝나지 않았지만, 문장을 나눈 것을 볼 수 있다.


한글 문장 토큰화도 진행해 보았다.

```python
import kss

# 테스트 문자열
test_string_kor = "파이썬에서는 리스트(list), 튜플(tuple), 딕셔너리(dictionary)를 사용할 수 있어! 하지만 set도 잊으면 안 돼. \"Hello, world!\"를 출력하려면 print(\"Hello, world!\")를 입력하면 돼—단, 따옴표를 제대로 확인해야 해! 여기 수학 공식 예제가 있어: a^2 + b^2 = c^2, 또는 ∑(n=1부터 ∞까지) 1/n^2 = π^2/6. 흥미롭지 않아? 또한 이메일 주소는 user@example.com처럼 쓸 수도 있고, user_name-123@domain.co.kr 같은 형식도 가능해—둘 다 유효한 형식이야."
print('-- kss --\n', "\n".join(kss.split_sentences(test_string_kor)))
```

```plain
-- kss --
 파이썬에서는 리스트(list), 튜플(tuple), 딕셔너리(dictionary)를 사용할 수 있어!
하지만 set도 잊으면 안 돼.
"Hello, world!"를 출력하려면 print("Hello, world!")를 입력하면 돼—단, 따옴표를 제대로 확인해야 해!
여기 수학 공식 예제가 있어
: a^2 + b^2 = c^2, 또는 ∑(n=1부터 ∞까지) 1/n^2 = π^2/6. 흥미롭지 않아?
또한 이메일 주소는 user@example.com처럼 쓸 수도 있고, user_name-123@domain.co.kr 같은 형식도 가능해—둘 다 유효한 형식이야.
```

한글도 문장 토큰화가 잘 진행된 것을 알 수 있다. 그러나, '여기 수학 공식 예제가 있어' 부분에 문장이 끝나지 않았지만, 토큰이 나누어진 겻을 알 수 있다.

# 품사 태깅 (Part-of-speech tagging)
단어는 품사에 따라 다른 뜻을 가질 수 있다. 따라서 품사를 태깅할 필요가 있다.

```python
from nltk.tag import pos_tag

print('pos_tag:', pos_tag(word_tokenize(test_string)))
```

```plain
pos_tag: [('In', 'IN'), ('Python', 'NNP'), (',', ','), ('you', 'PRP'), ('can', 'MD'), ('use', 'VB'), ('lists', 'NNS'), (',', ','), ('tuples', 'NNS'), (',', ','), ('and', 'CC'), ('dictionaries', 'NNS'), ('!', '.'), ('But', 'CC'), ('don', 'JJ'), ('’', 'NNP'), ('t', 'NN'), ('forget', 'NN'), ('about', 'RB'), ('set', 'VBN'), ('as', 'RB'), ('well', 'RB'), ('.', '.'), ('If', 'IN'), ('you', 'PRP'), ('want', 'VBP'), ('to', 'TO'), ('print', 'VB'), ('``', '``'), ('Hello', 'NNP'), (',', ','), ('world', 'NN'), ('!', '.'), ('``', '``'), (',', ','), ('just', 'RB'), ('type', 'JJ'), ('print', 'NN'), ('(', '('), ('``', '``'), ('Hello', 'NNP'), (',', ','), ('world', 'NN'), ('!', '.'), ('``', '``'), (')', ')'), ('—but', 'NN'), ('be', 'VB'), ('sure', 'JJ'), ('to', 'TO'), ('check', 'VB'), ('the', 'DT'), ('quotation', 'NN'), ('marks', 'NNS'), ('!', '.'), ('Here', 'RB'), ("'s", 'VBZ'), ('a', 'DT'), ('math', 'JJ'), ('example', 'NN'), (':', ':'), ('a^2', 'NN'), ('+', 'NNP'), ('b^2', 'NN'), ('=', 'NNP'), ('c^2', 'NN'), (',', ','), ('or', 'CC'), ('even', 'RB'), ('∑', 'NNP'), ('(', '('), ('n=1', 'JJ'), ('to', 'TO'), ('∞', 'VB'), (')', ')'), ('1/n^2', 'CD'), ('=', 'JJ'), ('π^2/6', 'NN'), ('.', '.'), ('Fascinating', 'NNP'), (',', ','), ('isn', 'NN'), ('’', 'NNP'), ('t', 'NN'), ('it', 'PRP'), ('?', '.'), ('Also', 'RB'), (',', ','), ('an', 'DT'), ('email', 'NN'), ('address', 'NN'), ('can', 'MD'), ('look', 'VB'), ('like', 'IN'), ('user', 'JJ'), ('@', 'NNP'), ('example.com', 'NN'), (',', ','), ('or', 'CC'), ('even', 'RB'), ('something', 'NN'), ('like', 'IN'), ('user_name-123', 'JJ'), ('@', 'NNP'), ('domain.co.kr—both', 'NN'), ('formats', 'NNS'), ('are', 'VBP'), ('valid', 'JJ'), ('.', '.')]
```

이런식으로 품사를 구분할 수 있다.

한국어는 조사, 어미 등이 붙는 교착어기 때문에 토큰화가 매우 어렵다.
따라서, 한국어는 영어와 다르게 형태소 단위로 토큰화를 해야 한다.

한국어의 경우 KoNLPy를 사용하여 형태소 태깅이 가능하다.

```python
from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

print('Okt:', okt.pos(test_string_kor))
print('Kkma:', kkma.pos(test_string_kor))
```

```plain
Okt: [('파이썬', 'Noun'), ('에서는', 'Josa'), ('리스트', 'Noun'), ('(', 'Punctuation'), ('list', 'Alpha'), ('),', 'Punctuation'), ('튜플', 'Noun'), ('(', 'Punctuation'), ('tuple', 'Alpha'), ('),', 'Punctuation'), ('딕셔', 'Noun'), ('너리', 'Noun'), ('(', 'Punctuation'), ('dictionary', 'Alpha'), (')', 'Punctuation'), ('를', 'Noun'), ('사용', 'Noun'), ('할', 'Verb'), ('수', 'Noun'), ('있어', 'Adjective'), ('!', 'Punctuation'), ('하지만', 'Conjunction'), ('set', 'Alpha'), ('도', 'Noun'), ('잊으면', 'Verb'), ('안', 'Noun'), ('돼', 'Verb'), ('.', 'Punctuation'), ('"', 'Punctuation'), ('Hello', 'Alpha'), (',', 'Punctuation'), ('world', 'Alpha'), ('!"', 'Punctuation'), ('를', 'Noun'), ('출력', 'Noun'), ('하려면', 'Verb'), ('print', 'Alpha'), ('("', 'Punctuation'), ('Hello', 'Alpha'), (',', 'Punctuation'), ('world', 'Alpha'), ('!")', 'Punctuation'), ('를', 'Noun'), ('입력', 'Noun'), ('하면', 'Verb'), ('돼', 'Verb'), ('—', 'Foreign'), ('단', 'Noun'), (',', 'Punctuation'), ('따옴표', 'Noun'), ('를', 'Josa'), ('제대로', 'Noun'), ('확인', 'Noun'), ('해야', 'Verb'), ('해', 'Noun'), ('!', 'Punctuation'), ('여기', 'Noun'), ('수학', 'Noun'), ('공식', 'Noun'), ('예제', 'Noun'), ('가', 'Josa'), ('있어', 'Adjective'), (':', 'Punctuation'), ('a', 'Alpha'), ('^', 'Punctuation'), ('2', 'Number'), ('+', 'Punctuation'), ('b', 'Alpha'), ('^', 'Punctuation'), ('2', 'Number'), ('=', 'Punctuation'), ('c', 'Alpha'), ('^', 'Punctuation'), ('2', 'Number'), (',', 'Punctuation'), ('또는', 'Adverb'), ('∑', 'Foreign'), ('(', 'Punctuation'), ('n', 'Alpha'), ('=', 'Punctuation'), ('1', 'Number'), ('부터', 'Noun'), ('∞', 'Foreign'), ('까지', 'Josa'), (')', 'Punctuation'), ('1', 'Number'), ('/', 'Punctuation'), ('n', 'Alpha'), ('^', 'Punctuation'), ('2', 'Number'), ('=', 'Punctuation'), ('π', 'Foreign'), ('^', 'Punctuation'), ('2/6', 'Number'), ('.', 'Punctuation'), ('흥미롭지', 'Adjective'), ('않아', 'Verb'), ('?', 'Punctuation'), ('또한', 'Noun'), ('이메일', 'Noun'), ('주소', 'Noun'), ('는', 'Josa'), ('user@example.com', 'Email'), ('처럼', 'Noun'), ('쓸', 'Verb'), ('수도', 'Noun'), ('있고', 'Adjective'), (',', 'Punctuation'), ('user_name-123@domain.co.kr', 'Email'), ('같은', 'Adjective'), ('형식', 'Noun'), ('도', 'Josa'), ('가능해', 'Adjective'), ('—', 'Foreign'), ('둘', 'Noun'), ('다', 'Adverb'), ('유효한', 'Adjective'), ('형식', 'Noun'), ('이야', 'Josa'), ('.', 'Punctuation')]
Kkma: [('파이', 'NNG'), ('썰', 'VV'), ('ㄴ', 'ETD'), ('에', 'VV'), ('서', 'ECD'), ('는', 'JX'), ('리스트', 'NNG'), ('(', 'SS'), ('list', 'OL'), (')', 'SS'), (',', 'SP'), ('튜플', 'UN'), ('(', 'SS'), ('tuple', 'OL'), (')', 'SS'), (',', 'SP'), ('딕', 'UN'), ('시', 'VV'), ('어', 'ECS'), ('너리', 'NNG'), ('(', 'SS'), ('dictionary', 'OL'), (')', 'SS'), ('를', 'JKO'), ('사용', 'NNG'), ('하', 'XSV'), ('ㄹ', 'ETD'), ('수', 'NNB'), ('있', 'VV'), ('어', 'ECD'), ('!', 'SF'), ('하지만', 'MAC'), ('set', 'OL'), ('도', 'JX'), ('잊', 'VV'), ('으면', 'ECD'), ('안', 'MAG'), ('되', 'VV'), ('어', 'ECS'), ('.', 'SF'), ('"', 'SS'), ('Hello', 'OL'), (',', 'SP'), ('world', 'OL'), ('!', 'SF'), ('"', 'SS'), ('를', 'JKO'), ('출력', 'NNG'), ('하', 'XSV'), ('려면', 'ECE'), ('print', 'OL'), ('(', 'SS'), ('"', 'SS'), ('Hello', 'OL'), (',', 'SP'), ('world', 'OL'), ('!', 'SF'), ('"', 'SS'), (')', 'SS'), ('를', 'JKO'), ('입력', 'NNG'), ('하', 'XSV'), ('면', 'ECE'), ('되', 'VV'), ('어', 'ECS'), ('—', 'SW'), ('단', 'NNG'), (',', 'SP'), ('따옴표', 'NNG'), ('를', 'JKO'), ('제대로', 'MAG'), ('확인', 'NNG'), ('하', 'XSV'), ('어야', 'ECD'), ('하', 'VV'), ('어', 'ECS'), ('!', 'SF'), ('여기', 'NP'), ('수학', 'NNG'), ('공식', 'NNG'), ('예제', 'NNG'), ('가', 'JKS'), ('있', 'VV'), ('어', 'ECD'), (':', 'SP'), ('a', 'OL'), ('^', 'SW'), ('2', 'NR'), ('+', 'SW'), ('b', 'OL'), ('^', 'SW'), ('2', 'NR'), ('=', 'SW'), ('c', 'OL'), ('^', 'SW'), ('2', 'NR'), (',', 'SP'), ('또는', 'MAG'), ('∑', 'SW'), ('(', 'SS'), ('n', 'OL'), ('=', 'SW'), ('1', 'NR'), ('부터', 'JX'), ('∞', 'SW'), ('까지', 'JX'), (')', 'SS'), ('1', 'NR'), ('/', 'SP'), ('n', 'OL'), ('^', 'SW'), ('2', 'NR'), ('=', 'SW'), ('π', 'SW'), ('^', 'SW'), ('2', 'NR'), ('/', 'SP'), ('6', 'NR'), ('.', 'SF'), ('흥미롭', 'VA'), ('지', 'ECD'), ('않', 'VXV'), ('아', 'ECD'), ('?', 'SF'), ('또', 'MAG'), ('한', 'MDN'), ('이메일', 'NNG'), ('주소', 'NNG'), ('는', 'JX'), ('user', 'OL'), ('@', 'SW'), ('example', 'OL'), ('.', 'SF'), ('com', 'OL'), ('처럼', 'JKM'), ('쓰', 'VV'), ('ㄹ', 'ETD'), ('수', 'NNB'), ('도', 'JX'), ('있', 'VV'), ('고', 'ECE'), (',', 'SP'), ('user', 'OL'), ('_', 'SW'), ('name-123', 'OL'), ('@', 'SW'), ('domain', 'OL'), ('.', 'SF'), ('co', 'OL'), ('.', 'SF'), ('kr', 'OL'), ('같', 'VA'), ('은', 'ETD'), ('형식', 'NNG'), ('도', 'JX'), ('가능', 'NNG'), ('하', 'XSV'), ('어', 'ECS'), ('—', 'SW'), ('둘', 'NNG'), ('다', 'MAG'), ('유효', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('형식', 'NNG'), ('이', 'VCP'), ('야', 'EFN'), ('.', 'SF')]
```

역시 Okt와 Kkma 형태소 분석 결과가 다르다. 단순히 봤을 때는 Okt가 의미적으로 더 잘 구분하는 것 같지만, 작업에 따라 적절한 분석기를 선택할 필요가 있다.
