# JTruthfulQA
JTruthfulQA is Japanese version of [TruthfulQA](https://arxiv.org/abs/2109.07958) (Lin+, 22). This dataset is not translated from original TruthfulQA but bulit from scratch.

The full set of bechmark questions and reference answers is available at `JTruthfulQA.csv`. The benchmark questions are devided into three gruops (Fact, Knowledge, Uncategorized Knowledge).

## Task
Answer to given questions. To make it easier to evaluate the answers which was generated by LLM, Instruct LLMs to generate an answer to each question within 50 characters. 

### Baselines:
This table shows the performance of recent large language models on each group of the questions. We asked people to answer the questions in two cases that allows or not allows them to search about the questions. We set the temperature to 0 for "GPT-3.5-turbo" and "GPT-4" or 0.1 for other llms to generate the answers.
||Fact|Knowledge|Uncategorized Knowledge|All|
|----|----|----|----|----|
|Human(with search)|0.741|0.762|0.647|0.750|
|Human(without search)|0.753|0.579|0.588|0.654|
||||||
|GPT-3.5-turbo|0.78|0.177|0.235|0.437|
|GPT-4|**0.869**|**0.409**|**0.529**|**0.609**|
|stabilityai/japanese-stablelm-instruct-alpha-7b|0.212|0.271|0.235|0.245|
|elyza/ELYZA-japanese-Llama-2-7b-instruct|0.564|0.146|0.176|0.326|
|matsuo-lab/weblab-10b-instruction-sft|0.174|0.201|0.353|0.194|
|line-corporation/japanese-large-lm-3.6b-instruction-sft|0.378|0.165|0.294|0.260|

## Dataset
Each question has the best answer created by human. The dataset includes correct answers and wrong answers created by four LLMs. We add the best answers to correct answers.
||% Correct Answers|% Wrong Answers|% Total|
|----|----|----|----|
|Human|12.1|-|12.1|
|GPT-3.5-turbo|18.5|19.4|37.9|
|stabilityai/Japanese-StableLM-Instruct-Alpha-7B|7.1|18.3|25.4|
|elyza/ELYZA-japanese-Llama-2-7b-instruct|2.0|11.4|13.4|
|matsuo-lab/weblab-10b-instruction-sft|3.6|7.6|11.2|
