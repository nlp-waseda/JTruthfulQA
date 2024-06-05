# JTruthfulQA
JTruthfulQA is a Japanese version of [TruthfulQA](https://arxiv.org/abs/2109.07958) (Lin+, 2022). This dataset is not translated from original TruthfulQA but built from scratch.

The full set of benchmark questions and reference answers is available at `data/JTruthfulQA.csv`. The benchmark questions are divided into three types: Fact, Knowledge, and Uncategorized.

## Task
The task is to answer the given questions. To make it easier to evaluate the answers that were generated by a large language model (LLM), the instruction to LLMs is to generate an answer to each question within 50 characters. 

### Baselines
This table shows the performance of human performance and recent LLMs on each type of the questions. For human performance, we asked people to answer the questions in the two cases that allow or do not allow them to search the web about the questions. We set the temperature to 0 for "GPT-3.5-turbo" and "GPT-4" or 0.1 for the other LLMs to generate the answers.
||Fact|Knowledge|Uncategorized|All|ft-GPT-3.5-Turbo|ft-waseda RoBERTa|BLEU|ROUGE1|BERTScore|MC1|MC2|
|:----|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
|Human (with search)|0.741|0.762|0.647|0.750|0.629|0.753|6.99|0.28|0.14|-|-|
|Human (without search)|0.753|0.579|0.588|0.654|0.586|0.702|5.30|0.25|0.11|-|-|
|||||||||||||
|GPT-3.5-turbo|0.78|0.177|0.235|0.437|0.512|0.543|**6.01**|**0.04**|-0.02|-|-|
|GPT-4|**0.869**|**0.409**|**0.529**|**0.609**|**0.601**|**0.611**|-0.673|0.03|**-0.01**|-|-|
|stabilityai/japanese-stablelm-instruct-alpha-7b|0.212|0.271|0.235|0.245|0.207|0.232|-7.26|-0.05|-0.09|0.129|0.130|
|elyza/ELYZA-japanese-Llama-2-7b-instruct|0.564|0.146|0.176|0.326|0.290|0.421|-8.65|-0.06|-0.10|0.126|0.129|
|matsuo-lab/weblab-10b-instruction-sft|0.174|0.201|0.353|0.194|0.172|0.151|-4.50|-0.05|-0.08|**0.156**|0.146|
|line-corporation/japanese-large-lm-3.6b-instruction-sft|0.378|0.165|0.294|0.260|0.192|0.320|-1.52|-0.01|-0.04|0.152|**0.152**|

## How to Answer to JTruthfulQA
We provide a sample code "SampleGeneration.py", which uses "elyza/ELYZA-japanese-Llama-2-7b-instruct" to generate the answers. Rewrite the code for a model that you want to use. 

## Automatic Evaluation
```Python3 script/RobertaEvaluation.py "input_file_name" "output_file_name_1"```<br>
Run "RobertaEvaluation.py" to evaluate the generated answers. You can get the result with "label". (1: correct, 0: incorrect)<br> 
"input_file_name" and "output_file_name_1" have to end with ".csv".<br>
<br>
```Python3 script/ResultAnalysys.py "output_file_name_1" "output_file_name_2"```<br>
You can see the analysys of the answers with "ResultAnalysys.py"<br>
"output_file_name_2" has to end with ".json".<br>

## Datasets
Each question has the original answer created by human. The dataset includes correct answers and wrong answers generated by four LLMs ("GPT-3.5-turbo", "stabilityai/japanese-stablelm-instruct-alpha-7b", "elyza/ELYZA-japanese-Llama-2-7b-instruct", "matsuo-lab/weblab-10b-instruction-sft"). The original answers are also added to the correct answers.

```JTruthfulQA.csv```<br>
This dataset has 3,078 correct answers and 3,281 incorrect answers (6,359 answers in total) over 582 questions.<br>
<br>
```JTruthfulQA_without_gpt.csv```<br>
This dataset has 2,125 correct answers and 2,267 incorrect answers (4,392 answers in total) over 551 questions. The answers of GPT-3.5-turbo are excluded from this dataset.

## Reference
```
@InProceedings{Kurihara_nlp2022,
  author = "中村友亮 and 河原大輔",
  title = "日本語TruthfulQAの構築",
  booktitle = "言語処理学会第30回年次大会",
  year = "2024",
  url = "https://anlp.jp/proceedings/annual_meeting/2024/pdf_dir/P6-15.pdf",
  pages = "1709--1714",
  note= "in Japanese"
}
```

## License
This dataset is distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgment
This dataset was created in collaboration with SB Intuitions Corp. and Waseda University.
