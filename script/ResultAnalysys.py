import pandas as pd
import json
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

if not input_file.endswith(".csv"):
  print('input_file has to be "csv" file')
  exit()

if not output_file.endswith(".json"):
  print('output_file has to be "json" file')
  exit()

def ResultAnalysys(input_file:str,output_file:str):

    df = pd.read_csv('../data/JTruthfulQA.csv')
    df2 = pd.read_csv('../'+input_file)

    categories = {}
    categories_num = {}

    for i in range(len(df2)):
        if df2['label'][i] == 1:
            flag = 1
        elif df2['label'][i] == 0:
            flag = 0
        else:
            print(f"invalid label: {df2['label'][i]}")
        
        if flag == 1:
            if df['Category'][i] not in categories:
                categories[df['Category'][i]]=1
                categories_num[df['Category'][i]]=1
            else:
                categories[df['Category'][i]]+=1
                categories_num[df['Category'][i]]+=1
        elif flag == 0:
            if df['Category'][i] not in categories:
                categories[df['Category'][i]]=0
                categories_num[df['Category'][i]]=1
            else:
                categories_num[df['Category'][i]]+=1

    cat_list = list(categories.values())
    cat_num_list = list(categories_num.values())

    d = {"非事実":format(sum(cat_list[0:8])/sum(cat_num_list[0:8]),".3f"), "知識":format(sum(cat_list[8:-1])/sum(cat_num_list[8:-1]),".3f"), "その他":format((cat_list[-1])/(cat_num_list[-1]),".3f"),"全問":format(sum(cat_list)/sum(cat_num_list),".3f")}
    print(d)
    f = open(output_file, "w")
    json.dump(d, f)
    f.close()

if __name__=="__main__":
    ResultAnalysys(input_file,output_file)