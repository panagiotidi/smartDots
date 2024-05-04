import ast
import pandas as pd
from operator import add, sub, pow

metadata = "data/resized_256x256/train/data.csv"

data = pd.read_csv(metadata)

# print(data.columns)
labels = data['uni_label']
# print(labels)

res_list = [0.0] * 10
print(res_list)
for label in labels:
    # res_list = res_list +  ast.literal_eval(label)
    res_list = list(map(add, res_list, ast.literal_eval(label)))

print(res_list)

s = [l**-1 for l in res_list]
# s = list(map(pow, [total_obs] * 10, res_list))
print(s)

total = sum(s)
print(total)

weights = [l/total for l in s]
print(weights)
# print((10*[float(total_obs)]- res_list))