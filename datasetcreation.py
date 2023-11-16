import os
import pandas as pd

folder_path_judgement = 'dataset/IN-Abs/train-data/judgement'
folder_path_summary = 'dataset/IN-Abs/train-data/summary'
main_path = 'dataset/IN-Abs/train-data/'

# Initialize an empty DataFrame
df = pd.DataFrame(columns=['judgement', 'summary'])
# Iterate through all files in the 'judgement' folder
dict ={'judgement' : [] , 'summary' :[]}
for filename in os.listdir(folder_path_judgement):
    judgement_file_path = os.path.join(folder_path_judgement, filename)
    summary_file_path = os.path.join(folder_path_summary, filename)

    with open(judgement_file_path, 'r', encoding='utf-8') as judgement_file:
                judgement_content = judgement_file.read()

    with open(summary_file_path, 'r', encoding='utf-8') as summary_file:
                summary_content = summary_file.read()


    dict['judgement'].append(judgement_content)
    dict['summary'].append(summary_content)


df =  pd.DataFrame(dict)


csv_file_path = 'Train_data.csv'


# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)