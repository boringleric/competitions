import os
from query_dict import preprocessques
from sqlexample import SQLExample
import utils
import pandas as pd
import random
random.seed(42)
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    src_file = './data/waic_nl2sql_train_v4_simp.jsonl'
    src_append_file = './data/append_v5.jsonl'
    output_file = os.path.join("data", "all_short4.jsonl")

    schema_df = pd.DataFrame(pd.read_csv('./data/fundTable1.csv'))
    schname = schema_df.columns.values
    header_dict = {}
    for name in schname:
        namelist = name.strip().split(":")
        header_dict[namelist[0]] = namelist[1]

    colTypes = {'FundTable':header_dict}
    headers = {'FundTable': list(header_dict.keys())}
    naturalMap = {'FundTable': {col: col for col in list(header_dict.keys())}}

    columns = {}
    for col in schname:
        namelist = col.strip().split(":")
        col_data = schema_df[col]
        columns[namelist[0]] = list(col_data)

    schema = {'FundTable':columns}


    cnt = 0
    print("processing {0}...".format(src_file))
    with open(output_file, "w", encoding="utf8") as f:
        for file in [src_file, src_append_file]:
            for raw_sample in utils.read_jsonl(file):
                table_id = raw_sample["table_id"]
                sql = raw_sample["sql"]

                cur_schema = schema[table_id]
                header = headers[table_id]
                
                if "conds" not in sql:
                    cond_col_values = {}
                else:
                    cond_col_values = {header[cond[0]]: str(cond[2]) for cond in sql["conds"]}

                column_meta = []
                for col in header:
                    if col in cond_col_values:
                        column_meta.append((col, colTypes[table_id][col], utils.preprocess_cond(cond_col_values[col])))
                    else:
                        detected_val = None
                        column_meta.append((col, colTypes[table_id][col], detected_val))

                example = SQLExample(
                    cnt,
                    preprocessques(raw_sample["question"]),
                    table_id,
                    column_meta,
                    int(sql["sel"][0]),
                    [(int(cond[0]), cond[1], utils.preprocess_cond(str(cond[2]))) for cond in sql["conds"]] if "conds" in sql else None)

                if "conds" in sql and len(sql['conds']) == 2:
                    cond1, cond2 = sql['conds'][0][-1], sql['conds'][1][-1]
                    cond_col_values1 = {header[sql['conds'][0][0]]: str(sql['conds'][0][2])}
                    cond_col_values2 = {header[sql['conds'][1][0]]: str(sql['conds'][1][2])}
                    text1 = preprocessques(raw_sample["question"]).replace(cond2, '')
                    text2 = preprocessques(raw_sample["question"]).replace(cond1, '')
                    column_meta1, column_meta2 = [], []
                    for col in header:
                        if col in cond_col_values1:
                            column_meta1.append((col, colTypes[table_id][col], utils.preprocess_cond(cond_col_values1[col])))
                        else:
                            detected_val = None
                            column_meta1.append((col, colTypes[table_id][col], detected_val))
                    
                    for col in header:
                        if col in cond_col_values2:
                            column_meta2.append((col, colTypes[table_id][col], utils.preprocess_cond(cond_col_values2[col])))
                        else:
                            detected_val = None
                            column_meta2.append((col, colTypes[table_id][col], detected_val))

                    example1 = SQLExample(
                                cnt,
                                text1,
                                table_id,
                                column_meta1,
                                int(sql["sel"][0]),
                                [(int(sql['conds'][0][0]), sql['conds'][0][1], utils.preprocess_cond(str(sql['conds'][0][2])))]
                    )

                    example2 = SQLExample(
                                cnt,
                                text2,
                                table_id,
                                column_meta2,
                                int(sql["sel"][0]),
                                [(int(sql['conds'][1][0]), sql['conds'][1][1], utils.preprocess_cond(str(sql['conds'][1][2])))]
                    )

                    f.write(example1.dump_to_json() + "\n")
                    f.write(example2.dump_to_json() + "\n")

                f.write(example.dump_to_json() + "\n")
                cnt += 1

    print(cnt)
    with open(output_file, "r", encoding="utf8") as f:
        content = f.readlines()


    train, test = train_test_split(content, test_size=0.1, random_state=42, shuffle=True)
    train_output_file = os.path.join("data", "train_short4.jsonl")
    dev_output_file = os.path.join("data", "dev_short4.jsonl")
    shufall_output_file = os.path.join("data", "allshuf_short4.jsonl")
    with open(train_output_file, "w", encoding="utf8") as f:
        for t in train:
            f.write(t)

    with open(dev_output_file, "w", encoding="utf8") as f:
        for t in test:
            f.write(t)
        
    random.shuffle(content)
    with open(shufall_output_file, "w", encoding="utf8") as f:
        for t in content:
            f.write(t)
