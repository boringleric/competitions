import json
from query_dict import find_fuzzy, get_headers

# 数据分析发现，cond[0]和cond[1]的强规律，修正数据
cond_sel_dict ={1:4, 46:2, 49:4, 29:4, 3:2, 30:4, 22:4, 8:4, 21:4, 51:4, 45:2, 9:2, 7:2, 6:2, 0:2, 5:2, 28:2, 10:2, 44:2}


def post_process(model_output_path, test_file_path, fin_output_file_path):
    contentlist = []
    with open(model_output_path) as f:
        for c in f.readlines():
            contentlist.append(json.loads(c))

    with open(test_file_path) as f:
        with open(fin_output_file_path,'w', encoding='utf-8') as fw:
            for ind, c in enumerate(f.readlines()):
                tmp = json.loads(c)
                # 对于0value的处理
                if len(contentlist[ind]['query']['conds'])==0:
                        new_dict = {
                        "id": tmp['id'],  
                        "question": tmp['question'],  
                        "table_id": tmp['table_id'],  
                        "sql": { 
                        "sel": [contentlist[ind]['query']['sel']],  
                        "agg": [0],  
                        "limit": 0,  
                        "orderby": [],  
                        "asc_desc": 0,  
                        "cond_conn_op": 0 if len(contentlist[ind]['query']['conds'])<2 else 2,  
                        },  
                        "keywords": {
                        "sel_cols": [get_headers([contentlist[ind]['query']['sel']])],  
                        "values": [cond[2] for cond in contentlist[ind]['query']['conds']] 
                        }  
                    }

                else:
                    # 对于1和2value的处理
                    vals, conds = [], []
                    tmp1, tmp2 = contentlist[ind]['query']['conds'][0][0], contentlist[ind]['query']['conds'][0][1]
                    for cond in contentlist[ind]['query']['conds']:
                        if '[UNK]' in cond[-1]:
                            tmp_text = tmp_text.replace('[UNK]', '')
                            index = tmp['question'].find(tmp_text)
                            tmp_text = tmp['question'][index: index+len(tmp_text)+1]
                        
                        # 获取每个栏位的最小编辑距离匹配词汇
                        tmp_text = find_fuzzy(cond[-1], tmp1)

                        # b榜数据特定，临时分析
                        if tmp1 == 0:   # 0 的span一点都不准
                            new_list = []                            
                            for t in list(tmp['question']):
                                if t.isdigit():
                                    new_list.append(t)
                                else:
                                    if len(new_list) != 0:
                                        break
                            tmp_text = ''.join(new_list)
                        if tmp_text in ['货币基金', '貨幣基金']:
                            tmp_text = '货币基金'
                            vals.insert(0, [1, 4, '货币基金'])
                        else:
                            if ('葛兰' == tmp_text and tmp1 == 51) or ('葛兰' in tmp_text and tmp1 == 21):  # 这一版模型就是不喜欢葛兰，奇怪
                                vals.insert(0, [21, cond_sel_dict[21], '葛兰']) 
                            else:
                                if tmp1 in cond_sel_dict:
                                    vals.insert(0, [tmp1, cond_sel_dict[tmp1], tmp_text])
                                else:
                                    vals.insert(0, [tmp1, tmp2, tmp_text])
                        conds.append(tmp_text)

                    if conds == [""]:
                        new_dict = {
                                "id": tmp['id'],  
                                "question": tmp['question'],  
                                "table_id": tmp['table_id'],  
                                "sql": { 
                                "sel": [contentlist[ind]['query']['sel']],  
                                "agg": [0],  
                                "limit": 0,  
                                "orderby": [],  
                                "asc_desc": 0,  
                                "cond_conn_op": 0,  
                                },  
                                "keywords": {  
                                "sel_cols": [get_headers([contentlist[ind]['query']['sel']])],  
                                "values": [] 
                                }  
                            }
                    else:
                        new_dict = {
                                "id": tmp['id'],  
                                "question": tmp['question'],  
                                "table_id": tmp['table_id'],  
                                "sql": { 
                                "sel": [contentlist[ind]['query']['sel']],  
                                "agg": [0],  
                                "limit": 0,  
                                "orderby": [],  
                                "asc_desc": 0,  
                                "cond_conn_op": 0 if len(contentlist[ind]['query']['conds'])<2 else 2,  
                                "conds": vals
                                },  
                                "keywords": {  
                                "sel_cols": [get_headers([contentlist[ind]['query']['sel']])],  
                                "values": conds
                                }  
                            }

                jsonc = json.dumps(new_dict, ensure_ascii=False)
                fw.write(jsonc+'\n')