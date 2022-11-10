import pandas as pd
import numpy as np
from rapidfuzz import process

schema_df = pd.DataFrame(pd.read_csv('./data/fundTable1.csv'))
schname = schema_df.columns.values
header_dict = {}
for name in schname:
    namelist = name.strip().split(":")
    header_dict[namelist[0]] = namelist[1]

headers = list(header_dict.keys())
columns = {}
colTypes = {'FundTable':header_dict}

for ind, col in enumerate(schname):
    namelist = col.strip().split(":")
    col_data = schema_df[col]
    columns[namelist[0]] = list(col_data)

column_meta = []
#header = headers['FundTable']
for col in headers:
    detected_val = None
    column_meta.append((col, colTypes['FundTable'][col], detected_val))
    

schname = schema_df.columns.values

with open('./data/banlist_v2.txt', 'r', encoding='utf-8') as f:
    bancontent = f.readlines()
    banlist = [c.strip() for c in bancontent]

yiliao = set()
jijinmingcheng = list(set(list(schema_df['基金名称:list'])))
zhongcang = list(set(list(schema_df['重仓:list'])))
for jijin in jijinmingcheng:
    jjlist = jijin[1:-1].split(',')
    for jj in jjlist:
        if "医疗" in jj:
            if "医疗" == jj.strip():
                    continue
            yiliao.add(jj.strip())

for jijin in zhongcang:
    if jijin is not np.nan:
        jjlist = jijin[1:-1].split(',')
        for jj in jjlist:
            if "医疗" in jj:
                if "医疗" == jj.strip():
                    print(123)
                yiliao.add(jj.strip())    



def preprocessques(ques, isTrain=True):
    global yiliao
    
    tmp_text = ques
    reverse_dict = {
        '劵':'券',
        '稳建':'稳健',
        '保本保息':'保本型',
        '这半年里':'六个月',
        '近半年':'六个月',
        '中等风险':'中风险',
        '风险中':'中风险',
        '风险低':'低风险',
        '稳健类':'低风险',
        '稳赚不赔':'低风险',
        '稳健理财':'低风险',
        '风险比较高':'高风险',
        '风险高':'高风险',
        '食品类':'食品饮料',
        '饮料类':'食品饮料',
        '医疗类': '医药',
        '医疗生物':'医药生物',
        '小米':'小米概念',
        '风险偏高':'中高风险',
        '券商':'证券',
        '时间短期':'短期理财',
        '指数基金':'指数型',
        '债基':'债券型',
        '债券基金':'债券型',
        '？':""
    }

    reverse_dict_train_ver = {
        '这半年里':'六个月',
        '近半年':'六个月',
        '中等风险':'中风险',
        '风险中':'中风险',
        '风险低':'低风险',
        '稳健类':'低风险',
        '稳赚不赔':'低风险',
        '稳健理财':'低风险',
        '风险比较高':'高风险',
        '风险高':'高风险',
        '食品类':'食品饮料',
        '饮料类':'食品饮料',
        '医疗类': '医药',
        '医疗生物':'医药生物',
        '小米':'小米概念',
        '风险偏高':'中高风险',
        '券商':'证券',
        '时间短期':'短期理财',
        '指数基金':'指数型',
        '债基':'债券型',
        '债券基金':'债券型',
        '？':""
    }
    if isTrain:
        reverse_dict = reverse_dict_train_ver
        
    for key in reverse_dict.keys():
        if key in tmp_text:
            tmp_text = tmp_text.replace(key, reverse_dict[key])

    if not isTrain:
        for key in banlist:
            if key in tmp_text:
                tmp_text = tmp_text.replace(key, "")

    tmp_text = tmp_text.replace('\\n', '')           

    if '食品饮料' not in tmp_text and '食品加工' not in tmp_text and ('食品' in tmp_text or '饮料' in tmp_text):
        if '食品' in tmp_text:
            tmp_text = tmp_text.replace('食品', '食品饮料')
        else:
            tmp_text = tmp_text.replace('饮料', '食品饮料')

    if '医疗的' in tmp_text:
        flag = True
        for t in yiliao:
            if t in tmp_text:
                flag = False
                break
        if flag:
            tmp_text = tmp_text.replace('医疗的', '医药')

    # only for b
    if not isTrain:
        if '保本' in tmp_text and '保本型' not in tmp_text:
            tmp_text = tmp_text.replace('保本', '保本型')

    tmp_text = tmp_text.strip('，')
    return tmp_text



compdict = {}
for i in [1, 3, 5, 6, 7, 8, 9, 10, 21, 22, 28, 29, 30, 44, 45, 46, 49, 51]:
    fullset = set()
    name = schname[i]
    content = list(schema_df[name])
    for jijin in content:
        if jijin is not np.nan:
            if isinstance(jijin, int):
                fullset.add(str(jijin).lower())
                continue

            if '[' not in jijin:
                if jijin.strip().lower() == 'none':
                    continue
                fullset.add(jijin.strip().lower().replace(" ",""))
            else:
                jjlist = jijin[1:-1].split(',')
                for jj in jjlist:
                    if jijin.strip().lower() == 'none':
                        continue
                    fullset.add(jj.strip().lower().replace(" ",""))

    compdict[i] = list(fullset)

def find_fuzzy(text, index, threshold=80):
    if index not in compdict:
        return text
    
    if text == 'qdll':
        return 'qdii'
       

    if index == 3:
        threshold = 60
    
    if index == 51:
        threshold = 74
        
    retlist = process.extract(text, compdict[index], limit=10)
    new_text = text

    backup_dict = {}
    maxscore = 0        
    for ret in retlist:
        if ret[1] == 100:
            new_text = ret[0]
            break
        
        if ret[1] < threshold:
            break

        if maxscore == 0:
            maxscore = ret[1]
            last_text = ret[0]
            backup_dict[ret[0]] = len(ret[0])
            continue
        
        if ret[1] < maxscore:
            break

        backup_dict[ret[0]] = len(ret[0])
    
    if len(backup_dict) != 0 and len(backup_dict) <= 3:
        backup_dict_sorted = sorted(backup_dict.items(), key=lambda x: x[1], reverse=True)
        nd = [r[0] for r in backup_dict_sorted]
        if index != 8:
            nd = sorted(nd)
            new_text = nd[0]
        else:
            new_text = nd[0]

    return new_text


def get_headers(index):
    return headers[index[0]]

def get_meta():
    return column_meta