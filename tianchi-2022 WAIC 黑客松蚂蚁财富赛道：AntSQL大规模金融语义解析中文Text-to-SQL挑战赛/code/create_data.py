import json
import pandas as pd
import numpy as np
import random
from nlpcda import Homophone
hmp = Homophone(create_num=2, change_rate=0.5)  # 同音字替换

random.seed(42)

with open('./data/banlist_v2.txt', 'r', encoding='utf-8') as f:
    bancontent = f.readlines()
    banlist = [c.strip() for c in bancontent]

with open('./data/waic_nl2sql_train.jsonl', encoding='utf-8') as f:
    content = f.readlines()

textset = set()
for c in content:
    clist = json.loads(c)
    if 'conds' in clist['sql']:       
        cond = clist['sql']['conds']
        for cd in cond:
            textset.add(cd[2])


schema_df = pd.DataFrame(pd.read_csv('./data/fundTable1.csv'))
jjdm = list(schema_df['基金代码:id']) # 0
jjmc = list(schema_df['基金名称:list']) # 1 a
gz = list(schema_df['估值:cate']) # 46
gzzs = list(schema_df['跟踪指数:list']) # 49
bk = list(schema_df['板块:list']) # 29  # 种植业 物流 地面兵装
jjlx = list(schema_df['基金类型:cate']) # 3 all
hy = list(schema_df['行业:list']) # 30 all
ztgn = list(schema_df['主题/概念:list']) # 22  美伊冲突 医用器械 华为产业链 充电桩 上海本地 功率半导体 纯碱 数据中心 工业大麻
jjgsmc = list(schema_df['基金公司名称:list']) # 8 
jjjl = list(schema_df['基金经理:list']) # 21    all
zc = list(schema_df['重仓:list']) # 51  all
jjtzfg = list(schema_df['基金投资风格描述:cate']) # 45
fhfs = list(schema_df['分红方式:cate']) # 9
fxdj = list(schema_df['风险等级:cate']) # 7
kxs = list(schema_df['是否可销售:cate']) # 6
xss = list(schema_df['销售状态:cate']) # 5
tzsc = list(schema_df['投资市场:cate']) # 28
shzt = list(schema_df['赎回状态:cate']) # 10
ztzc = list(schema_df['主投资产类型:cate']) # 44


def replacedata(tmpstr): 
    global banlist
    for ban in banlist:
        tmpstr = tmpstr.replace(ban, '')

    tmpstr = tmpstr.strip('，')
    return tmpstr

for t in textset:
    #text = '推荐几个红利再投资的行业'
    ret = replacedata(t)
    if ret != t:
        print(123)

zcfullset = set()
for jijin in zc:
    if jijin is not np.nan:
        if isinstance(jijin, int):
            zcfullset.add(str(jj).lower())
            continue

        if '[' not in jijin:
            if jijin.strip().lower() == 'none':
                continue
            zcfullset.add(jijin.strip().lower().replace(" ",""))
        else:
            jjlist = jijin[1:-1].split(',')
            for jj in jjlist:
                if jj.strip().lower() == 'none':
                    continue
                zcfullset.add(jj.strip().lower().replace(" ",""))

def preprocess_cond(text):
    tmp = text
    if '基金管理有限公司' in tmp:
        tmp = tmp.replace('基金管理有限公司', '')

    if '资产管理有限公司' in tmp:
        tmp = tmp.replace('资产管理有限公司', '')

    if '基金管理股份有限公司' in tmp:
        tmp = tmp.replace('基金管理股份有限公司', '')

    if '基金管理有限责任公司' in tmp:    
        tmp = tmp.replace('基金管理有限责任公司', '')

    if '股份有限公司' in tmp:    
        tmp = tmp.replace('股份有限公司', '')

    if '有限公司' in tmp:    
        tmp = tmp.replace('有限公司', '')

    return tmp

# 今天国际 zc
print(123)  
ind = 0
schname = schema_df.columns.values
all_data = list()

cond_sel_dict ={1:4, 46: 2, 49:4, 29:4, 3:2, 30:4, 22:4, 8:4, 21:4, 51:4, 45:2, 9:2, 7:2, 6:2, 0:2, 5:2, 28:2, 10:2, 44:2}

for i in [0, 1, 3, 5, 6, 7, 8, 9, 10, 21, 22, 28, 29, 30, 44, 45, 46, 49, 51]:
    fullset = set()
    name = schname[i]
    content = list(schema_df[name])
    for jijin in content:
        if jijin is not np.nan:
            if isinstance(jijin, int):
                fullset.add(str(jj).lower())
                continue

            if '[' not in jijin:
                fullset.add(jijin.strip().lower().replace(" ",""))
            else:
                jjlist = jijin[1:-1].split(',')
                for jj in jjlist:
                    fullset.add(jj.strip().lower().replace(" ",""))

    list_create = []
    fullset = sorted(list(fullset))
    random.shuffle(fullset)
    for jj in fullset:
        if jj not in textset:
            list_create.append(jj)

    list_demo_all = []
    
    if len(list_create) < 100:
        for _ in range(100):
            list_create.extend(list(fullset))

    namelist = name.strip().split(':')
    for l in list_create:
        l = preprocess_cond(l)
        if random.random() > 0.9:
            question = l + namelist[0]
        elif random.random() > 0.5:
            question = namelist[0] + l
        else:
            question = l

        if i == 21: # 基金经理类型修正
            if question == l:
                if random.random() > 0.8:
                    question = question + '管理'
                elif random.random() > 0.6:
                    question = question + '经理'

        if i == 3:  # 尝试形近字扩充
            if question == l:
                if random.random() > 0.7:
                    text_new = hmp.replace(question)
                    if len(text_new) > 1:
                        text_new = text_new[1]
                    else:
                        text_new = text_new[0]
                    if text_new != question:
                        question = l = text_new
            if random.random() > 0.8:
                question = question + random.choice(['基金', '理财'])
            elif random.random() > 0.5:
                question = random.choice(['基金', '理财']) + question

        new_dict ={
                    "id": ind,  
                    "question": question,  
                    "table_id": 'FundTable',  
                    "sql":{ 
                    "sel": [1],  
                    "agg": [0],  
                    "limit": 0,  
                    "orderby": [],  
                    "asc_desc": 0,  
                    "cond_conn_op": 0,  
                    "conds": [[i, cond_sel_dict[i], l]]
                    },  
                    "keywords":{  
                    "sel_cols": ["基金名称"],  
                    "values": [l]
                    }  
                }
                
        list_demo_all.append(new_dict)
    
    random.shuffle(list_demo_all)

    if i == 0:  # 基金代码不需要那么多
        list_demo_all = list_demo_all[:100]

    for demo in list_demo_all[:2000]:
        jsonc = json.dumps(demo, ensure_ascii=False)
        all_data.append(jsonc)

alldata = sorted(list(set(all_data)))
random.shuffle(alldata)
with open('./data/append_v5.jsonl','w',encoding='utf-8') as fw:    
    for jsonc in alldata:        
        fw.write(jsonc + '\n')