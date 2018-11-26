
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from functools import wraps


# In[2]:


import telepot
def send_notify(text):
    with open('../../_access_keys/telegram_token.txt', 'r') as content_file:
        token = content_file.read()
    with open('../../_access_keys/user.txt', 'r') as content_file:
        user = content_file.read()
    try:
        TelegramBot = telepot.Bot(token)
        TelegramBot.sendMessage(int(user), text)
    except:
        pass


# In[3]:


TRAIN = True


# In[4]:


DEBUG = True


# In[5]:


def execution_time_log(func):
    @wraps(func)
    def result_func(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        if DEBUG:
            print('execution time', func.__name__, datetime.now() - start)
        return result
    return result_func


# In[6]:


path = 'C:/data/mlbootcamp/dataset/'
path_train = path + 'train/'
path_test = path + 'test/'
pathfeatures = 'features/'
if TRAIN:
    pathfiles = path_train
    path_subs_bs_consumption = 'subs_bs_consumption_train.csv'
    path_subs_bs_data_session = 'subs_bs_data_session_train.csv'
    path_subs_bs_voice_session = 'subs_bs_voice_session_train.csv'
    path_subs_features = 'subs_features_train.csv'
    path_subs_csi = 'subs_csi_train.csv'
else:
    pathfiles = path_test
    path_subs_bs_consumption = 'subs_bs_consumption_test.csv'
    path_subs_bs_data_session = 'subs_bs_data_session_test.csv'
    path_subs_bs_voice_session = 'subs_bs_voice_session_test.csv'
    path_subs_features = 'subs_features_test.csv'
    path_subs_csi = 'subs_csi_test.csv'


# In[7]:


subs_csi = pd.read_csv(pathfiles + path_subs_csi, sep=';', decimal=',')


# In[8]:


subs_csi.head()


# In[9]:


def convert_date(x):
    dt = datetime.strptime(str(x), '%d.%m')
    return 100 * dt.month + dt.day


# In[10]:


user_dates = {i[0]:convert_date(i[1]) for i in zip(list(subs_csi['SK_ID']), list(subs_csi['CONTACT_DATE']))}


# In[11]:


indexes_users = {i[1]:i[0] for i in enumerate(subs_csi['SK_ID'].unique())}


# In[12]:


subs_bs_voice_session = pd.read_csv(pathfiles + path_subs_bs_voice_session, sep=';', decimal=',')


# In[13]:


subs_bs_voice_session['day'] = subs_bs_voice_session['START_TIME'].apply(lambda x:datetime.strptime(x, '%d.%m %H:%M:%S'))


# In[14]:


subs_bs_voice_session.head(1)


# In[15]:


subs_bs_data_session = pd.read_csv(pathfiles + path_subs_bs_data_session, sep=';', decimal=',')


# In[16]:


subs_bs_data_session['day'] = subs_bs_data_session['START_TIME'].apply(lambda x:datetime.strptime(x, '%d.%m %H:%M:%S'))


# In[17]:


subs_bs_data_session.head(1)


# In[18]:


user_data_session = defaultdict()
for i in subs_bs_data_session.values:
    sk_id = i[0]
    dt = i[4]
    cell_id = i[1]
    key = (sk_id, dt.month, dt.day)
    user_data_session.setdefault(key, defaultdict()).setdefault(cell_id, 0)
    user_data_session[key][cell_id] += i[2]


# In[19]:


user_voice_session = defaultdict()
for i in subs_bs_voice_session.values:
    sk_id = i[0]
    dt = i[4]
    cell_id = i[1]
    key = (sk_id, dt.month, dt.day)
    user_voice_session.setdefault(key, defaultdict()).setdefault(cell_id, 0)
    user_voice_session[key][cell_id] += i[2]


# In[20]:


columns = ['CELL_AVAILABILITY_2G', 'CELL_AVAILABILITY_3G',
       'CELL_AVAILABILITY_4G', 'CSSR_2G', 'CSSR_3G',
       'ERAB_PS_BLOCKING_RATE_LTE', 'ERAB_PS_BLOCKING_RATE_PLMN_LTE',
       'ERAB_PS_DROP_RATE_LTE', 'HSPDSCH_CODE_UTIL_3G',
       'NODEB_CNBAP_LOAD_HARDWARE', 'PART_CQI_QPSK_LTE', 'PART_MCS_QPSK_LTE',
       'PROC_LOAD_3G', 'PSSR_2G', 'PSSR_3G', 'PSSR_LTE',
       'RAB_CS_BLOCKING_RATE_3G', 'RAB_CS_DROP_RATE_3G',
       'RAB_PS_BLOCKING_RATE_3G', 'RAB_PS_DROP_RATE_3G', 'RBU_AVAIL_DL',
       'RBU_AVAIL_DL_LTE', 'RBU_AVAIL_UL', 'RBU_OTHER_DL', 'RBU_OTHER_UL',
       'RBU_OWN_DL', 'RBU_OWN_UL', 'RRC_BLOCKING_RATE_3G',
       'RRC_BLOCKING_RATE_LTE', 'RTWP_3G', 'SHO_FACTOR', 'TBF_DROP_RATE_2G',
       'TCH_DROP_RATE_2G', 'UTIL_BRD_CPU_3G', 'UTIL_CE_DL_3G',
       'UTIL_CE_HW_DL_3G', 'UTIL_CE_UL_3G', 'UTIL_SUBUNITS_3G',
       'UL_VOLUME_LTE', 'DL_VOLUME_LTE', 'TOTAL_DL_VOLUME_3G',
       'TOTAL_UL_VOLUME_3G']


# In[21]:


columns = ['UTIL_BRD_CPU_3G', 'UTIL_CE_DL_3G',
       'UTIL_CE_HW_DL_3G', 'UTIL_CE_UL_3G', 'UTIL_SUBUNITS_3G',
       'UL_VOLUME_LTE', 'DL_VOLUME_LTE', 'TOTAL_DL_VOLUME_3G',
       'TOTAL_UL_VOLUME_3G']


# In[22]:


bs_avg_kpi = pd.read_csv(path + 'bs_avg_kpi.csv', sep=';', usecols=['T_DATE', 'CELL_LAC_ID'])


# In[23]:


bs_avg_kpi['T_DATE'] = bs_avg_kpi['T_DATE'].apply(lambda x:datetime.strptime(str(x), '%d.%m'))


# In[24]:


dates_line = list(bs_avg_kpi['T_DATE'].apply(lambda x:100*x.month + x.day))
cells_line = list(bs_avg_kpi['CELL_LAC_ID'])


# In[25]:


start_date = bs_avg_kpi['T_DATE'].min()
finish_date = bs_avg_kpi['T_DATE'].max()


# In[26]:


indexes_cells = {i[1]:i[0] for i in enumerate(bs_avg_kpi['CELL_LAC_ID'].unique())}


# In[27]:


from datetime import timedelta
indexes_dates = {}
for i in enumerate(range((finish_date - start_date).days + 1)):
    cur_date = start_date + timedelta(days=i[1])
    indexes_dates[100*cur_date.month + cur_date.day] = i[0]


# In[28]:


periods = [i for i in range(63)]


# In[29]:


#@execution_time_log
def get_values(date_key, cells):
    all_sm = np.sum([i for i in cells.values()])
    result = []
    result_weigths = []
    result_a = []
    result_weigths_a = []
    for cell in cells:
        weight = cells[cell]
        item = None
        date_index = indexes_dates[date_key]
        if cell in indexes_cells:
            cell_index = indexes_cells[cell]
            item = raw_values[cell_index, date_index]
            if not np.isnan(item):
                result_a.append(item)
                result_weigths_a.append(weight)
        else:
            cell_index = None
        if item is None or np.isnan(item):
            if cell_index is None:
                item = np.nanmedian(raw_values[:, date_index])
            else:
                m1 = np.nanmedian(raw_values[:, date_index])
                m2 = np.nanmedian(raw_values[cell_index, :])
                if np.isnan(m1) and not np.isnan(m2):
                    item = m2
                elif np.isnan(m2) and not np.isnan(m1):
                    item = m1
                else:
                    item = (m1 + m2) / 2
        if item is not None and not np.isnan(item):
            result.append(item)
            result_weigths.append(weight)
    sm = np.sum(result_weigths)
    weights = [i / sm for i in result_weigths]
    min_v = np.min(result) if len(result) > 0 else 0
    max_v = np.max(result) if len(result) > 0 else 0
    avg_v = np.average(result, weights=weights) if len(result) > 0 else 0
    
    sm_a = np.sum(result_weigths_a)
    weights_a = [i / sm_a for i in result_weigths_a]
    min_a = np.min(result_a) if len(result_a) > 0 else 0
    max_a = np.max(result_a) if len(result_a) > 0 else 0
    avg_a = np.average(result_a, weights=weights_a) if len(result_a) > 0 else 0
    return all_sm, len(cells), min_v, avg_v, max_v, min_a, avg_a, max_a


# In[30]:


def sub_date(date_user, date_data):
    #print(date_user, data_user)
    dif = datetime(1902, int(date_user / 100), date_user % 100) - datetime(1902, int(date_data / 100), date_data % 100)
    return dif.days


# In[31]:


from multiprocessing.dummy import Pool as ThreadPool 


# In[ ]:


def multi_set_value(key_type):
    sk_id = key_type[0]
    date_key = 100 * key_type[1] + key_type[2]
    result = get_values(date_key, type_data[1][key_type])
    user_date = user_dates[sk_id]
    index_date = sub_date(user_date, date_key)
    index_user = indexes_users[sk_id]
    table[index_user, 8 * index_date + sum_offset + 1] = result[sum_offset]
    table[index_user, 8 * index_date + cnt_offset + 1] = result[cnt_offset]
    table[index_user, 8 * index_date + min_offset + 1] = result[min_offset]
    table[index_user, 8 * index_date + avg_offset + 1] = result[avg_offset]
    table[index_user, 8 * index_date + max_offset + 1] = result[max_offset]
    table[index_user, 8 * index_date + min_offseta + 1] = result[min_offseta]
    table[index_user, 8 * index_date + avg_offseta + 1] = result[avg_offseta]
    table[index_user, 8 * index_date + max_offseta + 1] = result[max_offseta]


# In[ ]:


sum_offset, cnt_offset, min_offset, avg_offset, max_offset, min_offseta, avg_offseta, max_offseta = 0, 1, 2, 3, 4, 5, 6, 7

for analyze_column in columns:
    print('Load file ', analyze_column)
    dataframe = pd.read_csv(path + 'bs_avg_kpi.csv', sep=';', usecols=[analyze_column], decimal=',')
    dataframe_values = np.array(dataframe[analyze_column])
    
    print('Process file ', analyze_column)
    raw_values = np.zeros((len(indexes_cells), len(indexes_dates)))
    for v in zip(cells_line, dates_line, dataframe_values):
        ind_cell = indexes_cells[v[0]]
        ind_date = indexes_dates[v[1]]
        raw_values[ind_cell, ind_date] = v[2]
    data_columns = ['sk_id']
    data_columns += [str(u) + ' ' + c for u in periods for c in ['sum', 'count', 'min', 'avg', 'max', 'mina', 'avga', 'maxa']]
    table = np.zeros((len(subs_csi.values), 8 * len(periods) + 1))
    for type_data in (['data', user_data_session], ['voice', user_voice_session]):
        
        cnt = len(type_data[1])
        pool = ThreadPool(8)
        pool.map(multi_set_value, type_data[1])
        
        pool.close() 
        pool.join() 
        
        '''
        for key_type_ind in enumerate(type_data[1]):
            key_type = key_type_ind[1]
            #clear_output(wait=True)
            #print('Parse ', type_data[0], " ", analyze_column, " ", key_type_ind[0]/cnt)
            sk_id = key_type[0]
            date_key = 100 * key_type[1] + key_type[2]
            #result = get_values(date_key, type_data[1][key_type], raw_values)
            user_date = user_dates[sk_id]
            index_date = sub_date(user_date, date_key)
            #index_user = indexes_users[sk_id]
            print(user_date, date_key, index_date)
            
            table[index_user, 5 * index_date + sum_offset + 1] = result[sum_offset]
            table[index_user, 5 * index_date + cnt_offset + 1] = result[cnt_offset]
            table[index_user, 5 * index_date + min_offset + 1] = result[min_offset]
            table[index_user, 5 * index_date + avg_offset + 1] = result[avg_offset]
            table[index_user, 5 * index_date + max_offset + 1] = result[max_offset]
        '''
        output_frame = pd.DataFrame(data=table, columns=data_columns)
        output_frame['sk_id'] = subs_csi['SK_ID']
        output_frame.to_csv(pathfiles + pathfeatures + '{}_{}.csv'.format(type_data[0], analyze_column), index=False)
        send_notify(pathfiles + pathfeatures + '{}_{}.csv'.format(type_data[0], analyze_column))


# In[ ]:


send_notify('Complete office')

