import pickle
import os
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

pd.set_option('display.max_columns', 30)

save_path = '/home/datamininglab/Downloads/Bicycle/JEONG/'
save_path_D = '/media/datamininglab/새 볼륨/Dataset/Bicycle/'
file_path = '/home/datamininglab/Downloads/Bicycle'
file_list = sorted(os.listdir(file_path))


# TODO : 18.01 ~ 18.06 rent merge (X)
rent_ = pd.DataFrame()

for f in ['rent_1801_1.csv', 'rent_1802_1.csv', 'rent_1802_2.csv', 'rent_1802_3.csv', 'rent_1802_4.csv']:
    rent_i = pd.read_csv(os.path.join(file_path, f), encoding='949')
    rent_len = len(rent_)
    rent_ = pd.concat([rent_, rent_i])
    assert len(rent_) == rent_len + len(rent_i)

rent_.columns = ['자전거번호','대여일시','대여대여소번호','대여대여소명','대여거치대','반납일시','반납대여소번호','반납대여소명','반납거치대','이용시간(분)','이용거리(M)']
rent_ = rent_[['대여일시','대여대여소번호','반납일시','반납대여소번호','이용시간(분)','이용거리(M)']]

len(np.unique(list(rent_['반납대여소번호'])))
list(set(list(np.unique(rent_['반납대여소번호']))) - set(list(np.unique(rent_['대여대여소번호']))))

# remove (')
for col_name in rent_.columns:
    if col_name not in ['이용시간(분)','이용거리(M)']:
        rent_[col_name] = rent_[col_name].map(lambda x: x.replace('\'',''))

# filtering
with open(os.path.join(file_path, 'total_rent.pkl'), 'rb') as f: rent = pickle.load(f)
ut89 = pd.DataFrame({'대여대여소번호':list(np.unique(rent.대여대여소번호))})
rent_89_ = pd.merge(rent_, ut89, how='right', on='대여대여소번호')
ut89 = pd.DataFrame({'반납대여소번호':list(np.unique(rent.대여대여소번호))})
rent_89 = pd.merge(rent_89_, ut89, how='right', on='반납대여소번호')

# save
rent_89.to_pickle(os.path.join(save_path, 'station_filter.pkl'))
rent_89.to_csv(os.path.join(save_path, 'station_filter.csv'))


'''
# rent, station merge
station = pd.read_csv(os.path.join(file_path, file_list[-1]))
original_station_col = station.columns.copy()
station['대여소번호'] = station['대여소번호'].map(lambda x: str(x))
station.columns = ['대여구분','대여대여소번호','대여대여소명','대여대여소 주소','대여거치대수','대여위도','대여경도']
station_89 = pd.merge(station, ut89, how='right', on='대여대여소번호')
rent_89_1 = pd.merge(rent_89, station_89, how='right', on='대여대여소번호')
station_89.columns = ['반납구분','반납대여소번호','반납대여소명','반납대여소 주소','반납거치대수','반납위도','반납경도']
rent_89_2 = pd.merge(rent_89_1, station_89, how='outer', on='반납대여소번호')

# delete columns
print(rent_89_2.columns)
rent = rent_89_2.drop('대여대여소명_y', 1)
rent = rent.drop('반납대여소명_y', 1)
rent.columns = ['자전거번호', '대여일시', '대여대여소번호', '대여대여소명', '대여거치대', '반납일시', '반납대여소번호', '반납대여소명', '반납거치대', '이용시간(분)', '이용거리(M)', '날짜', '대여구분', '대여대여소 주소', '대여거치대수', '대여위도', '대여경도', '반납구분', '반납대여소 주소', '반납거치대수', '반납위도', '반납경도']
print(rent.columns)

#rent.columns = ['bike_num','rent_date','rent_st','rent_st_name','rent_holder_num','to_date','to_st','to_st_name','to_holder']

# save
rent.to_pickle(os.path.join(file_path, 'total_rent.pkl'))
rent.to_csv(os.path.join(file_path, 'total_rent.csv'))
'''


# TODO : rent / return aggregate x time bundle (X)
with open(os.path.join(save_path, 'station_filter.pkl'), 'rb') as f: rent_ = pickle.load(f)
rent_.columns = ['rend_date','rent_no','return_date','return_no','use_time(min)','use_dist(m)']

print('unique rent_no :', len(np.unique(rent_.rent_no)))
print('unique return_no :', len(np.unique(rent_.return_no)))

# convert to timestamp type
def to_timestamp(obj):
    return pd.Timestamp(int(obj[:4]), int(obj[5:7]), int(obj[8:10]), int(obj[11:13]), int(obj[14:16]), int(obj[17:19]))

rent_timestamp = [to_timestamp(rent_.iloc[i,0]) for i in range(len(rent_))]
return_timestamp = [to_timestamp(rent_.iloc[i,2]) for i in range(len(rent_))]
rent_['rend_date'] = rent_timestamp
rent_['return_date'] = return_timestamp

# ex) 00:00:00 ~ 00:04:59 -> 00:00:00
# ex) 11:12:25 -> 11:10:00
# ex) 05:49:59 -> 05:45:00
def to_split_5min(obj):
    return pd.Timestamp(obj.year, obj.month, obj.day, obj.hour, obj.minute//5*5, 0)

rent_['rend_date'] = [to_split_5min(rent_.iloc[i,0]) for i in range(len(rent_))]
rent_['return_date'] = [to_split_5min(rent_.iloc[i,2]) for i in range(len(rent_))]

'''
### dataset ---> 1 hour unit
# ex) 00:00:00 ~ 00:59:59 -> 00:00:00
# ex) 11:12:25 -> 11:00:00
# ex) 05:49:59 -> 05:00:00
def to_split_1hour(obj):
    return pd.Timestamp(obj.year, obj.month, obj.day, obj.hour, 0, 0)

rent_['rend_date'] = [to_split_1hour(rent_.iloc[i,0]) for i in range(len(rent_))]
rent_['return_date'] = [to_split_1hour(rent_.iloc[i,2]) for i in range(len(rent_))]
'''

# group by date, station
rent_agg_5min = rent_[['rend_date','rent_no','use_dist(m)']].groupby(['rend_date','rent_no']).count().reset_index()
rent_agg_5min.columns = ['rent_date_5min', 'rent_no', 'count']
rent_agg_5min = rent_agg_5min.sort_values(by=['rent_no','rend_date_5min'])

return_agg_5min = rent_[['return_date','return_no','use_dist(m)']].groupby(['return_date','return_no']).count().reset_index()
return_agg_5min.columns = ['return_date_5min', 'return_no', 'count']
return_agg_5min = return_agg_5min.loc[return_agg_5min.return_date_5min <= pd.Timestamp(2018,6,30,23,59,59)]
return_agg_5min = return_agg_5min.sort_values(by=['return_no','return_date_5min'])

# insert 0 data
START_DATE = pd.to_datetime('2018-01-01')
END_DATE = pd.to_datetime('2018-07-01')
_5min = pd.Timedelta('5min')

all_time = [START_DATE + _5min * i for i in range(int(1e6)) if (START_DATE + _5min * i) < END_DATE]

# rent_
rent_agg_5min_ = pd.DataFrame()
all_time_df = pd.DataFrame({'rent_date_5min':all_time})

for ind, station in enumerate(list(np.unique(rent_agg_5min.rent_no))):
    station_i = rent_agg_5min[rent_agg_5min.rent_no == station]
    station_meg = pd.merge(all_time_df, station_i, how='outer', on='rent_date_5min')
    station_meg['rent_no'] = station_meg['rent_no'].fillna(station)
    station_meg['count'] = station_meg['count'].fillna(0)
    tmp_length = len(rent_agg_5min_)
    rent_agg_5min_ = pd.concat([rent_agg_5min_, station_meg])
    assert len(rent_agg_5min_) == tmp_length + len(station_meg)
    if (ind+1) % 100 == 0: print("{}/{}".format(ind+1, 810))

# return_
return_agg_5min_ = pd.DataFrame()
all_time_df = pd.DataFrame({'return_date_5min':all_time})

for ind, station in enumerate(list(np.unique(return_agg_5min.return_no))):
    station_i = return_agg_5min[return_agg_5min.return_no == station]
    station_meg = pd.merge(all_time_df, station_i, how='outer', on='return_date_5min')
    station_meg['return_no'] = station_meg['return_no'].fillna(station)
    station_meg['count'] = station_meg['count'].fillna(0)
    tmp_length = len(return_agg_5min_)
    return_agg_5min_ = pd.concat([return_agg_5min_, station_meg])
    assert len(return_agg_5min_) == tmp_length + len(station_meg)
    if (ind+1) % 100 == 0: print("{}/{}".format(ind+1, 810))

# vis
vis_data = rent_agg_5min_[rent_agg_5min_.rent_no == '1009']
vis_data = vis_data.sort_values(by='rent_date_5min')
plt.plot(vis_data['count'])
plt.hist(vis_data['count'])

# save
rent_agg_5min_.to_pickle(os.path.join(save_path_D, 'rent_5min_X.pkl'))
return_agg_5min_.to_pickle(os.path.join(save_path_D, 'return_5min_X.pkl'))




# TODO : bulid weather data (X)
# https://data.kma.go.kr/cmmn/main.do
air = pd.read_csv(file_path+'/JEONG/airpol.csv', encoding='949')
air = air.drop('지점',1)
wea = pd.read_csv(file_path+'/JEONG/weather.csv', encoding='949')
wea = wea.drop('지점',1)

weather = pd.merge(wea, air, how='outer', on='일시')
weather.columns = ['day','temperature','precipitation','wind_speed','humidity','dew_point','insolation','visibility_10m','dust']

# missing value
fill_zero_col_list = ['precipitation', 'insolation']
for fill_zero_col in fill_zero_col_list:
    if weather[fill_zero_col].isnull().sum() != 0:
        weather[fill_zero_col] = weather[fill_zero_col].fillna(0)

fill_interpolation_col_list = ['temperature','wind_speed','humidity','dew_point','visibility_10m','dust']
for fill_interpolation_col in fill_interpolation_col_list:
    if weather[fill_interpolation_col].isnull().sum() != 0:
        weather[fill_interpolation_col] = weather[fill_interpolation_col].interpolate(method='values')

weather.iloc[0,-1] = 9.0
weather.iloc[1,-1] = 9.0
print('Number of missing values \n', weather.isnull().sum())

# add precipitation_yn
# refer to http://web.kma.go.kr/notify/epeople/faq.jsp?mode=view&num=186
prec_yn = [0 if i == 0.0 else 1 for i in list(weather.precipitation)]
weather['precipitation_yn'] = prec_yn

# save
weather.to_pickle(os.path.join(save_path, 'weather_total_X.pkl'))










##### TODO : TOTAL build data (total_X_Lookup)
import pickle
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

pd.set_option('display.max_columns', 30)

save_path = '/home/datamininglab/Downloads/Bicycle/JEONG/'
save_path_D = '/media/datamininglab/새 볼륨/Dataset/Bicycle/'
file_path = '/home/datamininglab/Downloads/Bicycle'
file_list = sorted(os.listdir(file_path))

### load 5min rent/return data
with open(os.path.join(save_path_D, 'rent_5min_X.pkl'), 'rb') as f: rent_5min = pickle.load(f)
with open(os.path.join(save_path_D, 'return_5min_X.pkl'), 'rb') as f: return_5min = pickle.load(f)
# delete 01.01~03.31 (04.01~06.30)
rent_5min = rent_5min.loc[rent_5min.rent_date_5min >= pd.Timestamp(2018,4,1,0,0,0)]
return_5min = return_5min.loc[return_5min.return_date_5min >= pd.Timestamp(2018,4,1,0,0,0)]
# concat
rent_5min['return_no'] = return_5min['return_no']
rent_5min['return_count'] = return_5min['count']
rent_5min.columns = ['date_5min', 'station', 'rent_count', 'drop_column', 'return_count']
demand_5min = rent_5min.drop('drop_column', 1)


### build one-hot data (month, weekday, hour, quarter-hour)
time_df = pd.DataFrame({'date_5min': [pd.Timestamp(2018,4,1,0,0,0) + pd.Timedelta('5min') * i for i in range(int(1e5)) if (pd.Timestamp(2018,4,1,0,0,0) + pd.Timedelta('5min') * i < pd.Timestamp(2018,7,1,0,0,0))]})
time_df['month'] = [str(time_df.iloc[i,0].month) for i in range(len(time_df))]
time_df['weekday'] = [str(time_df.iloc[i,0].weekday()) for i in range(len(time_df))]
time_df['hour'] = [str(time_df.iloc[i,0].hour) for i in range(len(time_df))]
quarter_hour = []
for i in range(len(time_df)):
    quarter_ = time_df.iloc[i,0].minute
    if quarter_ < 15: quarter_hour.append('0')
    elif quarter_ >= 15 and quarter_ < 30: quarter_hour.append('1')
    elif quarter_ >= 30 and quarter_ < 45: quarter_hour.append('2')
    elif quarter_ >= 45: quarter_hour.append('3')
    else: raise StopIteration
time_df['quarter_hour'] = quarter_hour

# dummy
time_df = pd.get_dummies(time_df, columns=['month','weekday','hour','quarter_hour'], drop_first=True)


### load weather data
with open(os.path.join(save_path, 'weather_total_X.pkl'), 'rb') as f: weather = pickle.load(f)
# delete precipitation_yn
weather = weather.drop('precipitation_yn', 1)

# convert to timestamp type
def weather_to_timestamp(obj):
    return pd.Timestamp(int(obj[:4]), int(obj[5:7]), int(obj[8:10]), int(obj[11:13]), 0, 0)
converted_day = [weather_to_timestamp(weather.iloc[i,0]) for i in range(len(weather))]
weather['day'] = converted_day

# (04.01~06.30)
weather = weather.loc[weather.day >= pd.Timestamp(2018,4,1,0,0,0)]
weather = weather.loc[weather.day < pd.Timestamp(2018,7,1,0,0,0)]

# normalize
def normalize_data(df, col):
    min_max_scalar = preprocessing.MinMaxScaler()
    df[col] = min_max_scalar.fit_transform(df[col].values.reshape(-1,1))
    return df[col]

for col in weather.columns:
    if col == 'day': continue
    weather[col] = normalize_data(weather, col)
    assert np.min(weather[col]) == 0 and np.max(weather[col]) <= 1

# time 5min split
weather_time_df = pd.DataFrame({'date_5min': [pd.Timestamp(2018,4,1,0,0,0) + pd.Timedelta('5min') * i for i in range(int(1e5)) if (pd.Timestamp(2018,4,1,0,0,0) + pd.Timedelta('5min') * i < pd.Timestamp(2018,7,1,0,0,0))]})
weather_time_df['day'] = [pd.Timestamp(weather_time_df.iloc[i,0].year, weather_time_df.iloc[i,0].month, weather_time_df.iloc[i,0].day, weather_time_df.iloc[i,0].hour, 0, 0) for i in range(len(weather_time_df))]
weather_time_df = pd.merge(weather_time_df, weather, how='right', on='day')
weather_time_df = weather_time_df.drop('day', 1)

### ALL X_DATA MERGE
demand_5min_df = pd.merge(demand_5min, time_df, how='inner', on='date_5min')
demand_5min_df = pd.merge(demand_5min_df, weather_time_df, how='inner', on='date_5min')
demand_5min_df = demand_5min_df.sort_values(by=['station','date_5min']) # sort by station, time

#demand_5min_df['Ind'] = [str(demand_5min_df.iloc[i,0]) + ' S' + demand_5min_df.iloc[i,1] for i in tqdm(range(len(demand_5min_df)))]
#demand_5min_df = demand_5min_df.drop('date_5min', 1)
#demand_5min_df = demand_5min_df.drop('station', 1)

lookup_index = ['S'+demand_5min_df.iloc[i,1]+' '+str(demand_5min_df.iloc[i,0]) for i in tqdm(range(len(demand_5min_df)))]
demand_5min_df['lookup_index'] = lookup_index
demand_5min_df = demand_5min_df.drop('date_5min', 1)
demand_5min_df = demand_5min_df.drop('station', 1)
cols = [demand_5min_df.columns[-1]] + list(demand_5min_df.columns[:-1])
demand_5min_df_ = demand_5min_df[cols]

demand_5min_df_.to_pickle(os.path.join(save_path_D, 'demand_lookup.pkl'))





##### TODO : TOTAL build data (total_X, total_Y)
demand = demand_5min_df[['date_5min', 'station','rent_count','return_count']]
demand = demand.sort_values(by=['station', 'date_5min'])
demand.index = range(len(demand))

demand_xy = []
for st_ind, e_station in enumerate(list(np.unique(demand.station))):
    demand_ = demand[demand.station == e_station]
    print("{}/{}".format(st_ind, len(np.unique(demand.station))))
    for i in range(0, len(demand_)-12, 12):
        # station
        inst = [demand_.iloc[i,1]]
        # x
        for j in range(i, i+12):
            inst.append(demand_.iloc[j,0])
        # y
        rent_total = 0
        return_total = 0
        for k in range(i+12, i+12*2):
            rent_total += demand_.iloc[k,2]
            return_total += demand_.iloc[k,3]
        inst.append(rent_total)
        inst.append(return_total)
        demand_xy.append(inst)

demand_xy_df = pd.DataFrame(demand_xy, columns=['station','5min_01','5min_02','5min_03','5min_04','5min_05','5min_06','5min_07','5min_08','5min_09','5min_10','5min_11','5min_12','rent_next_1hour','return_next_1hour'])

demand_xy_df_rev = pd.DataFrame()
for col_ind, col in enumerate(demand_xy_df.columns):
    if col in ['station']:
        continue
    elif col in ['rent_next_1hour','return_next_1hour']:
        demand_xy_df_rev[col] = demand_xy_df[col]
    else:
        col_rev = ['S'+demand_xy_df.iloc[i,0]+' '+str(demand_xy_df.iloc[i,col_ind]) for i in range(len(demand_xy_df[col]))]
        demand_xy_df_rev[col] = col_rev
    print("{} comp".format(col))

demand_xy_df_rev.to_pickle(os.path.join(save_path_D, 'demand_xy.pkl'))



##### TODO : build VOCAB
with open(os.path.join(save_path_D, 'demand_xy.pkl'), 'rb') as f: demand_xy = pickle.load(f)
demand_ = demand_xy.iloc[:,0:12]

vocab_key = demand_.values
vocab_key = list(vocab_key.flatten())

## add (S950 2018-06-30 23:00:00 ~ S950 2018-06-30 23:55:00), which is last target
vocab_key.extend(['S950 2018-06-30 23:20:00', 'S950 2018-06-30 23:50:00', 'S950 2018-06-30 23:05:00', 'S950 2018-06-30 23:55:00', 'S950 2018-06-30 23:10:00', 'S950 2018-06-30 23:15:00', 'S950 2018-06-30 23:40:00', 'S950 2018-06-30 23:30:00', 'S950 2018-06-30 23:00:00', 'S950 2018-06-30 23:45:00', 'S950 2018-06-30 23:25:00', 'S950 2018-06-30 23:35:00'])

demand_vocab = {key_:(value_+1) for value_, key_ in enumerate(vocab_key)}
with open(os.path.join(save_path_D, 'demand_vocab.pkl'), 'wb') as f: pickle.dump(demand_vocab, f)
