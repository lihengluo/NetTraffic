import pandas as pd
import numpy as np
import warnings
from sklearn import preprocessing
from sklearn.utils import resample

warnings.filterwarnings('ignore')
df = pd.read_csv(r'capture20110810.binetflow')
df.drop(['StartTime'],axis=1,inplace = True)
df.rename(columns = {list(df)[0]:'Duration',list(df)[1]:'Protocol',list(df)[2]:'Source_IP',list(df)[3]:'Source_Port',
                    list(df)[5]:'Destination_IP',list(df)[6]:'Destination_Port',list(df)[7]:'Flags',list(df)[8]:'Source_Type_of_Service',
                    list(df)[9]:'Dest_Type_of_Service',list(df)[10]:'Packets',list(df)[11]:'Total_Bytes',list(df)[12]:'Source_Bytes'}, inplace = True)
direction = ['   ->','  <->','  <-']
df = df.loc[df['Dir'].isin(direction)]
df.replace(['   ->','  <->','  <-'], ['outgoing','two-way','incoming'],inplace=True)
df.rename(columns = {list(df)[4]:'Direction'}, inplace = True)
df['Label'] = df['Label'].str.replace(r'flow=', '')
df['Label'] = df['Label'].str.replace(r'From-', '')
df['Label'] = df['Label'].str.replace(r'To-', '')
df['Label'] = df['Label'].str.extract(r'(\w+)')
df.dropna(inplace=True)

label_encoder = preprocessing.LabelEncoder()

"""
Direction
'outgoing' 'two-way' 'incoming'
    1          2         0
"""
cat_dir_data= label_encoder.fit_transform(df[['Direction']])
New_cat_dir = pd.DataFrame(cat_dir_data)
New_cat_dir = New_cat_dir.rename(columns={0:'Direction'})

"""
Protocol
'tcp' 'udp' 'rtp' 'icmp' 'rtcp' 'udt'
  3     4     2     0       1     5
"""
Protocol_Copy= label_encoder.fit_transform(df[['Protocol']])
New_cat_protocol = pd.DataFrame(Protocol_Copy)
New_cat_protocol=New_cat_protocol.rename(columns={0:'Protocol'})

"""
Flags
169 in total
"""
cat_flags_data= label_encoder.fit_transform(df[['Flags']]) 
New_cat_flags = pd.DataFrame(cat_flags_data)
New_cat_flags=New_cat_flags.rename(columns={0:'Flags'})
# print(df.Flags.unique())
# print(New_cat_flags.Flags.unique())

"""
Ports Selected
"""
imp_ports = ['22', '443', '80', '53', '389', '25', '113', '123', '554', '520', '161', '995', '67', '993', '631', '110',
 '143', '0', '445', '137', '427', '138', '524', '514', '139', '1000', '784', '12', '465', '592', '587', '88', '2', '888', '21',
 '500', '544', '81', '418', '294', '34', '98', '68', '709', '23', '8', '625', '768', '579', '135', '104', '916', '877', '310',
 '490', '1', '82', '369', '1013', '83', '832', '843', '471', '118']
df['ports']=np.where(df.Destination_Port.isin(imp_ports), df['Destination_Port'] , 'NaN')
df = df[df.ports != 'NaN']

# Merge lists
Interval_Variables = df[['Source_IP','Destination_IP','Duration','Packets','Total_Bytes','Source_Bytes','ports']]
a = pd.merge(Interval_Variables, New_cat_flags, right_index=True,left_index=True)
b = pd.merge(a, New_cat_protocol, right_index=True,left_index=True)
d = pd.merge(b, New_cat_dir, right_index=True,left_index=True)
d.dropna(inplace=True)

# Lable transform
Target = df.iloc[:,-2:-1]
lable_mlc = ['Botnet']
Target['yes/no'] = np.where(Target['Label'].isin(lable_mlc), 'yes', 'no')
final_df = pd.merge(Target, d, right_index=True, left_index=True)

majo = final_df.loc[final_df['yes/no']=='no']
mino = final_df.loc[final_df['yes/no']=='yes']

# downsample majority class, 'yes' has 22590 items
maj_dsampled = resample(majo, replace=False, n_samples=22590, random_state=1)
dsampled = pd.concat([mino, maj_dsampled])
y=dsampled[['yes/no']]
X=dsampled.drop(['yes/no','Label','Source_IP','Destination_IP'],axis=1)

# save the data， 第一行是列名
y.to_csv("labels.csv",index=False)
X.to_csv("features.csv", index=False)

