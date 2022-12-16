# import pyreadr
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class MarketAirlineDataset(Dataset):

    def __init__(self,filename):
        df = pd.read_csv(filename)
        x = df[['YEAR','QUARTER','avg_price','tot_passengers','avg_dist','Closed',\
                'ID','AIRLINE_ID']].values
        y = df['avg_price_t1'].values
        self.x_train = torch.tensor(x,dtype=torch.float32)
        self.y_train = torch.tensor(y,dtype=torch.float32)

    def process_data(self,filename):
        if filename.split('.')[1]=='R':
            df = pyreadr.read_r(filename)
            df = df['market_airline_level']
        else:
            df = pd.read_csv(filename)
        carriers = pd.read_excel('mergers.xlsx')
        df = df.merge(carriers,on=['TICKET_CARRIER','YEAR','QUARTER'],how='left')
        df['TICKET_CARRIER'] = pd.Categorical(df['TICKET_CARRIER'])
        df['TICKET_CARRIER_FACT'] = df['TICKET_CARRIER'].cat.codes
        df['Closed'] = df['Closed'].fillna(0)
        df['Announced'] = df['Announced'].fillna(0)
        df['Experienced Merge'] = df['Experienced Merge'].fillna(0)
        test = df[(df['QUARTER']==1)&((df['YEAR']==2014)|(df['YEAR']==2015))]
        duplicate = test[test.duplicated(['ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID','TICKET_CARRIER'])]
        duplicate = duplicate.groupby(['ORIGIN_AIRPORT_ID','DEST_AIRPORT_ID','TICKET_CARRIER'])['YEAR','QUARTER','tot_passengers','avg_dist','Closed'].agg(' '.join).reset_index()
        return df

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self,idx):
        return self.x_train[idx],self.y_train[idx]

if __name__=='__main__':
    filename = 'df.csv'
    train_size=0.7
    val_size=0.3
    dataset = MarketAirlineDataset(filename)
    train_dataset,val_dataset = torch.utils.data.random_split(dataset,[train_size,val_size])
    print(len(train_dataset),len(val_dataset))
    train_loader = DataLoader(train_dataset,batch_size=4,shuffle=False)
    '''
    for i,(data,labels) in enumerate(train_loader):
        print(data.shape,labels.shape)
        print(data,labels)
    '''
