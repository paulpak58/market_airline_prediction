import torch
import pandas as pd
from collections import OrderedDict
from nn import NN 

if __name__=='__main__':
    
    model = NN(in_features=8,out_features=1)
    ckpt_dir = 'lightning_logs/largestNN_lagged_epoch400_batch256_lr4e-5/checkpoints/epoch=399-step=206800.ckpt'
    model_ckpt = torch.load(ckpt_dir)
    state_dict = OrderedDict()
    for k,v in model_ckpt['state_dict'].items():
        if 'model' in k:
            name = k[6:]
            state_dict[name] = v
    msg = model.load_state_dict(state_dict,strict=False)
    print(msg)

    df = pd.read_csv('data/final_df_with_lag.csv')

    find = df.loc[(df['YEAR']==2019)&(df['QUARTER']==3)&(df['AIRLINE_ID']==7)]
    x = find[['YEAR','QUARTER','avg_price','tot_passengers','avg_dist','Closed',\
               'ID','AIRLINE_ID']].values
    # y = find['avg_price_t1'].values
    pred = model(torch.tensor(x).float())
    avg_pred = torch.mean(pred)
    print('Counterfactual market fare prediction pre-merger: ', avg_pred.item())

    x_cf = find[['YEAR','QUARTER','avg_price','tot_passengers','avg_dist','Closed',\
               'ID','AIRLINE_ID']]
    x_cf = x_cf.assign(Closed=1.).values
    cf_pred = model(torch.tensor(x_cf).float())
    avg_cf_pred = torch.mean(cf_pred)
    print('Counterfactual market fare prediction post-merger: ', avg_cf_pred.item())
