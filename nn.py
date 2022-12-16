import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# from data import MarketAirlineDataset
from lagged_data import MarketAirlineDataset
from torch.utils.data import DataLoader

class NN(nn.Module):
    def __init__(self,in_features=22,hidden=128,hidden1=256,hidden2=512,hidden3=256,hidden4=128,out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features,hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden,hidden*2)
        self.bn2 = nn.BatchNorm1d(hidden*2)
        self.fc3 = nn.Linear(hidden*2,hidden*4)
        self.bn3 = nn.BatchNorm1d(hidden*4)
        self.fc4 = nn.Linear(hidden*4,hidden*8)
        self.bn4 = nn.BatchNorm1d(hidden*8)
        self.fc5 = nn.Linear(hidden*8,hidden*16)
        self.bn5 = nn.BatchNorm1d(hidden*16)
        self.fc6 = nn.Linear(hidden*16,hidden*16)
        self.bn6 = nn.BatchNorm1d(hidden*16)
        self.fc7 = nn.Linear(hidden*16,hidden*8)
        self.bn7 = nn.BatchNorm1d(hidden*8)
        self.fc8 = nn.Linear(hidden*8,hidden*4)
        self.bn8 = nn.BatchNorm1d(hidden*4)
        self.out = nn.Linear(hidden*4,out_features)

    def forward(self,x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = F.relu(self.bn8(self.fc8(x)))
        x = self.out(x)
        return x

class MarketTrainer(pl.LightningModule):
    def __init__(self,model=None,in_features=22,hidden1=256,hidden2=128,lr=1e-3,ckpt_dir=''):
        super().__init__()
        self.model = NN(in_features,hidden1,hidden2) if model is None else model
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.init_weights(ckpt_dir)

    def init_weights(self,ckpt_dir):
        self.apply(self.__init__weights)
        if ckpt_dir!='':
            ckpt = torch.load(ckpt_dir)
            from collections import OrderedDict
            state_dict = OrderedDict()
            for k,v in ckpt['state_dict'].items():
                if 'model' in k:
                    name = k[6:]
                    state_dict[name] = v
            msg = self.model.load_state_dict(state_dict)
            print(msg)

    def __init__weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m,nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.LayerNorm):
                torch.nn.init.constant_(m.bias,0)
                torch.nn.init.constant_(m.weight,1.0)

    def training_step(self,batch,batch_idx):
        x,y = batch
        y_pred = self.model(x)
        y = y.flatten()
        y_pred = y_pred.flatten()
        loss = self.loss_fn(y,y_pred)
        self.log('train_loss:',loss)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_pred = self.model(x)
        y = y.flatten()
        y_pred = y_pred.flatten()
        loss = self.loss_fn(y,y_pred)
        self.log('val_loss:',loss,sync_dist=True)
        return loss

    def validation_epoch_end(self,outputs)->None:
        avg_loss = sum(outputs)/len(outputs)
        print('\nEpoch Average Loss: ',avg_loss,'\n')

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),lr=self.lr)

if __name__=='__main__':
    # Configs
    batch_size=256
    num_workers=4
    gpus=0
    lr=4e-5

    # filename = 'market_airline_level.R'
    filename = 'final_df_with_lag.csv'
    dataset = MarketAirlineDataset(filename)
    train_size = math.floor(0.7*len(dataset))
    val_size = len(dataset)-train_size
    train_dataset,val_dataset = torch.utils.data.random_split(dataset,[train_size,val_size])
    print(len(train_dataset),len(val_dataset))
    # Retrieve datasets
    dataloader_train = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    backbone = NN(
        in_features=8,
        hidden1=256,
        hidden2=128,
        out_features=1
    )
    model = MarketTrainer(
        model = backbone,
        lr=lr
    )
    
    if torch.cuda.is_available():
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            strategy='ddp',
            num_nodes=1,
            check_val_every_n_epoch=1,
            default_root_dir=None,
            max_epochs=400,
            # accumulate_grad_batches=16,
            # precision=16
        )
    else:
        trainer = pl.Trainer(
            # accelerator='gpu',
            devices=1,
            strategy='ddp',
            num_nodes=1,
            check_val_every_n_epoch=1,
            default_root_dir=None,
            max_epochs=400,
            # accumulate_grad_batches=16,
            # precision=16
        )
    # trainer.fit(model,dataloader_train,dataloader_val,ckpt_path='lightning_logs/less_regressors1/checkpoints/epoch=249-step=10500.ckpt')
    trainer.fit(model,dataloader_train,dataloader_val)
