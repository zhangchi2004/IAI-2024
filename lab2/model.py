import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
def calculate_f_score(predictions, labels):
    tp = ((predictions==1) & (labels==1)).sum().item()
    fp = ((predictions==1) & (labels==0)).sum().item()
    fn = ((predictions==0) & (labels==1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f_score

class BaseModel(pl.LightningModule):
    def __init__(self):
        super(BaseModel,self).__init__()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        predictions = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits, y)
        self.log("training_loss", loss)
        acc = (predictions == y).float().mean()
        self.log("train_accuracy", acc)
        f_score = calculate_f_score(predictions, y)
        self.log("train_f_score", f_score)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        predictions = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits, y)
        self.log("validation_loss", loss)
        acc = (predictions == y).float().mean()
        self.log("validation_accuracy", acc)
        f_score = calculate_f_score(predictions, y)
        self.log("validation_f_score", f_score)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        predictions = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits, y)
        self.log("test_loss", loss)
        acc = (predictions == y).float().mean()
        self.log("test_accuracy", acc)
        f_score = calculate_f_score(predictions, y)
        self.log("test_f_score", f_score)
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        

class MLP(BaseModel):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.embed_dim = config["embed_dim"] # 50
        self.pad_len = config["pad_len"]
        self.dropout_rate = config["mlp_config"]["dropout_rate"]
        self.ln1 = nn.Linear(self.embed_dim*self.pad_len, 128)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.ln2 = nn.Linear(128,64)
        self.ln3 = nn.Linear(64,2)
        self.lr = config["learning_rate"]
        
    def forward(self,x):
        x = x.to(torch.float)
        out = (self.ln3(self.dropout(F.relu(self.ln2(self.dropout(F.relu(self.ln1(x.view(-1, self.embed_dim*self.pad_len)))))))))
        out = F.softmax(out, dim=-1)
        return out
    
        
class CNN(BaseModel):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.embed_dim = config["embed_dim"]
        self.filter_window_lens = config["cnn_config"]["filter_window_lens"]
        self.feature_map_num = config["cnn_config"]["feature_map_num"]
        self.pad_len = config["pad_len"]
        self.dropout_rate = config["cnn_config"]["dropout_rate"]
        self.conv0 = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.feature_map_num, kernel_size=self.filter_window_lens[0])
        self.conv1 = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.feature_map_num, kernel_size=self.filter_window_lens[1])
        self.conv2 = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.feature_map_num, kernel_size=self.filter_window_lens[2])
        
        self.full_connect = nn.Linear(len(self.filter_window_lens)*self.feature_map_num, 2)
        self.drop_out = nn.Dropout(self.dropout_rate)
        self.lr = config["learning_rate"]
        
    def forward(self,x):
        # x: batch_size x text_len x embed_dim
        x = x.transpose(-1,-2).to(torch.float)
        feat0 = self.conv0(x)
        feat0 = F.tanh(F.max_pool1d(feat0,feat0.shape[-1]).squeeze(-1))
        feat1 = self.conv0(x)
        feat1 = F.tanh(F.max_pool1d(feat1,feat1.shape[-1]).squeeze(-1))
        feat2 = self.conv0(x)
        feat2 = F.tanh(F.max_pool1d(feat2,feat2.shape[-1]).squeeze(-1))
        feats = torch.cat((feat0,feat1,feat2),dim=-1)
        out = self.full_connect(self.drop_out(feats))
        out = F.softmax(out, dim=-1)
        return out


class LSTM(BaseModel):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.embed_dim = config["embed_dim"]
        self.pad_len = config["pad_len"]
        self.hidden_size = config["lstm_config"]["hidden_size"]
        self.num_layers = config["lstm_config"]["num_layers"]
        self.dropout_rate = config["lstm_config"]["dropout_rate"]
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout = self.dropout_rate
        )
        self.full_connect = nn.Linear(2*self.hidden_size, 2)
        self.lr = config["learning_rate"]
    
    def forward(self,x):
        # x: batch_size x text_len x embed_dim
        x = x.to(torch.float)
        out, _ = self.lstm(x)
        out = self.full_connect(out[:, -1, :])
        out = F.softmax(out, dim=-1)
        return out