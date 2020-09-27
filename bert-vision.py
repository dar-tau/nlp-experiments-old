paperspace = True
guy_folder = "/content/"
if paperspace: 
    guy_folder = "/notebooks/"

cache_dir = guy_folder+"/cache/transformer_cache"
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import ImageNet, ImageFolder, CIFAR10, CIFAR100
from torchvision import transforms
from torchvision.models import resnet101, resnet50
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW
import wandb


class DummyLayer(nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__()
    pass
  def forward(self, x, *args, **kwargs):
    return x

class PlainBERT(nn.Module):
    def __init__(self, n_tokens, min_layer = None):
        super().__init__()
        self.nLayers = 6
        self.nHeads = 12
        self.seqLen = 512


        bert = AutoModel.from_pretrained('distilbert-base-uncased', cache_dir = cache_dir)
        self.position_embeddings = nn.Parameter(
            torch.Tensor(bert.embeddings.position_embeddings(torch.arange(self.seqLen)).detach().numpy()))
        if min_layer is None:
          self.bert = bert.transformer
        else:
          raise NotImplementedError
          bert_ = bert.transformer
          for n, m in bert_.layer.named_children():
            if int(n) < min_layer:
              setattr(bert_.layer, n, DummyLayer())
        
          self.bert = bert_

        self.bert.requires_grad_(False)


    def forward(self, x):
        return self.bert.forward(x + self.position_embeddings, attn_mask = torch.ones(x.size(0), 512).to(x.device),
                                head_mask = torch.ones(self.nLayers, x.size(0), 
                                                       self.nHeads, self.seqLen, self.seqLen).to(x.device))

class BertVision(nn.Module):
    def __init__(self,  n_classes, img_dim):
        super().__init__()
        self.with_classifier = True
        self.n_tokens = np.prod(img_dim)
        self.top = nn.Sequential(
                                 nn.Conv2d(3, 32, 3, padding = 1 ),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(32, 100, 3, padding = 1),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(100, 200, 3, padding = 1),
                                 nn.LeakyReLU(0.2),
                                 nn.Conv2d(200, 768, 3, stride = (1, 2), padding = 1),
                                 nn.LeakyReLU(0.2)
                                )
        
        self.top.apply(self._init_top)
        self.bert = PlainBERT(n_tokens = self.n_tokens)
        self.fc = nn.Linear(768 * self.n_tokens//2, n_classes)
        self.layer_norm = nn.LayerNorm((512,))

    def toggleIntermediate(self):
        self.with_classifier = not self.with_classifier
    
    
    def _init_top(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        pass
    def forward(self, x):
        x = self.top(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.transpose(1,2)
        x = self.bert(x)
        x = torch.stack(x).squeeze(0)
        x = x.transpose(1,2).contiguous()
#         x = self.layer_norm(x)
#         x = torch.mean(x, dim = (-2,))
        x = x.view(x.size(0), -1)
        if self.with_classifier:
            x = self.fc(x)
        return x.squeeze(1)
    
    
    
device = 'cuda'
train_ds = CIFAR100("{}/data/cifar100".format(guy_folder), download = True, transform=transforms.ToTensor())
test_ds = CIFAR100("{}/data/cifar100".format(guy_folder), download = True, transform=transforms.ToTensor(), train = False)


batch_size = 8

lr = {'bert-vision': [1e-6, 5e-6, 1e-5, 5e-5],
      'resnet': [3e-4, 1e-3, 3e-3, 1e-2]
      }

optimizerDict = {'adam': torch.optim.Adam,
                 'adamw': AdamW,
                 'sgd': torch.optim.SGD, # No momentum
                 }

def makeModel(modelName):
  if modelName == 'resnet':
    model_resnet = resnet50(pretrained = True)
    model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 100)
    model_resnet.to(device)
    model = model_resnet
  elif modelName == 'bert-vision':
    model = BertVision(len(train_ds.classes), (32,32)).to(device)
  else:
    model = Sequential()
  return model

def train(config):
  
  optimizerAlg = optimizerDict[config.optimizer]
  if config.model == 'bert-vision' and config.optimizer == 'adam':
    optimizerAlg = optimizerDict['adamw']
  modelName = config.model
  lr_idx = config.lr_idx
  model = makeModel(modelName)


  criterion = nn.CrossEntropyLoss()
  optimizer = optimizerAlg(model.parameters(), lr = lr[modelName][lr_idx])
  train_dataloader = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
  test_dataloader = DataLoader(test_ds, batch_size = batch_size, shuffle = True)
  pbar = tqdm(train_dataloader, leave = True, position = 0)
  acc_sum = 0

  for i, (x,y) in enumerate(pbar):
      model.train()    
      optimizer.zero_grad()
      y = y.to(device)
      yhat = model(x.to(device))
      loss = criterion(yhat, y)
      acc_sum += (yhat.argmax(dim =  -1) == y).sum()    
      wandb.log({'loss': loss.item(), 
                 'acc': acc_sum.item() / (batch_size * (i+1))})

      loss.backward()
      optimizer.step()

wandb.init()
config = wandb.config
train(config)
