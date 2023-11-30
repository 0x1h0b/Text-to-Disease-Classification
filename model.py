import torch
import torch.nn as nn
import transformers



class Model(nn.Module):
    def __init__(self,model_path,num_classes):
        super().__init__()
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = transformers.AutoModel.from_pretrained(self.model_path)
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024,self.num_classes),
            nn.Sigmoid()
        )
    
    def forward(self,ids,mask,token_type_ids,targets=None):
        _, out = self.model(input_ids=ids,attention_mask=mask,token_type_ids=token_type_ids)
        print(out,type(out))
        out = self.fc_layer(out)
        return out