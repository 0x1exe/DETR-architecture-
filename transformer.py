import torch.nn as nn
import torch.nn.functional as F
import torch 

class TransformerEncoderLayer(nn.Module):
  def __init__(self,dim_model,n_head,hidden_dim=2048,dropout=0.1):
    super().__init__()
    self.attn = nn.MultiheadAttention(dim_model,n_head,dropout=dropout)

    self.linear1=nn.Linear(dim_model,hidden_dim)
    self.dropout0 = nn.Dropout(dropout)
    self.linear2=nn.Linear(hidden_dim,dim_model)

    self.norm1 = nn.LayerNorm(dim_model)
    self.norm2= nn.LayerNorm(dim_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    self.activation = F_.relu

  def apply_pos(self,tensor,pos):
    return tensor if pos is None else tensor + pos

  def forward(self,src,mask=None,key_padding_mask=None,pos=None):

    q = k = self.apply_pos(src,pos)
    src2 = self.attn(q,k,value = src,attn_mask = mask,key_padding_mask = key_padding_mask)[0]
    src = self.norm1(src + self.dropout1(src2))
    src2= self.linear2(self.dropout0(self.activation(self.linear1(src))))
    src = self.norm2(src + self.dropout2(src2))
    return src
  
class TransformerEncoder(nn.Module):
  def __init__(self,encoder_layer,n_layers,norm=None):
    super().__init__()
    self.layers = stack_layers(encoder_layer,n_layers)
    self.n_layers = n_layers
    self.norm = norm

  def forward(self,src,mask=None,key_padding_mask=None,pos=None):
    output = src

    for layer in self.layers:
      output = layer(output,mask,key_padding_mask,pos)

    if self.norm is not None:
      output = self.norm(output)

    return output
  
class TransformerDecoderLayer(nn.Module):
  def __init__(self,dim_model,n_head,hidden_dim,dropout):
    super().__init__()
    self.src_attn=nn.MultiheadAttention(dim_model,n_head,dropout=dropout)
    self.memory_attn=nn.MultiheadAttention(dim_model,n_head,dropout=dropout)

    self.linear1 = nn.Linear(dim_model,hidden_dim)
    self.dropout0 = nn.Dropout(dropout)
    self.linear2 = nn.Linear(hidden_dim,dim_model)

    self.norm1 = nn.LayerNorm(dim_model)
    self.norm2 = nn.LayerNorm(dim_model)
    self.norm3 = nn.LayerNorm(dim_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3  = nn.Dropout(dropout)

    self.activation = F_.relu
  def apply_pos(self,tensor,pos):
    return tensor if pos is None else tensor+pos

  def forward(self,src,memory,
             src_mask=None,memory_mask=None,
             src_key_padding_mask=None,memory_key_padding_mask=None,
             query_pos=None,pos=None):
    q = k = self.apply_pos(src,query_pos)
    src2 = self.src_attn(q,k,src,attn_mask = src_mask,key_padding_mask = src_key_padding_mask)[0]
    src = self.norm1(src + self.dropout1(src2))
    src2 = self.memory_attn(self.apply_pos(src,query_pos),self.apply_pos(memory,pos),
                            memory,attn_mask = memory_mask,key_padding_mask = memory_key_padding_mask)[0]
    src = self.norm2(src + self.dropout2(src2))
    src2 = self.linear2(self.dropout0(self.activation(self.linear1(src))))
    src = self.norm3(src + self.dropout3(src2))
    return src
  
class TransformerDecoder(nn.Module):
  def __init__(self,decoder_layer,num_layers,norm=None,return_intermediate=False):
    super().__init__()
    self.layers=stack_layers(decoder_layer,num_layers)
    self.num_layers=num_layers
    self.norm=norm
    self.return_intermediate=return_intermediate
  def forward(self,src,memory,
              src_mask=None,memory_mask=None,
              src_key_padding_mask=None,memory_key_padding_mask=None,
              query_pos=None,pos=None):
    output=src
    intermediate=[]
    for layer in self.layers:
      output = layer(output,memory,src_mask,memory_mask,src_key_padding_mask,memory_key_padding_mask,query_pos=query_pos,pos=pos)
      if self.return_intermediate:
        intermediate.append(self.norm(output))

    if self.norm is not None:
      output = self.norm(output)
      if self.return_intermediate:
        intermediate.pop()
        intermediate.append(output)
    if self.return_intermediate:
      return torch.stack([intermediate])

    return output.unsqueeze(0)
  
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Transformer(nn.Module):
  def __init__(self,dim_model=512,num_head=8,num_encoder=6,num_decoder=6,
               hidden_dim = 2048,dropout=0.1,return_intermediate=False):
    super().__init__()

    encoder_layer=TransformerEncoderLayer(dim_model,num_head,hidden_dim,dropout)
    self.encoder = TransformerEncoder(encoder_layer,num_encoder)

    decoder_layer=TransformerDecoderLayer(dim_model,num_head,hidden_dim,dropout)
    decoder_norm = nn.LayerNorm(dim_model)
    self.decoder = TransformerDecoder(decoder_layer,num_decoder,norm=decoder_norm,return_intermediate=return_intermediate)

    self._reset_parameters()

    self.dim_model=dim_model
    self.num_head = num_head

  def _reset_parameters(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
  def forward(self,src,mask,query_embed,pos_embed):
    bs,c,h,w=src.shape

    src = src.flatten(2).permute(2,0,1)
    pos_embed=pos_embed.flatten(2).permute(2,0,1)
    query_embed=query_embed.unsqueeze(1).repeat(1,bs,1)
    mask=mask.flatten(1)

    tgt = torch.zeros_like(src)
    memory = self.encoder(src,key_padding_mask = mask,pos = pos_embed)
    hs = self.decoder(tgt,memory,memory_key_padding_mask=mask,
                      pos = pos_embed,query_pos = query_embed)

    return hs.transpose(1,2),memory.permute(1,2,0).view(bs,c,h,w)