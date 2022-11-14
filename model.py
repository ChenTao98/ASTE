import torch
import torch.nn as nn
from transformers import BertPreTrainedModel,BertForMaskedLM
from transformers.models.bert.modeling_bert import BertEncoder,BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import sys,json,math
from packaging import version
with open("pos_tag.json") as in_fp:
    pos_tag_map=json.load(in_fp)

# class PtuneEmbeding(nn.Embedding):
    
#     def forward(self,input_ids):
#         self
        
        

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.pos_tag_embeddings=nn.Embedding(len(pos_tag_map), config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,pos_tag_id=None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if(pos_tag_id is not None):
            pos_tag_embedding=self.pos_tag_embeddings(pos_tag_id)
            embeddings+=pos_tag_embedding
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pos_tag_id=None
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            pos_tag_id=pos_tag_id
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss

class MrcOpinionModel(nn.Module):
    def __init__(self,args,tokenizer):
        super(MrcOpinionModel,self).__init__()
        self.bert=BertModel.from_pretrained(args.model_path)
        self.start_classfier=nn.Linear(768,1)
        self.end_classfier=nn.Linear(768,1)
        nn.init.xavier_normal_(self.start_classfier.weight)
        nn.init.xavier_normal_(self.end_classfier.weight)

        # self.classifier=nn.Linear(768,2)
        # nn.init.xavier_normal_(self.classifier.weight)
    
    def forward(self,input_ids,token_type_ids,attention_mask,start_pos=None,end_pos=None):
        output=self.bert(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        output=output[0]
        start_logits=self.start_classfier(output)
        end_logits=self.end_classfier(output)

        # logits=self.classifier(output)
        # start_logits,end_logits=logits.split(1,dim=-1)
        
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss=None
        if(start_pos is not None and end_pos is not None):
            start_loss = distant_cross_entropy(start_logits, start_pos)
            end_loss = distant_cross_entropy(end_logits, end_pos)
            total_loss = (start_loss + end_loss) / 2
        return start_logits,end_logits,total_loss

class MrcAspectOpinionModel(nn.Module):
    def __init__(self,args,tokenizer):
        super(MrcAspectOpinionModel,self).__init__()
        self.bert=BertModel.from_pretrained(args.model_path)
        self.start_classfier=nn.Linear(768,1)
        self.end_classfier=nn.Linear(768,1)

        self.aspect_start_classfier=nn.Linear(768,1)
        self.aspect_end_classfier=nn.Linear(768,1)
        
        self.opinion_map_aspect_start_classfier=nn.Linear(768,1)
        self.opinion_map_aspect_end_classfier=nn.Linear(768,1)
        self.polarity_classifier=nn.Linear(768,3)
        self.polarity_loss_fct=nn.CrossEntropyLoss()
        
        # nn.init.xavier_normal_(self.start_classfier.weight)
        # nn.init.xavier_normal_(self.end_classfier.weight)
        # nn.init.xavier_normal_(self.aspect_start_classfier.weight)
        # nn.init.xavier_normal_(self.aspect_end_classfier.weight)
        # nn.init.xavier_normal_(self.opinion_map_aspect_start_classfier.weight)
        # nn.init.xavier_normal_(self.opinion_map_aspect_end_classfier.weight)
        # nn.init.xavier_normal_(self.polarity_classifier.weight)

        # self.classifier=nn.Linear(768,2)
        # nn.init.xavier_normal_(self.classifier.weight)
    
    def forward(self,input_ids,token_type_ids,attention_mask,start_pos=None,end_pos=None,polarity_label=None,mode="aspect",pos_tag_id=None):
        output=self.bert(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,pos_tag_id=pos_tag_id)
        polarity_loss=None
        if(polarity_label is not None):
            polarity_emb=output[0][:,0]
            polarity_logits=self.polarity_classifier(polarity_emb).view(-1,3)
            if(polarity_label != "predict"):
                polarity_loss=self.polarity_loss_fct(polarity_logits,polarity_label.view(-1))
            
        output=output[0]
        if(mode=="opinion"):
            start_logits=self.start_classfier(output)
            end_logits=self.end_classfier(output)
        elif(mode=="opinion_map_aspect"):
            start_logits=self.opinion_map_aspect_start_classfier(output)
            end_logits=self.opinion_map_aspect_end_classfier(output)
        else:
            start_logits=self.aspect_start_classfier(output)
            end_logits=self.aspect_end_classfier(output)
        


        # logits=self.classifier(output)
        # start_logits,end_logits=logits.split(1,dim=-1)
        
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        total_loss=None
        if(start_pos is not None and end_pos is not None):
            start_loss = distant_cross_entropy(start_logits, start_pos)
            end_loss = distant_cross_entropy(end_logits, end_pos)
            total_loss = (start_loss + end_loss) / 2
        if(polarity_label is not None):
            return start_logits,end_logits,total_loss,torch.softmax(polarity_logits,dim=-1),polarity_loss
        return start_logits,end_logits,total_loss



def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], -1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], -1)
    neg_loss = torch.logsumexp(y_pred_neg, -1)
    pos_loss = torch.logsumexp(y_pred_pos, -1)
    return neg_loss + pos_loss

def global_pointer_crossentropy(y_pred, y_true):
    bh = y_pred.size(0) * y_pred.size(1)
    y_pred = y_pred.view(bh, -1)
    y_true = y_true.view(bh, -1)
    # print(y_pred)
    return multilabel_categorical_crossentropy(y_pred, y_true).mean()

def sequence_masking(x, mask, value=0.0, axis=1):
    if mask is None: return x
    for _ in range(axis-1): 
        mask = torch.unsqueeze(mask, 1)
    for _ in range(x.dim() - mask.dim()): 
        mask = torch.unsqueeze(mask, mask.dim())
    return x * mask + value * (1 - mask)

def pu_loss(y_pred, y_true):
    bh = y_pred.size(0) * y_pred.size(1)
    y_pred = y_pred.view(bh, -1)
    y_true = y_true.view(bh, -1)
    y_pred=torch.sigmoid(y_pred)
    pos = - torch.sum(torch.mul(y_true, torch.log(y_pred+1e-4)),dim=1) / torch.maximum(torch.tensor(1e-4), torch.sum(y_true,dim=1))
    neg = torch.sum(torch.mul((1-y_true), y_pred),dim=1) / torch.maximum(torch.tensor(1e-4), torch.sum(1-y_true,dim=1))
    mid = torch.sum(y_true,dim=1)/ (torch.sum(1-y_true,dim=1)+torch.sum(y_true,dim=1))
    x = torch.abs(neg - mid)
    neg = - torch.log(1 + 1e-6 - x)
    return torch.sum(pos + neg)

# def pu_loss(y_pred,y_true):
#     bh = y_pred.size(0) * y_pred.size(1)
#     y_pred = y_pred.view(bh, -1)
#     y_true = y_true.view(bh, -1)
#     # print(y_pred.shape)
#     print(torch.sum(y_true * torch.log(y_pred+1e-4)))
#     print(torch.maximum(torch.tensor(1e-4), torch.sum(y_true)))
#     pos = - torch.sum(y_true * torch.log(y_pred+1e-4)) / torch.maximum(torch.tensor(1e-4), torch.sum(y_true))
#     print(pos)
#     neg = torch.sum((1-y_true) * y_pred) / torch.maximum(torch.tensor(1e-4), torch.sum(1-y_true))
#     print(neg)
#     mid = torch.sum(y_true)/ (torch.sum(1-y_true)+torch.sum(y_true))
#     print(mid)
#     x = torch.abs(neg - mid)
#     neg = - torch.log(1 + 1e-6 - x)
#     print(pos+neg)
#     return pos + neg

# def pu_loss(y_pred,y_true):
#     print(y_pred)
#     bh = y_pred.size(0) * y_pred.size(1)
#     y_pred = y_pred.view(bh, -1)
#     y_true = y_true.view(bh, -1)
#     print(y_pred)
#     print(torch.log(y_pred+1e-4))
#     print(torch.mul(y_true, torch.log(y_pred+1e-4)))
#     print(torch.sum(torch.mul(y_true, torch.log(y_pred+1e-4)),dim=1))
#     pos = - torch.sum(torch.mul(y_true, torch.log(y_pred+1e-4)),dim=1) / torch.maximum(torch.tensor(1e-4), torch.sum(y_true,dim=1))
#     print(pos)
#     neg = torch.sum(torch.mul((1-y_true), y_pred),dim=1) / torch.maximum(torch.tensor(1e-4), torch.sum(1-y_true,dim=1))
#     mid = torch.sum(y_true,dim=1)/ (torch.sum(1-y_true,dim=1)+torch.sum(y_true,dim=1))
#     x = torch.abs(neg - mid)
#     neg = - torch.log(1 + 1e-6 - x)
#     return torch.sum(pos + neg)

# def pu_loss(y_pred,y_true):
#     print(y_pred.size)
#     bh = y_pred.size(0) * y_pred.size(1)
#     y_pred = y_pred.view(bh, -1)
#     y_true = y_true.view(bh, -1)
#     pos = - torch.sum(y_true * torch.log(y_pred+1e-4)) / torch.maximum(1e-4, torch.sum(y_true))
#     neg = torch.sum((1-y_true) * y_pred) / torch.maximum(1e-4, torch.sum(1-y_true))
#     mid = torch.sum(y_true)/ (torch.sum(1-y_true)+torch.sum(y_true))
#     x = torch.abs(neg - mid)
#     neg = - torch.log(1 + 1e-6 - x)
#     return pos + neg

class PositionalEncoding(nn.Module):
    # [bst, seq, fea]
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class GlobalPointerModel(nn.Module):
    def __init__(self,args,tokenizer, heads=1, head_size=64,sentiment_label_index=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.model_path)
        masklm=BertForMaskedLM.from_pretrained(args.model_path)
        self.vocab_size=masklm.config.vocab_size
        self.polarity_cls_aspect_sentiment=masklm.cls
        masklm=BertForMaskedLM.from_pretrained(args.model_path)
        self.polarity_cls_opinion_sentiment=masklm.cls
        self.polarity_classifier=nn.Linear(768,3)
        self.polarity_classifier_dual=nn.Linear(768,3)
        self.polarity_loss_fct=nn.CrossEntropyLoss()
        self.heads = heads
        self.head_size = head_size
        self.gpfc = nn.Linear(768, self.heads * self.head_size * 2)
        self.pe = PositionalEncoding(self.head_size)
        self.gpfc_aspect = nn.Linear(768, self.heads * self.head_size * 2)
        self.pe_aspect = PositionalEncoding(self.head_size)
        self.gpfc_dual_opinion = nn.Linear(768, self.heads * self.head_size * 2)
        self.pe_dual_opinion = PositionalEncoding(self.head_size)
        self.gpfc_opinion_map_aspect = nn.Linear(768, self.heads * self.head_size * 2)
        self.pe_opinion_map_aspect = PositionalEncoding(self.head_size)
        self.sentiment_label_index=sentiment_label_index

    def get_gp_output(self, z,pe, mask):
        zsize = z.size()  # b x l x (heads * headsz * 2)
        gpin = z.view(zsize[0], zsize[1], self.heads, self.head_size, 2)
        qw, kw = gpin[...,0], gpin[...,1]

        pos = pe(qw)
        cos_pos = torch.repeat_interleave(pos[..., None, 1::2], 2, -1)
        sin_pos = torch.repeat_interleave(pos[..., None, ::2], 2, -1)

        qw2 = torch.cat([-qw[..., 1::2, None], qw[..., ::2, None]], -1).view(qw.size())
        kw2 = torch.cat([-qw[..., 1::2, None], qw[..., ::2, None]], -1).view(kw.size())
        qw = qw * cos_pos + qw2 * sin_pos
        kw = kw * cos_pos + kw2 * sin_pos

        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        if mask:
            logits = sequence_masking(logits, mask, -1e9, 2)
            logits = sequence_masking(logits, mask, -1e9, 3)

        trimask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - trimask * 1e11
        return logits / (self.head_size**0.5)
        
    def forward(self, input_ids,token_type_ids,attention_mask,global_label=None,polarity_label=None,mode="aspect",pos_tag_id=None, mask=None,mask_index=None):
        output = self.bert(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask,pos_tag_id=pos_tag_id)
        polarity_loss=None
        # if(polarity_label is not None):
        #     polarity_emb=output[0][:,0]
        #     if(mode=="aspect_sentiment"):   
        #         polarity_logits=self.polarity_classifier(polarity_emb).view(-1,3)
        #     else:
        #         polarity_logits=self.polarity_classifier_dual(polarity_emb).view(-1,3)
        #     if(polarity_label != "predict"):
        #         polarity_loss=self.polarity_loss_fct(polarity_logits,polarity_label.view(-1))
        #     return torch.softmax(polarity_logits,dim=-1),polarity_loss
        if(polarity_label is not None):
            polarity_emb=output[0]
            if(mode=="aspect_sentiment"):   
                polarity_logits=self.polarity_cls_aspect_sentiment(polarity_emb)
            else:
                polarity_logits=self.polarity_cls_opinion_sentiment(polarity_emb)
            rang_a=list(range(polarity_emb.shape[0]))
            polarity_logits=polarity_logits[rang_a,mask_index]
            polarity_logits=polarity_logits[:,self.sentiment_label_index]
            if(polarity_label != "predict"):
                polarity_loss=self.polarity_loss_fct(polarity_logits.view(-1, 3),polarity_label.view(-1))
            return torch.softmax(polarity_logits,dim=-1),polarity_loss
        output=output[0]
        if(mode=="aspect_map_opinion"):
            logits=self.get_gp_output(self.gpfc(output),self.pe, mask)
        elif(mode=="opinion_map_aspect"):
            logits=self.get_gp_output(self.gpfc_opinion_map_aspect(output),self.pe_opinion_map_aspect, mask)
        elif(mode=="dual_opinion"):
            logits=self.get_gp_output(self.gpfc_dual_opinion(output),self.pe_dual_opinion, mask)
        else:
            logits=self.get_gp_output(self.gpfc_aspect(output),self.pe_aspect, mask)
        total_loss=None
        if(global_label is not None):
            total_loss = global_pointer_crossentropy(logits, global_label)
        return logits,total_loss