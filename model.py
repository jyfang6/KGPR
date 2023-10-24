
import torch
import torch.nn as nn 
from typing import Optional, Tuple, Union

from transformers import LukeModel
from transformers.models.luke.configuration_luke import LukeConfig
from transformers.models.luke.modeling_luke import BaseLukeModelOutputWithPooling


class RelationLukeModel(LukeModel):

    def __init__(self, config: LukeConfig, add_pooling_layer: bool = True):
        super().__init__(config, add_pooling_layer)

    def init_relations(self, num_rels=None, rel_dim=None, rel_embed=None, path_mlp_dim=1024, \
                       train_rel_embed=False, rel_only=True):

        entity_embed_size = self.config.entity_emb_size
        self.rel_embeddings = nn.Embedding(num_rels+1, rel_dim, padding_idx=-1)

        if rel_embed is None:
            assert num_rels is not None and rel_dim is not None # num_rels and rel_dim must be specified when rel_embed=None
            # randomly initialize embeddings 
            self.num_relations = num_rels
            self.relation_embed_size = rel_dim
            self.rel_embeddings.weight.data[:num_rels].copy_(torch.normal(mean=0.0, std=0.02, size=(num_rels, rel_dim)))
            self.rel_embeddings.weight.data[num_rels:].copy_(torch.zeros(size=(1, rel_dim)))
            train_rel_embed = True
        else:
            self.num_relations, self.relation_embed_size = rel_embed.shape
            if not torch.is_tensor(rel_embed):
                rel_embed = torch.tensor(rel_embed, dtype=torch.float)
            self.rel_embeddings.weight.data[:num_rels].copy_(rel_embed)
            self.rel_embeddings.weight.data[num_rels:].copy_(torch.zeros(size=(1, rel_dim)))
            train_rel_embed = train_rel_embed
        
        self.rel_only = rel_only
        path_input_dim = self.relation_embed_size if rel_only else 2*entity_embed_size + self.relation_embed_size
        self.path_embedding_dense = nn.Sequential(
            nn.Linear(path_input_dim, path_mlp_dim),
            nn.GELU(),
            nn.Linear(path_mlp_dim, self.config.hidden_size),
            nn.Dropout(0.1), 
        )

        self._init_weights(self.path_embedding_dense) 
 
    def get_extended_attention_mask(
            self, 
            word_attention_mask: torch.LongTensor, 
            entity_attention_mask: Optional[torch.LongTensor],
            path_attention_mask: Optional[torch.LongTensor],
        ):
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=-1)
        if path_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, path_attention_mask], dim=-1)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min 

        return extended_attention_mask

    def get_path_embeddings(
            self, 
            path_ids: torch.LongTensor, 
            path_position_ids: torch.LongTensor, 
            path_token_type_ids: torch.LongTensor = None,
        ):
        
        batch_size = path_ids.shape[0]
        path_seq_length = path_ids.shape[1]
        if path_token_type_ids is None:
            path_token_type_ids = torch.zeros((batch_size, path_seq_length), device=path_ids.device)

        relation_embeddings = self.rel_embeddings(path_ids[:, :, 2:])
        relation_mask = (path_ids[:, :, 2:] != self.num_relations).type_as(relation_embeddings).unsqueeze(-1) 
        relation_embeddings = relation_embeddings * relation_mask
        relation_embeddings = torch.sum(relation_embeddings, dim=-2) / relation_mask.sum(dim=-2).clamp(min=1.0) 
        if not self.rel_only:
            src_entity_embeddings = self.entity_embeddings.entity_embeddings(path_ids[:, :, 0]) 
            tgt_entity_embeddings = self.entity_embeddings.entity_embeddings(path_ids[:, :, 1]) 
            path_embedding_inputs = torch.cat([src_entity_embeddings, relation_embeddings, tgt_entity_embeddings], dim=-1)
        else:
            path_embedding_inputs = relation_embeddings

        path_embeddings = self.path_embedding_dense(path_embedding_inputs)

        position_embeddings = self.entity_embeddings.position_embeddings(path_position_ids.clamp(min=0)) 
        position_embeddings_mask = (path_position_ids != -1).type_as(position_embeddings).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embeddings_mask
        position_embeddings = torch.sum(position_embeddings, dim=-2) / position_embeddings_mask.sum(dim=-2).clamp(min=1.0)
        
        token_type_embeddings = self.entity_embeddings.token_type_embeddings(path_token_type_ids)

        embeddings = path_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.entity_embeddings.LayerNorm(embeddings)
        embeddings = self.entity_embeddings.dropout(embeddings)

        return embeddings

    def forward(
            self, 
            input_ids, 
            attention_mask: Optional[torch.FloatTensor] = None, 
            token_type_ids: Optional[torch.LongTensor] = None, 
            position_ids: Optional[torch.LongTensor] = None, 
            entity_ids: Optional[torch.LongTensor] = None,  
            entity_attention_mask: Optional[torch.FloatTensor] = None, 
            entity_token_type_ids: Optional[torch.LongTensor] = None, 
            entity_position_ids: Optional[torch.LongTensor] = None, 
            path_ids: Optional[torch.LongTensor] = None,  
            path_attention_mask: Optional[torch.FloatTensor] = None, 
            path_token_type_ids: Optional[torch.LongTensor] = None, 
            path_position_ids: Optional[torch.LongTensor] = None, 
            head_mask: Optional[torch.FloatTensor] = None, 
            inputs_embeds: Optional[torch.FloatTensor] = None, 
            output_attentions: Optional[bool] = None, 
            output_hidden_states: Optional[bool] = None, 
            return_dict: Optional[bool] = None
        ) -> Union[Tuple, BaseLukeModelOutputWithPooling]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        if entity_ids is not None:
            entity_seq_length = entity_ids.shape[1]
            if entity_attention_mask is None:
                entity_attention_mask = torch.ones((batch_size, entity_seq_length), device=device)
            if entity_token_type_ids is None:
                entity_token_type_ids = torch.zeros((batch_size, entity_seq_length), dtype=torch.long, device=device)

        if path_ids is not None:
            path_seq_length = path_ids.shape[1]
            if path_attention_mask is None:
                path_attention_mask = torch.ones((batch_size, path_seq_length), device=device)
            if path_token_type_ids is None:
                path_token_type_ids = torch.zeros((batch_size, path_seq_length), dtype=torch.long, device=device)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        word_embedding_output = self.embeddings(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            inputs_embeds = None 
        )

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, entity_attention_mask, path_attention_mask)

        if entity_ids is not None:
            entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_token_type_ids)

        if path_ids is not None:
            path_embedding_output = self.get_path_embeddings(path_ids, path_position_ids, path_token_type_ids)
        
        if entity_ids is None and path_ids is None:
            subgraph_embedding_output = None 
        if entity_ids is not None and path_ids is None:
            subgraph_embedding_output = entity_embedding_output
        if entity_ids is None and path_ids is not None:
            subgraph_embedding_output = path_embedding_output
        if entity_ids is not None and path_ids is not None:
            subgraph_embedding_output = torch.cat([entity_embedding_output, path_embedding_output], dim=-2)

        encoder_outputs = self.encoder(
            word_embedding_output, 
            subgraph_embedding_output, 
            attention_mask = extended_attention_mask, 
            head_mask = head_mask,
            output_attentions = output_attentions, 
            output_hidden_states = output_hidden_states, 
            return_dict = return_dict
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None 
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseLukeModelOutputWithPooling(
            last_hidden_state = sequence_output, 
            pooler_output = pooled_output, 
            hidden_states = encoder_outputs.hidden_states, 
            attentions = encoder_outputs.attentions, 
            entity_last_hidden_state = encoder_outputs.entity_last_hidden_state, 
            entity_hidden_states = encoder_outputs.entity_hidden_states, 
        )

class RelationLukeCrossEncoder(nn.Module):

    def __init__(self, configs):

        super().__init__()

        self.configs = configs
        luke_model_name = configs["model_name"]
        self.use_cls = self.configs["use_cls"]

        num_rels = self.configs["num_rels"]
        rel_dim = self.configs["rel_dim"]
        rel_embed = self.configs["rel_embed"]
        rel_only = self.configs["rel_only"]

        print(f"Loading pretrained model from {luke_model_name}")
        self.encoder = RelationLukeModel.from_pretrained(luke_model_name)
        self.encoder.init_relations(num_rels=num_rels, rel_dim=rel_dim, rel_embed=rel_embed, \
                                    path_mlp_dim=2048, train_rel_embed=True, rel_only=rel_only)
        self.embedding_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.embedding_size, 2, bias=True)
        self.loss_funct = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, inputs):
        
        outputs = self.encoder(**{k:v for k, v in inputs.items()})
        if self.use_cls:
            hidden_states = self.dropout(outputs.last_hidden_state[:, 0])
        else:
            hidden_states = self.dropout(outputs.pooler_output)

        logits = self.classifier(hidden_states)
        return logits

    def loss(self, inputs, labels):
        logits = self.forward(inputs)
        loss = self.loss_funct(logits, labels)
        return loss, logits