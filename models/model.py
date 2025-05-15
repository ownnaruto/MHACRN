from einops import rearrange
import numpy as np
import pandas as pd
import torch
import torch.nn as nn   
import torch.nn.functional as F 
import random  


class STEmbedding(nn.Module):
    def __init__(self, hidden_dim, time_embed_dim, pos_embed_dim, node_embed_dim, out_embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim + time_embed_dim + pos_embed_dim + node_embed_dim, out_embed_dim)
        # self.relu = nn.ReLU() 
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(out_embed_dim, out_embed_dim)
        
    def forward(self, hidden_embed, time_embed, pos_embed, node_embed):
        B = time_embed.shape[0]
        N = node_embed.shape[0] 
        time_embed = time_embed[:, None, :].repeat(1, N, 1)
        pos_embed = pos_embed[None, None, :].repeat(B, N, 1)
        node_embed = node_embed[None, :, :].repeat(B, 1, 1)
        x = torch.concat([hidden_embed, time_embed, pos_embed, node_embed], dim=-1)
        res = self.fc1(x)
        return self.fc2(self.act(res)) + res 
    

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, num_graphs):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.num_graphs = num_graphs
        self.weights = nn.Parameter(torch.FloatTensor(num_graphs*cheb_k*dim_in, dim_out)) 
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, graph_list):
        
        batch_size = x.shape[0]
        x_g = []        
        total_graph_set = []
        if self.cheb_k >= 2:
            for i, graph in enumerate(graph_list):
                if len(graph.shape) == 2:
                    cheb_graph_list = [torch.eye(graph.shape[0]).to(graph.device), graph]
                elif len(graph.shape) == 3:
                    cheb_graph_list = [torch.eye(graph.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1).to(graph.device), graph]
                
                for k in range(2, self.cheb_k):
                    cheb_graph_list.append(torch.matmul(2 * graph, cheb_graph_list[-1]) - cheb_graph_list[-2]) 
                total_graph_set.extend(cheb_graph_list)
        else:
            total_graph_set.extend(graph_list)
        for graph in total_graph_set:
            if len(graph.shape) == 2:
                x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
            elif len(graph.shape) == 3:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1) 
        
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  
        return x_gconv


class GCRUCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_graphs):
        super(GCRUCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, num_graphs)
        self.update = AGCN(dim_in+self.hidden_dim, dim_out, cheb_k, num_graphs)

    def forward(self, x, state, graph_list):
        
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, graph_list))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, graph_list))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)




class GCRU_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers, num_graphs):
        super(GCRU_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GCRUCell(node_num, dim_in, dim_out, cheb_k, num_graphs))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GCRUCell(node_num, dim_out, dim_out, cheb_k, num_graphs))

    def forward(self, x, init_state, graph_list):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1] 

        
        current_inputs = x
        output_hidden = []
        
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length): 

                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, graph_list)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)    


class GCRU_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers, num_graphs):
        super(GCRU_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GCRUCell(node_num, dim_in, dim_out, cheb_k, num_graphs))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GCRUCell(node_num, dim_out, dim_out, cheb_k, num_graphs))

    def forward(self, xt, init_state, graph_list):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        try:  
            assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        except:
            import pdb; pdb.set_trace() 
            print(xt.shape)
            print(self.input_dim) 
            
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], graph_list)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden
    

class MHACRN(nn.Module):
    def __init__(self,
                num_nodes,
                input_dim=1,
                output_dim=1,
                node_dim=20,
                horizon=12,
                rnn_dim=96,
                encoder_rnn_layers=1, 
                decoder_rnn_layers=2,
                ycov_dim=2,
                cheb_k=2,
                graph_num=1,
                tf_decay_steps=2000,
                use_teacher_forcing=True,
                decoder_embed_dim=16,
                st_embed_dim=20,
                time_embed_dim=32, 
                decoder_patch_len=3,
                shared_pred=True,
                seperate_pred=False, 
                min_tf_decay_rate=0.8,
                ):
        super().__init__()

        self.min_tf_decay_rate = min_tf_decay_rate 
        
        self.num_nodes = num_nodes 
        self.input_dim = input_dim  
        self.rnn_dim = rnn_dim
        self.output_dim = output_dim 
        self.horizon = horizon 
        
        self.encoder_rnn_layers = encoder_rnn_layers 
        self.decoder_rnn_layers = decoder_rnn_layers 
        
        self.cheb_k = cheb_k 
        self.ycov_dim = ycov_dim
        self.tf_decay_steps = tf_decay_steps
        self.use_teacher_forcing = use_teacher_forcing 
        self.graph_num = graph_num
        
        self.decoder_embed_dim = decoder_embed_dim
            
        self.node_embedding = nn.init.xavier_normal_(
            nn.Parameter(torch.empty(num_nodes, node_dim))
        )
        
        self.time_embed_dim = time_embed_dim 
        
        self.tid_embedding = nn.Embedding(288, time_embed_dim)
        self.dow_embedding = nn.Embedding(7, time_embed_dim) 

        encoder_dim = rnn_dim  
        self.encoder = GCRU_Encoder(self.num_nodes, self.input_dim, encoder_dim, self.cheb_k, self.encoder_rnn_layers, self.graph_num) 
        decoder_dim = encoder_dim 
        self.decoder = GCRU_Decoder(self.num_nodes, node_dim + decoder_patch_len + 2 * self.time_embed_dim + self.decoder_embed_dim, decoder_dim, self.cheb_k, self.decoder_rnn_layers, self.graph_num + 1) 
        self.decoder_patch_len = decoder_patch_len 
        self.num_decoder_patch = self.horizon // decoder_patch_len   
        self.decoder_position_embedding = nn.init.xavier_normal_(
            nn.Parameter(torch.randn(self.num_decoder_patch, self.decoder_embed_dim))
        )
        

        self.st_embedding = STEmbedding(encoder_dim, 2 * self.time_embed_dim, decoder_embed_dim, node_dim, st_embed_dim) 
        
        pred_dim = decoder_dim 
        
        self.hidden_weight = nn.Parameter(
            torch.ones(pred_dim, decoder_patch_len)
        )
        self.shared_pred = shared_pred
        if shared_pred:
            self.shared_pred_layer = nn.Linear(pred_dim, output_dim) 
        
        self.seperate_pred = seperate_pred 
        if seperate_pred:
            self.seperate_pred_weight = nn.Parameter(
                torch.randn(horizon, pred_dim, output_dim)
            )
            self.seperate_pred_bias = nn.Parameter(
                torch.randn(horizon, output_dim)
            )
        
    def compute_sampling_threshold(self, batches_seen):
        return max(self.tf_decay_steps / (self.tf_decay_steps + np.exp(batches_seen / self.tf_decay_steps)), self.min_tf_decay_rate)
     
        
    def forward(self, x, y_cov, labels=None, batches_seen=None):
        batch_size = x.shape[0]
        his = x[..., [0]] 
        graph = F.softmax(F.relu(self.node_embedding @ self.node_embedding.T), dim=-1)
        graph_list = [graph] 
        init_state = self.encoder.init_hidden(x.shape[0])  
        init_state = init_state.to(x.device) 
        encoder_hidden_last_layer, _ = self.encoder(x, init_state, graph_list)  
        encoder_hidden = encoder_hidden_last_layer[:, -1]
        decoder_graph_list = graph_list
        decoder_hidden = [encoder_hidden_last_layer[:, -1]] * self.decoder_rnn_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.decoder_patch_len), device=x.device)
        out = []
        for t in range(self.num_decoder_patch):
            position_embedding = self.decoder_position_embedding[t] 
            tid = y_cov[:, t * self.decoder_patch_len:(t+1) * self.decoder_patch_len, 0, 0] 
            dow = y_cov[:, t * self.decoder_patch_len:(t+1) * self.decoder_patch_len, 0, 1] 
            tid_emb = self.tid_embedding(
                (tid * 288).long()  
            ) 
            dow_emb = self.dow_embedding(dow.long())
            time_embedding = torch.cat([tid_emb, dow_emb], dim=-1).mean(dim=1) 
            st_embedding = self.st_embedding(encoder_hidden, time_embedding, position_embedding, self.node_embedding)
            st_graph = F.softmax(F.relu(st_embedding @ st_embedding.transpose(1, 2)), dim=-1)
            
            decoder_hidden_last_layer, decoder_hidden = self.decoder(
                torch.cat(
                    [go, time_embedding[:, None, :].repeat(1, self.num_nodes, 1), 
                     position_embedding[None, None, :].repeat(batch_size, self.num_nodes, 1),
                     self.node_embedding[None, :, :].repeat(batch_size, 1, 1)
                     ], 
                    dim=-1), 
                decoder_hidden, 
                decoder_graph_list + [st_graph]
            )  
            

            hidden = decoder_hidden_last_layer 
            hidden = torch.einsum('bnd,dl -> bnld', hidden, self.hidden_weight) 

            go = 0
            if self.shared_pred:
                go += self.shared_pred_layer(hidden)
            if self.seperate_pred:
                weight = self.seperate_pred_weight[t * self.decoder_patch_len:(t+1) * self.decoder_patch_len]
                bias = self.seperate_pred_bias[t * self.decoder_patch_len:(t+1) * self.decoder_patch_len]
                go += (torch.einsum('bnld,ldo->bnlo', hidden, weight) + bias)
            
            go = go.squeeze(-1) 
            out.append(go)
            if self.training and self.use_teacher_forcing:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t * self.decoder_patch_len:(t+1) * self.decoder_patch_len, :, 0]
                    go = go.transpose(1, 2) 
        output = torch.stack(out, dim=1)
        output = rearrange(output, 'b s n (l d) -> b (s l) n d', d=1)
        
        return output


