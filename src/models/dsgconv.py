import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min
from torch_scatter.composite import scatter_softmax

class DSGConv(nn.Module):
    def __init__(self, 
                 in_dim, # l-1번째 레이어의 노드 임베딩 차원
                 out_dim, # l번째 레이어의 노드 임베딩 차원
                 num_factors, # K
                 act_fn, # 활성화 함수
                 aggr_type, # 집계 방식 ['sum', 'mean', 'max', 'attn']
                 num_neigh_type=4): # 이웃 타입의 수 DINES의 경우 + , -, ->, <- 4개
        """
        Build a DSGConv layer
        
        Args:
            in_dim: input embedding dimension (d_(l-1))
            out_dim: output embedding dimension (d_l)
            num_factors: number of factors (K)
            act_fn: torch activation function 
            aggr_type: aggregation type ['sum', 'mean', 'max', 'attn']
            num_neigh_type: number of neighbor's type (|D|)
        """
        super(DSGConv, self).__init__()
        
        # 입력 파라미터 설정
        self.d_in = in_dim
        self.d_out = out_dim
        self.K = num_factors
        self.num_neigh_type = num_neigh_type
        self.act_fn = act_fn
        self.aggr_type = aggr_type

        # 레이어 초기화 함수 호출
        self.setup_layers()
    
    # 레이어 초기화 함수
    def setup_layers(self):

        # 집계 방식이 'attn'(attention)인 경우
        if self.aggr_type == 'attn':
            self.disen_attn_weights = nn.ModuleList() # nn.ModuleList()를 사용하여 리스트를 만들어줌

            # 이웃 타입의 수 만큼 attention 가중치를 저장
            for _ in range(self.num_neigh_type):
                disen_attn_w = nn.Parameter(torch.empty(self.K, 2*self.d_in//self.K))
                torch.nn.init.xavier_uniform_(disen_attn_w)
                self.disen_attn_weights.append(disen_attn_w)
        
        # 집계 방식이 'max'(max pooling)인 경우
        elif self.aggr_type == 'max':
            self.disen_max_weights = nn.ParameterList()
            for _ in range(self.num_neigh_type):
                disen_max_w = nn.Parameter(torch.empty(self.K, self.d_in//self.K, self.d_in//self.K))
                torch.nn.init.xavier_uniform_(disen_max_w)
                self.disen_max_weights.append(disen_max_w)
            
        # 노드 임베딩 업데이트를 위한 가중치와 편향 초기화
        self.disen_update_weights = nn.Parameter(torch.empty(self.K, (self.num_neigh_type+1)*self.d_in//self.K, self.d_out//self.K))
        self.disen_update_bias = nn.Parameter(torch.zeros(1, self.K, self.d_in//self.K))
        torch.nn.init.xavier_uniform_(self.disen_update_weights)
    
    def forward(self, f_in, edges_each_type):
        """
        For each factor, aggregate the neighbors' embedding and update the anode embeddings using aggregated messages and before layer embedding
        
        Args:
            f_in: disentangled node embeddings of before layer
            edges_each_type: collection of edge lists of each neighbor type
        Returns:
            f_out: aggregated disentangled node embeddings
        """
        
        m_agg = [] # 집계된 메시지를 저장할 리스트
        m_agg.append(f_in) # 이전 임베딩 추가
        
        # 이웃 타입별로 메시지 집계
        print("edges_each_type: ", edges_each_type)
        for neigh_type_idx, edges_delta in enumerate(edges_each_type):
            m_delta = self.aggregate(f_in, edges_delta, neigh_type_idx=neigh_type_idx)
            m_agg.append(m_delta)

        f_out = self.update(m_agg)
        return f_out

    # 집계 함수
    def aggregate(self, f_in, edges_delta, neigh_type_idx):
        """
        Aggregate messsages for each factor by considering neighbor type and aggregator type
        torch_scatter: https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
        
        Args:
            f_in: disentangled node embeddings of before layer
            edges_delta: edge list of delta-type neighbors
            neigh_type_idx: index of neighbor type
            
        Returns:
            m_delta: aggregated meesages of delta-type neighbors
        """
        print("edges_delta: ", edges_delta)
        
        # 소스와 타겟 노드 추출
        src, dst = edges_delta[:, 0], edges_delta[:, 1]
        
        # 집계된 메시지를 저장할 변수 초기화
        out = f_in.new_zeros(f_in.shape)
        
        # SUM 집계 방식
        if self.aggr_type == 'sum':
            """
            scatter_add(src, index, dim=0, out=None)
            src: 집계할 값
            index: 집계 대상 노드의 index
            dim: 연산이 수행될 차원
            out: 출력 텐서
            """
            m_delta = scatter_add(f_in[dst], src, dim=0, out=out)
            
        elif self.aggr_type == 'attn':
            f_edge = torch.concat([f_in[src], f_in[dst]], dim=2)
            score = F.leaky_relu(torch.einsum("ijk,jk->ij", f_edge, self.disen_attn_weights[neigh_type_idx])).unsqueeze(2) 
            norm = scatter_softmax(score, src, dim=0) 
            m_delta = scatter_add(f_in[dst]*norm, src, dim=0, out=out)
            
        elif self.aggr_type == 'mean':
            m_delta = scatter_mean(f_in[dst], src, dim=0, out=out)
            
        elif self.aggr_type == 'max':
            f_in_max = torch.einsum("ijk,jkl->ijl", f_in, self.disen_max_weights[neigh_type_idx])
            m_delta = scatter_max(f_in_max[dst], src, dim=0, out=out)[0]
        
        return m_delta
    
    def update(self, m_agg):
        """
        Update node embeddings using aggregated messages and before layer embedding
        The einsum function used in forward operates as follows:
            (Concatenation of messages of each factor (N, K, 5*d_in/K) X FC Weights (K, 5*d_in/K, d_out/K)) + FC biases (1, K, d_out/K)
                -> Updated disentangled embedding (N, K, d_out/K)

        Args:
            m_agg: list of aggregated meesages and before layer embedding

        Returns:
            f_out: updated node embeddings
        """
        f_out = torch.einsum("ijk,jkl->ijl", torch.concat(m_agg, dim=2), self.disen_update_weights) + self.disen_update_bias 
        f_out = F.normalize(self.act_fn(f_out), dim=2)
        return f_out