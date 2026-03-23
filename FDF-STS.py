class dyna_f(nn.Module):
    def __init__(self):
        super(dyna_f, self).__init__()
        self.net = nn.Sequential(nn.Linear(128, 256,bias=False),nn.Tanh(),nn.Linear(256,128,bias=False))
    def forward(self, t, y):
        return self.net(y)
# input x has shape [B, T, C] (Batch, Tokens, Channels)

class DyT(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Tanh()
    def forward(self, x):
        return self.act(x)
        
class SelfAttentionT(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # 定义Q、K、V的线性变换层
        self.Wq = nn.Linear(input_dim, input_dim)
        self.Wk = nn.Linear(input_dim, input_dim)
        self.Wv = nn.Linear(input_dim, input_dim)
    def forward(self, x):
        # 输入形状: [B, N, T, D]
        B, N, T, D = x.shape
        # 合并B和N维度，便于批量处理
        x_flat = x.permute(0,1,3,2).view(B*N,D,T)
        # 计算Q, K, V
        Q = self.Wq(x_flat) 
        K = self.Wk(x_flat)  
        V = self.Wv(x_flat) 
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (self.input_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B*N, D, D]
        # 应用注意力权重到V
        output = torch.matmul(attn_weights, V)  # [B*N, D, T]
        # 恢复原始形状
        output = output.view(B,N,D,T)
        return output

from torchdiffeq import odeint_adjoint as odeint
class FDF_STS(nn.Module):
    def __init__(self):
        super().__init__()
        # 节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(25,10)) ##### 前面的是节点数
        # 时域上时间与空间特征提取
        self.Linear_temporal = nn.Linear(1,32)
        self.act = DyT()
        self.map_temporal = nn.Sequential(nn.Linear(96,96),DyT(),nn.Dropout(0.2)) ##### 后面的是预测步长
        self.Linear_graph = nn.Linear(1,32,bias=False)
        self.map_graph = nn.Sequential(nn.Linear(96,96),DyT(),nn.Dropout(0.2)) ##### 后面的是预测步长
        # 频域上的特征提取
        #self.freq_encoder = nn.Sequential(nn.Linear(25,32),nn.Dropout(0.2)) ##### 前面的是节点数
        self.freq_encoder = nn.Sequential(SelfAttentionT(25),DyT(),nn.Dropout(0.2),nn.Linear(25,32),DyT(),nn.Dropout(0.2)) ##### 前面的是节点数
        self.transform_real = nn.Parameter(torch.randn(49, 32, 32))
        self.transform_imag = nn.Parameter(torch.randn(49, 32, 32))
        self.freq_decoder = nn.Sequential(nn.Linear(32,25),nn.Dropout(0.2)) ##### 后面的是节点数
        self.map_freq = nn.Sequential(nn.Linear(96,32),DyT(),nn.Dropout(0.2),nn.Linear(32,1))
        self.Linear_freq = nn.Linear(1,32)
        # ODE相关
        self.ode_att = nn.Sequential(SelfAttentionT(96),DyT(),nn.Dropout(0.2))
        self.eps = nn.Parameter(torch.zeros(1))
        self.ode_encoder = nn.Sequential(nn.Linear(64,128),DyT(),nn.Dropout(0.2))
        self.ode = dyna_f()
        self.param = nn.Sequential(nn.Linear(128,49),nn.Dropout(0.2)) 
        # 输出层
        self.decoder = nn.Sequential(nn.Linear(32*3,64),DyT(),nn.Dropout(0.2),nn.Linear(64,32),DyT(),nn.Dropout(0.2),nn.Linear(32,1))

    def forward(self, data):
        # 图结构计算
        node_emb = self.node_embeddings
        norm = torch.norm(node_emb, p=2, dim=1, keepdim=True)
        graph_data = torch.mm(node_emb/norm, (node_emb/norm).t())
        
        # 时域上时空特征提取
        B,N,T = data["flow_x"].shape
        data_temporal = self.act(self.Linear_temporal(data["flow_x"].unsqueeze(3))).permute(0,1,3,2) #[B,N,D,T]
        data_temporal_map = self.map_temporal(data_temporal).permute(0,1,3,2)  #[B,N,Pred_len,D]
        data_graph = self.act(self.Linear_graph(torch.matmul(graph_data,data["flow_x"]).unsqueeze(3))).permute(0,1,3,2) #[B,N,D,T]
        data_graph_map = self.map_graph(data_graph).permute(0,1,3,2)  #[B,N,Pred_len,D] 
        feature_spatio_temporal = torch.cat((data_temporal,data_graph),dim=2).permute(0,1,3,2) #[B,N,T,D*2]
            
        # 频域上特征提取
        freq_features_init = self.freq_encoder(data["flow_x"].unsqueeze(1)).squeeze() #[B,T,D]   可替换成更复杂的attension
        freq = torch.fft.rfft(freq_features_init, dim=1, norm='ortho').unsqueeze(1)
        xf = freq.repeat(1, freq.shape[2], 1, 1) #[B,49,49,D]
        mask = torch.eye(49).byte().to(xf.device) 
        #这里的乘法是点乘(就是*操作符),先把tl扩展成btlc,然后对应位置乘,目的就是把每个频率分量孤立开进行特征提取
        fre = torch.einsum('btlc, tl -> btlc', xf, mask) #[B,49,49,D]
        processed_real = torch.matmul(fre.real,self.transform_real)-torch.matmul(fre.imag,self.transform_imag)  #[B,49,49,D]
        processed_imag = torch.matmul(fre.real,self.transform_imag)+torch.matmul(fre.imag,self.transform_real)  #[B,49,49,D]
        processed = torch.view_as_complex(torch.stack([processed_real, processed_imag], dim=-1)) #[B,49,49,D]
        data_frequence_intemporal = torch.fft.irfft(processed,dim=2, norm='ortho') #[B,49,T,D]
        data_frequence = self.freq_decoder(data_frequence_intemporal).permute(0,3,2,1) #[B,N,T,49]
        
        # ODE动力学
        feature_temporal_ode = torch.sum(self.ode_att(feature_spatio_temporal),dim=3).squeeze() #[B,N,D*2]
        feature_spatio_temporal_ode = torch.sum(((1+self.eps)*feature_temporal_ode+torch.matmul(graph_data,feature_temporal_ode)),dim=1)#[B,D*2]
        ini_ode = self.ode_encoder(feature_spatio_temporal_ode.squeeze())
        t_span = torch.arange(0.1,9.7,step=0.1,dtype=torch.float32).to(device)  ##### 后面的96是预测步长
        hid_ode = odeint(self.ode, ini_ode, t_span) #[97,B,32]
        fuse_param = F.softmax(self.param(hid_ode),dim=2).permute(1,2,0)  # (B,49,Pedict_len)

        # 频域特征加权
        data_frequence_map_hid = torch.einsum('bnhf, bfp -> bnhp',data_frequence,fuse_param) #[B,N,T,49] [B,49,Pedict_len] -> [B,N,T,Pedict_len]
        data_frequence_map = self.Linear_freq(self.act(self.map_freq(data_frequence_map_hid.permute(0,1,3,2)).squeeze().unsqueeze(3))) #[B,N,Pedict_len,D]
        
        # 解码获得预测
        final_feature = torch.cat((data_temporal_map,data_graph_map,data_frequence_map),dim=3) #[B,N,Pedict_len,D*3]
        output = self.decoder(final_feature).squeeze()
        return output
