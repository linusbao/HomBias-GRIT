import torch
import torch.nn as nn
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder, register_network
from torch_geometric.nn.models import MLP
from torch_geometric.nn.pool.glob import global_add_pool

from torch_geometric.nn.encoding import PositionalEncoding

def compute_gradual_width(dim_in, dim_out, layers):
    #returns a channel_list of MLP widths that gradually changes from dim_in to dim_out

    if (dim_in - dim_out) > 0:
        increasing = False
    elif (dim_in - dim_out) < 0:
        increasing = True
    else:
        return [dim_in for i in range(layers)]

    step_size = abs((dim_in - dim_out)/layers)
    channel_list = [dim_in]

    current_dim = dim_in
    for i in range(layers-1):
        if increasing:
            current_dim = round(current_dim + step_size)
        else:
            current_dim = round(current_dim - step_size)
        channel_list.append(current_dim)
    channel_list.append(dim_out)

    print(f'Gradual channel list (for MLPcountEnc) is: {channel_list}')
    return channel_list

class trig_encoder(torch.nn.Module):
    #Applied the sin/cos positional encoding from "Attention is All You Need"

    def __init__(self, d):
        super().__init__()

        self.d = d
        self.trig = PositionalEncoding(d)

    def forward(self, counts):

        og_size = counts.size()
        h = torch.flatten(counts)
        h = self.trig(h)
        h = h.view((og_size[0], og_size[1]*self.d))

        return h

    # def forward(self, batch): #might have to use this forward method if module.children() call in GPS forward() doesn't play well with trig_encoder

    #     if hasattr(batch, "counts_wl"):
    #         wl_flag = True
    #         h = batch.counts_wl
    #         og_size =  batch.counts_wl.size()
    #     elif hasattr(batch, "counts"):
    #         wl_flag = False:
    #         h = batch.counts
    #         og_size =  batch.counts.size()
    #     else:
    #         raise Exception('No appropriate homcounts avalible')

    #     h = torch.flatten(h)
    #     h = self.trig(h)
    #     h = h.view((og_size[0], og_size[1]*self.d))

    #     if wl_flag:
    #         batch.counts_wl = h
    #     else:
    #         batch.counts = h

    #     return batch

@register_node_encoder('NodeCountEnc')
class MLPNodeCountEncoder(torch.nn.Module):
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()

        dim_in = cfg.share.dim_in
        dim_ogct = cfg.ctenc_NodeCountEnc.dim_ogct
        dim_ct = cfg.ctenc_NodeCountEnc.dim_ct
        grad_width_change = cfg.ctenc_NodeCountEnc.grad_width_change if hasattr(cfg.ctenc_NodeCountEnc, "grad_width_change") else False
        self.pass_as_var = cfg.ctenc_NodeCountEnc.pass_as_var if hasattr(cfg, "ctenc_NodeCountEnc.pass_as_var") else False

        if dim_emb - dim_ct < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"Count dim size {dim_ct} is too large for "
                             f"desired embedding size of {dim_emb}.")
        
        if expand_x and dim_emb - dim_ct > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_ct)
        self.expand_x = expand_x and dim_emb - dim_ct > 0
        
        if cfg.ctenc_NodeCountEnc.raw_bn:
            self.raw_norm = nn.BatchNorm1d(dim_ogct)
        else:
            self.raw_norm = None

        # Trig Encoder pre MLP
        self.use_mlp = True #use MLP by default
        self.use_trig = False
        if hasattr(cfg.ctenc_NodeCountEnc, "trig_enc"):
            tcfg = cfg.ctenc_NodeCountEnc.trig_enc
            if tcfg.use:
                self.use_trig = True
                self.use_mlp = False if hasattr(tcfg, "post_trig_fc") and not tcfg.post_trig_fc else True #turn off MLP if post_trig_fc attribute exists in cfg and is set tp false
                self.trig_enc = trig_encoder(tcfg.d)
                dim_ogct = dim_ogct * tcfg.d #scale up the MLP's input dimension

        if self.use_mlp:
            if hasattr(cfg.ctenc_NodeCountEnc, "norm") and cfg.ctenc_NodeCountEnc.norm != 'none' and cfg.ctenc_NodeCountEnc.norm != 'None':
                mlp_norm = cfg.ctenc_NodeCountEnc.norm
            else:
                mlp_norm = None
            if grad_width_change:
                self.encoder = MLP(channel_list=compute_gradual_width(dim_ogct, dim_ct, cfg.ctenc_NodeCountEnc.n_layers), dropout=cfg.ctenc_NodeCountEnc.dropout, norm=mlp_norm)
            else:
                self.encoder = MLP(in_channels=dim_ogct, hidden_channels=cfg.ctenc_NodeCountEnc.dim_hid, out_channels=dim_ct, num_layers=cfg.ctenc_NodeCountEnc.n_layers, dropout=cfg.ctenc_NodeCountEnc.dropout, norm=mlp_norm)

    def forward(self, batch):
        if not hasattr(batch, 'counts') and (not hasattr(batch, 'counts_spasm')) and (not hasattr(batch, "graph_hom") or tuple(batch.graph_hom.size())[0] != tuple(batch.x.size())[0]):
            raise ValueError(f'No counts given in data')
        
        if hasattr(batch, 'counts') and hasattr(batch, "graph_hom"):
            count_enc = getattr(batch, 'counts')
        elif hasattr(batch, 'counts') and not hasattr(batch, "graph_hom"):
            count_enc = getattr(batch, 'counts')
        elif not hasattr(batch, 'counts') and hasattr(batch, "graph_hom"):
            assert tuple(batch.graph_hom.size())[0] == tuple(batch.x.size())[0]
            count_enc = getattr(batch, 'graph_hom')
        elif hasattr(batch, 'counts_spasm'):
            count_enc = getattr(batch, 'counts_spasm')

        if self.raw_norm != None:
            count_enc = self.raw_norm(count_enc)

        #apply trig encoder
        if self.use_trig:
            count_enc = self.trig_enc(count_enc)

        #apply main MLP encoder
        if self.use_mlp:
            count_enc = self.encoder(count_enc)

        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        
        batch.x = torch.cat((h, count_enc), 1)

        if self.pass_as_var:
            setattr(batch, 'count_emb', count_enc)

        return batch


@register_network('GraphCountEnc') #registering as a network because torch_geometric.graphgym.register does not support "graph_encoder"
class MLPGraphCountEncoder(torch.nn.Module):
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()

        if hasattr(cfg, "ctenc_GraphCountEnc"):
            c = cfg.ctenc_GraphCountEnc
        else:
            raise Exception('Give ctenc_GraphCountEnc')
        self.dim_in = c.dim_ogct
        self.dim_hid = c.dim_hid
        self.dim_out = c.dim_ct
        self.L = c.n_layers
        self.dropout = c.dropout
        self.norm = c.norm

        if c.raw_bn:
            self.raw_norm = nn.BatchNorm1d(dim_ogct)
        else:
            self.raw_norm = None

        self.encoder = MLP(in_channels=self.dim_in, hidden_channels=self.dim_hid, out_channels=self.dim_out, num_layers=self.L, dropout=self.dropout, norm=self.norm)

    def forward(self, batch):
        assert hasattr(batch, "graph_hom")
        if tuple(batch.graph_hom.size())[0] == tuple(batch.y.size())[0]:#if graph_hom file is truly graph level (pre-pooled)
            homs = batch.graph_hom
        elif tuple(batch.graph_hom.size())[0] == tuple(batch.x.size())[0]:#if graph_hom file really has node level counts, we should pool to get graph level
            homs = global_add_pool(batch.graph_hom, batch.batch)
            assert tuple(homs.size())[0] == tuple(batch.y.size())[0]

        if self.raw_norm != None:
            homs = self.raw_norm(homs)
        homs = self.encoder(homs)

        batch.graph_hom = homs

        return batch

@register_node_encoder('NodeCountEncX2') #for encoding both counts_spasm and counts_wl (use in conjunction with dataset.format PyG-ZINC-Spasm+WLtree)
class MLPNodeCountEncoderX2(torch.nn.Module): #Has two seperate MLPs for processing spasm counts and wl counts for the sake of consistency with other composed encoders, could try one big MLP with width spasm_width+wl_width to allow for mixing of count info
    def __init__(self, dim_emb, expand_x=True):
        super().__init__()

        dim_in = cfg.share.dim_in
        dim_ogct_s = cfg.ctenc_NodeCountEncX2.spasm.dim_ogct
        dim_ct_s = cfg.ctenc_NodeCountEncX2.spasm.dim_ct
        dim_ogct_w = cfg.ctenc_NodeCountEncX2.wl.dim_ogct
        dim_ct_w = cfg.ctenc_NodeCountEncX2.wl.dim_ct
        dim_ct = cfg.ctenc_NodeCountEncX2.dim_ct

        assert dim_ct == dim_ct_s+dim_ct_w, "dim_ct_s+dim_ct_w must equal dim_ct"
        if dim_emb - dim_ct < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"Count dim size {dim_ct} is too large for "
                             f"desired embedding size of {dim_emb}.")
        
        if expand_x and dim_emb - dim_ct > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_ct)
        self.expand_x = expand_x and dim_emb - dim_ct > 0
        
        if cfg.ctenc_NodeCountEncX2.spasm.raw_bn:
            self.raw_norm_s = nn.BatchNorm1d(dim_ogct_s)
        else:
            self.raw_norm_s = None
        if cfg.ctenc_NodeCountEncX2.wl.raw_bn:
            self.raw_norm_w = nn.BatchNorm1d(dim_ogct_w)
        else:
            self.raw_norm_w = None

        self.encoder_s = MLP(in_channels=dim_ogct_s, hidden_channels=cfg.ctenc_NodeCountEncX2.spasm.dim_hid, out_channels=dim_ct_s, num_layers=cfg.ctenc_NodeCountEncX2.spasm.n_layers, dropout=cfg.ctenc_NodeCountEncX2.spasm.dropout, norm=cfg.ctenc_NodeCountEncX2.spasm.norm)
        self.encoder_w = MLP(in_channels=dim_ogct_w, hidden_channels=cfg.ctenc_NodeCountEncX2.wl.dim_hid, out_channels=dim_ct_w, num_layers=cfg.ctenc_NodeCountEncX2.wl.n_layers, dropout=cfg.ctenc_NodeCountEncX2.wl.dropout, norm=cfg.ctenc_NodeCountEncX2.wl.norm)

    def forward(self, batch):
        assert hasattr(batch, 'counts_spasm') and hasattr(batch, 'counts_wl'), "Missing counts"
        count_s = getattr(batch, 'counts_spasm')
        count_w = getattr(batch, 'counts_wl')

        if self.raw_norm_s != None:
            count_s = self.raw_norm_s(count_s)
        count_s = self.encoder_s(count_s)
        if self.raw_norm_w != None:
            count_w = self.raw_norm_w(count_w)
        count_w = self.encoder_w(count_w)

        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        
        batch.x = torch.cat([h, count_w, count_s], 1)

        return batch


@register_node_encoder('NodeCountSum') #never use this alone, you must use otherEncoders+NodeCountSum
class NodeCountSum(torch.nn.Module):

    def __init__(self, dim_emb):
        super().__init__()

        dim_ogct = cfg.ctenc_NodeCountSum.wl.dim_ogct #original input dim of WLtree counts
        dim_h = cfg.ctenc_NodeCountSum.dim_h #output dim of the MLP which encoders the WLtree counts (which is also used as the input dim for the sum_encoder)
        self.post_sum_layers = cfg.ctenc_NodeCountSum.sum.n_layers

        # Trig Encoder pre MLP
        self.use_mlp = True #use MLP by default
        self.use_trig = False
        if hasattr(cfg.ctenc_NodeCountSum, "trig_enc"):
            tcfg = cfg.ctenc_NodeCountSum.trig_enc
            if tcfg.use:
                self.use_trig = True
                self.use_mlp = False if hasattr(tcfg, "post_trig_fc") and not tcfg.post_trig_fc else True #turn off MLP if post_trig_fc attribute exists in cfg and is set tp false
                self.trig_enc = trig_encoder(tcfg.d)
                dim_ogct = dim_ogct * tcfg.d #scale up the MLP's input dimension

        grad_width_change = cfg.ctenc_NodeCountSum.wl.grad_width_change if hasattr(cfg.ctenc_NodeCountSum.wl, "grad_width_change") else False

        assert (dim_h == dim_emb) or (self.use_trig and not self.use_mlp and dim_ogct == dim_emb), "output of sum_encoder must have same dim as node embedding dim of main module"
        # if dim_emb - dim_ct < 0: # formerly 1, but you could have zero feature size
        #     raise ValueError(f"Count dim size {dim_ct} is too large for "
        #                     f"desired embedding size of {dim_emb}.")            
        
        if cfg.ctenc_NodeCountSum.wl.raw_bn:
            self.raw_norm_wl = nn.BatchNorm1d(dim_ogct)
        else:
            self.raw_norm_wl = None
        if cfg.ctenc_NodeCountSum.sum.raw_bn:
            self.raw_norm_sum = nn.BatchNorm1d(dim_h)
        else:
            self.raw_norm_sum = None
        
        if self.use_mlp:
            if hasattr(cfg.ctenc_NodeCountSum.wl, "norm") and cfg.ctenc_NodeCountSum.wl.norm != 'none' and cfg.ctenc_NodeCountSum.wl.norm != 'None':
                mlp_norm_wl = cfg.ctenc_NodeCountSum.wl.norm
            else:
                mlp_norm_wl = None
            if grad_width_change:
                self.wl_enc = MLP(channel_list=compute_gradual_width(dim_ogct, dim_h, cfg.ctenc_NodeCountSum.wl.n_layers), dropout=cfg.ctenc_NodeCountSum.wl.dropout, norm=mlp_norm_wl)
            else:
                self.wl_enc = MLP(in_channels=dim_ogct, hidden_channels=cfg.ctenc_NodeCountSum.wl.dim_hid, out_channels=dim_h, num_layers=cfg.ctenc_NodeCountSum.wl.n_layers, dropout=cfg.ctenc_NodeCountSum.wl.dropout, norm=mlp_norm_wl)
        if self.post_sum_layers > 0:
            if hasattr(cfg.ctenc_NodeCountSum.sum, "norm") and cfg.ctenc_NodeCountSum.sum.norm != 'none' and cfg.ctenc_NodeCountSum.sum.norm != 'None':
                mlp_norm_s = cfg.ctenc_NodeCountSum.sum.norm
            else:
                mlp_norm_s = None
            if self.use_mlp:
                self.encoder_s = MLP(in_channels=dim_h, hidden_channels=dim_h, out_channels=dim_h, num_layers=self.post_sum_layers, dropout=cfg.ctenc_NodeCountSum.sum.dropout, norm=mlp_norm_s) #flat MLP
            else:
                self.encoder_s = MLP(in_channels=dim_ogct, hidden_channels=dim_ogct, out_channels=dim_ogct, num_layers=self.post_sum_layers, dropout=cfg.ctenc_NodeCountSum.sum.dropout, norm=mlp_norm_s) #flat MLP

    def forward(self, batch):

        if hasattr(batch, "counts_wl"):
            wl_counts = batch.counts_wl
        elif hasattr(batch, 'counts'):
            wl_counts = batch.counts
        else:
            raise Exception("Batch must include WLtree counts")

        h = batch.x

        if self.use_trig:
            wl_counts = self.trig_enc(wl_counts)

        if self.use_mlp:
            wl_counts = self.wl_enc(wl_counts)

        x = wl_counts + h
        if self.post_sum_layers > 0:
            x = self.encoder_s(x)
        batch.x = x

        return batch