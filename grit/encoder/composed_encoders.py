import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.encoder import AtomEncoder
from torch_geometric.graphgym.register import register_node_encoder

from grit.encoder.ast_encoder import ASTNodeEncoder
from grit.encoder.kernel_pos_encoder import RWSENodeEncoder, \
    HKdiagSENodeEncoder, ElstaticSENodeEncoder
from grit.encoder.laplace_pos_encoder import LapPENodeEncoder
from grit.encoder.ppa_encoder import PPANodeEncoder
from grit.encoder.signnet_pos_encoder import SignNetNodeEncoder
from grit.encoder.voc_superpixels_encoder import VOCNodeEncoder
from grit.encoder.type_dict_encoder import TypeDictNodeEncoder
from grit.encoder.linear_node_encoder import LinearNodeEncoder
from grit.encoder.equivstable_laplace_pos_encoder import EquivStableLapPENodeEncoder

#for homcounts #ADDITION
from grit.encoder.MLP_count_encoder import MLPNodeCountEncoder, MLPGraphCountEncoder, MLPNodeCountEncoderX2, NodeCountSum


def concat_node_encoders(encoder_classes, pe_enc_names):
    """
    A factory that creates a new Encoder class that concatenates functionality
    of the given list of two or three Encoder classes. First Encoder is expected
    to be a dataset-specific encoder, and the rest PE Encoders.

    Args:
        encoder_classes: List of node encoder classes
        pe_enc_names: List of PE embedding Encoder names, used to query a dict
            with their desired PE embedding dims. That dict can only be created
            during the runtime, once the config is loaded.

    Returns:
        new node encoder class
    """

    class Concat2NodeEncoder(torch.nn.Module):
        """Encoder that concatenates two node encoders.
        """
        enc1_cls = None
        enc2_cls = None
        enc2_name = None

        def __init__(self, dim_emb):
            super().__init__()
            
            if cfg.posenc_EquivStableLapPE.enable: # Special handling for Equiv_Stable LapPE where node feats and PE are not concat
                self.encoder1 = self.enc1_cls(dim_emb)
                self.encoder2 = self.enc2_cls(dim_emb)
            else:
                # PE dims can only be gathered once the cfg is loaded.
                enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe if hasattr(cfg, f"posenc_{self.enc2_name}") else getattr(cfg, f"ctenc_{self.enc2_name}").dim_ct
                
                # scale the output dimension up if we use a trig_encoder (without using an MLP after to downproject back to dim_ct)
                if hasattr(cfg, f"ctenc_{self.enc2_name}") and hasattr(getattr(cfg, f"ctenc_{self.enc2_name}"), "trig_enc") and getattr(cfg, f"ctenc_{self.enc2_name}").trig_enc.use and not getattr(cfg, f"ctenc_{self.enc2_name}").trig_enc.post_trig_fc:
                    enc_cfg = getattr(cfg, f"ctenc_{self.enc2_name}")
                    d = enc_cfg.trig_enc.d
                    #find the OG count input dim (which will be scaled by d)
                    if hasattr(enc_cfg, "dim_ogct"): #in the case we are just using a usual NodeCountEnc
                        input_dim = enc_cfg.dim_ogct
                    enc2_dim_pe = input_dim * d

                if (not hasattr(cfg, f"ctenc_{self.enc2_name}") or not hasattr(getattr(cfg, f"ctenc_{self.enc2_name}"), 'stack_on_h')) and (not hasattr(cfg, f"posenc_{self.enc2_name}.stack_on_h") or not hasattr(getattr(cfg, f"posenc_{self.enc2_name}"), 'stack_on_h')):
                    self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe)
                elif hasattr(cfg, f"ctenc_{self.enc2_name}") and hasattr(getattr(cfg, f"ctenc_{self.enc2_name}"), 'stack_on_h'):
                    self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe) if getattr(getattr(cfg, f"ctenc_{self.enc2_name}"), "stack_on_h") == False else self.enc1_cls(dim_emb)
                elif hasattr(cfg, f"posenc_{self.enc2_name}") and hasattr(getattr(cfg, f"posenc_{self.enc2_name}"), 'stack_on_h'):
                    self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe) if getattr(getattr(cfg, f"posenc_{self.enc2_name}"), "stack_on_h") == False else self.enc1_cls(dim_emb)

                self.encoder2 = self.enc2_cls(dim_emb, expand_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            return batch

    class Concat3NodeEncoder(torch.nn.Module):
        """Encoder that concatenates three node encoders.
        """
        enc1_cls = None
        enc2_cls = None
        enc2_name = None
        enc3_cls = None
        enc3_name = None

        def __init__(self, dim_emb):
            super().__init__()
            # PE dims can only be gathered once the cfg is loaded.
            enc2_dim_pe = getattr(cfg, f"posenc_{self.enc2_name}").dim_pe if hasattr(cfg, f"posenc_{self.enc2_name}") else getattr(cfg, f"ctenc_{self.enc2_name}").dim_ct
            enc3_dim_pe = getattr(cfg, f"posenc_{self.enc3_name}").dim_pe if hasattr(cfg, f"posenc_{self.enc3_name}") else getattr(cfg, f"ctenc_{self.enc3_name}").dim_ct

            # scale the output dimension up if we use a trig_encoder (without using an MLP after to downproject back to dim_ct)
            if hasattr(cfg, f"ctenc_{self.enc2_name}") and hasattr(getattr(cfg, f"ctenc_{self.enc2_name}"), "trig_enc") and getattr(cfg, f"ctenc_{self.enc2_name}").trig_enc.use and not getattr(cfg, f"ctenc_{self.enc2_name}").trig_enc.post_trig_fc:
                enc_cfg = getattr(cfg, f"ctenc_{self.enc2_name}")
                d = enc_cfg.trig_enc.d
                #find the OG count input dim (which will be scaled by d)
                if hasattr(enc_cfg, "dim_ogct"): #in the case we are just using a usual NodeCountEnc
                    input_dim = enc_cfg.dim_ogct
                enc2_dim_pe = input_dim * d
            if hasattr(cfg, f"ctenc_{self.enc3_name}") and hasattr(getattr(cfg, f"ctenc_{self.enc3_name}"), "trig_enc") and getattr(cfg, f"ctenc_{self.enc3_name}").trig_enc.use and not getattr(cfg, f"ctenc_{self.enc3_name}").trig_enc.post_trig_fc:
                enc_cfg = getattr(cfg, f"ctenc_{self.enc3_name}")
                d = enc_cfg.trig_enc.d
                #find the OG count input dim (which will be scaled by d)
                if hasattr(enc_cfg, "dim_ogct"): #in the case we are just using a usual NodeCountEnc
                    input_dim = enc_cfg.dim_ogct
                enc3_dim_pe = input_dim * d

            if (hasattr(cfg, f"ctenc_{self.enc2_name}") and hasattr(getattr(cfg, f"ctenc_{self.enc2_name}"),"stack_on_h" ) and getattr(cfg, f"ctenc_{self.enc2_name}").stack_on_h == True) and (hasattr(cfg, f"posenc_{self.enc3_name}") and hasattr(getattr(cfg, f"posenc_{self.enc3_name}"), 'stack_on_h') and getattr(cfg, f"posenc_{self.enc3_name}").stack_on_h == True):
                dim1 = dim_emb
                dim2 = dim_emb
                dim3 = dim_emb
            elif (hasattr(cfg, f"ctenc_{self.enc3_name}") and hasattr(getattr(cfg, f"ctenc_{self.enc3_name}"),"stack_on_h" ) and getattr(cfg, f"ctenc_{self.enc3_name}.stack_on_h") == True) and (hasattr(cfg, f"posenc_{self.enc2_name}.stack_on_h") and hasattr(getattr(cfg, f"posenc_{self.enc2_name}"),"stack_on_h" ) and getattr(cfg, f"posenc_{self.enc2_name}").stack_on_h == True):
                dim1 = dim_emb
                dim2 = dim_emb
                dim3 = dim_emb
            else:
                dim1 = dim_emb - enc2_dim_pe - enc3_dim_pe
                dim2 = dim_emb - enc3_dim_pe
                dim3 = dim_emb

            self.encoder1 = self.enc1_cls(dim1)
            self.encoder2 = self.enc2_cls(dim2, expand_x=False)
            self.encoder3 = self.enc3_cls(dim3, expand_x=False)
            # self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe - enc3_dim_pe)
            # self.encoder2 = self.enc2_cls(dim_emb - enc3_dim_pe, expand_x=False)
            # self.encoder3 = self.enc3_cls(dim_emb, expand_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            batch = self.encoder3(batch)
            return batch

    # Configure the correct concatenation class and return it.
    if len(encoder_classes) == 2:
        Concat2NodeEncoder.enc1_cls = encoder_classes[0]
        Concat2NodeEncoder.enc2_cls = encoder_classes[1]
        Concat2NodeEncoder.enc2_name = pe_enc_names[0]
        return Concat2NodeEncoder
    elif len(encoder_classes) == 3:
        Concat3NodeEncoder.enc1_cls = encoder_classes[0]
        Concat3NodeEncoder.enc2_cls = encoder_classes[1]
        Concat3NodeEncoder.enc3_cls = encoder_classes[2]
        Concat3NodeEncoder.enc2_name = pe_enc_names[0]
        Concat3NodeEncoder.enc3_name = pe_enc_names[1]
        return Concat3NodeEncoder
    else:
        raise ValueError(f"Does not support concatenation of "
                         f"{len(encoder_classes)} encoder classes.")

#EDITED: added a factory which adds the encoders that sum WL_full with the composed-encoder embeddings of node label and pe
def add_WLfembed_to_encoders(encoder_classes, pe_enc_names):

    class WLf_sum_encoder(torch.nn.Module):

        composed_enc=None
        WLtree_enc=None

        def __init__(self, dim_emb):
            super().__init__()
            self.comp_enc = self.composed_enc(dim_emb)
            self.wl_enc = self.WLtree_enc(dim_emb)

        def forward(self, batch):
            batch = self.comp_enc(batch)
            batch = self.wl_enc(batch)
            return batch

    # get the main (composed) encoder module
    if len(encoder_classes) == 1: #composed_encoder should just be the dataset specific (initial node label embedder) encoder
        WLf_sum_encoder.composed_enc = encoder_classes[0]
    else: #composed encoder should be a concatenation of dataset specific encoders along with some positional encoders
        WLf_sum_encoder.composed_enc = concat_node_encoders(encoder_classes, pe_enc_names)
    WLf_sum_encoder.WLtree_enc = NodeCountSum

    return WLf_sum_encoder


# Dataset-specific node encoders.
ds_encs = {'Atom': AtomEncoder,
           'ASTNode': ASTNodeEncoder,
           'PPANode': PPANodeEncoder,
           'TypeDictNode': TypeDictNodeEncoder,
           'VOCNode': VOCNodeEncoder,
           'LinearNode': LinearNodeEncoder}

# Positional Encoding node encoders.
pe_encs = {'LapPE': LapPENodeEncoder,
           'RWSE': RWSENodeEncoder,
           'HKdiagSE': HKdiagSENodeEncoder,
           'ElstaticSE': ElstaticSENodeEncoder,
           'SignNet': SignNetNodeEncoder,
           'EquivStableLapPE': EquivStableLapPENodeEncoder,}

# Count Encoding node encoders.
ct_encs = {'NodeCountEnc': MLPNodeCountEncoder,
           'GraphCountEnc': MLPGraphCountEncoder,
           'NodeCountEncX2': MLPNodeCountEncoderX2}

# Concat dataset-specific and PE encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for pe_enc_name, pe_enc_cls in pe_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{pe_enc_name}",
            concat_node_encoders([ds_enc_cls, pe_enc_cls],
                                 [pe_enc_name])
        )

# Combine both LapPE and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+LapPE+RWSE",
        concat_node_encoders([ds_enc_cls, LapPENodeEncoder, RWSENodeEncoder],
                             ['LapPE', 'RWSE'])
    )

# Combine both SignNet and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+SignNet+RWSE",
        concat_node_encoders([ds_enc_cls, SignNetNodeEncoder, RWSENodeEncoder],
                             ['SignNet', 'RWSE'])
    )

# Concat dataset-specific and count encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for ct_enc_name, ct_enc_cls in ct_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{ct_enc_name}",
            concat_node_encoders([ds_enc_cls, ct_enc_cls],
                                 [ct_enc_name])
        )

# Combine counts with RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for ct_enc_name, ct_enc_cls in ct_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{ct_enc_name}+RWSE",
            concat_node_encoders([ds_enc_cls, ct_enc_cls, RWSENodeEncoder],
                                [ct_enc_name, 'RWSE'])
        )

# WLtree sum encoders:

# Sum WL with dataset-specific encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+NodeCountSum",
            add_WLfembed_to_encoders([ds_enc_cls],
                                 [pe_enc_name])
        )

# Sum WL with dataset-specific and PE encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for pe_enc_name, pe_enc_cls in pe_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{pe_enc_name}+NodeCountSum",
            add_WLfembed_to_encoders([ds_enc_cls, pe_enc_cls],
                                 [pe_enc_name])
        )

# Sum WL with (ds encoder and) both LapPE and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+LapPE+RWSE+NodeCountSum",
        add_WLfembed_to_encoders([ds_enc_cls, LapPENodeEncoder, RWSENodeEncoder],
                             ['LapPE', 'RWSE'])
    )

# Sum WL with dataset-specific and count encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for ct_enc_name, ct_enc_cls in ct_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{ct_enc_name}+NodeCountSum",
            add_WLfembed_to_encoders([ds_enc_cls, ct_enc_cls],
                                 [ct_enc_name])
        )

# Sum WL with (ds and) counts with RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for ct_enc_name, ct_enc_cls in ct_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{ct_enc_name}+RWSE+NodeCountSum",
            add_WLfembed_to_encoders([ds_enc_cls, ct_enc_cls, RWSENodeEncoder],
                                [ct_enc_name, 'RWSE'])
        )