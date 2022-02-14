from models import pointnet, pointnet2

def build_extractor(config):
    if config.TYPE == 'pointnet':
        model = pointnet.PointNetfeat_lite(
            feat_dim=config.feat_dim, hidden_size=config.hidden_size, global_feat=config.global_feat
        )
    elif config.TYPE == 'pointnet2':
        model = pointnet2.pointnet2_feat_msg(
            feat_dim=config.feat_dim, hidden_size=config.hidden_size
        )
    else:
        raise NotImplementedError(f"For model type {config.TYPE}")
    return model