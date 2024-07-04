from models.normalizing_flow import NormalizingFlow
from models.flow_matching import FlowMatching
from models.flow_matching_cond import FlowMatchingCond


def get_model(cfg):
    if cfg.model.name == "normalizing_flow":
        return NormalizingFlow(layer_dim=2, num_layers=cfg.model.num_layers)
    if cfg.model.name == "flow_matching" and not cfg.data.conditioned:
        return FlowMatching(layer_dim=2, dt=cfg.model.dt)
    if cfg.model.name == "flow_matching" and cfg.data.conditioned:
        return FlowMatchingCond(layer_dim=2, dt=cfg.model.dt)
    else:
        raise Exception(f"Model {cfg.model.name} not found")



