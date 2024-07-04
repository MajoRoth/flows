from pathlib import Path
from trainers.normalizing_flow_trainer import NormalizingFlowTrainer
from trainers.flow_matching_trainer import FlowMatchingTrainer
from trainers.flow_matching_cond_trainer import FlowMatchingCondTrainer




def get_trainer(cfg):
    if cfg.model.name == "normalizing_flow":
        return NormalizingFlowTrainer
    if cfg.model.name == "flow_matching" and not cfg.data.conditioned:
        return FlowMatchingTrainer
    if cfg.model.name == "flow_matching" and cfg.data.conditioned:
        return FlowMatchingCondTrainer
    else:
        raise Exception(f"Model {cfg.model.name} not found")



