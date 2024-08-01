REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .updet_agent import UPDeT
REGISTRY['updet'] = UPDeT

from .phase_updet_agent import PhaseUPDeT1
REGISTRY['phase_updet1'] = PhaseUPDeT1

from .phase_updet_agent import PhaseUPDeT2
REGISTRY['phase_updet2'] = PhaseUPDeT2

from .phase_updet_agent import PhaseUPDeT3
REGISTRY['phase_updet3'] = PhaseUPDeT3

from .transformer_agg_agent import TransformerAggregationAgent
REGISTRY['transformer_aggregation'] = TransformerAggregationAgent