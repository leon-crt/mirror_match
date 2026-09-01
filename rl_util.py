from torch import nn
from model import ResBlockMLP
from torchrl.modules import ProbabilisticActor, ValueOperator, LSTMModule
from torchrl.envs.transforms import Transform
from torch.distributions import Bernoulli, Independent
from tensordict.nn import TensorDictModule, TensorDictSequential
import torch

MAX_X = 928
MIN_X = 93
MAX_Y = 226
MIN_Y = -42

HOST = "127.0.0.1"
PORT = 42069

# lstm initialization
device = torch.device(0 if torch.cuda.is_available() else 'cpu')
hidden_size = 512
out_size = 10 # number of pressable buttons same as targets
threshold = 0.3
input_size = 26
num_layers = 2
num_blocks= 1

class InitZeroState(Transform):
    """Initializes specified keys with zero tensors on env.reset()."""
    def __init__(self, keys: list, feature_dims: int):
        super().__init__()
        self.keys = keys
        self.feature_dims = feature_dims

    def _reset(self, tensordict, tensordict_reset):
        # Infer device and batch size directly from the reset TensorDict
        batch_shape = tensordict_reset.shape
        device = tensordict_reset.device

        zeros = []
        for feature_dim in self.feature_dims:
            zeros.append(torch.zeros(
                (*batch_shape, feature_dim), 
                device=device, 
                dtype=torch.float32
            ))
        for i in range(len(self.keys)):
            tensordict_reset.set(self.keys[i], zeros[i])

        return tensordict_reset

class MaskInitState(nn.Module):
    """Zeros out previous outputs whenever is_init is True."""
    def forward(self, prev_output: torch.Tensor, is_init: torch.Tensor) -> torch.Tensor:
        # Ensure is_init matches dimensions for broadcasting (*batch_shape, 1) -> (*batch_shape, feature_dim)
        if is_init.dim() < prev_output.dim():
            is_init = is_init.unsqueeze(-1)
        
        # Replace prev_output values with zeros where is_init is True
        return torch.where(is_init, torch.zeros_like(prev_output), prev_output)

class IndependentBernoulli(Independent):
    def __init__(self, probs=None, logits=None):
        base_dist = Bernoulli(probs=probs, logits=logits)
        super().__init__(base_dist, reinterpreted_batch_ndims=1)

def recurrent_body(prefix, state_dict_mlp=None, state_dict_lstm=None):
    input_mlp = TensorDictModule(
                module=nn.Sequential(
                    nn.Linear(input_size, 4*input_size),
                    nn.ReLU(),
                    nn.Linear(4 * input_size, hidden_size)
                ),
                in_keys=["observation"],
                out_keys=[f"{prefix}_embed"],
            )
    LSTM = LSTMModule(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                in_keys=[f"{prefix}_embed", f"{prefix}_rs", f"{prefix}_rc", "is_init"],
                out_keys=[f"{prefix}_features", ("next", f"{prefix}_rs"), ("next", f"{prefix}_rc")],
                recurrent_backend="auto",
            )
    if state_dict_lstm != None:
        LSTM.lstm.load_state_dict(state_dict_lstm)
    if state_dict_mlp != None:
        input_mlp.module.load_state_dict(state_dict_mlp)

    return TensorDictSequential(
        input_mlp,
        LSTM
    )

def transpose_weights_nn_to_rl(checkpoint, model):
    action_head_net = nn.Sequential(*[ResBlockMLP(hidden_size, hidden_size) for _ in range(num_blocks)])

    action_head = TensorDictModule(
        module=action_head_net,
        in_keys=["actor_features"],
        out_keys=["action_head_out"]
    )

    # Output layer
    lin_out_layer = nn.Linear(hidden_size, out_size)
    fc_out_pol_net = nn.Sequential(nn.ReLU(), lin_out_layer, nn.Sigmoid())
    fc_out_pol = TensorDictModule(
        module=fc_out_pol_net,
        in_keys=["action_head_out"],
        out_keys=["probs"]
    )

    actor_rec = recurrent_body("actor")

    policy_module = ProbabilisticActor(
        module=TensorDictSequential(
            actor_rec,
            action_head,
            fc_out_pol
        ),
        in_keys=["probs"],
        distribution_class=IndependentBernoulli,
        return_log_prob=True,
    )


    # 1. Load checkpoint into policy_module
    policy_module.load_state_dict(checkpoint["model_state_dict"])

    # 2. Extract components using exact sub-module paths:

    # A. Input MLP (actor_rec -> input_mlp -> inner nn.Sequential)
    model.input_mlp.load_state_dict(
        policy_module.module[0][0].module[0].module.state_dict()
    )

    # B. LSTM Core (actor_rec -> LSTMModule -> inner nn.LSTM)
    model.lstm.load_state_dict(
        policy_module.module[0][0].module[1].lstm.state_dict()
    )

    # C. Residual Blocks (action_head -> inner nn.Sequential)
    model.res_blocks.load_state_dict(
        policy_module.module[0][1].module.state_dict()
    )

    # D. Linear Output Head (fc_out_pol -> inner nn.Sequential -> index 1 nn.Linear)
    model.fc_out.load_state_dict(
        policy_module.module[0][2].module[1].state_dict()
    )

    return model