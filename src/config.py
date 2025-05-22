from dataclasses import dataclass
from typing import ClassVar
from jax import Array
from jax import numpy as jnp

@dataclass
class Config():
    case: str = 'base'
    # Model Def
    d_in: int = 1 # not including t
    d_out: int = 1
    d_hidden: int = 64
    num_layers: int = 8
    activation: str = 'swish'
    four_emb: bool = True
    emb_dim: int = 256
    emb_scale: float = 1
    skip_conn: bool = True
    save_layers: tuple = (1,3,5,7)
    skip_layers: tuple = (3,5,7,9)

    # Domain Def
    x_range: tuple = ((-1,1),)
    t_range: tuple = (0,1)
    plot_range: tuple = (-1,1)

    # Training Def
    batch_pde: int = 256
    batch_bc: int = 256
    optim: str = 'adam'
    lr: float = 1e-3
    iter: int = 250000
    loss_method: str = 'pinns'
    additional_losses: bool = False
    
    # PINNS Loss Def
    pde_scale: float = 1
    ic_scale: float = 1

    # BSDE Loss Def
    traj_len: int = 50
    delta_t: float = 1e-2
    reset_u: bool = True
    skip_len: int = 5
 
    #extras
    save_to_wandb: bool = False
    track_pinns_loss: bool = False
    track_bsde_loss: bool = False
    track_fspinns_loss: bool = False
    checkpointing: bool = False
    save_sol: bool = False
    ref_sol: bool = True
    custom_eval: bool = False
    periodic: bool = False

    def get_train_config(self):
        return TrainConfig(batch_pde=self.batch_pde,
                           batch_bc=self.batch_bc,
                           optim=self.optim,
                           lr=self.lr,
                           iter=self.iter,
                           loss_method=self.loss_method,
                           pde_scale=self.pde_scale,
                           ic_scale=self.ic_scale,
                           traj_len=self.traj_len,
                           delta_t=self.delta_t)

@dataclass
class TrainConfig():
    # Training Def
    batch_pde: int = 256
    batch_bc: int = 256
    optim: str = 'adam'
    lr: float = 1e-3
    iter: int = 250000
    loss_method: str = 'pinns'

    # PINNS Loss Def
    pde_scale: float = 1
    bc_scale: float = 1
    ic_scale: float = 1
    bsde_scale: float = 1
    traj_scale:float = 1

    # BSDE Loss Def
    traj_len: int = 50
    delta_t: float = 1e-2