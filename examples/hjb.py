from src.problems import *
from src.solver import Controller
import jax

# Enabling f64 calculations
jax.config.update("jax_enable_x64", True)

# Choosing HJB for training
Case_Solver = HJB_Solver

config = Case_Solver.get_base_config()

# Option to save results to weights & biases
config.save_to_wandb = True

# Learning rate schedule
lr = [1e-3,1e-4,1e-5]
iter = [50000,25000,25000]

# Random seed for initialization & SDE rollouts
seed = 1234

# Setting config for run
config.loss_method = "bsdeheun"
config.batch_pde = 64

# Additional Configs for LR scheduling
config.additional_losses = True
config.lr = lr[0]
config.iter = iter[0]
config2 = config.get_train_config()
config2.lr = lr[1]
config2.iter = iter[1]
config3 = config.get_train_config()
config3.lr = lr[2]
config3.iter = iter[2]

# Initialize and run solver & controller
svr = Case_Solver(config)
ctr = Controller(svr,seed=seed)
ctr.append_train_config(config2)
ctr.append_train_config(config3)
ctr.solve()