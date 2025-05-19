from problems import *
from solver import Controller
import jax

jax.config.update("jax_enable_x64", True)

Case_Solver = Pendulum_Solver

config = Case_Solver.get_base_config()

config.save_to_wandb = True

lr1 = 1e-4
lr2 = 1e-5
lr3 = 1e-5
iter1 = 50000
iter2 = 25000
iter3 = 25000

tag = "pendulum test"
seed = 1234

config.loss_method = "bsde"
config.batch_pde = 64
config.additional_losses = True
config.lr = lr1
config.iter = iter1
config2 = config.get_train_config()
config2.lr = lr2
config2.iter = iter2
config3 = config.get_train_config()
config3.lr = lr3
config3.iter = iter3

svr = Case_Solver(config)
svr.wandb_tags(tag)
ctr = Controller(svr,seed=seed)
ctr.append_train_config(config2)
ctr.append_train_config(config3)
ctr.solve()

# Heun
config.loss_method = "bsdeheun"
config.batch_pde = 64
config.additional_losses = True
config.lr = lr1
config.iter = iter1
config2 = config.get_train_config()
config2.lr = lr2
config2.iter = iter2
config3 = config.get_train_config()
config3.lr = lr3
config3.iter = iter3

svr = Case_Solver(config)
svr.wandb_tags(tag)
ctr = Controller(svr,seed=seed)
ctr.append_train_config(config2)
ctr.append_train_config(config3)
ctr.solve()

jax.config.update("jax_enable_x64", False)