import sys
sys.path.append("./src")

from problems import *
from solver import Controller
import jax
import argparse

parser = argparse.ArgumentParser(description='Main Run Script')
parser.add_argument('-c','--case', help='Case to Run (hjb, bsb, bz)', required=True,type=str)
parser.add_argument('-l','--loss', help='Loss method (pinns, fspinns, bsde, bsdeheun, bsdeskip, bsdeheunskip)', required=True,type=str)
parser.add_argument('-f','--float',help="Float 32 vs 64 (0=f32, 1=f64(default))",type=int,default=1)
parser.add_argument('-d','--disc',help="Trajectory length, L (T/L = dt) (default = 50)",type=int,default=50)
parser.add_argument('-s','--skip',help="Trajectory skip length, only for skip losses (default=5)",type=int,default=5)

args = vars(parser.parse_args())

# Enabling f64 calculations
if args["float"]==1:
    print("f64 Enabled")
    jax.config.update("jax_enable_x64", True)

# Choosing Case for training
def choose_case(case):
    match case:
        case "hjb":
            print("case: hjb")
            return HJB_Solver
        case "bsb":
            print("case: bsb")
            return BSB_Solver
        case "bz":
            print("case: bz")
            return BZ_Solver
        case _:
            Exception("Invalid Case")
Case_Solver = choose_case(args["case"])
print("trajectory length: "+str(args["disc"]))
config = Case_Solver.get_base_config(traj_len=args["disc"])

# Option to save results to weights & biases
config.save_to_wandb = True

# Learning rate schedule
lr = [1e-3,1e-4,1e-5]
iter = [50000,25000,25000]

# Random seed for initialization & SDE rollouts
seed = 1234

# Setting config for run
print("loss method: "+str(args["loss"]))
config.loss_method = args["loss"]
config.batch_pde = 64
print("skip length: "+str(args["skip"]))
config.skip_len = args["skip"]
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

if args["float"] == 64:
    jax.config.update("jax_enable_x64", False)