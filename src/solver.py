import sys
import jax.flatten_util
sys.path.append('..')
sys.path.append('.')
import jax
import optax
from jax import numpy as jnp
import numpy as np
from flax import linen as nn
import tqdm
import wandb
import pandas
import matplotlib.pyplot as plt
from model import PINNs
from config import *
from utils import *
from functools import partial
import copy

class Solver():

    ### Init & PyTree ###

    # for tree flatten/unflattening
    def __init__(self,config: Config):
        self.config = copy.deepcopy(config)

        self.model = self.create_model()
        self.optimizer = self.create_opt()
        if self.config.ref_sol:
            self.sol,self.interp = self.get_ref_sol()
        if self.config.custom_eval:
            self.eval_point = self.get_eval_point()
        if self.config.save_to_wandb:
            self.init_wandb()
        
    def init_solver(self,key:Key):
        params = self.init_model(key)
        opt_state = self.init_opt(params)
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        if self.config.save_to_wandb:
            wandb.config['# Params'] =  num_params
        return params,opt_state
    
    ### Methods ###
    
    # Init Methods
    def create_model(self):
        return PINNs(self.config)
    
    def init_model(self,key: Key):
        x_pde,t_pde = self.sample_domain(key,self.config.batch_pde)
        return self.model.init(key.newkey(),x_pde,t_pde)

    def tab_model(self,key:Key):
        x_pde,t_pde = self.sample_domain(key,self.config.batch_pde)
        tab_fn = nn.tabulate(self.model,key.newkey())
        print(tab_fn(x_pde,t_pde))

    def create_opt(self):
        match self.config.optim:
            case "adam":
                return optax.inject_hyperparams(optax.adam)(learning_rate = self.config.lr)
            case _:
                raise Exception("Optimizer '"+self.config.optim+"' Not Implemented")
            
    def init_opt(self,params):
        return self.optimizer.init(params)
    
    def init_wandb(self):
        print("Initializing wandb")
        wandb.init(project="heunbsde",config=vars(self.config))

    def wandb_tags(self,tag):
        wandb.run.tags = wandb.run.tags + (tag,)

    def close(self):
        wandb.finish()

    def get_base_config():
        return Config(case='multi-dim')
    
    # Util Methods
    def sample_domain(self,key:Key,batch_size): # Initiates normal dist for x and uniform t
        x_pde = []
        for i in range(self.config.d_in):
            x_pde.append(2*jax.random.normal(key.newkey(),(batch_size,1)))
        x_pde = jnp.concatenate(x_pde,axis=-1)
        t_pde = jax.random.uniform(key.newkey(),
                                   (batch_size,1),
                                   minval=self.config.t_range[0],
                                   maxval=self.config.t_range[1])
        return x_pde,t_pde
    
    def sample_domain_bsde(self,key,batch_size):
        x_pde = []
        for i in range(self.config.d_in):
            x_pde.append(jnp.zeros((batch_size,1)))
        x_pde = jnp.concatenate(x_pde,axis=-1)
        t_pde = jnp.zeros((batch_size,1))
        return x_pde,t_pde
    
    def inbounds(self,x,t,config:Config):
        inbound = jnp.ones(len(x)).astype(bool)
        #print(inbound.shape)
        for i in range(config.d_in):
            inbound = inbound & (x[:,i]>=config.x_range[i][0]) & (x[:,i] <=config.x_range[i][1])
            #print(inbound.shape)
        inbound = inbound[...,jnp.newaxis]
        inbound = inbound & ((t)<=config.t_range[1])
        #print(inbound.shape)
        return inbound
    
    def plot_pred(self,params):
        pred = self.calc_u(params,self.sol[1][...,0:-1],self.sol[1][...,-1:])[...,0]
        fig = plt.figure(figsize=(4,3))
        plt.plot(self.sol[1][...,-1].T,pred.T,"k")
        plt.plot(self.sol[1][...,-1].T,self.sol[0].T,":k")
        plt.xlabel('t')
        plt.ylabel('u')
        plt.title('Prediction') 
        wandb.summary['Prediction'] = wandb.Image(fig)
        if self.config.save_sol:
            pred_values = pandas.DataFrame(np.array(jnp.reshape(jnp.stack([self.sol[1][...,-1].T,pred.T,self.sol[0].T]),(3,-1))))
            pred_values.to_csv(self.config.loss_method+"_"+str(self.config.reset_u)+"_save.csv", header=False, index=False)
    
    def plot_err(self,params):
        pred = self.calc_u(params,self.sol[1][...,0:-1],self.sol[1][...,-1:])[...,0]
        fig = plt.figure(figsize=(4,3))
        plt.plot(self.sol[1][...,-1].T,jnp.abs(self.sol[0]-pred).T,"k")
        plt.title('Error')
        plt.xlabel('t')
        plt.ylabel('u')
        wandb.summary['Error'] = wandb.Image(fig)
    
    def plot_eval(self,params):
        pass

    def get_ref_sol(self):
        u_ana = jax.jit(self.u_ana)
        num_traj = 5
        x = jnp.zeros((num_traj,51,self.config.d_in))
        x_init,_ = self.sample_domain_bsde(Key.create_key(1),num_traj)
        x = x.at[:,0,:].set(x_init)
        x = x.reshape(num_traj,51,self.config.d_in)
        w = jax.random.normal(jax.random.key(1),(num_traj,50,self.config.d_in))
        w = jnp.concatenate((jnp.zeros((num_traj,1,self.config.d_in)),w),axis=1)
        t = jnp.repeat(jnp.linspace(0,1,51)[jnp.newaxis,...,jnp.newaxis],num_traj,axis=0)
        for i in range(50):
            d_x = jnp.sqrt(.02)*jnp.matmul(self.sigma(x[:,i,:],u_ana(jnp.concatenate((x[:,i,:],t[:,i,:]),axis=-1))[...,jnp.newaxis]),w[:,i+1,:,jnp.newaxis])[...,0]
            x = x.at[:,i+1,:].set(x[:,i,:]+d_x)
        print(x.flatten().mean())
        print(x.flatten().std())
        xt = jnp.concatenate((x,t),axis=-1)
        u = u_ana(xt)
        return (u,xt),u_ana

    def get_eval_point(self):
        pass

    def select_loss(self,loss_method):
        match loss_method:
            case "pinns":
                loss_fn = self.pinns_loss
            case "bsde":
                loss_fn = self.bsde_loss
            case "bsdeskip":
                loss_fn = self.bsde_skip_loss
            case "bsdeheun":
                loss_fn = self.bsde_huen_loss
            case "bsdeheunskip":
                loss_fn = self.bsde_heun_skip_loss
            case "fspinns":
                loss_fn = self.fspinns_loss
            case "regress":
                loss_fn = self.reg_loss
            case _:
                raise Exception("Loss Method '" +loss_method+ "' Not Implemented")
        return loss_fn
    
    # Model Output Methods
    def calc_u(self,params,*args):
        return self.model.apply(params,*args)
    
    def calc_ut(self,params,x,t):
        # calculates wrt t, returns (u(batch,d_out),u_t(batch,d_out))
        def t_func(x,t):
            model_fn  = lambda t: self.model.apply(params,x,t)
            u,du_dt = jax.vjp(model_fn,t)
            u_t = jax.vmap(du_dt,in_axes=0)(jnp.eye(len(u)))[0]
            return u,u_t
        return jax.vmap(t_func,in_axes=(0,0))(x,t)
    
    def calc_uxx(self,params,x,t,output_pos= (0,)):
        #Only calculates for x inputs, returns (u:(batch,d_out),u_x:(batch,d_out,d_in),u_xx(batch,d_out,d_in,d_in))
        def jacrev2(x,t):
            def func(x):
                u = self.model.apply(params,x,t)
                u = u[...,output_pos]
                return u
            def jacrev(x):
                u,vjp_fun = jax.vjp(func,x)
                ret = jax.vmap(vjp_fun,in_axes=0)(jnp.eye(len(u)))
                return ret[0],u
            func2 = lambda s: jax.jvp(jacrev,(x,),(s,),has_aux=True)
            u_x,u_xx,u = jax.vmap(func2,in_axes=1,out_axes=(None,1,None))(jnp.eye(len(x)))
            return u,u_x,u_xx
        return jax.vmap(jacrev2,in_axes=(0,0))(x,t)

    def calc_ux(self,params,x,t,output_pos = (0,)):
        # calculates for x and t, returns (u:(batch,d_out),u_x(batch,d_out,d_in),u_t(batch,d_out,d_in))
        def func(x,t):
            u = self.model.apply(params,x,t)
            u = u[...,output_pos]
            return u
        def jacrev(x,t):
            u,vjp_fun = jax.vjp(func,x,t)
            ret = jax.vmap(vjp_fun, in_axes=0)(jnp.eye(len(u)))
            return u,ret[0],ret[1]
        return jax.vmap(jacrev,in_axes=0)(x,t)

    # Definition Functions, Losses & RL Methods (Implemented in sub-class)
    def u_ana(self,x):
        pass # implement analytical solution in child class

    def ic_fn(self,x):
        pass

    def b(self,x): # in: batch x d_in, out: batch x d_in
        return jnp.zeros_like(x)
    
    def b_heun(self,x,u,ux):
        return jnp.zeros_like(x)

    def sigma(self,x,y): # in: batch x d_in, out: batch x d_in x d_in
        return jnp.repeat(jnp.expand_dims(jnp.eye(x.shape[-1]),axis=0),x.shape[0],axis=0)
    
    def h(self,x,y,z,t): # in: batch x [d_in,d_out,d_out x d_in], out: batch x d_out x d_in
        return jnp.zeros_like(y)

    def c(self,x,u,u_x,u_xx):
        return .5*jnp.trace(jnp.matmul(jnp.matmul(self.sigma(x,u),self.sigma(x,u)),u_xx[:,0]),axis1=-1,axis2=-2)[...,jnp.newaxis]
    
    def c_direct(self,x,u,u_x,u_xx):
        return .5*jnp.trace(jnp.matmul(jnp.matmul(self.sigma(x,u),self.sigma(x,u)),u_xx[:,0]),axis1=-1,axis2=-2)[...,jnp.newaxis]
    
    def ic_grad(self,x):
        ic = lambda x: self.ic_fn(x)[0]
        ic_grad = jax.grad(ic)
        return jax.vmap(ic_grad,in_axes=0)(x)

    def pinns_pde_loss(self,key,params,x,t):
        pass # should be defined in child class

    def pinns_pde(self,key,params):
        x_pde,t_pde = self.sample_domain(key,self.config.batch_pde)
        return self.pinns_pde_loss(key,params,x_pde,t_pde)
    
    def pinns_bc(self,key,params): # No BCs only IC
        x_bc,t_bc = self.sample_domain(key,self.config.batch_bc)
        x_bcs = x_bc
        t_bcs = jnp.ones_like(t_bc)
        u,u_x,_ = self.calc_ux(params,x_bcs,t_bcs)
        loss = jnp.mean((u-self.ic_fn(x_bcs))**2)
        loss += jnp.mean((jnp.squeeze(u_x)-self.ic_grad(x_bcs))**2)
        return (loss,)

    def pinns_loss(self,key:Key,params):
        (pde_loss,) = self.pinns_pde(key,params)
        (ic_loss,) = self.pinns_bc(key,params)
        return (jnp.asarray(pde_loss)*self.config.pde_scale,jnp.asarray(ic_loss)*self.config.ic_scale)

    @partial(jax.jit,static_argnums = 0)
    def jit_pinns_loss(self,key: Key,params):
        return jnp.sum(jnp.asarray(self.pinns_loss(key,params)))
    
    def fspinns_loss(self,key:Key,params):
        x_pde,t_pde = self.sample_domain_bsde(key,self.config.batch_pde)
        dt = jnp.zeros((self.config.batch_pde,self.config.traj_len+1,1))
        dw = jnp.zeros((self.config.batch_pde,self.config.traj_len+1,self.config.d_in))
        dt = dt.at[:,1:,:].set(self.config.delta_t)
        dw = dw.at[:,1:,:].set(jnp.sqrt(self.config.delta_t)*jax.random.normal(key.newkey(),(self.config.batch_pde,self.config.traj_len,self.config.d_in)))
        t = jnp.cumsum(dt,axis=1)
        x = jnp.zeros((self.config.batch_pde,self.config.traj_len+1,self.config.d_in))
        x = x.at[:,0,:].set(x_pde)
        def loop(i,input):
            x = input
            u,u_x,_ = self.calc_ux(params,x[:,i-1,:],t[:,i-1,:])
            x = x.at[:,i,:].set(x[:,i-1,:]+self.b(x[:,i-1,:],u_x)*self.config.delta_t+jnp.matmul(self.sigma(x[:,i-1,:],u),dw[:,i,:,jnp.newaxis])[...,0])
            return x
        
        x = jax.lax.fori_loop(1,self.config.traj_len+1,loop,(x))
        x_ic = x[:,-1,:]
        t_ic = t[:,-1,:]
        x = jnp.reshape(x,(-1,self.config.d_in))
        t = jnp.reshape(t,(-1,1))
        temp = jnp.concatenate([x,t],axis=-1)
        temp = jax.random.choice(key.newkey(),temp,(temp.shape[0]//10,),axis=0)
        x = temp[:,0:-1]
        t = temp[:,-1:]

        (pinns_loss,) = self.pinns_pde_loss(key,params,x,t)

        u,u_x,_ = self.calc_ux(params,x_ic,t_ic)
        term_loss = jnp.sum((u-self.ic_fn(x_ic))**2)

        ic = lambda x: self.ic_fn(x)[0]
        ic_grad = jax.grad(ic)

        term_loss += jnp.sum((jnp.squeeze(u_x)-jax.vmap(ic_grad,in_axes=0)(x_ic))**2)

        return (pinns_loss,term_loss)

    @partial(jax.jit, static_argnums = 0)
    def jit_fspinns_loss(self,key:Key,params):
        return jnp.sum(jnp.asarray(self.fspinns_loss(key,params)))

    def bsde_loss(self,key:Key,params):
        x,t = self.sample_domain_bsde(key,self.config.batch_pde)
        u,u_x,_ = self.calc_ux(params,x,t)
        step_loss = jnp.zeros(self.config.traj_len)
        def traj_calc(i,input):
            key,x,t,u,u_x,step_loss = input

            xi = jax.random.normal(key.newkey(),(self.config.batch_pde,self.config.d_in)) # random generator
            x_prop1 = self.b(x,u_x)*self.config.delta_t
            x_prop2 = jnp.matmul(self.sigma(x,u),xi[...,jnp.newaxis])[...,0] * jnp.sqrt(self.config.delta_t) # (d_in x d_in) * (d_in) = (d_in)
            x_new = x + x_prop1+x_prop2

            u_temp1 = self.h(x,u,u_x,t) * self.config.delta_t
            u_temp2 = jnp.matmul(jnp.matmul(u_x,self.sigma(x,u)),xi[...,jnp.newaxis])[...,0]*jnp.sqrt(self.config.delta_t)
            u_new = u + u_temp1+u_temp2
            t_new = t + self.config.delta_t

            u_calc,u_x_calc,_ = self.calc_ux(params,x_new,t_new)

            step_loss = step_loss.at[i].set(jnp.sum((u_new-u_calc)**2))
            if self.config.reset_u:
                return key,x_new,t_new,u_calc,u_x_calc,step_loss
            else:
                return key,x_new,t_new,u_new,u_x_calc,step_loss
        key,x,t,u,u_x,step_loss = jax.lax.fori_loop(0,self.config.traj_len,traj_calc,(key,x,t,u,u_x,step_loss))
        sr_loss = jnp.sum(step_loss)

        
        term_loss = jnp.sum((u-self.ic_fn(x))**2)
        term_loss += jnp.sum((jnp.squeeze(u_x)-self.ic_grad(x))**2)

        return (sr_loss,term_loss)
    
    def bsde_skip_loss(self,key:Key,params):
        x,t = self.sample_domain_bsde(key,self.config.batch_pde)
        u,u_x,_ = self.calc_ux(params,x,t)
        step_loss = jnp.zeros(self.config.traj_len)
        def traj_calc(i,input):
            key,x,t,u,u_x,step_loss = input

            xi = jax.random.normal(key.newkey(),(self.config.batch_pde,self.config.d_in)) # random generator
            x_prop1 = self.b(x,u_x)*self.config.delta_t
            x_prop2 = jnp.matmul(self.sigma(x,u),xi[...,jnp.newaxis])[...,0] * jnp.sqrt(self.config.delta_t) # (d_in x d_in) * (d_in) = (d_in)
            x_new = x + x_prop1+x_prop2

            u_temp1 = self.h(x,u,u_x,t) * self.config.delta_t
            u_temp2 = jnp.matmul(jnp.matmul(u_x,self.sigma(x,u)),xi[...,jnp.newaxis])[...,0]*jnp.sqrt(self.config.delta_t)

            u_new = u + u_temp1+u_temp2

            t_new = t + self.config.delta_t

            u_calc,u_x_calc,_ = self.calc_ux(params,x_new,t_new)

            step_loss = step_loss.at[i].set(jax.lax.select(i%self.config.skip_len==0,jnp.sum((u_new-u_calc)**2),0.))
            if self.config.reset_u:
                u_new = jax.lax.select(i%self.config.skip_len==0,u_calc,u_new)

            return key,x_new,t_new,u_new,u_x_calc,step_loss

        key,x,t,u,u_x,step_loss = jax.lax.fori_loop(0,self.config.traj_len,traj_calc,(key,x,t,u,u_x,step_loss))
        sr_loss = jnp.sum(step_loss)

        
        term_loss = jnp.sum((u-self.ic_fn(x))**2)
        term_loss += jnp.sum((jnp.squeeze(u_x)-self.ic_grad(x))**2)

        return (sr_loss,term_loss)
    
    def bsde_huen_loss(self,key:Key,params):
        x,t = self.sample_domain_bsde(key,self.config.batch_pde)
        if self.config.checkpointing:
            calc_uxx = jax.checkpoint(self.calc_uxx)
        else:
            calc_uxx = self.calc_uxx
        u,u_x,u_xx = calc_uxx(params,x,t)
        step_loss = jnp.zeros(self.config.traj_len)
        def traj_calc(i,input):
            key,x,t,u,u_x,u_xx,step_loss = input

            xi = jax.random.normal(key.newkey(),(self.config.batch_pde,self.config.d_in)) # random generator
            x_temp1_1 = self.b_heun(x,u,u_x)*self.config.delta_t
            x_temp2_1 = jnp.matmul(self.sigma(x,u),xi[...,jnp.newaxis])[...,0] * jnp.sqrt(self.config.delta_t) # (d_in x d_in) * (d_in) = (d_in)
            x_temp = x + x_temp1_1+x_temp2_1

            u_temp1_1 = (self.h(x,u,u_x,t)-self.c(x,u,u_x,u_xx)) * self.config.delta_t
            u_temp2_1 = jnp.matmul(jnp.matmul(u_x,self.sigma(x,u)),xi[...,jnp.newaxis])[...,0]*jnp.sqrt(self.config.delta_t)

            t_new = t + self.config.delta_t

            u_calc_1,u_x_calc_1,u_xx_calc_1 = calc_uxx(params,x_temp,t_new)

            x_temp1_2 = self.b_heun(x_temp,u_calc_1,u_x_calc_1)*self.config.delta_t
            x_temp2_2 = jnp.matmul(self.sigma(x_temp,u_calc_1),xi[...,jnp.newaxis])[...,0] * jnp.sqrt(self.config.delta_t) # (d_in x d_in) * (d_in) = (d_in)
            x_new = x + .5*(x_temp1_1+
                           x_temp1_2+
                           x_temp2_1+
                           x_temp2_2)

            u_temp1_2 = (self.h(x_temp,u_calc_1,u_x_calc_1,t_new)-self.c(x_temp,u_calc_1,u_x_calc_1,u_xx_calc_1)) * self.config.delta_t
            u_temp2_2 = jnp.matmul(jnp.matmul(u_x_calc_1,self.sigma(x_temp,u_calc_1)),xi[...,jnp.newaxis])[...,0]*jnp.sqrt(self.config.delta_t)
            u_new = u + .5*(u_temp1_1+
                           u_temp1_2+
                           u_temp2_1+
                           u_temp2_2)

            u_calc,u_x_calc,u_xx_calc = calc_uxx(params,x_new,t_new)

            step_loss = step_loss.at[i].set(jnp.sum((u_new-u_calc)**2))
            return key,x_new,t_new,u_calc,u_x_calc,u_xx_calc,step_loss

        key,x,t,u,u_x,u_xx,step_loss = jax.lax.fori_loop(0,self.config.traj_len,traj_calc,(key,x,t,u,u_x,u_xx,step_loss))
        sr_loss = jnp.sum(step_loss)

        
        term_loss = jnp.sum((u-self.ic_fn(x))**2)
        term_loss += jnp.sum((jnp.squeeze(u_x)-self.ic_grad(x))**2)

        return (sr_loss,term_loss)

    def bsde_heun_skip_loss(self,key:Key,params):
        x,t = self.sample_domain_bsde(key,self.config.batch_pde)
        if self.config.checkpointing:
            calc_uxx = jax.checkpoint(self.calc_uxx)
        else:
            calc_uxx = self.calc_uxx
        u,u_x,u_xx = calc_uxx(params,x,t)
        step_loss = jnp.zeros(self.config.traj_len)
        def traj_calc(i,input):
            key,x,t,u,u_x,u_xx,step_loss = input

            xi = jax.random.normal(key.newkey(),(self.config.batch_pde,self.config.d_in)) # random generator
            x_temp1_1 = self.b_heun(x,u,u_x)*self.config.delta_t
            x_temp2_1 = jnp.matmul(self.sigma(x,u),xi[...,jnp.newaxis])[...,0] * jnp.sqrt(self.config.delta_t) # (d_in x d_in) * (d_in) = (d_in)
            x_temp = x + x_temp1_1+x_temp2_1

            u_temp1_1 = (self.h(x,u,u_x,t)-self.c(x,u,u_x,u_xx)) * self.config.delta_t
            u_temp2_1 = jnp.matmul(jnp.matmul(u_x,self.sigma(x,u)),xi[...,jnp.newaxis])[...,0]*jnp.sqrt(self.config.delta_t)

            t_new = t + self.config.delta_t

            u_calc_1,u_x_calc_1,u_xx_calc_1 = calc_uxx(params,x_temp,t_new)

            x_temp1_2 = self.b_heun(x_temp,u_calc_1,u_x_calc_1)*self.config.delta_t
            x_temp2_2 = jnp.matmul(self.sigma(x_temp,u_calc_1),xi[...,jnp.newaxis])[...,0] * jnp.sqrt(self.config.delta_t) # (d_in x d_in) * (d_in) = (d_in)
            x_new = x + .5*(x_temp1_1+
                           x_temp1_2+
                           x_temp2_1+
                           x_temp2_2)

            u_temp1_2 = (self.h(x_temp,u_calc_1,u_x_calc_1,t_new)-self.c(x_temp,u_calc_1,u_x_calc_1,u_xx_calc_1)) * self.config.delta_t
            u_temp2_2 = jnp.matmul(jnp.matmul(u_x_calc_1,self.sigma(x_temp,u_calc_1)),xi[...,jnp.newaxis])[...,0]*jnp.sqrt(self.config.delta_t)
            u_new = u + .5*(u_temp1_1+
                           u_temp1_2+
                           u_temp2_1+
                           u_temp2_2)

            u_calc,u_x_calc,u_xx_calc = calc_uxx(params,x_new,t_new)

            step_loss = step_loss.at[i].set(jax.lax.select(i%self.config.skip_len==0,jnp.sum((u_new-u_calc)**2),0.))
            if self.config.reset_u:
                u_new = jax.lax.select(i%self.config.skip_len==0,u_calc,u_new)

            return key,x_new,t_new,u_new,u_x_calc,u_xx_calc,step_loss
        key,x,t,u,u_x,u_xx,step_loss = jax.lax.fori_loop(0,self.config.traj_len,traj_calc,(key,x,t,u,u_x,u_xx,step_loss))
        sr_loss = jnp.sum(step_loss)
        
        term_loss = jnp.sum((u-self.ic_fn(x))**2)
        term_loss += jnp.sum((jnp.squeeze(u_x)-self.ic_grad(x))**2)

        return (sr_loss,term_loss)

    @partial(jax.jit,static_argnums = 0)
    def jit_bsde_loss(self,key: Key,params):
        return jnp.sum(jnp.asarray(self.bsde_loss(key,params)))
    
    def reg_loss(self,key: Key,params):
        _,rl2 = self.interp_rl(key,params)
        return rl2

    def calc_rl(self,params):
        pred = self.calc_u(params,self.sol[1][...,0:-1],self.sol[1][...,-1:])[...,0]

        rl1 = jnp.sum(jnp.abs(self.sol[0]-pred))/jnp.sum(jnp.abs(self.sol[0]))
        rl2 = jnp.sqrt(jnp.sum((self.sol[0]-pred)**2) / jnp.sum(self.sol[0]**2))

        return rl1,rl2
    
    def calc_eval(self,params):
        pass # Custom Eval Definition
    
    def interp_rl(self,key:Key,params):
        x_pde,t_pde = self.sample_domain(key,self.config.batch_pde)
        pred = self.calc_u(params,x_pde,t_pde)
        u = self.interp(jnp.concatenate((x_pde,t_pde),axis=-1))[...,jnp.newaxis]

        rl1 = jnp.sum(jnp.abs(u-pred))/jnp.sum(jnp.abs(u))
        rl2 = jnp.sqrt(jnp.sum((u-pred)**2) / jnp.sum(u**2))
        return rl1,rl2

    @partial(jax.jit,static_argnums = 0)
    def jit_calc_rl(self,params):
        return self.calc_rl(params)

    @partial(jax.jit,static_argnums = 0)
    def jit_calc_eval(self,params):
        return self.calc_eval(params)

    # Optimization Methods
    @partial(jax.jit,static_argnums = [0,1])
    def optimize(self,loss_method,key:Key,params,opt_state,loss_weights):
        def loss_fn(key,params):
            losses = jnp.asarray(self.select_loss(loss_method)(key,params))
            loss = jnp.sum(losses)
            return loss, losses
        (loss,losses),grad = jax.value_and_grad(loss_fn,argnums=1,has_aux=True)(key,params)
        updates,opt_state = self.optimizer.update(grad,opt_state)
        params = optax.apply_updates(params,updates)

        return loss, losses, params, opt_state, key
    
class Controller():
    def __init__(self,solver: Solver,seed = 1234):
        self.solver = solver
        self.key = Key.create_key(seed)
        self.params, self.opt_state = self.solver.init_solver(self.key)
        self.loss_weights = jnp.ones((len(solver.select_loss(self.solver.config.loss_method)(self.key,self.params)),))
        self.track = []
        self.training_configs = []
        # adaptive loss tracking
        self.loss_track = []
        self.loss_roc = []

    def step(self,loss_method,i):
        self.key.newkey()
        loss,losses,self.params,self.opt_state,self.key = self.solver.optimize(loss_method,self.key,self.params,self.opt_state,self.loss_weights)
        if self.solver.config.ref_sol:
            rl1,rl2 = self.solver.jit_calc_rl(self.params)
        if self.solver.config.custom_eval:
            custom_eval = self.solver.jit_calc_eval(self.params)
        self.track.append(loss)
        # Saving Logs
        if self.solver.config.save_to_wandb:
            if self.solver.config.track_pinns_loss:
                pinns_loss = self.solver.jit_pinns_loss(self.key,self.params)
                wandb.log({"pinns Loss": pinns_loss},commit=False)
            if self.solver.config.track_bsde_loss:
                bsde_loss = self.solver.jit_bsde_loss(self.key,self.params)
                wandb.log({"bsde Loss": bsde_loss},commit=False)
            if self.solver.config.track_fspinns_loss:
                fspinns_loss = self.solver.jit_fspinns_loss(self.key,self.params)
                wandb.log({"fspinns Loss": fspinns_loss},commit=False)
            temp = {"loss"+str(k+1): v for k,v in dict(enumerate(losses)).items()}
            wandb.log(temp,commit=False)
            if self.solver.config.ref_sol:
                wandb.log({"RL1": rl1,"RL2":rl2},commit=False)
            if self.solver.config.custom_eval:
                wandb.log({"Eval": custom_eval},commit=False)
            wandb.log({"loss": jnp.sum(losses)})

    def solve(self):
        for i in tqdm.tqdm(range(self.solver.config.iter)):
            self.step(self.solver.config.loss_method,i)
            # Additional Losses
        if self.solver.config.additional_losses:
            loss_num = 1
            for config in self.training_configs:
                self.change_config(config)
                if self.solver.config.save_to_wandb:
                    self.save_config(config,loss_num)
                for i in tqdm.tqdm(range(self.solver.config.iter)):
                    self.step(self.solver.config.loss_method,i)
                loss_num+=1
        if self.solver.config.save_to_wandb:
            if self.solver.config.ref_sol:
                self.solver.plot_pred(self.params)
                self.solver.plot_err(self.params)
            if self.solver.config.custom_eval:
                self.solver.plot_eval(self.params)
        self.solver.close()

    ## Utils
    def tab_model(self):
        self.solver.tab_model(self.key)

    def append_train_config(self,config: TrainConfig):
        self.training_configs.append(config)

    def change_config(self,config:TrainConfig):
        for var in vars(config):
            setattr(self.solver.config,var,vars(config)[var])
        self.opt_state.hyperparams['learning_rate'] = self.solver.config.lr

        # reset loss weighing
        self.loss_weights = jnp.ones((len(self.solver.select_loss(self.solver.config.loss_method)(self.key,self.params)),))
        self.loss_track = []
        self.loss_roc = []

    def save_config(self,config:TrainConfig,i: int):
        tempdict = dict(vars(config))
        for var in vars(config):
            tempdict[var+str(i)] = tempdict.pop(var)
        wandb.config.update(tempdict)