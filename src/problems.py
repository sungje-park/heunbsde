from solver import Solver
from config import Config
from utils import *
import jax
from jax import numpy as jnp
import wandb

### Multi-Dim HJB Solver
class HJB_Solver(Solver):
    def __init__(self,config:Config):
        super().__init__(config)

    def sample_domain(self,key:Key,batch_size): # Initiates normal dist for x and uniform t
        x_pde = []
        for i in range(self.config.d_in):
            x_pde.append(jnp.sqrt(2)*jax.random.normal(key.newkey(),(batch_size,1)))
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

    def u_ana(self,x):
            def u_inner(x):
                w = jax.random.normal(jax.random.key(10),(100000,self.config.d_in,)) # batch x d_in
                return -jnp.log(jnp.mean(jnp.exp(-self.ic_fn(x[...,0:-1]+jnp.sqrt(2*(self.config.t_range[1]-x[...,-1]))*w))))
            return jax.vmap(jax.vmap(u_inner,in_axes=0),in_axes=0)(x)

    def ic_fn(self,x):
        return jnp.log(.5*(1+jnp.sum(x**2,keepdims=True,axis=-1)))


    def b(self,x,u_x): # in: batch x d_in, out: batch x d_in
        return super().b(x)
    
    def sigma(self,x,y): # in: batch x d_in, out: batch x d_in x d_in
        return jnp.sqrt(2)*super().sigma(x,y)
    
    def h(self,x,y,z,t): # in: batch x [d_in,d_out,d_in x d_out], out: batch x d_out x d_in
        return jnp.sum(z**2,axis=-1)
    
    def pinns_pde_loss(self,key,params,x,t):
        _,u_x,u_xx = self.calc_uxx(params,x,t)
        _,u_t = self.calc_ut(params,x,t)
        loss = jnp.mean((u_t[...,0]+jnp.trace(u_xx,axis1=-2,axis2=-1)-jnp.sum(u_x**2,axis=-1))**2)
        return (loss,)
    
    def get_base_config(d_in=1,traj_len=50):
        T = 1
        return Config(case="hjb",
                      d_in = d_in,
                      d_hidden = 256,
                      num_layers=5,
                      batch_pde = 128,
                      batch_bc = 128,
                      traj_len=traj_len,
                      delta_t=T/traj_len,
                      ic_scale=10,
                      iter = 100000)    
    def get_ref_sol(self):
        u_ana = jax.jit(self.u_ana)
        num_traj = 5
        x = jnp.zeros((num_traj,51,self.config.d_in))
        x_init,_ = self.sample_domain_bsde(jax.random.key(1),num_traj)
        x = x.at[:,0,:].set(x_init)
        x = x.reshape(num_traj,51,self.config.d_in)
        w = jax.random.normal(jax.random.key(1),(num_traj,50,self.config.d_in))
        w = jnp.concatenate((jnp.zeros((num_traj,1,self.config.d_in)),w),axis=1)
        t = jnp.repeat(jnp.linspace(0,1,51)[jnp.newaxis,...,jnp.newaxis],num_traj,axis=0)
        for i in range(50):
            d_x = jnp.sqrt(.02)*jnp.matmul(self.sigma(x[:,i,:],None),w[:,i+1,:,jnp.newaxis])[...,0]
            x = x.at[:,i+1,:].set(x[:,i,:]+d_x)
        xt = jnp.concatenate((x,t),axis=-1)
        u = u_ana(xt)
        return (u,xt),u_ana

### Multi-Dim BSB_Solver
class BSB_Solver(Solver):
    def __init__(self,config:Config):
        super().__init__(config)

    def sample_domain(self,key:Key,batch_size): # Initiates normal dist for x and uniform t
        x_pde = []
        for i in range(self.config.d_in):
            x_pde.append(.75+jax.random.normal(key.newkey(),(batch_size,1)))

        x_pde = jnp.concatenate(x_pde,axis=-1)
        t_pde = jax.random.uniform(key.newkey(),
                                   (batch_size,1),
                                   minval=self.config.t_range[0],
                                   maxval=self.config.t_range[1])
        return x_pde,t_pde
    
    def sample_domain_bsde(self,key,batch_size):
        x_pde = []
        for i in range(self.config.d_in):
            if i%2 == 1:
                x_pde.append(jnp.ones((batch_size,1))/2)
            else:
                x_pde.append(jnp.ones((batch_size,1)))
        x_pde = jnp.concatenate(x_pde,axis=-1)
        t_pde = jnp.zeros((batch_size,1))
        return x_pde,t_pde

    def u_ana(self,x):
        return (jnp.exp((.05+.4**2)*(1-x[...,-1:]))*self.ic_fn(x[...,0:-1]))[...,0]

    def ic_fn(self,x):
        return jnp.sum(x**2,keepdims=True,axis=-1)

    def b(self,x,u_x): # in: batch x d_in, out: batch x d_in
        return super().b(x)
    
    def sigma(self,x,y): # in: batch x d_in, out: batch x d_in x d_in
        return .4*jax.vmap(jnp.diag,in_axes=0)(x)
    
    def h(self,x,y,z,t): # in: batch x [d_in,d_out,d_out x d_in], out: batch x d_out x d_in
        return .05*(y-jnp.matmul(z,x[...,jnp.newaxis])[...,0])
        #return (y-jnp.matmul(z,x[...,jnp.newaxis])[...,0])
    
    def c(self,x,u,u_x,u_xx):
        return .5*.4**2*(jnp.matmul(u_x,x[...,jnp.newaxis])[...,0]+
          jnp.trace(jnp.matmul(jax.vmap(jnp.diag,in_axes=0)(x**2),u_xx[:,0]),axis1=-1,axis2=-2)[...,jnp.newaxis])
    
    def b_heun(self,x,u,ux):
        return -(.4**2)*.5*x
    
    def pinns_pde_loss(self,key,params,x,t):
        u,u_x,u_xx = self.calc_uxx(params,x,t)
        _,u_t = self.calc_ut(params,x,t)
        loss = jnp.mean((u_t[...,0]+
                         .5*jnp.trace(self.sigma(x,u)**2 @ u_xx[...,0,:,:],axis1=-2,axis2=-1)[...,jnp.newaxis]-
                         .05*(u-jnp.matmul(u_x,x[...,jnp.newaxis])[...,0]))**2)
        return (loss,)

    def drift(self,t,y,args):
            x = y[0:self.config.d_in]
            y = y[-1:]
            t = jnp.asarray(t)
            t = jnp.reshape(t,(1,))
            params = args
            u,u_x,u_xx = self.calc_uxx(params,x[jnp.newaxis],t[jnp.newaxis])
            u_xx = u_xx[0]
            u_x = u_x[0]
            x_drift = self.b(x)
            y_drift = self.h(x,y,u_x) - .5*(.4**2*jnp.matmul(u_x,x[...,jnp.newaxis])[...,0]+jnp.trace(jnp.matmul(jnp.matmul(self.sigma(x[jnp.newaxis]),self.sigma(x[jnp.newaxis]))[0],u_xx),axis1=-1,axis2=-2))
            d_y = jnp.concatenate([x_drift,y_drift],axis=-1)
            return d_y

    def get_base_config(d_in = 1,traj_len = 50):
        T = 1
        return Config(case='bsb',
                      d_in = d_in,
                      d_hidden = 256,
                      num_layers=5,
                      batch_pde = 128,
                      batch_bc = 128,
                      traj_len=traj_len,
                      delta_t=T/traj_len,
                      ic_scale=10,
                      iter = 100000)
    
# Example toy problem from Bender & Zhang 2008
class BZ_Solver(Solver):
    def __init__(self,config:Config):
        super().__init__(config)

    def sample_domain(self,key:Key,batch_size): # Initiates normal dist for x and uniform t
        x_pde = []
        for i in range(self.config.d_in):
            x_pde.append(jnp.pi/2+2*jax.random.normal(key.newkey(),(batch_size,1)))
        x_pde = jnp.concatenate(x_pde,axis=-1)
        t_pde = jax.random.uniform(key.newkey(),
                                   (batch_size,1),
                                   minval=self.config.t_range[0],
                                   maxval=self.config.t_range[1])
        return x_pde,t_pde
    
    def sample_domain_bsde(self,key,batch_size):
        x_pde = []
        for i in range(self.config.d_in):
            x_pde.append(jnp.ones((batch_size,1))*jnp.pi/2)
        x_pde = jnp.concatenate(x_pde,axis=-1)
        t_pde = jnp.zeros((batch_size,1))
        return x_pde,t_pde

    def u_ana(self,x):
        t = x[...,-1:]
        x = x[...,0:-1]
        return (jnp.exp(-.1*(self.config.t_range[-1]-t))*.1*jnp.sum(jnp.sin(x),axis=-1,keepdims=True))[...,0]

    def ic_fn(self,x):
        return .1*jnp.sum(jnp.sin(x),axis=-1,keepdims=True)

    def b(self,x,u_x): # in: batch x d_in, out: batch x d_in
        return super().b(x)
    
    def sigma(self,x,y): # in: batch x d_in, batch x d_out, out: batch x d_in x d_in
        return .3*jax.vmap(lambda x,y: x*y,in_axes=(0,0))(y,super().sigma(x,y))
    
    def h(self,x,y,z,t): # in: batch x [d_in,d_out,d_in x d_out], out: batch x d_out x d_in
        return -(-.1*y+.5*jnp.exp(-.3*(self.config.t_range[-1]-t))*.3**2*(.1*jnp.sum(jnp.sin(x),axis=-1,keepdims=True))**3)
    
    def get_base_config(d_in = 1,traj_len = 50):
        T = 1
        return Config(case='bz',
                    d_in = d_in,
                    d_hidden = 256,
                    num_layers=5,
                    batch_pde = 128,
                    batch_bc = 128,
                    traj_len=traj_len,
                    delta_t=T/traj_len,
                    ic_scale=10,
                    iter = 100000)
    
    def pinns_pde_loss(self,key,params,x,t):
        u,u_x,u_xx = self.calc_uxx(params,x,t)
        _,u_t = self.calc_ut(params,x,t)
        loss = jnp.mean((u_t+.5*.3**2*u**2*jnp.trace(u_xx,axis1=-1,axis2=-2)-self.h(x,u,u_x,t))**2)
        return (loss,)

    def b_heun(self,x,u,ux):
        return jnp.matmul(u[...,jnp.newaxis],-.3**2*.5*ux)[...,0,:]
    
    def c(self,x,u,u_x,u_xx):
        return jnp.matmul(u_x,self.b_heun(x,u,u_x)[...,jnp.newaxis])[...,0]+self.c_direct(x,u,u_x,u_xx)