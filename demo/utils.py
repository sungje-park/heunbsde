import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as onp

@jax.tree_util.register_pytree_node_class
class Key:
    def __init__(self,key):
        self.key = key

    @classmethod
    def create_key(cls,seed):
        temp = cls.__new__(cls)
        temp.__init__(jax.random.key(seed))
        return temp
    
    # JAX PyTree Definitions
    def tree_flatten(self):
        children = (self.key,)
        aux_data = {}
        return (children,aux_data)

    @classmethod
    def tree_unflatten(cls,aux_data,children):
        return cls(*children, **aux_data)
    
    def newkey(self):
        self.key,ret_key = jax.random.split(self.key)
        return ret_key
    
def plot1d(config,u,title_str):
    fig = plt.figure(figsize=(4,3))
    plt.imshow(u, extent=config.x_range[0]+config.t_range,aspect='auto',vmin=config.plot_range[0],vmax=config.plot_range[1],origin='lower')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(title_str)
    plt.colorbar()
    plt.tight_layout()
    return fig

def plot2dstream(x,y,u,v,title_str,axis_ranges = (0,1,0,1),axis_names = ['x','y']):
    fig = plt.figure(figsize=(4,4))
    plt.streamplot(x,y,u,v,density=2,linewidth=1)
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.axis(axis_ranges)
    plt.gca().set_aspect('equal',adjustable='box')
    plt.title(title_str)
    plt.tight_layout()
    return fig

def plot2d(u,title_str,axis_ranges = (0,1,0,1),axis_names = ['x','y'],plot_range = (0,1)):
    fig = plt.figure(figsize=(4,3))
    plt.imshow(u,extent=axis_ranges,aspect='equal',vmin=plot_range[0],vmax=plot_range[1],origin='lower')
    plt.xlabel(axis_names[0])
    plt.ylabel(axis_names[1])
    plt.title(title_str)
    plt.colorbar()
    plt.tight_layout()
    return fig

def hvp(f, primals, tangents):
  return jax.jvp(jax.grad(f), primals, tangents)[1]
