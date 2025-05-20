import jax.numpy as jnp
import flax.linen as nn
import jax
from config import Config

class  WaveAct(nn.Module):

    @nn.compact
    def __call__(self,x):
        w1 = self.param('w1',nn.initializers.normal(.1), (x.shape[-1],))
        w2= self.param('w2',nn.initializers.normal(.1),(x.shape[-1],))
        return jnp.asarray(w1) * jnp.sin(x) + jnp.asarray(w2) * jnp.cos(x)
    
class FourierEmbs(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", jax.nn.initializers.normal(self.config.emb_scale), (x.shape[-1], self.config.emb_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y
  
class PINNs(nn.Module):
    config: Config

    def setup(self):
        adaptive_act = False

        match self.config.activation:
            case "wave":
                self.activation = WaveAct
                adaptive_act = True
            case "tanh":
                self.activation = nn.tanh
            case "swish":
                self.activation = nn.swish
            case _:
                raise Exception("Activation '"+self.config.activation+"' Not Implemented")

        layers = []

        if self.config.four_emb:
            self.four_layer = FourierEmbs(self.config)

        for i in range(self.config.num_layers):
            layers.append(nn.Dense(self.config.d_hidden))
            if adaptive_act:
                layers.append(self.activation())
            else:
                layers.append(self.activation)
        
        self.output_layer = nn.Dense(self.config.d_out)

        self.layers = layers
    
    def __call__(self,*args):
        src = args[0]
        for i in args[1:len(args)]:
            src = jnp.concatenate((src,i),axis=-1)
        if self.config.periodic:
            src = src.at[...,0].set(jnp.sin(src[...,0]))
            src = jnp.concatenate((src,jnp.cos(src[...,0:1])),axis=-1)

        if self.config.four_emb:
            src =self.four_layer(src)

        i = 1
        src_skip = self.layers[0](src)

        for layer in self.layers:
            src = layer(src)
            if self.config.skip_conn:
                src = jax.lax.select(jnp.any(i==jnp.asarray(self.config.skip_layers)*2),src+src_skip,src)
                src_skip = jax.lax.select(jnp.any(i==jnp.asarray(self.config.save_layers)*2),src,src_skip)
            i += 1
        
        src = self.output_layer(src)
        
        return src
