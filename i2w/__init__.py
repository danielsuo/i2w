import os
import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch
import tensorflow as tf
import optax
import chex
import h5py

from init2winit.dataset_lib import fastmri_dataset


def get_slice(path, slice_id):
    with tf.io.gfile.GFile(path, 'rb') as gf:
        path = gf
    with h5py.File(path, 'r') as hf:
        return (
            hf['kspace'][slice_id],
            hf['kspace'][slice_id].shape,
            hf['reconstruction_esc'][slice_id],
            hf['reconstruction_esc'][slice_id].shape,
            hf.attrs.get('max', 0.0)
        )
    

def create_batch(t_batch, base='../../singlecoil_train'):
    batch_size = len(t_batch.slice_num)
    inputs = [get_slice(os.path.join(base, t_batch.fname[i]), t_batch.slice_num[i]) for i in range(batch_size)]
    processed = [fastmri_dataset._process_example(*input, tf.cast(jax.random.PRNGKey(0), tf.int64)) for input in inputs]
    batched = [{key: tf.expand_dims(value, 0) for key, value in f.items()} for f in processed]

    f_batch = {}
    for key in batched[0].keys():
        f_batch[key] = tf.concat([batched[i][key] for i in range(batch_size)], axis=0).numpy()
        
    return f_batch


def convert_params(t_model, f_model, batch_size=1):
  out_channels = 1
  in_channels = 1
  w, h = 320, 320

  inp = np.random.uniform(size=(batch_size, in_channels, w, h))
  jax_inp = np.transpose(inp, axes=(0, 2, 3, 1))

  pytorch_conv = t_model
  #Pytorch wants NCHW
  tt = torch.tensor(inp, dtype=torch.float32)
  a = pytorch_conv(tt)
  # print(a.size())

  # for p in list(pytorch_conv.parameters()):
  #   print(p.name)
  #   print(p.data.shape)
  pytorch_params = list(pytorch_conv.parameters())
  # pytorch_params = list(pytorch_conv.parameters())
  # print(len(pytorch_params))
#   np_params = [p.data.numpy() for p in pytorch_params]
#   [print('pytorch params shape', p.shape) for p in np_params]
#   for name, param in t_model.named_parameters():
#     print(name, param.data.numpy().shape)

  # jax_data = [jnp.array(np.transpose(p, axes = (2, 3, 1, 0))) for p in np_params]
  # [print(p.shape) for p in jax_data]

  #jt = jnp.ones((1, h, h, in_channels))
  jt = jnp.array(jax_inp).squeeze(-1)
#   print(jt.shape)
  jax_conv = f_model
  params = jax_conv.init(jax.random.PRNGKey(0), jt)

  counter = 0
  params = flax.core.unfreeze(params)
  for k in params['params']:
    for l in params['params'][k]:
      # print(k, l)
      if 'ConvBlock' in k and not 'Transpose' in k:
        #print(k, l)
        # print(params['params'][k][l]['kernel'].shape)
        np_params = pytorch_params[counter].data.numpy()
#         print(np_params.shape)
        jax_data = jnp.array(np.transpose(np_params, axes = (2, 3, 1, 0)))
        params['params'][k][l]['kernel'] = jax_data
        # print(params['params'][k][l]['kernel'].shape)
        counter += 1
      elif 'TransposeConvBlock' in k:
        # print(k, l)
        #print(params['params'][k][l]['kernel'].shape)
        np_params = pytorch_params[counter].data.numpy()
#         print(np_params.shape)
        jax_data = jnp.array(np.transpose(np_params, axes = (2, 3, 0, 1)))
        params['params'][k][l]['kernel'] = jnp.flip(jax_data, axis = (0, 1))
        #print(params['params'][k][l]['kernel'].shape)
        counter += 1
      elif 'Conv_0' in k:
        # print(k, l)
        if l == 'kernel':
          # print(params['params'][k][l].shape)
          np_params = pytorch_params[-2].data.numpy()
          # print(np_params.shape)
          jax_data = jnp.array(np.transpose(np_params, axes = (2, 3, 1, 0)))
          params['params'][k][l] = jax_data
          # print(params['params'][k][l].shape)
        if l == 'bias':
          #print(params['params'][k][l].shape)
          np_params = pytorch_params[-1].data.numpy()
          jax_data = jnp.array(np_params)
          params['params'][k][l] = jax_data
          #print(params['params'][k][l].shape)

  # params['params']['Conv_1']['kernel'] = jax_data[1]

  ja = jax_conv.apply(params, jt)
#   print('jax output shape', ja.shape)
#   print('jax params shape', p['params']['Conv_0']['kernel'].shape)
#   jax_output = jnp.transpose(ja, axes=(0, 3, 1, 2))
  jax_output = ja

  # print('pytorch output shape', a.data.shape)
  # print('jax output modified shape', jax_output.shape)


  ## compare data 
  po_jax = jnp.array(a.data.numpy())
  print('Abs sum diff:', jnp.sum(jnp.abs(jax_output - po_jax)))

  return params


from typing import NamedTuple
import chex

class PreconditionBySecondMomentCoordinateWiseState(NamedTuple):
  """State for the Adam preconditioner."""
  count: chex.Array
  nu: optax.Updates

def _update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return jax.tree_multimap(lambda g, t: (1 - decay) * (g**order) + decay * t,
                           updates, moments)


def _bias_correction(moment, decay, count):
  """Perform bias correction. This becomes a no-op as count goes to infinity."""
  beta = 1 - decay**count
  return jax.tree_map(lambda t: t / beta.astype(t.dtype), moment)

def precondition_by_rms(decay: float = 0.99,
                        eps: float = 1e-8,
                        eps_root: float = 0.0,
                        debias: bool = False,
                        ) -> optax.GradientTransformation:
  """Preconditions updates according to the RMS Preconditioner from Adam.

  References:
    [Kingma, Ba 2015] https://arxiv.org/pdf/1412.6980.pdf

  Args:
    decay: decay rate for exponentially weighted average of moments of grads.
    eps: Term added to the denominator to improve numerical stability.
      The default is kept to 1e-8 to match optax Adam implementation.
    eps_root: term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    debias: whether to use bias correction or not

  Gotcha:
    Note that the usage of epsilon and defaults are different from optax's
    scale_by_rms. This matches optax's adam template.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    return PreconditionBySecondMomentCoordinateWiseState(
        count=jnp.zeros([], jnp.int32), nu=jax.tree_map(jnp.zeros_like, params))

  def update_fn(updates, state, params=None):
    del params
    nu = _update_moment(updates, state.nu, decay, 2)
    count = state.count + jnp.array(1, dtype=jnp.int32)
    nu_hat = nu if not debias else _bias_correction(nu, decay, count)
    updates = jax.tree_multimap(lambda u, v: u / (jnp.sqrt(v + eps_root) + eps),
                                updates, nu_hat)
    return updates, PreconditionBySecondMomentCoordinateWiseState(
        count=count, nu=nu)

  return optax.GradientTransformation(init_fn, update_fn)
