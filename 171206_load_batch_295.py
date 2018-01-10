import nems.main as nm
import nems.db as ndb
import nems.utilities as nu
import copy
import matplotlib.pyplot as plt
import nems.modules.metrics as mt
import numpy as np
import pandas as pd

'''
Single cell test run of SI calculation in batch 259
replaces data with oddball paradigm, evaluate to generate predicted data, uses predicted data as a proxy for
SI calculation.

'''

# Define batch, cellids, modelname
batch = 259
cells_in_batch = ndb.get_batch_cells(batch = batch)
cellids = cells_in_batch['cellid'].tolist()
modelname1 = 'env100_dlog_fir15_dexp_fit01'
modelname2 = 'env100_dlog_stp1pc_fir15_dexp_fit01'

example_cell = 'chn008b-c2'

# imports a ssa stack to extract the stimulus i.e. the oddball paradigm. 'gus019d-b1' has a reasonable stimulation
# example to use
ssa_stack = nu.io.load_single_model(cellid='gus019d-b1', batch=296, modelname='env100e_stp1pc_fir20_fit01_ssa')
oddball = copy.deepcopy(ssa_stack.data[1][0]) # Jittered input data

# import a stack from the data base.
original_stack = nu.io.load_single_model(example_cell, batch, modelname1)
stp_stack = nu.io.load_single_model(example_cell, batch, modelname2)



# normalized transform the oddball stimulus to be on pair with the range of the original
# stimulus used to fit the stack.
vocal = stp_stack.data[1][0]['stim']
normstim = oddball['stim'] * np.max(vocal)
oddball['stim'] = normstim

# takes the fitted stack (from the vocalization experiment) and changes the input data
chimera = copy.deepcopy(stp_stack)
chimera.modules[0].d_out = [oddball]
chimera.modules[1].d_in = [oddball]
chimera.evaluate()

# appends the ssa metric module
chimera.append(mt.ssa_index)

# print some example traces
nu_trances = 3
ssa = chimera.modules[-1]
stim = ssa.d_out[0]['stim']
resp = ssa.d_out[0]['resp']
pred = ssa.d_out[0]['pred']

fig, axes = plt.subplots(nu_trances)
axes = np.ravel(axes)

for ii, ax in enumerate(axes):
    ax.plot(stim[0,ii,:], color='C0')
    ax.plot(stim[1,ii,:], color='C1')
    ax.plot(resp[0,ii,:] + 2, color='green', label='actual')
    ax.plot(pred[0,ii,:] + 2, color='red', label='predicted')
    if ii==0:
        ax.legend()

vocal = stp_stack.data[1][0]['stim']

ssa = chimera.modules[-1]
ssa.do_plot(ssa)


