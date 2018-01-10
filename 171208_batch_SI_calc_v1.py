import nems.db as ndb
import nems.utilities as nu
import copy
import nems.modules.metrics as mt
import numpy as np
import pandas as pd
import joblib as jl

'''
Batch 259 SI calculation

Import the stacks, replace the data with an oddball paradigm and evaluate to generate a prediction
use said prediction to calculate SI by appending and ssa_index module at the end of the stack.
    
'''

# Define batch, cellids, modelname
batch = 259
cells_in_batch = ndb.get_batch_cells(batch = batch)
cellids = cells_in_batch['cellid'].tolist()
modelname1 = 'env100_dlog_fir15_dexp_fit01'
modelname2 = 'env100_dlog_stp1pc_fir15_dexp_fit01'
modelnames = [modelname1, modelname2]
filename = '/home/mateo/batch_259/171207_batch_SI_simulation_DF'
try:
    originalDF = jl.load(filename)
    readycells = originalDF.cellid.unique().tolist()
except:
    readycells = []
    originalDF = pd.DataFrame()

# compares the cells already analized with the whole cell list in the database, batch

newcells = [cell for cell in cellids if cell not in readycells]

# imports a ssa stack to extract the stimulus i.e. the oddball paradigm. 'gus019d-b1' has a reasonable stimulation
# example to use
# also updates a couple of fields in the ssa_index module, what are not present in the Data base
ssa_stack = nu.io.load_single_model(cellid='gus019d-b1', batch=296, modelname='env100e_stp1pc_fir20_fit01_ssa')
ssa_module = ssa_stack.modules[-2]
ssa_module.z_score = 'all'
ssa_module.significant_bins = 'window'
ssa_stack.evaluate()
oddball = copy.deepcopy(ssa_stack.data[1][0]) # Jittered input data


# initilizes soon to be DF
df = list()
errorcells = list()
stacks = list()
# iterates over cells

for cellid in newcells:
    print(' \n working on cell {} \n '.format(cellid))
    for model in modelnames:
        print(' \n working on model {} \n'.format(model))
        #test suit
        #cellid = cellids[1]
        #model = modelname2

        try:
            # import a stack from the data base.
            stack = nu.io.load_single_model(cellid, batch, model)

            # normalized transform the oddball stimulus to be on pair with the range of the original
            # stimulus used to fit the stack.
            vocal = stack.data[1][0]['stim']
            new_data = copy.deepcopy(oddball)

            stim_amp= 'max'

            if stim_amp == 'max':
                amplitude = np.nanmax(stack.data[1][0]['stim'])
            elif isinstance(stim_amp, float):
                amplitude = stim_amp

            normstim = new_data['stim'] * amplitude

            new_data['stim'] = normstim

            # takes the fitted stack (from the vocalization experiment) and changes the input data
            chimera = copy.deepcopy(stack)
            chimera.modules[0].d_out = [new_data]
            chimera.modules[1].d_in = [new_data]
            chimera.evaluate()

            # appends the ssa metric module
            chimera.append(mt.ssa_index, z_score = 'all', significant_bins='window')

            stacks.append(chimera)


            # extract relevant parameteres into dictionaries
            # starts with tau and u, only present in the model 2
            if model == 'env100_dlog_stp1pc_fir15_dexp_fit01':
                stp = chimera.modules[3]
                stp_mod_parameters = {'Tau': stp.tau, 'U': stp.u}
                for parameter, streams in stp_mod_parameters.items():
                    # organizes into streams
                    streams = {'stream0': streams[0][0], 'stream1': streams[1][0], 'mean': np.nanmean(streams)}
                    for stream, value in streams.items():
                        d = {'cellid': cellid,
                             'stream': stream,
                             'values': value,
                             'model_name': model,
                             'parameter': parameter}
                        df.append(d)

            SI = chimera.modules[-1].SI[0]['pred']
            for stream, value in SI.items():
                d = {'cellid': cellid,
                     'stream': stream,
                     'values': value,
                     'model_name': model,
                     'parameter': 'SI'}
                df.append(d)

        except:
            errorcells.append(cellid)

DF = pd.DataFrame(df)

originalDF = originalDF.append(DF, ignore_index=True)
filename = '171207_batch_SI_simulation_DF'
jl.dump(originalDF,filename)
