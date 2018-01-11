import joblib as jl
import pandas as pd
import nems.utilities.io as io
import matplotlib.pyplot as plt
import nems.db as ndb
import seaborn as sbn

'''
some cells have both an experiment in batch 259 and in batch 296, we can take advantage of this to compare
the virtual SI calculated in the batch 259 with the real SI from batch 296

'''
# filtering function (based on activity as z_score) for batch 296

def filterdf(in_DF, jitter=('On', 'Off'), stream='mean', parameter='activity', threshold='mean'):
    df = in_DF.copy()

    if parameter == None:
        return df.cellid.unique().tolist()

    filtered = df.loc[(df.parameter == parameter) &
                      (df.stream == stream) &
                      (df.model_name == 'env100e_stp1pc_fir20_fit01_ssa') &
                      (df.act_pred == 'actual') &
                      (df.Jitter.isin(jitter)),
                      ['cellid', 'values']].drop_duplicates(subset=['cellid'])

    if isinstance(threshold, float):
        metric = threshold
    elif isinstance(threshold, str):
        metric = getattr(filtered['values'], threshold)()
        print('{} {} threshold level: {}'.format(stream, parameter, metric))
    else:
        print('metric should be either a number or a dataframe method like mean()')
        return None

    thresholded = filtered.loc[(filtered['values'] >= metric), :].cellid.tolist()

    return thresholded

# check the list of cell ids in both batches and keeps the shared cellids

SIcells = ndb.get_batch_cells(batch=296).cellid.unique().tolist()
VOCcells = ndb.get_batch_cells(batch=259).cellid.unique().tolist()

sharedcells = [cellid for cellid in VOCcells if cellid in SIcells]

# import both batches digested DF
virtSIDB = jl.load('/home/mateo/batch_259/171207_batch_SI_simulation_DF')
realSIDB = jl.load('/home/mateo/batch_296/171113_refreshed_full_batch_DF')

# select the proper subset of data from each database

# shared columns
param = 'U'
stream = 'mean'
modelnames = ['env100e_stp1pc_fir20_fit01_ssa', 'env100_dlog_stp1pc_fir15_dexp_fit01']

# batch296 columns
goodcells = filterdf(realSIDB, jitter=['Off'])
goodcells = sharedcells # does not filter by activity level.
jitter = 'Off'
act_pred = 'predicted'

concat = pd.concat([virtSIDB, realSIDB])

filtered = concat.loc[((concat.Jitter == jitter) | (pd.isnull(concat.Jitter))) &
                      ((concat.act_pred == act_pred) | (pd.isnull(concat.act_pred))) &
                      (concat.model_name.isin(modelnames)) &
                      (concat.parameter == param) &
                      (concat.stream == stream) &
                      (concat.cellid.isin(sharedcells)) &
                      (concat.cellid.isin(goodcells)), :].drop_duplicates(['cellid', 'model_name'])

pivoted = filtered.pivot(index='cellid', columns='model_name', values='values')
f = sbn.jointplot(x=modelnames[0], y=modelnames[1], data=pivoted, kind='reg')
f.set_axis_labels(xlabel='oddball {} {}'.format(act_pred, param),
                  ylabel='vocalization predicted {}'.format(param))
f.fig.suptitle('{} comparison, stream: {}'.format(param, stream))
