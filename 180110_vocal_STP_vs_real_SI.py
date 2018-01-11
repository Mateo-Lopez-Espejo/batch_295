import joblib as jl
import pandas as pd
import nems.utilities.io as io
import matplotlib.pyplot as plt
import nems.db as ndb
import seaborn as sbn
import sklearn.linear_model as lm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

'''
comparison of common cells between baches 259 and 296

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

# 3d plotting function with multiple linear regression

def plot3d (tau, u, si, title='placeholder'):

    Tau = np.asarray(tau)
    U = np.absolute(np.asarray(u))
    SI = np.asarray(si)
    #act = np.asarray(pivoted.activity)

    data = np.c_[Tau, U, SI]#, act]
    # regular grid covering the domain of the data
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))

    # best-fit linear plane
    clf = lm.LinearRegression()
    clf.fit(np.c_[Tau, U], SI)
    # evaluate it on grid
    Z = clf.coef_[0] * X + clf.coef_[1] * Y + clf.intercept_
    r_est = clf.score(np.c_[Tau, U], SI)

    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    scat = ax.scatter(data[:, 0], data[:, 1], data[:, 2], picker=True,
               label='SI = {:.3g} * Tau + {:.3g} U + {:.3g} \nr_est = {:.3g}'.
               format(clf.coef_[0], clf.coef_[1], clf.intercept_, r_est))
    # formats axis, titles and stuff
    ax.axis('equal')
    ax.axis('tight')
    ax.set_xlabel('Tau')
    ax.set_ylabel('U')
    ax.set_zlabel('SI')
    ax.set_title(title)
    fig.legend(loc='lower right')
    #fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, pivoted.index.tolist()))
    plt.show()


# check the list of cell ids in both batches and keeps the shared cellids

SIcells = ndb.get_batch_cells(batch=296).cellid.unique().tolist()
VOCcells = ndb.get_batch_cells(batch=259).cellid.unique().tolist()

sharedcells = [cellid for cellid in VOCcells if cellid in SIcells]

# import both batches digested DF
virtSIDB = jl.load('/home/mateo/batch_259/171215_batch_SI_simulation_DF_v2')
realSIDB = jl.load('/home/mateo/batch_296/171113_refreshed_full_batch_DF')
# adds columns to virtSIDB to ease later analisis
virtSIDB['act_pred'] = np.nan
virtSIDB['Jitter'] = np.nan



################################################
# common plotting parameters
modelnames = ['env100e_stp1pc_fir20_fit01_ssa', 'env100_dlog_stp1pc_fir15_dexp_fit01']
jitter = 'Off'

stream = ['mean', 'cell']
act_pred = 'actual'
badcells = ['gus019d-b1']  #wierd outlier

##############################################################################
# mixed 3d plot, can take Tau and U or SI from one or other batch

goodcells = filterdf(realSIDB, jitter=['Off'])
#goodcells = sharedcells # does not filter by activity level.

# defines wich batch gives stp and which gives ssa
stpvals = realSIDB
sivals = virtSIDB


stps = stpvals.loc[(stpvals.parameter.isin(['Tau', 'U', 'activity'])), :]
sis = sivals.loc[(sivals.parameter == 'SI'), :]

concat = pd.concat([stps, sis])

if True: # filters cells based on activity
    act_filter = concat.loc[(concat.stream.isin(stream)) &
                        (concat.parameter == 'activity') &
                        (concat['values'] > 0), :]
else:
    act_filter = concat

filtered = concat.loc[(concat.model_name.isin(modelnames)) &
                  (concat.stream.isin(stream)) &
                  (concat.cellid.isin(goodcells)) &
                  (concat.cellid.isin(act_filter.cellid.tolist())) &
                  (concat.cellid.isin(sharedcells)) &
                  ((concat.Jitter == 'Off') | (pd.isnull(concat.Jitter)))&
                  ((concat.act_pred == act_pred) | (pd.isnull(concat.act_pred)))
                   ,:].drop_duplicates(['cellid', 'parameter'])


pivoted = filtered.pivot(index='cellid', columns='parameter', values='values').dropna(axis=0)
try:
    pivoted = pivoted.drop(badcells)
except:
    pass

figname = 'batch259 stp (Tau, U) vs batch296 {} ssa (SI), jitter {}, stream: {}'.format(
           act_pred,jitter, stream[0])
plot3d(pivoted.Tau, pivoted.U, pivoted.SI, figname)

###############################################################################

# there is some strong discrepancy with previous analisis: 171215. this at the
# level of the regression R value. here i parse some previous code for easier
# comparison. it seems the biggest issue is with filtering out negative values.

source = 1

if source == 1:
    DF = virtSIDB

elif source == 2:
    DF = realSIDB

goodcells = filterdf(realSIDB, jitter=['Off'])


if True: # filters cells based on activity
    act_filter = DF.loc[(DF.stream.isin(stream)) &
                        (DF.parameter == 'activity') &
                        (DF['values'] > 0), :]
else:
    act_filter = DF


if source == 1:
    filtered = DF.loc[(DF.model_name.isin(modelnames)) &
                      (DF.stream.isin(stream)) &
                      (DF.cellid.isin(goodcells)) &
                      (DF.cellid.isin(act_filter.cellid.tolist())) &
                      (DF.cellid.isin(sharedcells))
                       ,:].drop_duplicates(['cellid', 'parameter'])

elif source == 2:
    filtered = DF.loc[(DF.model_name.isin(modelnames)) &
                      (DF.stream.isin(stream)) &
                      (DF.cellid.isin(goodcells)) &
                      (DF.cellid.isin(act_filter.cellid.tolist())) &
                      (DF.cellid.isin(sharedcells)) &
                      ((DF.Jitter =='Off') | (pd.isnull(DF.Jitter)))&
                      ((DF.act_pred == act_pred) | (pd.isnull(DF.act_pred)))
                       ,:].drop_duplicates(['cellid', 'parameter'])

pivoted = filtered.pivot(index='cellid', columns='parameter', values='values').dropna(axis=0)

try:
    pivoted = pivoted.drop(badcells)
except:
    pass

title1 = 'batch 295, only excitatory cells, {} values'.format(stream[0])

plot3d(pivoted.Tau, pivoted.U, pivoted.SI, title1)

