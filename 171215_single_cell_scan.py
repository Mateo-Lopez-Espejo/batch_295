import joblib as jl
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import nems.utilities as nu
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import scipy.stats as st
import sklearn.linear_model as lm
import seaborn as sns

'''
This script is a set of utilities to plot interactive scatters in order to see in detail
the outlier of the population. In the past we have seen how sensitive SI is to not so active
cells, giving anomalously high (possitive or negative values)

esentially the same as 171208_batch_SI_stp_plot, however with the adition of interactive plotting and onpick stuff
'''


# define functions
def onpick(event, cellids):
    ind = event.ind
    for ii in ind:
        print('index: {}, cellid: {}'.format(ii, cellids[ii]))
        try:
            print('loading stack')
            modelname = 'env100_dlog_stp1pc_fir15_dexp_fit01'
            stack = nu.io.load_single_model(cellids[ii], 259, modelname)
            stp = stack.modules[3]
            fir = stack.modules[4]
            fig = plt.figure()
            outer = gspec.GridSpec(2,1)
            # pases outer for stp plotting
            stp.plot_fns[2](stp,figure=fig, outer=outer[0])
            # passes simple subplot for STRF
            mod_ax = plt.Subplot(fig, outer[1])
            fig.add_subplot(mod_ax)
            fir.do_plot(fir)
            fig.suptitle('{}  {}'.format(cellids[ii], modelname))
        except:
            print('error plotting: index: {}, cellid: {}'.format(ii, cellids[ii]))

def onpick3D(event):
    ind = event.ind[0]
    x, y, z = event.artist._offsets3d
    print(x[ind], y[ind], z[ind])


# load dataframe (with activity: v2)

# DF = jl.load('/home/mateo/batch_259/171207_batch_SI_simulation_DF')
DF = jl.load('/home/mateo/batch_259/171215_batch_SI_simulation_DF_v2')


# Proper data selectio and DF filtering.

stream = ['stream1'] # this is a pointy thing, given that in many cases one stream
                          # is excitatory while the other

if True: # filters cells based on activity
    act_filter = DF.loc[(DF.stream.isin(stream)) &
                        (DF.parameter == 'activity') &
                        (DF['values'] > 0), :]
else:
    act_filter = DF

filtered = DF.loc[(DF.model_name == 'env100_dlog_stp1pc_fir15_dexp_fit01') &
                   (DF.stream.isin(stream)) &
                   (DF.cellid.isin(act_filter.cellid.tolist())),:]
pivoted = filtered.pivot(index='cellid', columns='parameter', values='values')


title1 = 'batch 295, only excitatory cells, {} values'.format(stream[0])


############################################################################################
# plot SI vs Tau, interactive plot

fig, ax = plt.subplots()
ax.scatter(pivoted['Tau'],pivoted['SI'], picker=True)
ax.set_xlabel('Tau')
ax.set_ylabel('SI')
reg = st.linregress(pivoted['Tau'], pivoted['SI'])
XX = np.asarray(ax.get_xlim())
ax.plot(XX, reg.slope * XX + reg.intercept, color='black',)
fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, pivoted.index.tolist()))

############################################################################################
# plot SI vs U, interactive plot

fig, ax = plt.subplots()
ax.scatter(np.absolute(pivoted['U']),pivoted['SI'], picker=True)
ax.set_xlabel('U')
ax.set_ylabel('SI')
reg = st.linregress(pivoted['U'], pivoted['SI'])
XX = np.asarray(ax.get_xlim())
ax.plot(XX, reg.slope * XX + reg.intercept, color='black',)
fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, pivoted.index.tolist()))

############################################################################################
# interactive 3d plot

Tau = np.asarray(pivoted.Tau)
U = np.absolute(np.asarray(pivoted.U))
SI = np.asarray(pivoted.SI)
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
ax.set_title(title1)
fig.legend(loc='lower right')
fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, pivoted.index.tolist()))
plt.show()

############################################################################################
# 2d plot relating SI or tau or U to activity level

parameter = ['Tau', 'U', 'SI']
for par in parameter:
    f = sns.jointplot(x='activity', y=par, data=pivoted, kind='reg')


############################################################################################


