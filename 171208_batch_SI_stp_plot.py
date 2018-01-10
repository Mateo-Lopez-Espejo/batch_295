import joblib as jl
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import scipy.linalg
import numpy as np
import copy
import sklearn.linear_model as lm

'''
firs rough itertion of a linear regression between Si and (Tau and U), i.e. SI = f(tau, U)
in this case using the simulated SI, U and Tau values obtained by evaluating an arbitrary oddball parading.
'''

DF = jl.load('/home/mateo/batch_259/171207_batch_SI_simulation_DF')

stream = ['cell', 'mean']
parameter = ['U', 'Tau']


filtered = DF.loc[(DF.stream.isin(stream)) &
                  (DF.model_name == 'env100_dlog_stp1pc_fir15_dexp_fit01'),:]

pivoted = filtered.pivot(index='cellid', columns='parameter', values = 'values')


for par in parameter:
    f = sns.jointplot(x=par, y='SI', data=pivoted, kind='reg')

f = sns.jointplot(x='Tau', y='U', data=pivoted, kind='reg')

# combined facetgrid plot to check differences between streams
WDF = DF.copy()
WDF.loc[(WDF.stream == 'mean'),'stream'] = 'cell'
filtered = WDF.loc[(WDF.model_name == 'env100_dlog_stp1pc_fir15_dexp_fit01'),:]
indexed = filtered.set_index(['cellid', 'parameter', 'stream'])['values']
pivoted = indexed.unstack(['parameter']).reset_index(['stream'])
f = sns.lmplot(x='Tau', y='SI', col='stream', data=pivoted, truncate=True, size=5, fit_reg=True)

# 3D plot of SI as function of Tau and U
filtered = WDF.loc[(WDF.model_name == 'env100_dlog_stp1pc_fir15_dexp_fit01') &
                   (WDF.stream == 'cell'),:]
pivoted = filtered.pivot(index='cellid', columns='parameter', values='values')


def SI_prediction(Tau, U, SI):

    data = np.c_[Tau, U, SI]
    # regular grid covering the domain of the data

    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()

    order = 1  # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        clf = lm.LinearRegression()
        clf.fit(np.c_[Tau, U], SI)

        # evaluate it on grid
        Z = clf.coef_[0] * X + clf.coef_[1] * Y + clf.intercept_

        r_est = clf.score(np.c_[Tau, U], SI)

        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    elif order == 2:
        # best-fit quadratic curve
        clf = lm.LinearRegression()
        clf.fit(np.c_[Tau, U], SI ** 2)

        # evaluate it on a grid
        Z = clf.coef_[0] * X + clf.coef_[1] * Y + clf.intercept_

        r_est = clf.score(np.c_[Tau, U], SI)

    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2],label='SI = {:.3g} * Tau + {:.3g} U + {:.3g} \nr_est = {:.3g}'.format(
                        clf.coef_[0], clf.coef_[1], clf.intercept_, r_est))
    ax.axis('equal')
    ax.axis('tight')
    ax.set_xlabel('Tau')
    ax.set_ylabel('U')
    ax.set_zlabel('SI')
    ax.set_title('batch 259, predicted SI, Tau, U')
    ax.legend()
    plt.show()

SI_prediction(np.asarray(pivoted.Tau), np.absolute(np.asarray(pivoted.U)), np.asarray(pivoted.SI))
