"""
Clusters igts segments.

I hated this part methodology with a passion, please do not use.
"""
#!/usr/bin/env python
# coding: utf-8

# In[188]:


import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn.metrics import silhouette_score


def transform_multivariate_to_univariate(p):
    """
    Aggregates a multivariate pdf to a univariate one
    p comes from windowing, and has dimensionts number of variables x number
    of windows
    """
    pn = np.zeros(p.shape[0])
    fullsum = np.sum(p)
    if fullsum == 0:
        return pn
    return np.divide(np.sum(p, axis=1), fullsum)


def cluster_segments(segmentpickle):
    max_model_score = 0
    max_model = None
    sil_score = list()
    segments = pd.read_pickle(segmentpickle)
    segments['distribution'] = segments['seg'].apply(transform_multivariate_to_univariate)
    distributions = np.stack(np.array(segments['distribution']))
    print(distributions.shape)

    for clusters in range(3, 20):
        model = cluster.KMeans(n_clusters=clusters).fit(distributions)
        sil_avg = silhouette_score(distributions, model.labels_)
        print("Sil score: " + str(sil_avg))
        sil_score.append(sil_avg)

        if sil_avg > max_model_score:
            max_model = model
            max_model_score = sil_avg

    segments['label'] = max_model.labels_
    segments.to_pickle(segmentpickle)
