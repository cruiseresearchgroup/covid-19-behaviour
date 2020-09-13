#!/usr/bin/env python
# coding: utf-8

# In[188]:


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


def cluster_segments(filename):
    max_model_score = 0
    max_model = None
    sil_score = list()
    segments = np.load(filename, allow_pickle=True)
    print(len(segments))
    for segment in segments:
        if 'seg' in segment:
            print("Yippee")
        else:
            print(segment)
            print("Yaaaa")
    distributions = np.array([transform_multivariate_to_univariate(seg['seg'])
                              for seg in segments])

    print(distributions)
    for clusters in range(3, 30):
        model = cluster.KMeans(n_clusters=clusters).fit(distributions)
        print(model.labels_)
        sil_avg = silhouette_score(distributions, model.labels_)
        print(sil_avg)
        sil_score.append(sil_avg)

        if sil_avg > max_model_score:
            max_model = model
            max_model_score = sil_avg

    np.save(filename + "-labels", max_model.labels_)
