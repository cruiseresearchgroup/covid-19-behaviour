
import pandas as pd
import numpy as np
import pytz
import scipy.stats as stats


def corralate_segments(segmentsfile, labelsfile, surveysfile):
    surveys = pd.read_csv(surveysfile)
    labels = np.load(labelsfile)
    segments = np.load(segmentsfile, allow_pickle=True)

    ts_col = surveys.time
    surveys['time'] = pd.to_datetime(ts_col, unit='ms')

    rows = list()

    for i in range(len(labels)):
        print(segments[i])
        start_time = segments[i]['from']
        end_time = segments[i]['to']
        survey_end = end_time + pd.Timedelta(hours=1.5)
        segment_surveys = surveys[(surveys['time'] > start_time) & (surveys['time'] < survey_end)]
        if len(segment_surveys) > 0:
            valence = segment_surveys.mean()['valence']
            arousal = segment_surveys.mean()['arousal']
            row = {'label':  labels[i],
                   'valence': valence,
                   'arousal': arousal
                   }
            rows.append(row)
    df = pd.DataFrame(rows)
    label_count = max(labels) + 1

    rows = list()
    for i in range(label_count):
        (valence_corr, valence_p) = stats.pointbiserialr(df['label'] == i, df['valence'])
        (arousal_corr, arousal_p) = stats.pointbiserialr(df['label'] == i, df['arousal'])
        row = {'valence_corr': valence_corr,
               'valence_p': valence_p,
               'arousal_corr': arousal_corr,
               'arousal_p': arousal_p
               }
        rows.append(row)
    pd.DataFrame(rows).to_csv("correlations.csv")
