
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytz
import scipy.stats as stats
from dateutil.parser import parse

def corralate_text(segmentsfile, surveytextfile, usersfile, clusternamesfile):
    surveys = pd.read_csv(surveytextfile).fillna(0)
    segments = pd.read_pickle(segmentsfile)
    users = pd.read_csv(usersfile)
    all_users = users['code'].unique()
    clusternames = pd.read_csv(clusternamesfile)
    females = users[users['gender'] == 0]['code'].unique()
    males = users[users['gender'] == 1]['code'].unique()
    employed = users[users['employed']]['code'].unique()
    not_employed = users[~users['employed']]['code'].unique()
    age_18_20 = users[(users['age'] <= 20) & (users['age'] >= 18)]['code'].unique()
    age_21_25 = users[(users['age'] <= 25) & (users['age'] >= 21)]['code'].unique()
    age_26_30 = users[(users['age'] <= 30) & (users['age'] >= 26)]['code'].unique()
    age_30_plus = users[users['age'] >= 30]['code'].unique()

    demographics = [all_users]
    # Q6 = Happiness, Q7 = Work Life Balance, Q11 = Energy, Q10 = Difference from yesterday
    questions = ['Q6', 'Q7', 'Q11', 'Q10']

    surveys['record_time'] = surveys['record_time'].apply(parse)
    features = ['cogproc', 'negemo','posemo', 'work']

    demographic_count = 0
    for demographic in demographics:
        top_k_rows = list()
        top_k_index = list()
        for question in questions:
            rows = list()
            for _, segment in segments.iterrows():
                start_time = segment['from']
                end_time = segment['to']
                code = segment['code']
                if code in demographic:
                    daily_end = end_time + pd.Timedelta(hours=24)
                    segment_dailies = surveys[(surveys['record_time'] > start_time) & (surveys['record_time'] < daily_end) & (surveys['record_code'] == code) & (surveys['record_question'] == question)]
                    if len(segment_dailies) > 0:
                        first = segment_dailies.iloc[0]
                        means = segment_dailies[segment_dailies['record_time'] == first['record_time']].sum()
                        row = {'label': segment['label']}
                        for feature in features:
                            row[feature] = means[feature]
                        rows.append(row)
            df = pd.DataFrame(rows)
            features = list(set(df.columns) - {'label'})
            label_count = max(segments['label']) + 1
            print(df)

            rows = list()
            corr_rows = list()
            for i in range(label_count):
                row = {}
                corr_row = {}
                for feature in features:
                    filtered = df[['label', feature]].dropna()
                    if len(filtered) > 1:
                        (corr, p_value) = stats.spearmanr(filtered['label'] == i, filtered[feature])
                        row[feature + '_corr'] = corr
                        corr_row[feature] = corr
                        row[feature + '_p'] = p_value
                    else:
                        row[feature + '_corr'] = float('nan')
                        corr_row[feature] = float('nan')
                        row[feature + '_p'] = float('nan')
                rows.append(row)
                corr_rows.append(corr_row)

            p_values = pd.DataFrame(rows)
            corr_df = pd.DataFrame(corr_rows)
            for column in corr_df:
                sorted_column = corr_df[column].sort_values()
                head = sorted_column.head(1)
                tail = sorted_column.tail(1)
                notable_correlations = head.append(tail)
                new_labels = []
                for index,value in notable_correlations.sort_values().iteritems():
                    cluster = index
                    p_value = p_values[column + '_p'][cluster]
                    r = p_values[column + '_corr'][cluster]
                    label = "{} (r: {: .2f}, p: {:.1e})".format(clusternames.at[cluster,'name'], r, p_value)
                    new_labels.append(label)
                top_k_rows.append(new_labels)
                top_k_index.append("{} & {}".format(question, column))

            p_values.to_csv("correlations_text_{}_{}.csv".format(demographic_count,question))
            print("Demographic {} {} done".format(demographic_count, question))
        demographic_count += 1
    top_k = pd.DataFrame(top_k_rows,index=top_k_index)
    print(top_k.to_latex(escape=False))
    top_k.to_csv("top_k.csv")

def corralate_segments(segmentsfile, surveysfile, endofday_file, usersfile, clusternamesfile):
    surveys = pd.read_csv(surveysfile)
    segments = pd.read_pickle(segmentsfile)
    users = pd.read_csv(usersfile)
    clusternames = pd.read_csv(clusternamesfile)

    all_users = users['code'].unique()
    females = users[users['gender'] == 0]['code'].unique()
    males = users[users['gender'] == 1]['code'].unique()
    age_18_24 = users[(users['age'] <= 24) & (users['age'] >= 18)]['code'].unique()
    age_25_plus = users[users['age'] >= 25]['code'].unique()

    demographics = { 'All': all_users
                   , 'Male': males
                   , 'Female': females
                   , 'Age 18-24': age_18_24
                   , 'Age >= 25': age_25_plus
                   }

    
    ts_col = surveys.time
    surveys['time'] = pd.to_datetime(ts_col, unit='ms')
    surveys['working'] = surveys['role'].replace({1:0, 3:0.5, 2:1})
    surveys['valence'] = surveys['valence'].replace({2:1, 3:2, 4:3,5:3})
    surveys['arousal'] = surveys['arousal'].replace({2:1, 3:2, 4:3,5:3})
    surveys = surveys.replace({-1: np.nan})

    daily_surveys = pd.read_csv(endofday_file)
    daily_surveys['time'] = daily_surveys['StartDate'].apply(parse)
    daily_surveys['Q1'] = daily_surveys['Q1'].replace({1:1,2:1,3:2,4:3,5:3})
    daily_surveys['Q2'] = daily_surveys['Q2'].replace({1:1,2:1,3:2,4:3,5:3})


    demographic_count = 0
    top_k_rows = list()
    top_k_index = list()
    for demographic_name, demographic in demographics.items():
        rows = list()
        for _, segment in segments.iterrows():
            start_time = segment['from']
            end_time = segment['to']
            code = segment['code']
            if code in demographic:
                survey_end = end_time + pd.Timedelta(hours=1)
                segment_surveys = surveys[(surveys['time'] > start_time) & (surveys['time'] < survey_end) & (surveys['code'] == code)]
                if len(segment_surveys) > 0:
                    valence = segment_surveys['valence'].mean()
                    arousal = segment_surveys['arousal'].mean()
                    working = segment_surveys['working'].mean()
                    row = {'label':  segment['label'],
                           'happiness': valence,
                           'energy': arousal,
                           }
                    rows.append(row)

                role_end = end_time + pd.Timedelta(hours=0.25)
                segment_surveys = surveys[(surveys['time'] > start_time) & (surveys['time'] < survey_end) & (surveys['code'] == code)]
                if len(segment_surveys) > 0:
                    working = segment_surveys['working'].mean()
                    row = {'label':  segment['label'],
                           'working': working
                           }
                    rows.append(row)
                
                daily_end = end_time + pd.Timedelta(hours=24)
                segment_dailies = daily_surveys[ (daily_surveys['time'] > start_time) & (daily_surveys['time'] < daily_end) & (daily_surveys['code'] == code)]
                if len(segment_dailies) > 0:
                    next_response = segment_dailies.iloc[0]
                    row = {'productivity': - next_response['Q1']
                          ,'interruptions': - next_response['Q2']
                          ,'label': segment['label']
                          }
                    rows.append(row)
        features = ['happiness', 'energy', 'working', 'productivity','interruptions']
        df = pd.DataFrame(rows,columns=features + ['label'])
        label_count = max(segments['label']) + 1

        rows = list()
        corr_rows = list()
        for i in range(label_count):
            row = {}
            corr_row = {}
            for feature in features:
                filtered = df[['label',feature]].dropna()
                if len(filtered) > 1:
                    (corr, p_value) = stats.spearmanr(filtered['label'] == i, filtered[feature])
                    row[feature + '_corr'] = corr
                    corr_row[feature] = corr
                    row[feature + '_p'] = p_value
                else:
                    row[feature + '_corr'] = float('nan')
                    corr_row[feature] = float('nan')
                    row[feature + '_p'] = float('nan')
            rows.append(row)
            corr_rows.append(corr_row)

        p_values = pd.DataFrame(rows)
        corr_df = pd.DataFrame(corr_rows)
        table = {}
        for column in corr_df:
            sorted_column = corr_df[column].sort_values()
            head = sorted_column.head(1)
            tail = sorted_column.tail(1)
            notable_correlations = head.append(tail)
            new_labels = []
            for index,value in notable_correlations.sort_values().iteritems():
                cluster = index
                p_value = p_values[column + '_p'][cluster]
                r = p_values[column + '_corr'][cluster]
                label = "{} (r: {: .2f}, p: {:.1e})".format(clusternames.at[cluster,'name'], r, p_value)
                new_labels.append(label)
            top_k_rows.append(new_labels)
            top_k_index.append("{} & {}".format(demographic_name, column))

        p_values.to_csv("correlations_{}.csv".format(demographic_count))
        print("Demographic {} done".format(demographic_count))
        demographic_count += 1
    top_k = pd.DataFrame(top_k_rows,index=top_k_index)
    print(top_k.to_latex(escape=False))
    top_k.to_csv("top_k.csv")

def safe_point_biserial(dich, value):
    df = pd.DataFrame({'dich': dich, 'value': value})
    _, p = stats.shapiro(df[df.dich == False]['value'])
    if p < 0.017:
        print("NOT NORMAL UNDER FALSE ({})".format(p))
        print(df[df.dich == False]['value'].value_counts().sort_index())
        return (np.nan, np.nan)
    _, p = stats.shapiro(df[df.dich == True]['value'])
    if p < 0.017:
        print("NOT NORMAL UNDER TRUE ({})".format(p))
        print(df[df.dich == True]['value'].value_counts().sort_index())
        return (np.nan, np.nan)
    _, p = stats.levene(df[df.dich == False]['value'], df[df.dich == True]['value'])
    if p < 0.017:
        print("NOT EQUAL VARIANCE")

    return stats.pointbiserialr(dich, value)


def normalise_frame(df, columns):
    def normalise_user(user_df):
        frame = pd.DataFrame(user_df) 
        for column in columns:
            frame[column] = normalise_scores(user_df[column])
        return frame
    return df.groupby('code').apply(normalise_user)


def normalise_scores(scores):
    mean = scores.mean()
    std = scores.std()
    return (scores - mean)/std
