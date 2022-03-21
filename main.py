""" 
A python tool for the analysis of COVID-19 Social roles
You should be able to run this with:
> python main.py [command]

"""
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv
import util
import base64
import math

# These imports contain other python files for specific tasks
import segment
import cluster
import correlate
import demographics
import difference
import venn

import pytz
from dateutil.parser import parse
import scipy.stats as stats
import re

# This is a blacklist of applications that should not appear in the application
# Log, mainly because they represent events that are not app usages, as well
# as usage of the tracking application
blacklist = {
     'SCREEN',
     'LOCATION',
     'AUDIO',
     'ACTIVITY',
     'Keyboard',
     'Mouse clicked',
     'ESM',
     'Location',
     'Mouse scrolled',
     'Battery',
     'WIFI',
     'BLUETOOTH',
     'Balance'
     }

required_columns = {
    'code',
    'application_name',
    'application_title',
    'time_issued',
    'application_package',
    'esm_role',
    'esm_arousal',
    'esm_task',
    'esm_valence',
    'esm_interruption',
    'time_zone',
    'event_type',
    'time_issued',
    'bluetooth_address',
    'bluetooth_rssi',
    'wifi_bssid',
    'wifi_rssi'
    }


def load_data():
    """
    Loads the data, performing validation checks and warnings
    """
    try:
        df = pd.read_csv(sys.argv[2])
        util.check_columns(df, required_columns)

        non_apps = set(df.application_name.unique()).intersection(blacklist)

        if len(non_apps) > 0:
            print("""
            Warning, there folowing records in this CSV are not app usage.
            Call python main.py filter [infile] [outfile]
            to get a filtered version
            """)
            print(non_apps)

        return df
    except FileNotFoundError:
        print("Could not find file")
        sys.exit(1)


def nb64(x):
    length = math.ceil(math.log(x + 2, 256))
    b64_bytes = base64.b64encode((x + 1).to_bytes(length, byteorder='big'))
    return b64_bytes.decode('utf-8')


def get_coded(codemap, new_item):
    given_code = new_item
    new_code = ""
    if given_code in codemap:
        new_code = codemap[given_code]
    else:
        new_code = str(len(codemap))
        codemap[given_code] = new_code
    return new_code


def filter_data(infilename):
    """
    Filtering data takes an events file and turns it into a activity.csv and
    surveys.csv for app usage activity and surveys respectively.

    The events file should be events_anon.csv in the data directory
    """
    codes = {}
    apps = {}
    min_time = float('inf')
    with open(infilename, 'r') as infile:
        with open('activity.csv', 'w') as outfile:
            with open('surveys.csv', 'w') as surveyoutfile:
                header = True
                index_map = None
                writer = csv.writer(outfile)
                surveywriter = csv.writer(surveyoutfile)
                line = 0
                for row in csv.reader(iter(infile.readline, '')):
                    if header:
                        index_map = {
                           name: row.index(name) for name in required_columns
                        }
                        header = False
                        writer.writerow([
                            'time',
                            'device_type',
                            'code',
                            'app',
                            'bluetooth_rssi',
                            'bluetooth_address',
                            'wifi_bssid',
                            'wifi_rssi'
                            ])
                        surveywriter.writerow([
                            'user',
                             'valence',
                             'arousal',
                             'role',
                             'interruption',
                             'task',
                             'device_type',
                             'time'
                             ])
                    else:
                        if row[index_map['esm_valence']] != '-1':
                            surveywriter.writerow([
                                row[index_map['code']],
                                row[index_map['esm_valence']],
                                row[index_map['esm_arousal']],
                                row[index_map['esm_role']],
                                row[index_map['esm_interruption']],
                                row[index_map['esm_task']],
                                row[index_map['event_type']],
                                row[index_map['time_issued']]
                                ])

                        if row[index_map['application_name']] not in blacklist:
                            min_time = min(min_time, index_map['time_issued'])
                            writer.writerow([
                                row[index_map['time_issued']],
                                row[index_map['event_type']],
                                row[index_map['code']],
                                row[index_map['application_name']],
                                row[index_map['bluetooth_rssi']],
                                row[index_map['bluetooth_address']],
                                row[index_map['wifi_bssid']],
                                row[index_map['wifi_rssi']]
                                ])

    # Prints the earliest time for the purpose of determining what
    # collection period this was
    print("First time recorded on this dataset")
    print(min_time)
    with open('users.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['user_id', 'code'])
        for key, value in codes.items():
            writer.writerow([key, value])

    with open('apps.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['app_package', 'code'])
        for key, value in apps.items():
            writer.writerow([key, value])


# The main function
def main():
    """
    Usage: python main.py users  [file]
                          apps   [file]
                          segment [file]
                          cluster [files]
                          filter [infile]
                          demographics [usersfile]
                          clusterprototypes [segments] [
                          correlate [segments] [labels] [surveys]
    """


    if len(sys.argv) == 3 and sys.argv[1] == "filter":
        # Filters the event information, creating the activity and surveys file
        filter_data(sys.argv[2])

    elif len(sys.argv) == 4 and sys.argv[1] == "segment":
        # Takes an activity file (from filter) and categories.csv to segment
        segment.segment(sys.argv[2], sys.argv[3])
    
    elif len(sys.argv) == 3 and sys.argv[1] == "cluster":
        # Clusters the result of segments. BAD METHODOLOGY: DO NOT USE
        cluster.cluster_segments(sys.argv[2])

    elif len(sys.argv) == 7 and sys.argv[1] == "correlate":
        # Correlates the results of clusters. DISGUSTING METHODOLOGY: DO NOT USE
        correlate.alternative_corralate_segments(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

    elif len(sys.argv) == 6 and sys.argv[1] == "correlatetext":
        # Correlates the results of clusters with text using LIWC. BAD METHODOLOGY: DO NOT USE
        correlate.corralate_text(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

    elif len(sys.argv) == 3 and sys.argv[1] == "demographics":
        # Creates basic demographic information from the raw users.csv
        demographics.create_demographic_graphs(sys.argv[2])

    elif len(sys.argv) == 6 and sys.argv[1] == "clusterprototypes":
        # Creates prototypes for cluster activities. CONFUSING METHODOLOGY: DO NOT USE
        cluster_prototypes(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

    elif len(sys.argv) == 5 and sys.argv[1] == "drawcorrelations":
        # Draws collecation graphs. MISLEADING METHODOLOGY: DO NOT USE
        draw_correlations(sys.argv[2], sys.argv[3], sys.argv[4])
        
    elif len(sys.argv) == 5 and sys.argv[1] == "drawtextcorrelations":
        # Draws collecation graphs for text. NOISY METHODOLOGY: DO NOT USE
        draw_text_correlations(sys.argv[2], sys.argv[3], sys.argv[4])

    elif len(sys.argv) == 3 and sys.argv[1] == "extracttext":
        # Extracts text for use with LIWC.
        extracttext(sys.argv[2])

    elif len(sys.argv) == 3 and sys.argv[1] == "surveydistribution":
        # Creates a graph with the amounts of surveys created by each person.
        # Takes the surveys.csv file from filter
        surveydistribution(sys.argv[2])

    elif len(sys.argv) == 3 and sys.argv[1] == "topcategories":
        # Creates a graph with the top categories used. Takes the categories.csv file
        top_categories(sys.argv[2])

    elif len(sys.argv) == 4 and sys.argv[1] == "clusterdistrib":
        # Creates a pie chart over activities. DO NOT USE UNDER ANY CIRCUMSTANCE
        cluster_distrib(sys.argv[2], sys.argv[3])

    elif len(sys.argv) == 3 and sys.argv[1] == "vennwork":
        # Creates charts detailing the amount of work each person did during each
        # time period. Considering we are only working on one time period now,
        # This may not be useful. However the code inside it could be
        venn.venn_work(sys.argv[2])

    elif len(sys.argv) == 4 and sys.argv[1] == "count":
        # Counts many items of interest in the dataset and outputs LaTeX for use
        # argument 1 are activities, argument 2 are surveys, both from filter.
        count_points(sys.argv[2], sys.argv[3])

    elif len(sys.argv) == 5 and sys.argv[1] == "correlateinfluence":
        # Calculates the influence that COVID-19 was reported to have on ESM results
        correlate_influence(sys.argv[2], sys.argv[3], sys.argv[4])

    elif len(sys.argv) == 6 and sys.argv[1] == "diffyesterday":
        # Calculates whether when someone reported yesterday was different actually
        # meant it was different. Oh my goodness. Don't use this please.
        difference.difference_yesterday(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

    elif len(sys.argv) > 2 and sys.argv[1] == "picklesegments":
        # Collects the segments charts in one file. DO NOT USE, this is to input
        # into the correlation portion
        picklesegments(sys.argv[2:])
    else:
        print(main.__doc__)

def picklesegments(files):
    segments = list()
    for filename in files:
        regex = re.compile("\/([^-]+)-([^-]+)-segments.npy")
        print(filename)
        match = regex.search(filename)
        code = match.group(1)
        device = match.group(2)
        print(f"{code} on {device}")
        user_segments = np.load(filename, allow_pickle=True)
        for segment in user_segments:
            segment['code'] = code
            segment['device'] = device
            segments.append(segment)
    df = pd.DataFrame(segments)
    df.to_pickle("segments.pkl")



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


def draw_text_correlations(correlationsfile, correlationsnamefile, title):
    correlations = pd.read_csv(correlationsfile)
    correlation_labels = pd.read_csv(correlationsnamefile)
    correlations = correlations.set_index(correlation_labels['name'])

    features = ['cogproc', 'negemo','posemo','friend','work']

    feature_ps = [feature + '_p' for feature in features]
    p_values = correlations[feature_ps]
    feature_corr = [feature + '_corr' for feature in features]
    coefficients = correlations[feature_corr]
    fig, ax = plt.subplots(figsize=(10,10))

    rows = []
    for _, label in correlations.iterrows():
        new_row = {}
        for feature in features:
            prefix = ""
            if label[feature + '_corr'] > 0:
                prefix = "(+)"
            else:
                prefix = "(-)"
            new_row[feature] ="{} {:.2e}".format(prefix, label[feature + '_p'])
        rows.append(new_row)
    plot_labels = pd.DataFrame(rows)
    print(plot_labels)
    print(p_values)

    plt.title(title + " p values")

    sns.heatmap(p_values, fmt="s",annot=plot_labels, ax=ax, ylabel="Cluster")
    plt.savefig("p_values_text.png", bbox_inches='tight')
    plt.close()


    fig, ax = plt.subplots(figsize=(10,10))
    plt.title(title + " coefficients")
    g = sns.heatmap(coefficients, annot=True, vmin=coefficients.min().min(), vmax=coefficients.max().max(), center=0, ax=ax)
    plt.savefig("coefficients_text.png", bbox_inches='tight')
    plt.close()


def draw_correlations(correlationsfile, correlationsnamefile, title):
    correlations = pd.read_csv(correlationsfile)
    correlation_labels = pd.read_csv(correlationsnamefile)
    features = ['happiness', 'alertness', 'productivity', 'interruptions']
#    if daily:
#        features = ['productivity','interruptions','influence_on_happiness','influence_on_energy','influence_on_work_life_balance', 'difference_from_yesterday']
    correlations = correlations.set_index(correlation_labels['name'])
    
    print(correlations)

    p_values = correlations.filter(regex="^(" + "|".join(features) + ")_p$")
    coefficients = correlations.filter(regex="^(" + "|".join(features) + ")_corr$")
    rows = []
    for _, label in correlations.iterrows():
        new_row = {}
        for feature in features:
            prefix = ""
            if label[feature + '_corr'] > 0:
                prefix = "(+)"
            else:
                prefix = "(-)"
            new_row[feature] ="{} {:.2e}".format(prefix, label[feature + '_p'])
        rows.append(new_row)
    plot_labels = pd.DataFrame(rows)

    print(plot_labels)
    print(p_values)

    fig, ax = plt.subplots(figsize=(6,6)) 
    plt.title(title + " p values")
    ax.axvline(2)
    sns.heatmap(p_values, annot=plot_labels, fmt="s", ax=ax)
    ax.set(xlabel="variable", ylabel="activity")
    plt.savefig("p_values.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6,6))
    plt.title(title + " coefficients")
    ax.axvline(2)
    coefficients.columns = [name[:-5] for name in coefficients.columns]
    print(coefficients)
    sns.heatmap(coefficients, annot=True, fmt=".4f", vmin=-0.2, vmax=0.2, center=0, ax=ax)
    ax.set(xlabel="variable", ylabel="activity")
    plt.savefig("coefficients.png", bbox_inches='tight')
    plt.close()


def extracttext(surveysfile):
    end_of_day = pd.read_csv(surveysfile)
    filtered = end_of_day[['code','StartDate','Q6','Q7','Q10','Q11']]
    pd.melt(filtered, id_vars=['code','StartDate'], value_vars=['Q6','Q7','Q10','Q11']).dropna().to_csv("surveytext.csv")




def cluster_prototypes(picklefile, activityfile, categoriesfile, clusternamesfile):
    segments = pd.read_pickle(picklefile)
    activity = pd.read_csv(activityfile)
    categories_df = pd.read_csv(categoriesfile)
    clusternames = pd.read_csv(clusternamesfile)

    ts_col = activity.time
    utc_times = pd.to_datetime(ts_col, unit='ms').dt.tz_localize(tz=pytz.utc)
    aus_times = utc_times.dt.tz_convert(
        tz=pytz.timezone('Australia/Melbourne'))
    activity['time'] = aus_times

    activity = pd.merge(left=activity, right=categories_df, left_on='app', right_on='app')
    activity.sort_values('time', inplace=True)
    all_segments = list()
    activity = activity[activity['category'] != 'Unknown']
    categories = activity.category.unique()
    all_categories = activity.category.unique()

    cluster_dist = pd.DataFrame(index=all_categories)
    for label in range(segments['label'].max() + 1):
        my_segments = segments[segments['label']==label]['seg']
        dist = my_segments.apply(transform_multivariate_to_univariate).mean(axis=0)
        cluster_dist[clusternames.at[label,'name']] = dist
    max_cat = cluster_dist.max(axis=1)
    print(max_cat.sort_values())
    cluster_dist = cluster_dist[max_cat > 0.1]
    plt.subplots(figsize=(4,3))
    g = sns.heatmap(cluster_dist.transpose(),xticklabels=True, yticklabels=True, cbar_kws={'label': 'Proportion in mean Segment'})
    plt.xlabel("App Category")
    plt.ylabel("Cluster")
    plt.savefig("clusterprototypes.png", bbox_inches='tight')


def surveydistribution(survey_file):
    survey_file = pd.read_csv(survey_file)
    plt.xlabel("Response Count")
    plt.ylabel("Participants")
    survey_file['code'].value_counts().sort_values().reindex().hist(grid=False)
    plt.savefig("surveydistribution.png")

def top_categories(categories_file):
    categories = pd.read_csv(categories_file)
    def pivot(group):
        return pd.Series(list(group.sort_values('count', ascending=False)['app']) + [""] * (3 - len(group['app'])), index =['first','second','third'])
    sorted_categories = categories[categories['count'] > 60].sort_values(['category','count'],ascending=False)
    top_three = sorted_categories.groupby('category').head(3).groupby('category').apply(pivot)
    print(sorted_categories[sorted_categories['category'] == 'Tools'].head(10))
    print(top_three.to_latex())

def cluster_distrib(pickle_file, cluster_names):
    cluster_names = pd.read_csv(cluster_names)
    segments = pd.read_pickle(pickle_file)
    details = pd.DataFrame()
    details['name'] = cluster_names['name']
    details['proportion'] = segments['label'].value_counts() / len(segments) * 100
    details['counts'] = segments['label'].value_counts()
    segments['length'] = (segments['to'] - segments['from']).dt.seconds / 60
    print(segments)
    fig, ax = plt.subplots(figsize=(10,10))
    details['counts'].plot.pie(labels=details['name'],ax=ax)
    details['average_length_of_segment'] = segments.groupby('label').mean()['length']
    plt.savefig("cluster_distrib.png")
    print(details.sort_index().to_latex(index=False, float_format="%.2f"))

end_stage_1 = pd.Timestamp(year=2020, month=3, day=1, hour=0,tz=pytz.timezone('Australia/Melbourne'))
end_stage_2 = pd.Timestamp(year=2020, month=6, day=1, hour=0,tz=pytz.timezone('Australia/Melbourne'))

def count_points(activityfile, surveyfile):
    activity = pd.read_csv(activityfile)
    surveys = pd.read_csv(surveyfile)

    def period(df):
        if df['time'] < end_stage_1:
            return 'before'
        elif df['time'] > end_stage_1 and df['time'] < end_stage_2:
            return 'first'
        else:
            return 'second'

    utc_times = pd.to_datetime(surveys['time'], unit='ms').dt.tz_localize(tz=pytz.utc)
    aus_times = utc_times.dt.tz_convert(tz=pytz.timezone('Australia/Melbourne'))
    surveys['time'] = aus_times

    utc_times = pd.to_datetime(activity['time'], unit='ms').dt.tz_localize(tz=pytz.utc)
    aus_times = utc_times.dt.tz_convert(tz=pytz.timezone('Australia/Melbourne'))
    activity['time'] = aus_times

    description = pd.DataFrame()
    surveys['period'] = surveys.apply(period, axis=1)
    description['Surveys'] = surveys.groupby('period').count()['time']

    activity['period'] = activity.apply(period, axis=1)
    description['App Usage Recordings'] = activity.groupby('period').count()['time'] 

    description['Participants'] = activity.groupby('period')['code'].nunique()
    description['Start'] = activity.groupby('period')['time'].min().dt.strftime('%Y-%m-%d')
    description['End'] = activity.groupby('period')['time'].max().dt.strftime('%Y-%m-%d')
    print(description.to_latex())

def correlate_influence(esmfile, eodfile, usersfile):
    esm = pd.read_csv(esmfile)
    eod = pd.read_csv(eodfile)
    eod['time'] = eod['StartDate'].apply(parse)
    eod['date'] = eod['time'].dt.date

    esm['time'] = pd.to_datetime(esm.time, unit='ms')
    esm['date'] = esm.time.dt.date
    esm['working'] = esm['role'].replace({1:1, 3:2, 2:3})
    esm['valence'] = esm['valence'].replace({2:1, 3:2, 4:3,5:3})
    esm['arousal'] = esm['arousal'].replace({2:1, 3:2, 4:3,5:3})

    users = pd.read_csv(usersfile)
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

    rows = list()
    for _,survey in eod.iterrows():
        esm_on_date = esm[(esm.code==survey.code) & (esm.date == survey.date)]
        if len(esm_on_date) > 0:
            mean_esm = esm_on_date.mean()
            rows.append({ 'influnce_happiness': - survey['Q3']
                        , 'influence_energy': - survey['Q8']
                        , 'influence_on_work_life_balance': - survey['Q4']
                        , 'happiness': mean_esm.valence
                        , 'energy': mean_esm.arousal
                        , 'working': mean_esm.working
                        , 'code': survey.code
                        })
    df = pd.DataFrame(rows)


    happiness_corr, happiness_p = stats.spearmanr(df['influnce_happiness'], df['happiness'])
    energy_corr, energy_p = stats.spearmanr(df['influence_energy'], df['energy'])
    work_corr, work_p = stats.spearmanr(df['influence_on_work_life_balance'], df['working'])
    results = pd.DataFrame({'correlation coefficient': [happiness_corr, energy_corr, work_corr], 'p value': [happiness_p, energy_p, work_p]}, index=['happiness', 'energy', 'working'])
    print(results.to_latex())

    for question in ['Q9', 'Q3', 'Q8', 'Q4']:
        fig, ax = plt.subplots(figsize=(6,6)) 
        eod[question] = eod[question].replace({5:1, 4:1, 3: 2, 2: 3, 1: 3})
        average_diff = eod.groupby('code').mean()['Q9'].mean()
        male_diff = eod[eod.code.isin(males)].groupby('code').mean()[question].mean() - average_diff
        female_diff = eod[eod.code.isin(females)].groupby('code').mean()[question].mean() - average_diff

        age_18_24_diff = eod[eod.code.isin(age_18_24)].groupby('code').mean()[question].mean() - average_diff
        age_25_plus_diff = eod[eod.code.isin(age_25_plus)].groupby('code').mean()[question].mean() - average_diff
        ax.axvline(0)
        pd.DataFrame([ { 'demographic': 'female', 'color': 'pink', 'diff': female_diff}
                     , { 'demographic': 'male', 'color': 'blue', 'diff': male_diff}
                     , { 'demographic': '18-24', 'color': 'red', 'diff': age_18_24_diff}
                     , { 'demographic': '>=25', 'color': 'green', 'diff': age_25_plus_diff}
                     ]).plot.barh(y='diff', x='demographic', ax=ax)

        _, gender_p = stats.ttest_ind(eod[eod.code.isin(males)][question], eod[eod.code.isin(females)][question])
        _, age_p = stats.ttest_ind(eod[eod.code.isin(age_18_24)][question], eod[eod.code.isin(age_25_plus)][question])
        print("Question {}".format(question))
        print("Gender p {}".format(gender_p))
        print("Age p {}".format(age_p))

    def esm_correlations(points):
        happiness_corr, happiness_p = stats.spearmanr(points['influnce_happiness'], points['happiness'])
        energy_corr, energy_p = stats.spearmanr(points['influence_energy'], points['energy'])
        work_corr, work_p = stats.spearmanr(points['influence_on_work_life_balance'], points['working'])
        return pd.Series({ 'happiness_corr': happiness_corr
                         , 'energy_corr': energy_corr
                         , 'work_corr': work_corr
                         })


    male_df = df[df.code.isin(males)].groupby('code').apply(esm_correlations)
    female_df = df[df.code.isin(females)].groupby('code').apply(esm_correlations)

    age_18_24_df = df[df.code.isin(age_18_24)].groupby('code').apply(esm_correlations)
    age_25_plus_df = df[df.code.isin(age_25_plus)].groupby('code').apply(esm_correlations)

    print(male_df)
    _, gender_happiness_p = stats.ttest_ind(male_df['happiness_corr'].dropna(), female_df['happiness_corr'].dropna())
    _, gender_energy_p = stats.ttest_ind(male_df['energy_corr'].dropna(), female_df['energy_corr'].dropna())
    _, gender_work_p = stats.ttest_ind(male_df['work_corr'].dropna(), female_df['work_corr'].dropna())

    _, age_happiness_p = stats.ttest_ind(age_18_24_df['happiness_corr'].dropna(), age_25_plus_df['happiness_corr'].dropna())
    _, age_energy_p = stats.ttest_ind(age_18_24_df['energy_corr'].dropna(), age_25_plus_df['energy_corr'].dropna())
    _, age_work_p = stats.ttest_ind(age_18_24_df['work_corr'].dropna(), age_25_plus_df['work_corr'].dropna())
    print("Correlation differences")
    print(pd.DataFrame({'happiness': [gender_happiness_p, age_happiness_p], 'energy': [gender_energy_p, age_energy_p], 'work': [gender_work_p, age_work_p]}, index=['gender', 'age']))

    male_df = df[df.code.isin(males)].groupby('code').mean()
    female_df = df[df.code.isin(females)].groupby('code').mean()

    age_18_24_df = df[df.code.isin(age_18_24)].groupby('code').mean()
    age_25_plus_df = df[df.code.isin(age_25_plus)].groupby('code').mean()

    print(male_df)
    _, gender_happiness_p = stats.ttest_ind(male_df['happiness'].dropna(), female_df['happiness'].dropna())
    _, gender_energy_p = stats.ttest_ind(male_df['energy'].dropna(), female_df['energy'].dropna())
    _, gender_work_p = stats.ttest_ind(male_df['working'].dropna(), female_df['working'].dropna())

    _, age_happiness_p = stats.ttest_ind(age_18_24_df['happiness'].dropna(), age_25_plus_df['happiness'].dropna())
    _, age_energy_p = stats.ttest_ind(age_18_24_df['energy'].dropna(), age_25_plus_df['energy'].dropna())
    _, age_work_p = stats.ttest_ind(age_18_24_df['working'].dropna(), age_25_plus_df['working'].dropna())
    print("ESM differences")
    print(pd.DataFrame({'happiness': [gender_happiness_p, age_happiness_p], 'energy': [gender_energy_p, age_energy_p], 'work': [gender_work_p, age_work_p]}, index=['gender', 'age']))

    happiness_corr, happiness_p = stats.spearmanr(df['influnce_happiness'], df['happiness'])
    energy_corr, energy_p = stats.spearmanr(df['influence_energy'], df['energy'])
    work_corr, work_p = stats.spearmanr(df['influence_on_work_life_balance'], df['working'])
    results = pd.DataFrame({'correlation coefficient': [happiness_corr, energy_corr, work_corr], 'p value': [happiness_p, energy_p, work_p]}, index=['happiness', 'energy', 'working'])
    print(results.to_latex())

    plt.savefig("demographic_diff.png", bbox_inches='tight')

if __name__ == "__main__":
    main()

