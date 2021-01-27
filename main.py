""" A python tool for the analysis of COVID-19 Social roles"""
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import csv
import util
import base64
import math
import segment
import cluster
import correlate
import demographics
import pytz
import venn
from dateutil.parser import parse
import scipy.stats as stats

blacklist = { 'SCREEN',
             'LOCATION',
             'AUDIO',
             'ACTIVITY',
             'Keyboard',
             'Mouse clicked',
             'ESM',
             'Location',
             'Mouse scrolled',
             'Battery'
             }

required_columns = {'code',
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
    codes = {}
    apps = {}
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
                        writer.writerow(['time', 'device_type', 'code', 'app', 'event_type', 'bluetooth_rssi', 'bluetooth_address', 'wifi_bssid', 'wifi_rssi'])
                        surveywriter.writerow(
                                ['user',
                                 'valence',
                                 'arousal',
                                 'role',
                                 'interruption',
                                 'task',
                                 'device_type',
                                 'time'
                                 ])
                    else:
                        new_code = get_coded(codes, row[index_map['code']])
                        if row[index_map['esm_valence']] != '-1':
                            surveywriter.writerow(
                                    [new_code,
                                     row[index_map['esm_valence']],
                                     row[index_map['esm_arousal']],
                                     row[index_map['esm_role']],
                                     row[index_map['esm_interruption']],
                                     row[index_map['esm_task']],
                                     row[index_map['event_type']],
                                     row[index_map['time_issued']]
                                     ]
                                     )

                        if row[index_map['application_name']] not in blacklist:
                            new_app = get_coded(
                                    apps,
                                    row[index_map['application_name']]
                            )
                            writer.writerow(
                                [row[index_map['time_issued']],
                                 row[index_map['event_type']],
                                 new_code,
                                 new_app,
                                 row[index_map['bluetooth_rssi']],
                                 row[index_map['bluetooth_address']],
                                 row[index_map['wifi_bssid']],
                                 row[index_map['wifi_rssi']]
                                 ]
                            )
                    line += 1
                    print(f'Line {line}\r', end='')
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
    if len(sys.argv) == 3 and sys.argv[1] == "users":
        df = load_data()

    elif len(sys.argv) == 3 and sys.argv[1] == "apps":
        df = load_data()

    elif len(sys.argv) == 3 and sys.argv[1] == "filter":
        filter_data(sys.argv[2])

    elif len(sys.argv) == 3 and sys.argv[1] == "segment":
        segment.segment(sys.argv[2])
    
    elif len(sys.argv) == 3 and sys.argv[1] == "cluster":
        cluster.cluster_segments(sys.argv[2])

    elif len(sys.argv) == 7 and sys.argv[1] == "correlate":
        correlate.corralate_segments(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])

    elif len(sys.argv) == 6 and sys.argv[1] == "correlatetext":
        correlate.corralate_text(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

    elif len(sys.argv) == 3 and sys.argv[1] == "demographics":
        demographics.create_demographic_graphs(sys.argv[2])

    elif len(sys.argv) == 5 and sys.argv[1] == "clusterprototypes":
        cluster_prototypes(sys.argv[2], sys.argv[3], sys.argv[4])

    elif len(sys.argv) == 5 and sys.argv[1] == "drawcorrelations":
        draw_correlations(sys.argv[2], sys.argv[3], sys.argv[4])
        
    elif len(sys.argv) == 5 and sys.argv[1] == "drawtextcorrelations":
        draw_text_correlations(sys.argv[2], sys.argv[3], sys.argv[4])

    elif len(sys.argv) == 3 and sys.argv[1] == "extracttext":
        extracttext(sys.argv[2])

    elif len(sys.argv) == 3 and sys.argv[1] == "surveydistribution":
        surveydistribution(sys.argv[2])

    elif len(sys.argv) == 3 and sys.argv[1] == "topcategories":
        top_categories(sys.argv[2])

    elif len(sys.argv) == 4 and sys.argv[1] == "clusterdistrib":
        cluster_distrib(sys.argv[2], sys.argv[3])

    elif len(sys.argv) == 3 and sys.argv[1] == "vennwork":
        venn.venn_work(sys.argv[2])

    elif len(sys.argv) == 4 and sys.argv[1] == "count":
        count_points(sys.argv[2], sys.argv[3])

    elif len(sys.argv) == 4 and sys.argv[1] == "correlateinfluence":
        correlate_influence(sys.argv[2], sys.argv[3])

    else:
        print(main.__doc__)


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

    sns.heatmap(p_values, fmt="s",annot=plot_labels, ax=ax)
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
    features = ['happiness', 'energy','working', 'productivity', 'interruptions']
#    if daily:
#        features = ['productivity','interruptions','influence_on_happiness','influence_on_energy','influence_on_work_life_balance', 'difference_from_yesterday']
    correlations = correlations.set_index(correlation_labels['name'])

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

    fig, ax = plt.subplots(figsize=(10,10)) 
    plt.title(title + " p values")
    ax.axvline(3)
    sns.heatmap(p_values, annot=plot_labels, fmt="s", ax=ax)
    plt.savefig("p_values.png", bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(10,10))
    plt.title(title + " coefficients")
    print(coefficients)
    ax.axvline(3)
    sns.heatmap(coefficients, annot=True, fmt=".4f", vmin=-0.2, vmax=0.2, center=0, ax=ax)
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
    s2_end = pd.Timestamp(year=2020, month=8, day=23, tz=pytz.timezone('Australia/Melbourne'))
    activity = activity[activity['time'] > s2_end]
    all_categories = activity.category.unique()

    cluster_dist = pd.DataFrame(index=all_categories)
    for label in range(segments['label'].max() + 1):
        my_segments = segments[segments['label']==label]['seg']
        dist = my_segments.apply(transform_multivariate_to_univariate).mean(axis=0)
        cluster_dist[clusternames.at[cluster,'name']] = dist
        fig, ax = plt.subplots(figsize=(7,4))

        result_frequencies.sort_values('frequency',ascending=False).head(5).plot.bar(ylim=(0,1),rot=0,ax=ax)
        plt.savefig("cluster-prototypes/cluster-" + str(label) + ".png")


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

def correlate_influence(esmfile, eodfile):
    esm = pd.read_csv(esmfile)
    eod = pd.read_csv(eodfile)
    eod['time'] = eod['StartDate'].apply(parse)
    eod['date'] = eod['time'].dt.date

    esm['time'] = pd.to_datetime(esm.time, unit='ms')
    esm['date'] = esm.time.dt.date
    esm['working'] = esm['role'].replace({1:1, 3:2, 2:3})

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


if __name__ == "__main__":
    main()

