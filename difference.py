
import pandas as pd
from dateutil.parser import parse
import datetime
import scipy.stats as stat
import pytz
import numpy as np
import matplotlib.pyplot as plt



def cosine_simularity(p,q):
    p_size = np.linalg.norm(p)
    q_size = np.linalg.norm(q)
    if p_size == 0 and q_size == 0:
        return 0
    if p_size == 0 or q_size == 0:
        return 1
    return 1 - (np.dot(p,q) / p_size / q_size)

def difference_yesterday(esmfile, dailyfile, activityfile, usersfile):
    daily_surveys = pd.read_csv(dailyfile).fillna(0)
    esm_surveys = pd.read_csv(esmfile)
    ts_col = esm_surveys.time
    utc_times = pd.to_datetime(ts_col, unit='ms').dt.tz_localize(tz=pytz.utc)
    aus_times = utc_times.dt.tz_convert(
        tz=pytz.timezone('Australia/Melbourne'))
    esm_surveys.time = aus_times

    users = daily_surveys.code.unique()
    daily_surveys['record_time'] = daily_surveys['StartDate'].apply(parse)
    rows = list()

    activity = pd.read_pickle(activityfile)
    activity['duration'] = activity['to'] - activity['from']
    diff_rows = list()

    users = pd.read_csv(usersfile)
    all_users = users['code'].unique()
    females = users[users['gender'] == 0]['code'].unique()
    males = users[users['gender'] == 1]['code'].unique()
    age_18_24 = users[(users['age'] <= 24) & (users['age'] >= 18)]['code'].unique()
    age_25_plus = users[users['age'] >= 25]['code'].unique()

    for user in all_users:
        # Q3 = Influence Happiness
        # Q8 = Influence energy
        # Q4 = Influence Work life
        # Q6 = Happiness text
        # Q7 = Work Life text
        # Q11 = Energy text
        # Q10 = Diff text
        # Q9 = Diff from yesterday
        gender = 'neither'
        if user in males:
            gender = 'male'

        if user in females:
            gender = 'female'

        age = 'Age 18-24'
        if user in age_25_plus:
            age = 'Age >=25'

        user_activity = activity[activity.code == user]
        user_daily_surveys = daily_surveys[daily_surveys.code == user]
        high_diff_surveys = user_daily_surveys[(user_daily_surveys['Q9'] == 1) | (user_daily_surveys['Q9'] == 2)]
        low_diff_surveys = user_daily_surveys[(user_daily_surveys['Q9'] == 4) | (user_daily_surveys['Q9'] == 5)]
        if len(high_diff_surveys) > 0 and len(low_diff_surveys) > 0:
            def find_diffs(eod_survey, variable):
                today = eod_survey.record_time.date()
                yesterday = today - datetime.timedelta(days=1)
                vector_today = esm_vector(esm_surveys, today, variable)
                vector_yesterday = esm_vector(esm_surveys, yesterday, variable)
                return cosine_simularity(vector_today, vector_yesterday)

            def find_activity_diff(eod_survey):
                today = eod_survey.record_time.date()
                yesterday = today - datetime.timedelta(days=1)
                vector_today = usage_vector(user_activity, today).dt.total_seconds()
                vector_yesterday = usage_vector(user_activity, yesterday).dt.total_seconds()
                return cosine_simularity(vector_today, vector_yesterday)


            high_valence_diffs = high_diff_surveys.apply(lambda x: find_diffs(x, 'valence'), axis=1)
            high_arousal_diffs = high_diff_surveys.apply(lambda x: find_diffs(x, 'arousal'), axis=1)
            high_activity_diffs = high_diff_surveys.apply(find_activity_diff, axis=1)

            for diff in high_activity_diffs:
                diff_rows.append({ 'code': user
                                 , 'diff': diff
                                 , 'user_diff': 'high'
                                 , 'gender': gender
                                 , 'age': age
                                 })

            low_valence_diffs = low_diff_surveys.apply(lambda x: find_diffs(x, 'valence'), axis=1)
            low_arousal_diffs = low_diff_surveys.apply(lambda x: find_diffs(x, 'arousal'), axis=1)
            low_activity_diffs = low_diff_surveys.apply(find_activity_diff, axis=1)

            for diff in low_activity_diffs:
                diff_rows.append({ 'code': user
                                 , 'diff': diff
                                 , 'user_diff': 'low'
                                 , 'gender': gender
                                 , 'age': age
                                 })

            valence_p = 0
            arousal_p = 0
            activity_p = 0

            valence_p = stat.ttest_ind(high_valence_diffs, low_arousal_diffs)[1] / 2
            arousal_p = stat.ttest_ind(high_arousal_diffs, low_arousal_diffs)[1] / 2
            activity_p = stat.ttest_ind(high_activity_diffs, low_activity_diffs)[1] / 2

            if low_valence_diffs.mean() > high_valence_diffs.mean():
                valence_p = 1

            if low_arousal_diffs.mean() > high_arousal_diffs.mean():
                arousal_p = 1

            if low_activity_diffs.mean() > high_activity_diffs.mean():
                activity_p = 1

            row = { 'code': user
                  , 'valence_test': valence_p
                  , 'arousal_test': arousal_p
                  , 'activity_test': activity_p
                  }

            rows.append(row)

    result = pd.DataFrame(rows)
    diff_df = pd.DataFrame(diff_rows)
    print(diff_df)
    diff_df['colour'] = diff_df.user_diff.replace({'high': 'blue', 'low': 'red'})
    diff_df.plot.scatter(x='diff',y='code', c='colour')
    plt.savefig("diff_scatters.png", bbox_inches='tight')

    diff_df[diff_df.gender=='male'].plot.scatter(x='diff', y='code', c='colour')
    plt.title("Male difference of app activity")
    plt.savefig("diff_scatters_gender_male.png", bbox_inches='tight')

    diff_df[diff_df.gender=='female'].plot.scatter(x='diff', y='code', c='colour')
    plt.title("Female difference of app activity")
    plt.savefig("diff_scatters_gender_female.png", bbox_inches='tight')

    diff_df[diff_df.age == 'Age 18-24'].plot.scatter(x='diff', y='code', c='colour')
    plt.title("Age 18-24 difference of app activity")
    plt.savefig("diff_scatters_age_18_24.png", bbox_inches='tight')
    diff_df[diff_df.age == 'Age >=25'].plot.scatter(x='diff', y='code', c='colour')
    plt.title("Age 25+ difference of app activity")
    plt.savefig("diff_scatters_age_25_plus.png", bbox_inches='tight')

                   
def usage_vector(activity, date):
    segments = activity[activity['from'].dt.date == date]
    duration_vector = segments.groupby('label').sum()['duration']
    return duration_vector.reindex(pd.RangeIndex(15)).fillna(pd.Timedelta(seconds=0))

def esm_vector(surveys, date, variablename):
    surveys_today = surveys[surveys.time.dt.date == date]
    section1 = surveys_today[(surveys_today.time.dt.hour >= 9) & (surveys_today.time.dt.hour < 12)]
    section2 = surveys_today[(surveys_today.time.dt.hour >= 12) & (surveys_today.time.dt.hour < 15)]
    section3 = surveys_today[(surveys_today.time.dt.hour >= 15) & (surveys_today.time.dt.hour < 18)]
    section4 = surveys_today[(surveys_today.time.dt.hour >= 18) & (surveys_today.time.dt.hour < 21)]
    return np.nan_to_num(np.array([section1[variablename].mean(), section2[variablename].mean(), section3[variablename].mean(), section4[variablename].mean()]))
