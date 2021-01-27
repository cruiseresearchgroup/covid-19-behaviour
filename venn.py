

import pandas as pd
import scipy.stats as stats
import pytz
import matplotlib.dates as md
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

def venn_role(df):
    if len(df) >0:
        role_freq = df['role'].value_counts()
        print(role_freq)
        role_freq = df['role'].value_counts() / role_freq.sum()
        role_freq = role_freq.round(2)
        return venn2(subsets=(role_freq[1],role_freq[2],role_freq[3]), set_labels = ('Private', 'Work'))
    else:
        plt.axis('off')

end_stage_1 = pd.Timestamp(year=2020, month=3, day=1, hour=0,tz=pytz.timezone('Australia/Melbourne'))
end_stage_2 = pd.Timestamp(year=2020, month=6, day=1, hour=0,tz=pytz.timezone('Australia/Melbourne'))

def venn_work(surveys_file):
    surveys = pd.read_csv(surveys_file)
    users = ['Z23ATY', 'C09MBB', 'S13NBG']
    surveys = surveys[surveys['code'].isin(users)]
    utc_times = pd.to_datetime(surveys['time'], unit='ms').dt.tz_localize(tz=pytz.utc)
    aus_times = utc_times.dt.tz_convert(tz=pytz.timezone('Australia/Melbourne'))
    aus_date = aus_times.dt.date
    surveys['date'] = aus_date
    surveys['time'] = aus_times

    xfmt = md.DateFormatter('%Y-%m-%d')

    def period(df):
        if df['time'] < end_stage_1:
            return 'before'
        elif df['time'] > end_stage_1 and df['time'] < end_stage_2:
            return 'first'
        else:
            return 'second'

    surveys['period'] = surveys.apply(period, axis=1)
    periods = ['before','first','second']
    fix, axs = plt.subplots(len(users), 3, figsize=(7,7))
    i = 0
    for user in users:
        user_data = surveys[surveys['code'] == user]
        j = 0
        for period in periods:
            plt.sca(axs[i,j])
            axs[i,j].set_title("Participant " + (str(i+1)) + " " + period)
            venn_role(user_data[user_data['period'] == period])
            j += 1
        i += 1
    plt.savefig("venn_work.png", bbox_inches='tight')
    surveys['working'] = ((surveys['role'] == 3) | (surveys['role'] == 2))

    def describe_period(period):
        def describe_day(day):
            description = pd.Series()
            description['happiness'] = day['valence'].mean() / 5
            description['energy'] = day['arousal'].mean() / 5
            description['work'] = day['working'].mean()
            description['private'] = (~day['working']).mean()
#            description['valence_std'] = day['valence'].std()
#            description['arousal_std'] = day['arousal'].std() / 5
#            description['work_std'] = day['working'].std()
            return description
        result = period.groupby('date').apply(describe_day).mean()
        return result
    fix, axs = plt.subplots(len(users), 1, figsize=(7,7), sharex=True)
    i = 0
    for user in users:
        axs[i].set_ylabel("Participant " + str(i+1))
        surveys[surveys['code'] == user].groupby('period').apply(describe_period).plot(kind='bar',ax=axs[i], legend=False,ylim=(0,1))
        i += 1

    handles, labels = axs[len(users) - 1].get_legend_handles_labels()
    fix.legend(handles, labels, loc='upper center')
    plt.savefig("variable_change.png", bbox_inches='tight')


    for feature in ['valence', 'arousal', 'working']:
        rows = list()
        for user in users:
            description = pd.Series()
            before = surveys[(surveys['code'] == user) & (surveys['period'] == 'before')].groupby('date').mean()[feature]
            first = surveys[(surveys['code'] == user) & (surveys['period'] == 'first')].groupby('date').mean()[feature]
            second = surveys[(surveys['code'] == user) & (surveys['period'] == 'second')].groupby('date').mean()[feature]
            (statistic, p_value) = stats.ttest_ind(before, first)
            description['before-first-p'] = p_value
            description['before-first-stat'] = statistic
            (statistic, p_value) = stats.ttest_ind(first, second)
            description['first-second-p'] = p_value
            description['first-second-stat'] = statistic
            (statistic, p_value) = stats.ttest_ind(before, second)
            description['before-second-p'] = p_value
            description['before-second-stat'] = statistic
            rows.append(description)
        t_test_df = pd.DataFrame(rows, index=["Participant 1", "Participant 2", "Participant 3"])
        print(feature)
        print(t_test_df.to_latex(float_format="%.6f"))

