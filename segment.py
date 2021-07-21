"""
Author: Sam Nolan, with code borrowed from Shohreh Deldari
IGTS module
"""

import pandas as pd
import matplotlib.pyplot as plt
import pytz
import numpy as np
import igts
import statistics


def prepare_data(data):
    """
    Adds end dates to the data and python dates
    """
    data['utc_end'] = data['time'] + 60000
    data['utc_start'] = data['time']
    ts_col = data.utc_end
    utc_times = pd.to_datetime(ts_col, unit='ms').dt.tz_localize(tz=pytz.utc)
    aus_times = utc_times.dt.tz_convert(
        tz=pytz.timezone('Australia/Melbourne'))
    aus_date = aus_times.dt.date
    data['time'] = aus_times
    data['date'] = aus_date
    return data


def plot_app_usage(data, apps, day, TT=[], step=1):
    fig = plt.figure(figsize=(7, 10))

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)

    ax.set_xlabel('Time steps(utc(m))')
    ax.set_ylabel('app usage')

    plt.title("Example segmentation")
    for i in range(0, apps.shape[0]):
        ax.plot((data[i, :] > 0) +2*step*i)
        ax.annotate(xy=[-5, 2*step*i+1], s=apps[i])

    if len(TT) > 0:
        for i in range(0, TT.shape[0]):
            # draw vertical line from (70,100) to (70, 250)
            ax.plot([TT[i], TT[i]], [0, 2*step*apps.shape[0]+1], 'r-', lw=2)
            # ax.annotate(xy=[TT[i]+0.5,2*step*apps.shape[0]+1],s=str(i))
        # fig.show()

    fig.savefig("segment-graphs/"+day.strftime("%d-%b-%Y")+'step'+str(step)+'.png', bbox_inches='tight')
      

def calculate_segments(day_data, day, categories):
    """Takes the days data and splits it up into an array of segments"""

    daily, categories = daily_timeline(day_data, day, categories)
    win_daily = windowing(daily,  60)
    TT, IG, knee = igts.TopDown(win_daily, 50, 1, 1)

    # plot_app_usage(win_daily, categories, day, TT=np.array(TT), step=60)
    first_time = day_data['utc_start'].iloc[0]
    last_segment = 0
    segments = []
    for t in np.sort(TT):
        segment = win_daily[:, last_segment:t]
        if len(segment[0]) > 0:
            from_time = (pd.to_datetime(first_time, unit='ms') +
                         pd.Timedelta(seconds=last_segment * 60))
            to_time = pd.to_datetime(
                first_time, unit='ms') + pd.Timedelta(seconds=t * 60)

            segments.append({'from': from_time, 'to': to_time,
                            'seg': segment, 'day': day})
        last_segment = t
    return win_daily, TT, knee


def windowing(data, w_size):
    num_ts, len_ts = data.shape
    num_w = (len_ts//w_size)+1
    win_data = np.zeros((num_ts, num_w))
    for i in range(0, num_w):
        win_data[:, i] = np.sum(data[:, i*w_size:(i+1)*w_size], axis=1)
    return win_data


def daily_timeline(data, day, categories=None, SAVE=False):
    # List of unique apps
    if categories is None:
        categories = data.category.unique()

    start_time = (np.array(data.utc_start.values) // 1000).astype(int)
    end_time = (np.array(data.utc_end.values) // 1000).astype(int)

    # Extract app usage timeline info
    start = start_time - start_time[0]
    endt = end_time - start_time[0]
    app_usage = np.zeros((categories.shape[0],  endt[-1]+1))
    for row in range(0, end_time.shape[0]):
        record = data.iloc[row, :]
        idx = np.where(categories == record.category)[0][0]
        app_usage[idx, start[row]:endt[row]+1] = 1
        # Save daily app usage

    return app_usage, categories


def segment(filename, categories_filename,draw_only=False):
    """
    Segments the dataframe using IGTS.

    activity file is assumed to have the following column names:
       - app
       - time
       - code
       - device_type

    category file is assumed to have the following column names:
       - app
       - category

    """
    df = prepare_data(pd.read_csv(filename))
    categories_df = pd.read_csv(categories_filename)
    print(categories_df.columns)
    #users = df.code.unique()
    users = ['B05JGB', 'S13NBG','Z23ATY', 'C09MBB', 'X20MJG']
    devices = df.device_type.unique()

    df = pd.merge(left=df, right=categories_df, left_on='app', right_on='app')
    df.sort_values('time', inplace=True)
    all_segments = list()
    df = df[df['category'] != 'Unknown']
    categories = df.category.unique()
    np.savetxt("category_names.csv", categories,delimiter=",",fmt='%s')

    for user in users:
        for device in devices:
            user_data = df[(df['code'] == user) &
                           (df['device_type'] == device)]

            if len(user_data) > 0:
                # List of days
                days = user_data.date.unique()
                all_segments = list()

                data_lengths = list()
                norm_ks = list()
                tts = list()
                win_dailies = list()
                daily_data = list()

                for day in days:
                    day_data = user_data[user_data['date'] == day]
                    print(str(day) + "-" + device + "-" + str(user))
                    print(day)
                    utc_start = day_data['utc_start'].iloc[0]
                    from_time = (pd.to_datetime(utc_start, unit='ms'))
                    print(pd.to_datetime(from_time))
                    tts.append([])
                    continue # Test data
                        
                    win_daily, tt_new, knee = calculate_segments(day_data, day, categories)
                    if draw_only:
                        plot_app_usage(win_daily,categories,day,tt_new[:knee])
                    tts.append(tt_new)
                    data_lengths.append(len(day_data))
                    norm_ks.append(knee / len(day_data))
                    win_dailies.append(win_daily)
                    daily_data.append(day_data)
                
                #median_k_ratio = statistics.median(norm_ks)
                segments = list()
                #print("K Ratio " + str(median_k_ratio))

                for i in range(len(tts)):
                    knee = int(data_lengths[i] * median_k_ratio)
                    last_segment = 0
                    first_time = daily_data[i]['utc_start'].iloc[0]
                    print(from_time)
                    print(day)
                    print(day - from_time.date)
                    continue
                    for t in np.sort(tts[i][:knee]):
                        segment = win_dailies[i][:, last_segment:t]
                        if len(segment[0]) > 0:
                            from_time = (pd.to_datetime(first_time, unit='ms') +
                                         pd.Timedelta(seconds=last_segment * 60))
                            to_time = pd.to_datetime(
                                first_time, unit='ms') + pd.Timedelta(seconds=t * 60)

                        last_segment = t
                        segments.append({'from': from_time, 'to': to_time,
                                        'seg': segment, 'day': day})

                print("Saving " + str(len(segments)))

                if not draw_only:
                    np.save("day-segments/" + str(user) + '-' + device +
                            "-segments", segments)
