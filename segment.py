"""
Author: Sam Nolan, with code borrowed from Shohreh Deldari
IGTS module
"""

import pandas as pd
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
    data['date'] = aus_date
    return data


def calculate_segments(day_data, day, apps):
    """Takes the days data and splits it up into an array of segments"""

    daily, categories = daily_timeline(day_data, day, apps)
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


def daily_timeline(data, day, apps=None, SAVE=False):
    # List of unique apps
    if apps is None:
        apps = data.app.unique()

    start_time = (np.array(data.utc_start.values) // 1000).astype(int)
    end_time = (np.array(data.utc_end.values) // 1000).astype(int)

    # Extract app usage timeline info
    start = start_time - start_time[0]
    endt = end_time - start_time[0]
    app_usage = np.zeros((apps.shape[0],  endt[-1]+1))
    for row in range(0, end_time.shape[0]):
        record = data.iloc[row, :]
        idx = np.where(apps == record.app)[0][0]
        app_usage[idx, start[row]:endt[row]+1] = 1
        # Save daily app usage

    return app_usage, apps


def segment(filename):
    """
    Segments the dataframe using IGTS.

    file is assumed to have the following column names:
       - app
       - time
       - code
       - device_type

    """
    df = prepare_data(pd.read_csv(filename))
    users = df.code.unique()
    devices = df.device_type.unique()
    for user in users:
        for device in devices:
            user_data = df[(df['code'] == user) &
                           (df['device_type'] == device)]

            # List of days
            apps = user_data.app.unique()
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
                win_daily, tt_new, knee = calculate_segments(day_data, day, apps)
                tts.append(tt_new)
                data_lengths.append(len(day_data))
                norm_ks.append(knee / len(day_data))
                win_dailies.append(win_daily)
                daily_data.append(day_data)
            
            median_k_ratio = statistics.median(norm_ks)
            segments = list()
            print("K Ratio " + str(median_k_ratio))

            for i in range(len(tts)):
                knee = int(data_lengths[i] * median_k_ratio)
                print("Knee " + str(knee))
                last_segment = 0
                print(day_data)
                first_time = daily_data[i]['utc_start'].iloc[0]
                for t in np.sort(tts[i][:knee]):
                    print("t " + str(t))
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

            np.save("day-segments/" + str(user) + '-' + device +
                    "-segments", segments)
