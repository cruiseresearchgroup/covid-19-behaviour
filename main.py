"""
A python tool for the analysis of COVID-19 Social roles
"""
import sys
import pandas as pd
import csv
import util
import base64
import math

blacklist = {'BLUETOOTH',
             'SCREEN',
             'WIFI',
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
                    'time_issued',
                    'application_package',
                    'esm_role',
                    'esm_arousal',
                    'esm_task',
                    'esm_valence',
                    'esm_interruption',
                    'time_zone',
                    'event_type',
                    'time_issued'
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
        new_code = nb64(len(codemap))
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
                        writer.writerow(['time', 'code', 'app'])
                        surveywriter.writerow(
                                ['user',
                                 'valence',
                                 'arousal',
                                 'role',
                                 'interruption',
                                 'task',
                                 'time'
                                 ])
                    else:
                        new_code = get_coded(codes, row[index_map['code']])
                        new_app = get_coded(
                                apps,
                                row[index_map['application_package']]
                        )

                        if row[index_map['esm_valence']] != '-1':
                            surveywriter.writerow(
                                    [new_code,
                                     row[index_map['esm_valence']],
                                     row[index_map['esm_arousal']],
                                     row[index_map['esm_role']],
                                     row[index_map['esm_interruption']],
                                     row[index_map['esm_task']],
                                     row[index_map['time_issued']]
                                     ]
                                     )

                        if row[index_map['application_name']] not in blacklist:
                            writer.writerow(
                                [nb64(int(float(row[index_map['time_issued']]))),
                                 new_code,
                                 new_app
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
                          filter [infile] [outfile]
    """
    if len(sys.argv) == 3 and sys.argv[1] == "users":
        df = load_data()
        print(df.code.unique())

    elif len(sys.argv) == 3 and sys.argv[1] == "apps":
        df = load_data()
        print(df.application_name.unique())

    elif len(sys.argv) == 3 and sys.argv[1] == "filter":
        filter_data(sys.argv[2])
    else:
        print(main.__doc__)


if __name__ == "__main__":
    main()
