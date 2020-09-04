import sys


def check_columns(df, required_columns):
    columns = set(df.columns)
    missing_columns = required_columns.difference(columns)
    extra_columns = columns.difference(required_columns)
    if len(missing_columns) > 0:
        print("Error, there data is required to have the following columns")
        print(missing_columns)
        sys.exit(1)

    if len(extra_columns) > 0:
        print("Warning, The data has extra columns, these columns can be removed with `filter` to load data faster")
        print(extra_columns)

