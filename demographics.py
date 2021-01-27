import pandas as pd
import matplotlib.pyplot as plt


def create_demographic_graphs(filename):
    df = pd.read_csv(filename)
    ages = df[df['age'] != 0].drop_duplicates('age')
    ages['age'].hist(grid=False)
    plt.title('Age Distributions')
    plt.xlabel('Age')
    plt.savefig('age_distro.png')

    # Now lets create the gender chart
    genders = df.sort_values('gender').drop_duplicates('code', keep='last')['gender']
    genders = genders.replace(0,'female')
    genders = genders.replace(1,'male')
    genders = genders.replace(-1,'unspecified')
    genders = genders.replace(3,'other')
    tallies = genders.value_counts()
    print(tallies)
    tallies.plot.pie()
    plt.title('Gender Distributions')
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('gender_distro.png')

