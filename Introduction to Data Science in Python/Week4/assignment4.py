#Question 1
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nhl_df=pd.read_csv("assets/nhl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

#functions for aligning city/region names
d1 = {
 'San Francisco Bay Area' : 'San Francisco',
 'Washington, D.C.' : 'Washington',
 'Minneapolis–Saint Paul' : 'Minneapolis',
 'Tampa Bay Area' : 'Tampa Bay',
 'Miami–Fort Lauderdale' : 'Miami',
 'New York City' : 'New York',
 'Dallas–Fort Worth' : 'Dallas',
}

def cities_to_area(loc):
    return d1.get(loc, loc)

d2 = {
 'Minnesota' : 'Minneapolis',
 'Vegas' : 'Las Vegas',
 'Florida' : 'Miami',
 'San Jose' : 'San Francisco',
 'Anaheim' : 'Los Angeles',
 'Colorado' : 'Denver',
 'Carolina' : 'Raleigh',
 'New Jersey' : 'New York',
 'Arizona' : 'Phoenix'
}

def team_to_area(team):
    loc = team.split()[0]

    if loc in ['Tampa', 'New', 'St.', 'San', 'Los']:
        loc = team.split()[0] + ' ' + team.split()[1]

    return d2.get(loc, loc)


def nhl_correlation():
    #reformatting nhl_df
    nhl_df = nhl_df[nhl_df['year'] == 2018]
    nhl_df.replace('\*', '', regex = True, inplace = True)
    nhl_df = nhl_df[['team', 'W', 'L']][~nhl_df['team'].str.contains('Division')]
    nhl_df['W'] = nhl_df['W'].astype('int64')
    nhl_df['L'] = nhl_df['L'].astype('int64')
    nhl_df['W/L Ratio'] = nhl_df['W'] / (nhl_df['W'] + nhl_df['L'])
    nhl_df['Area'] = nhl_df['team'].apply(team_to_area)

    #reformatting cities
    cities.replace('\[.*\]', '', regex = True, inplace = True)
    cities.replace('\—', np.NaN, regex = True, inplace = True)
    cities.replace('^\s*$', np.NaN, regex = True, inplace = True)
    cities.rename(columns = {'Population (2016 est.)[8]' : 'Population'}, inplace = True)
    cities['Population'] = cities['Population'].astype('int64')
    cities.drop(['NFL', 'MLB', 'NBA'], axis = 'columns', inplace = True)
    cities['Area'] = cities['Metropolitan area'].apply(cities_to_area)

    #merge
    combined = pd.merge(nhl_df, cities, how = 'inner').dropna().groupby(['Area']).agg({'Population' : np.mean, 'W/L Ratio' : np.mean})

    population_by_region = combined['Population'] # pass in metropolitan area population from cities
    win_loss_by_region = combined['W/L Ratio'] # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]

#Question 2
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nba_df=pd.read_csv("assets/nba.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

#functions for aligning city/region names
d1 = {
 'San Francisco Bay Area' : 'San Francisco',
 'Washington, D.C.' : 'Washington',
 'Minneapolis–Saint Paul' : 'Minneapolis',
 'Tampa Bay Area' : 'Tampa Bay',
 'Miami–Fort Lauderdale' : 'Miami',
 'New York City' : 'New York',
 'Dallas–Fort Worth' : 'Dallas',
}

def cities_to_area(loc):
    return d1.get(loc, loc)

d2 = {
 'Minnesota' : 'Minneapolis',
 'Vegas' : 'Las Vegas',
 'Florida' : 'Miami',
 'San Jose' : 'San Francisco',
 'Anaheim' : 'Los Angeles',
 'Colorado' : 'Denver',
 'Carolina' : 'Raleigh',
 'New Jersey' : 'New York',
 'Arizona' : 'Phoenix',
 'Golden' : 'San Francisco',
 'Brooklyn' : 'New York',
 'Utah' : 'Salt Lake City',
 'Indiana' : 'Indianapolis'
}

def team_to_area(team):
    loc = team.split()[0]

    if loc in ['Tampa', 'New', 'St.', 'San', 'Los', 'Oklahoma']:
        loc = team.split()[0] + ' ' + team.split()[1]

    return d2.get(loc, loc)


def nba_correlation():
    #reformatting nba_df
    nba_df = nba_df[nba_df['year'] == 2018]
    nba_df.replace('\*', '', regex = True, inplace = True)
    nba_df.replace('\(.*\)', '', regex = True, inplace = True)
    nba_df['W'] = nba_df['W'].astype('int64')
    nba_df['L'] = nba_df['L'].astype('int64')
    nba_df.rename(columns = {'W/L%' : 'W/L Ratio'}, inplace = True)
    nba_df['W/L Ratio'] = nba_df['W/L Ratio'].astype('float64')
    nba_df = nba_df[['team', 'W', 'L', 'W/L Ratio']][~nba_df['team'].str.contains('Division')]
    nba_df['Area'] = nba_df['team'].apply(team_to_area)


    #reformatting cities
    cities.replace('\[.*\]', '', regex = True, inplace = True)
    cities.replace('\—', np.NaN, regex = True, inplace = True)
    cities.replace('^\s*$', np.NaN, regex = True, inplace = True)
    cities.rename(columns = {'Population (2016 est.)[8]' : 'Population'}, inplace = True)
    cities['Population'] = cities['Population'].astype('int64')
    cities.drop(['NFL', 'MLB', 'NHL'], axis = 'columns', inplace = True)
    cities['Area'] = cities['Metropolitan area'].apply(cities_to_area)

    #merge
    combined = pd.merge(nba_df, cities, how = 'inner').dropna().groupby(['Area']).agg({'Population' : np.mean, 'W/L Ratio' : np.mean})

    population_by_region = combined['Population'] # pass in metropolitan area population from cities
    win_loss_by_region = combined['W/L Ratio'] # pass in win/loss ratio from nba_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]

#Question 3
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

mlb_df=pd.read_csv("assets/mlb.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

#functions for aligning city/region names
d1 = {
 'San Francisco Bay Area' : 'San Francisco',
 'Washington, D.C.' : 'Washington',
 'Minneapolis–Saint Paul' : 'Minneapolis',
 'Tampa Bay Area' : 'Tampa Bay',
 'Miami–Fort Lauderdale' : 'Miami',
 'New York City' : 'New York',
 'Dallas–Fort Worth' : 'Dallas',
}

def cities_to_area(loc):
    return d1.get(loc, loc)

d2 = {
 'Minnesota' : 'Minneapolis',
 'Vegas' : 'Las Vegas',
 'Florida' : 'Miami',
 'San Jose' : 'San Francisco',
 'Anaheim' : 'Los Angeles',
 'Colorado' : 'Denver',
 'Carolina' : 'Raleigh',
 'New Jersey' : 'New York',
 'Arizona' : 'Phoenix',
 'Golden' : 'San Francisco',
 'Brooklyn' : 'New York',
 'Utah' : 'Salt Lake City',
 'Indiana' : 'Indianapolis',
 'Oakland' : 'San Francisco',
 'Texas' : 'Dallas'
}

def team_to_area(team):
    loc = team.split()[0]

    if loc in ['Tampa', 'New', 'St.', 'San', 'Los', 'Oklahoma', 'Kansas']:
        loc = team.split()[0] + ' ' + team.split()[1]

    return d2.get(loc, loc)


def mlb_correlation():
    #reformatting mlb_df
    mlb_df = mlb_df[mlb_df['year'] == 2018]
    mlb_df.replace('\*', '', regex = True, inplace = True)
    mlb_df.replace('\(.*\)', '', regex = True, inplace = True)
    mlb_df['W'] = mlb_df['W'].astype('int64')
    mlb_df['L'] = mlb_df['L'].astype('int64')
    mlb_df.rename(columns = {'W-L%' : 'W/L Ratio'}, inplace = True)
    mlb_df['W/L Ratio'] = mlb_df['W/L Ratio'].astype('float64')
    mlb_df = mlb_df[['team', 'W', 'L', 'W/L Ratio']][~mlb_df['team'].str.contains('Division')]
    mlb_df['Area'] = mlb_df['team'].apply(team_to_area)

    #reformatting cities
    cities.replace('\[.*\]', '', regex = True, inplace = True)
    cities.replace('\—', np.NaN, regex = True, inplace = True)
    cities.replace('^\s*$', np.NaN, regex = True, inplace = True)
    cities.rename(columns = {'Population (2016 est.)[8]' : 'Population'}, inplace = True)
    cities['Population'] = cities['Population'].astype('int64')
    cities.drop(['NFL', 'NBA', 'NHL'], axis = 'columns', inplace = True)
    cities['Area'] = cities['Metropolitan area'].apply(cities_to_area)

    #merge
    combined = pd.merge(mlb_df, cities, how = 'inner').dropna().groupby(['Area']).agg({'Population' : np.mean, 'W/L Ratio' : np.mean})

    population_by_region = combined['Population'] # pass in metropolitan area population from cities
    win_loss_by_region = combined['W/L Ratio'] # pass in win/loss ratio from nba_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26, "Q3: There should be 26 teams being analysed for MLB"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]

#Question 4
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nfl_df=pd.read_csv("assets/nfl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

#functions for aligning city/region names
d1 = {
 'San Francisco Bay Area' : 'San Francisco',
 'Washington, D.C.' : 'Washington',
 'Minneapolis–Saint Paul' : 'Minneapolis',
 'Tampa Bay Area' : 'Tampa Bay',
 'Miami–Fort Lauderdale' : 'Miami',
 'New York City' : 'New York',
 'Dallas–Fort Worth' : 'Dallas',
}

def cities_to_area(loc):
    return d1.get(loc, loc)

d2 = {
 'Minnesota' : 'Minneapolis',
 'Vegas' : 'Las Vegas',
 'Florida' : 'Miami',
 'San Jose' : 'San Francisco',
 'Anaheim' : 'Los Angeles',
 'Colorado' : 'Denver',
 'Carolina' : 'Raleigh',
 'New Jersey' : 'New York',
 'Arizona' : 'Phoenix',
 'Golden' : 'San Francisco',
 'Brooklyn' : 'New York',
 'Utah' : 'Salt Lake City',
 'Indiana' : 'Indianapolis',
 'Oakland' : 'San Francisco',
 'Texas' : 'Dallas',
 'New England' : 'Boston',
 'Tennessee' : 'Nashville'
}

def team_to_area(team):
    loc = team.split()[0]

    if loc in ['Tampa', 'New', 'St.', 'San', 'Los', 'Oklahoma', 'Kansas', 'Green']:
        loc = team.split()[0] + ' ' + team.split()[1]

    return d2.get(loc, loc)


def nfl_correlation():
    #reformatting nfl_df
    nfl_df = nfl_df[nfl_df['year'] == 2018]
    nfl_df.replace('\*', '', regex = True, inplace = True)
    nfl_df.replace('\(.*\)', '', regex = True, inplace = True)
    nfl_df.replace('\+', '', regex = True, inplace = True)
    nfl_df.rename(columns = {'W-L%' : 'W/L Ratio'}, inplace = True)
    nfl_df = nfl_df[['team', 'W', 'L', 'W/L Ratio']][~nfl_df['team'].str.contains('FC')]
    nfl_df['W'] = nfl_df['W'].astype('int64')
    nfl_df['L'] = nfl_df['L'].astype('int64')
    nfl_df['W/L Ratio'] = nfl_df['W/L Ratio'].astype('float64')
    nfl_df['Area'] = nfl_df['team'].apply(team_to_area)

    #reformatting cities
    cities.replace('\[.*\]', '', regex = True, inplace = True)
    cities.replace('\—', np.NaN, regex = True, inplace = True)
    cities.replace('^\s*$', np.NaN, regex = True, inplace = True)
    cities.rename(columns = {'Population (2016 est.)[8]' : 'Population'}, inplace = True)
    cities['Population'] = cities['Population'].astype('int64')
    cities.drop(['MLB', 'NBA', 'NHL'], axis = 'columns', inplace = True)
    cities['Area'] = cities['Metropolitan area'].apply(cities_to_area)

    #merge
    combined = pd.merge(nfl_df, cities, how = 'inner').groupby(['Area']).agg({'Population' : np.mean, 'W/L Ratio' : np.mean})

    population_by_region = combined['Population'] # pass in metropolitan area population from cities
    win_loss_by_region = combined['W/L Ratio'] # pass in win/loss ratio from nba_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q4: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q4: There should be 29 teams being analysed for NFL"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]

#Question 5
import pandas as pd
import numpy as np
import scipy.stats as stats
import re

cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

#functions for aligning city/region names
d1 = {
 'San Francisco Bay Area' : 'San Francisco',
 'Washington, D.C.' : 'Washington',
 'Minneapolis–Saint Paul' : 'Minneapolis',
 'Tampa Bay Area' : 'Tampa Bay',
 'Miami–Fort Lauderdale' : 'Miami',
 'New York City' : 'New York',
 'Dallas–Fort Worth' : 'Dallas',
}

def cities_to_area(loc):
    return d1.get(loc, loc)

d2 = {
 'Minnesota' : 'Minneapolis',
 'Vegas' : 'Las Vegas',
 'Florida' : 'Miami',
 'San Jose' : 'San Francisco',
 'Anaheim' : 'Los Angeles',
 'Colorado' : 'Denver',
 'Carolina' : 'Raleigh',
 'New Jersey' : 'New York',
 'Arizona' : 'Phoenix',
 'Golden' : 'San Francisco',
 'Brooklyn' : 'New York',
 'Utah' : 'Salt Lake City',
 'Indiana' : 'Indianapolis',
 'Oakland' : 'San Francisco',
 'Texas' : 'Dallas',
 'New England' : 'Boston',
 'Tennessee' : 'Nashville'
}

def team_to_area(team):
    loc = team.split()[0]

    if loc in ['Tampa', 'New', 'St.', 'San', 'Los', 'Oklahoma', 'Kansas', 'Green']:
        loc = team.split()[0] + ' ' + team.split()[1]

    return d2.get(loc, loc)


def sports_team_performance():
    # YOUR CODE HERE
    #raise NotImplementedError()
    #reformatting cities
    cities.replace('\[.*\]', '', regex = True, inplace = True)
    cities.replace('\—', np.NaN, regex = True, inplace = True)
    cities.replace('^\s*$', np.NaN, regex = True, inplace = True)
    cities.rename(columns = {'Population (2016 est.)[8]' : 'Population'}, inplace = True)
    cities['Population'] = cities['Population'].astype('int64')
    cities['Area'] = cities['Metropolitan area'].apply(cities_to_area)


    def nhl_wl():
        nhl_df=pd.read_csv("assets/nhl.csv")

        #reformatting nhl_df
        nhl_df = nhl_df[nhl_df['year'] == 2018]
        nhl_df.replace('\*', '', regex = True, inplace = True)
        nhl_df = nhl_df[['team', 'W', 'L']][~nhl_df['team'].str.contains('Division')]
        nhl_df['W'] = nhl_df['W'].astype('int64')
        nhl_df['L'] = nhl_df['L'].astype('int64')
        nhl_df['W/L Ratio'] = nhl_df['W'] / (nhl_df['W'] + nhl_df['L'])
        nhl_df['Area'] = nhl_df['team'].apply(team_to_area)

        #merge cities and nhl_df
        combined = pd.merge(nhl_df, cities, how = 'inner').groupby(['Area']).agg({'Population' : np.mean, 'W/L Ratio' : np.mean})

        return combined


    def nba_wl():
        nba_df=pd.read_csv("assets/nba.csv")

        #reformatting nba_df
        nba_df = nba_df[nba_df['year'] == 2018]
        nba_df.replace('\*', '', regex = True, inplace = True)
        nba_df.replace('\(.*\)', '', regex = True, inplace = True)
        nba_df['W'] = nba_df['W'].astype('int64')
        nba_df['L'] = nba_df['L'].astype('int64')
        nba_df.rename(columns = {'W/L%' : 'W/L Ratio'}, inplace = True)
        nba_df['W/L Ratio'] = nba_df['W/L Ratio'].astype('float64')
        nba_df = nba_df[['team', 'W', 'L', 'W/L Ratio']][~nba_df['team'].str.contains('Division')]
        nba_df['Area'] = nba_df['team'].apply(team_to_area)

        #merge cities and nba_df
        combined = pd.merge(nba_df, cities, how = 'inner').groupby(['Area']).agg({'Population' : np.mean, 'W/L Ratio' : np.mean})

        return combined


    def mlb_wl():
        mlb_df=pd.read_csv("assets/mlb.csv")

        #reformatting mlb_df
        mlb_df = mlb_df[mlb_df['year'] == 2018]
        mlb_df.replace('\*', '', regex = True, inplace = True)
        mlb_df.replace('\(.*\)', '', regex = True, inplace = True)
        mlb_df['W'] = mlb_df['W'].astype('int64')
        mlb_df['L'] = mlb_df['L'].astype('int64')
        mlb_df.rename(columns = {'W-L%' : 'W/L Ratio'}, inplace = True)
        mlb_df['W/L Ratio'] = mlb_df['W/L Ratio'].astype('float64')
        mlb_df = mlb_df[['team', 'W', 'L', 'W/L Ratio']][~mlb_df['team'].str.contains('Division')]
        mlb_df['Area'] = mlb_df['team'].apply(team_to_area)

        #merge cities and mlb_df
        combined = pd.merge(mlb_df, cities, how = 'inner').groupby(['Area']).agg({'Population' : np.mean, 'W/L Ratio' : np.mean})

        return combined


    def nfl_wl():
        nfl_df=pd.read_csv("assets/nfl.csv")

        #reformatting nfl_df
        nfl_df = nfl_df[nfl_df['year'] == 2018]
        nfl_df.replace('\*', '', regex = True, inplace = True)
        nfl_df.replace('\(.*\)', '', regex = True, inplace = True)
        nfl_df.replace('\+', '', regex = True, inplace = True)
        nfl_df.rename(columns = {'W-L%' : 'W/L Ratio'}, inplace = True)
        nfl_df = nfl_df[['team', 'W', 'L', 'W/L Ratio']][~nfl_df['team'].str.contains('FC')]
        nfl_df['W'] = nfl_df['W'].astype('int64')
        nfl_df['L'] = nfl_df['L'].astype('int64')
        nfl_df['W/L Ratio'] = nfl_df['W/L Ratio'].astype('float64')
        nfl_df['Area'] = nfl_df['team'].apply(team_to_area)

        #merge
        combined = pd.merge(nfl_df, cities, how = 'inner').groupby(['Area']).agg({'Population' : np.mean, 'W/L Ratio' : np.mean})

        return combined


    def all_df(sport):
        if sport == 'NHL':
            return nhl_wl()
        elif sport == 'NBA':
            return nba_wl()
        elif sport == 'MLB':
            return mlb_wl()
        elif sport == 'NFL':
            return nfl_wl()

    # Note: p_values is a full dataframe, so df.loc["NFL","NBA"] should be the same as df.loc["NBA","NFL"] and
    # df.loc["NFL","NFL"] should return np.nan
    sports = ['NFL', 'NBA', 'NHL', 'MLB']
    p_values = pd.DataFrame({k:np.nan for k in sports}, index=sports)

    for s1 in sports:
        for s2 in sports:
            if s1 != s2:
                df = pd.merge(all_df(s1), all_df(s2), how = 'inner', on = 'Area')
                p_values.loc[s1, s2] = stats.ttest_rel(df['W/L Ratio_x'],df['W/L Ratio_y'])[1]

                #print(s1, s2, df.shape)


    assert abs(p_values.loc["NBA", "NHL"] - 0.02) <= 1e-2, "The NBA-NHL p-value should be around 0.02"
    assert abs(p_values.loc["MLB", "NFL"] - 0.80) <= 1e-2, "The MLB-NFL p-value should be around 0.80"
    return p_values
