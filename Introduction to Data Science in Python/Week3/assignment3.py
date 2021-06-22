#Question 1
import pandas as pd
import numpy as np

#ENERGY FILE
energy = pd.read_excel('assets/Energy Indicators.xls')

#get rid of unnecessary columns
energy.drop(energy.columns[[0,1]], inplace = True, axis = 'columns')

#rename columns to something understandable
energy.rename(columns = {'Unnamed: 2' : 'Country', 'Unnamed: 3' : 'Energy Supply',
                         'Unnamed: 4' : 'Energy Supply per Capita', 'Unnamed: 5' : '% Renewable'},
              inplace = True)

#remove unnecessary rows
energy = energy[17:244]

#convert placeholder w/ NaN
energy['Energy Supply'].replace('...', np.NaN, inplace = True)

#changing energy supply units to gigajoules from petajoules
#vectorization
for k, v in energy['Energy Supply'].iteritems():
    energy['Energy Supply'].set_value(k, v * 1000000)

replace_values = {"Republic of Korea": "South Korea",
                  "United States of America": "United States",
                  "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                  "China, Hong Kong Special Administrative Region": "Hong Kong"}

#function to remove numbers
def del_num(country):
    return ''.join(filter(lambda c : not c.isdigit(), country))

#function to remove unnecessary characters
def del_excess(country):
    index = country.find('(')
    if index != -1:
        return country[: index - 1]
    return country

energy['Country'] = energy['Country'].apply(del_num)
energy['Country'] = energy['Country'].apply(del_excess)

energy.replace({'Country' : replace_values}, inplace = True)


#GDP FILE
#skip header / unnecessary rows
GDP = pd.read_csv('assets/world_bank.csv', skiprows = 4)

replace_values = {"Korea, Rep.": "South Korea",
                  "Iran, Islamic Rep.": "Iran",
                  "Hong Kong SAR, China": "Hong Kong"}

GDP.replace({'Country Name' : replace_values}, inplace = True)

GDP.rename(columns = {'Country Name' : 'Country'}, inplace = True)

GDP

#SCIMAGO FILE
ScimEn = pd.read_excel('assets/scimagojr-3.xlsx')


def answer_one():
    global energy, GDP, ScimEn

    #merge energy and GDP
    combined = pd.merge(energy, GDP, how = 'inner', on = 'Country')

    #merge energy+GDP and ScimEn
    combined = pd.merge(combined, ScimEn, how = 'inner', on = 'Country')

    #only keep the top 15 countries
    combined = combined[combined['Rank'] <= 15]

    #sort in ascending order based on rank
    combined.sort_values(['Rank'], ascending = True, inplace = True)

    #make Country the index
    combined.set_index('Country', inplace = True)

    #keep only specified indicies
    combined = combined[['Rank', 'Documents', 'Citable documents', 'Citations',
                        'Self-citations', 'Citations per document', 'H index',
                        'Energy Supply', 'Energy Supply per Capita', '% Renewable',
                        '2006', '2007', '2008', '2009', '2010', '2011', '2012',
                        '2013', '2014', '2015']]

    return combined

#Question 2
import pandas as pd
import numpy as np

def answer_two():
    all_joined = pd.merge(pd.merge(energy, GDP, how = 'outer', on = 'Country'), ScimEn, how = 'outer', on = 'Country')
    center_joined = pd.merge(pd.merge(energy, GDP, how = 'inner', on = 'Country'), ScimEn, how = 'inner', on = 'Country')

    return len(all_joined) - len(center_joined)

#Question 3
import pandas as pd
import numpy as np

def answer_three():
    combined = answer_one()
    avgGDP = combined[['2006', '2007', '2008', '2009', '2010', '2011', '2012',
                    '2013', '2014', '2015']].mean(axis = 'columns').sort_values(ascending = False)

    return avgGDP

#Question 4
import pandas as pd
import numpy as np

def answer_four():
    combined = answer_one()
    combined['avgGDP'] = answer_three()
    combined.sort_values('avgGDP', ascending = False, inplace = True)

    return abs(combined.iloc[5]['2015'] - combined.iloc[5]['2006'])

#Question 5
import pandas as pd
import numpy as np

def answer_five():
    combined = answer_one()

    return combined['Energy Supply per Capita'].mean()

#Question 6
import pandas as pd
import numpy as np

def answer_six():
    combined = answer_one()

    combined.sort_values('% Renewable', ascending = False, inplace = True)
    return (combined.index[0], combined['% Renewable'][0])

#Question 7
import pandas as pd
import numpy as np

def answer_seven():
    combined = answer_one()
    combined['Citation Ratio'] = combined['Self-citations'] / combined['Citations']
    combined.sort_values('Citation Ratio', ascending = False, inplace = True)

    return (combined.index[0], combined['Citation Ratio'][0])

#Question 8
import pandas as pd
import numpy as np

def answer_eight():
    combined = answer_one()
    combined['Est Pop'] = combined['Energy Supply'] / combined['Energy Supply per Capita']
    combined.sort_values('Est Pop', ascending = False, inplace = True)

    return combined.index[2]

#Question 9
import pandas as pd
import numpy as np

def answer_nine():
    combined = answer_one()
    combined['Est Pop'] = combined['Energy Supply'] / combined['Energy Supply per Capita']
    combined['Citable Doc per Capita'] = combined['Citable documents'] / combined['Est Pop']

    return combined['Citable Doc per Capita'].astype('float64').corr(combined['Energy Supply per Capita'].astype('float64'))

#Question 10
import pandas as pd
import numpy as np

def answer_ten():
    combined = answer_one()
    med = combined['% Renewable'].median(axis = "rows")

    def score(percent):
        if percent >= med:
            return 1
        return 0

    combined['HighRenew'] = combined['% Renewable'].apply(score)
    return combined['HighRenew']

#Question 11
import pandas as pd
import numpy as np

def answer_eleven():
    ContinentDict  = {'China':'Asia',
                  'United States':'North America',
                  'Japan':'Asia',
                  'United Kingdom':'Europe',
                  'Russian Federation':'Europe',
                  'Canada':'North America',
                  'Germany':'Europe',
                  'India':'Asia',
                  'France':'Europe',
                  'South Korea':'Asia',
                  'Italy':'Europe',
                  'Spain':'Europe',
                  'Iran':'Asia',
                  'Australia':'Australia',
                  'Brazil':'South America'}

    out = pd.DataFrame(columns = ['size', 'sum', 'mean', 'std'])

    combined = answer_one()
    combined['Est Pop'] = combined['Energy Supply'] / combined['Energy Supply per Capita']

    for group, frame in combined.groupby(ContinentDict):
        out.loc[group] = [len(frame), frame['Est Pop'].sum(), frame['Est Pop'].mean(), frame['Est Pop'].std()]

    return out

#Question 12
import pandas as pd
import numpy as np

def answer_twelve():
    ContinentDict  = {'China':'Asia',
                  'United States':'North America',
                  'Japan':'Asia',
                  'United Kingdom':'Europe',
                  'Russian Federation':'Europe',
                  'Canada':'North America',
                  'Germany':'Europe',
                  'India':'Asia',
                  'France':'Europe',
                  'South Korea':'Asia',
                  'Italy':'Europe',
                  'Spain':'Europe',
                  'Iran':'Asia',
                  'Australia':'Australia',
                  'Brazil':'South America'}

    combined = answer_one()
    combined['Continent'] = combined.index.map(lambda c : ContinentDict[c])
    combined['% Renewable'] = pd.cut(combined['% Renewable'], 5)

    return combined.groupby(['Continent', '% Renewable']).size()

#Question 13
import pandas as pd
import numpy as np

def answer_thirteen():
    combined = answer_one()
    combined['Est Pop'] = combined['Energy Supply'] / combined['Energy Supply per Capita']

    #Convert a number into a string
    #Thousands separators w/ commas
    return combined['Est Pop'].apply(lambda s : '{:,}'.format(s))
