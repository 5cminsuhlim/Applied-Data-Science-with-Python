df = pd.read_csv('data/C2A2_data/BinnedCsvs_d400/fb441e62df2d58994928907a91895ec62c2c42e6cd075c2700843b89.csv')

#sort ID and Dates in ascending order
df.sort_values(['ID', 'Date'], ascending = True, inplace = True)

#remove leap years
df = df[~df['Date'].str.contains('02-29')].dropna()

df['Month_Day'] = df['Date'].apply(lambda x : x[5:])

#get max temps from non-2015 dates
#group max temps by the day
#get the max temp among the grouped days
highs = df[(df['Element'] == 'TMAX') & (~df['Date'].str.contains('2015'))].groupby('Month_Day').agg({'Data_Value' : np.max})
#365

#get min temps from non-2015 dates
#group min temps by the day
#get the min temp among the grouped days
lows = df[(df['Element'] == 'TMIN') & (~df['Date'].str.contains('2015'))].groupby('Month_Day').agg({'Data_Value' : np.min})

#repeat above for 2015's highs and lows
highs2015 = df[(df['Element'] == 'TMAX') & (df['Date'].str.contains('2015'))].groupby('Month_Day').agg({'Data_Value' : np.max})
lows2015 = df[(df['Element'] == 'TMIN') & (df['Date'].str.contains('2015'))].groupby('Month_Day').agg({'Data_Value' : np.min})

#gets the days in which 2015's highs and lows broke 2005-2014 highs and lows
highest2015 = np.where(highs2015['Data_Value'] > highs['Data_Value'])[0]
lowest2015 = np.where(lows2015['Data_Value'] < lows['Data_Value'])[0]


#visual
plt.figure()

#line
plt.plot(highs.values, 'salmon')
plt.plot(lows.values, 'skyblue')

#scatter
plt.scatter(highest2015, highs2015.iloc[highest2015], s = 5, c ='darkred')
plt.scatter(lowest2015, lows2015.iloc[lowest2015], s = 5, c ='darkblue')

#labels
plt.xlabel('Days')
plt.ylabel('Temperature (tenths of degrees C)')
plt.title('Ann Arbor, Michigan Weather')
plt.legend(['Max Temp', 'Min Temp', "Record High", "Record Low"], frameon = False)

#formatting
#fill
plt.gca().fill_between(range(len(highs)), highs['Data_Value'], lows['Data_Value'], facecolor = 'plum', alpha = 0.25)

#get rid of top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

#space x axis ticks every 30 days
#convert x axis days to actual dates
plt.xticks(range(0, len(highs), 30), highs.index[range(0, len(highs), 30)], rotation = '45')

#display
plt.show()
