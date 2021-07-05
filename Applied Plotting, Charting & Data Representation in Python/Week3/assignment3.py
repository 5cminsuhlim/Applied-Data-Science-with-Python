import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650),
                   np.random.normal(43000,100000,3650),
                   np.random.normal(43500,140000,3650),
                   np.random.normal(48000,70000,3650)],
                  index=[1992,1993,1994,1995])

mean = df.mean(axis = 'columns')
std = df.std(axis = 'columns')

n = df.shape[1]
z = 1.96

#get plus-minus portion of CI
pm = z * std / np.sqrt(n)

lbound = mean - pm
ubound = mean + pm

df['mean'] = mean
df['std'] = std
df['pm'] = pm
df['lbound'] = lbound
df['ubound'] = ubound

df = df[['mean', 'std', 'pm', 'lbound', 'ubound']]

voi = np.mean(df['mean'])

#function to determine bar color
def barColor(mean, pm, voi):
    if (mean - pm) <= voi and (mean + pm) >= voi:
        return 'white'
    elif mean < voi:
        return 'blue'
    return 'red'

#get a list of colors
colors = [barColor(df['mean'].iloc[x], df['pm'].iloc[x], voi) for x in range(df.shape[0])]

plt.figure()

fig, ax = plt.subplots()

plt.bar(range(df.shape[0]), mean, align = 'center', yerr = pm, color = colors, width = 0.5)

for spine in plt.gca().spines.values():
    spine.set_visible(False)

#threshold line
plt.axhline(y = voi, linestyle="--", color="orange")

ax.set_facecolor("gray")

plt.xticks(range(len(df.index)), df.index)
plt.title('Easy option')
plt.xlabel('Year')

plt.show()
