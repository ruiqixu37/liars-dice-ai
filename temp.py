import numpy as np
import pandas as pd 

col = ['parent', 'children', 'wins', 'visits', 'value']
df = pd.DataFrame(columns=col)
data={'parent': 'root', 'children': [], 'wins': 0, 'visits': 0, 'value': 0}
df.loc['123456789'] = data
df.loc['1'] = data

children = np.array(['123456789', '1'])

df['wins'] += 1
df['visits'] += 1

parent_visit = 1

children = df.loc[children]
children = children.loc[children['visits'] > 0]

uct_values = children['wins'] / children['visits'] + \
    0.1 * np.sqrt(np.log(parent_visit) / children['visits'].astype('float'))
print(df)

