import numpy as np
import pandas as pd 

col = ['parent', 'children', 'wins', 'visits', 'value']
df = pd.DataFrame(columns=col)
data={'parent': 'root', 'children': [], 'wins': 0, 'visits': 0, 'value': 0}
df.loc['123456789'] = data
df.loc['1'] = data

children = np.array(['123456789', '1'])
print(df)
