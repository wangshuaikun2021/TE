import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('TE数据集\d00.csv')
cols = df.columns.tolist()
print(cols)
df = df.values
df1 = pd.read_csv('TE数据集\d01.csv').values
df2 = pd.read_csv('TE数据集\d02.csv').values
for i in range(df.shape[1]):
    plt.figure(dpi=200)
    plt.plot(df[:, i])
    plt.plot(df2[:, i])
    plt.title(cols[i])
    plt.savefig(f'图\{cols[i]}.png', dpi=300)
    plt.close()
# plt.show()