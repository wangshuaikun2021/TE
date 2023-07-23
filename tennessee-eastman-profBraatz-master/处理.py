import functools
from multiprocessing import Pool
import pandas as pd
import os
from tqdm import tqdm


def read_dat(file, r_path, s_path, cols):
    if '.dat' in file:
        # print(file)
        df = pd.read_csv(fr'{r_path}\{file}', header=None, sep='\s+')
        print(df.shape)
        if df.shape[0] == 52:
            df = df.T
        df.columns = cols
        df.to_csv(fr'{s_path}\{file[:-4]}.csv', index=False)


if __name__ == '__main__':
    path = 'tennessee-eastman-profBraatz-master'
    files = os.listdir('tennessee-eastman-profBraatz-master')
    # df_concat = pd.DataFrame()
    cols = []
    for i in range(41):
        cols.append(f'XMEAS_{i + 1}')  # XMEAS(i)
    for i in range(11):
        cols.append(f'XMV_{i + 1}')  # XMV(i)
    print(cols)
    save_path = r'TE数据集'
    os.makedirs(save_path, exist_ok=True)
    multi_read_dat = functools.partial(read_dat, r_path=path, s_path=save_path, cols=cols)
    size = 12
    for i in tqdm(range(0, len(files), size)):
        if i + size <= len(files):
            fs = files[i:i + size]
        else:
            fs = files[i:]
        pool = Pool(size)
        pool.map(multi_read_dat, fs)
        pool.close()
        pool.join()
