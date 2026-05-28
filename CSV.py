import pandas as pd

# 1. 列名
cols = [
    (0, 22, 'RAdeg'),    (23, 45, 'DEdeg'),   (46, 53, 'rmag'),    (54, 60, 'e_rmag'),
    (61, 67, 'g_r'),     (68, 74, 'e_g_r'),   (75, 82, 'RVel'),    (83, 88, 'e_RVel'),
    (89, 93, 'Teff'),    (94, 97, 'e_Teff'),  (98, 104, 'logg'),   (105, 110, 'e_logg'),
    (111, 117, 'feh'),  (118, 123, 'e_feh'),(124, 131, 'DM5'),   (132, 139, 'DM16'),
    (140, 147, 'DM50'),  (148, 155, 'DM84'),  (156, 163, 'DM95'),  (164, 170, 'e_DM50'),
    (171, 178, 'rMag_d'),  (179, 185, 'e_rMag_d'),  (186, 194, 'D'),   (195, 202, 'e_D'),
    (203, 211, 'rGC'),  (212,219,'e_rGC')
]

# 2. 读取文件
df = pd.read_fwf(
    'ajacc9bbt1_mrt.txt',  
    colspecs=[(s, e) for s, e, _ in cols],
    names=[n for _, _, n in cols],
    skiprows=37,
    header=None
)

# 3. 保存为 CSV
df.to_csv('data.csv', index=False)

print("保存为 data.csv")