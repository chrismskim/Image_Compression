"""Real grayscale-image analysis; source image: Matplotlib Grace Hopper sample photo."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from svd_core import svd_from_scratch,reconstruct,metrics
BASE=Path(__file__).resolve().parents[1]
original=plt.imread(BASE/'data'/'grace_hopper.jpg')
# Luminance transformation produces a grayscale pixel matrix from real photo.
A=(original[:,:,:3].astype(float) @ np.array([0.299,0.587,0.114])) if original.ndim==3 else original.astype(float)
m,n=A.shape
U,s,Vt=svd_from_scratch(A)
ks=[5,10,20,40]
rows=[]
fig,axes=plt.subplots(1,5,figsize=(15,3.5))
axes[0].imshow(A,cmap='gray',vmin=0,vmax=255);axes[0].set_title('Original grayscale')
for i,k in enumerate(ks,1):
    Ak=reconstruct(U,s,Vt,k)
    rows.append(metrics(A,Ak,s,k))
    axes[i].imshow(np.clip(Ak,0,255),cmap='gray',vmin=0,vmax=255)
    axes[i].set_title(f'Rank {k}')
for ax in axes:ax.axis('off')
fig.tight_layout();fig.savefig(BASE/'figures'/'real_reconstructions.png',dpi=170);plt.close(fig)
df=pd.DataFrame(rows)
df.to_csv(BASE/'docs'/'real_image_results.csv',index=False)
energy=np.cumsum(s**2)/sum(s**2)
k95=int(np.searchsorted(energy,0.95)+1)
fig,ax=plt.subplots(figsize=(7,4));ax.plot(np.arange(1,len(s)+1),100*energy)
ax.axhline(95,color='gray',linestyle='--');ax.axvline(k95,color='gray',linestyle='--')
ax.set(xlabel='Retained rank k',ylabel='Energy retained (%)',title=f'Grayscale Photo: 95% Energy at k={k95}')
fig.tight_layout();fig.savefig(BASE/'figures'/'real_energy.png',dpi=170);plt.close(fig)
with open(BASE/'docs'/'real_generated.tex','w') as f:
    for name,v in [('ImageRows',m),('ImageCols',n),('ImageRank',len(s)),('ImageKAtNinetyFive',k95),('ImageMin',f'{A.min():.2f}'),('ImageMax',f'{A.max():.2f}')]:
        f.write('\\newcommand{\\'+name+'}{'+str(v)+'}\n')
    f.write(r'\newcommand{\RealTable}{'+'\n')
    f.write(r'\begin{tabular}{rrrrr}\toprule $k$ & Squared error & Storage & Ratio & Energy (\%)\\\midrule'+'\n')
    for d in rows:
        f.write(f"{d['rank']} & {d['sq_error']:.2f} & {d['stored_scalars']} & {d['compression_ratio']:.2f} & {d['energy_percent']:.2f} \\\\ \n")
    f.write(r'\bottomrule\end{tabular}'+'\n}\n')
print('Image size',A.shape,'pixel range',A.min(),A.max(),'nonfinite pixels',np.size(A)-np.isfinite(A).sum())
print('k95',k95, 'all ranks',len(s));print(df.to_string(index=False))
assert not np.isnan(A).any()
