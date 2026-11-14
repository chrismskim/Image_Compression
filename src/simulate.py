"""Reproducible rank-five plus Gaussian-noise simulation, based on assignment p. 6."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from svd_core import svd_from_scratch, reconstruct, metrics

BASE=Path(__file__).resolve().parents[1]
(BASE/'figures').mkdir(exist_ok=True)
rng=np.random.default_rng(0)
m,n,r=120,80,5
A0=rng.normal(size=(m,r)) @ rng.normal(size=(r,n))
A=A0+0.01*rng.normal(size=(m,n))
U,s,Vt=svd_from_scratch(A)
s_builtin=np.linalg.svd(A,compute_uv=False)  # final cross-check only
q4_error=abs(np.sum(A**2)-np.sum(s**2))
orth_error=np.max(abs(U[:,:5].T@U[:,:5]-np.eye(5)))
rows=[]
for k in [1,3,5,10]:
    Ak=reconstruct(U,s,Vt,k)
    rows.append(metrics(A,Ak,s,k))
df=pd.DataFrame(rows)
df.to_csv(BASE/'docs'/'simulation_results.csv',index=False)

fig,ax=plt.subplots(figsize=(7,4));ax.plot(np.arange(1,len(s)+1),s,'o-',markersize=3)
ax.axvline(5,linestyle='--',color='gray',label='True signal rank = 5')
ax.set(xlabel='Component index',ylabel='Singular value',title='Simulated Rank-5 Matrix: Singular Values');ax.legend();fig.tight_layout()
fig.savefig(BASE/'figures'/'simulation_spectrum.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));ax.plot(df['rank'],df['sq_error'],'o-',label='Actual squared Frobenius error');ax.plot(df['rank'],df['tail_energy'],'x--',label='Discarded singular-value energy');ax.set(xlabel='Retained rank k',ylabel='Squared error',title='Q5: Reconstruction Error Identity');ax.legend();fig.tight_layout();fig.savefig(BASE/'figures'/'simulation_error.png',dpi=170);plt.close(fig)

# Truly noiseless known-rank check complements the noisy assignment setup.
U0,s0,Vt0=svd_from_scratch(A0)
rank5_error=np.linalg.norm(A0-reconstruct(U0,s0,Vt0,5),'fro')
with open(BASE/'docs'/'simulation_generated.tex','w') as f:
    f.write(r'\newcommand{\SimTraceError}{'+f'{q4_error:.2e}'+'}\n')
    f.write(r'\newcommand{\SimOrthError}{'+f'{orth_error:.2e}'+'}\n')
    f.write(r'\newcommand{\SimNoiselessError}{'+f'{rank5_error:.2e}'+'}\n')
    f.write(r'\newcommand{\SimTable}{'+'\n')
    f.write(r'\begin{tabular}{rrrrr}\toprule Rank $k$ & Measured $\|A-A_k\|_F^2$ & $\sum_{i>k}\sigma_i^2$ & Scalars & Energy (\%) \\\midrule'+'\n')
    for d in rows:
        f.write(f"{d['rank']} & {d['sq_error']:.4f} & {d['tail_energy']:.4f} & {d['stored_scalars']} & {d['energy_percent']:.2f} \\\\ \n")
    f.write(r'\bottomrule\end{tabular}'+'\n}\n')
print('Simulation: eigenvalue/SVD Q4 difference',q4_error,'Q2 orthonormality max error',orth_error)
print('Noiseless rank-5 error',rank5_error,'Built-in singular values max difference',np.max(abs(s-s_builtin)))
print(df.to_string(index=False))
assert np.allclose(df['sq_error'],df['tail_energy'],rtol=1e-8,atol=1e-7)
assert q4_error<1e-6 and orth_error<1e-6
