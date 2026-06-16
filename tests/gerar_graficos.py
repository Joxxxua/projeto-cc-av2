import sys
sys.path.insert(0, '.')

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import os

os.makedirs('notebooks/img', exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted')

df = pd.read_parquet('data/final/f1_dataset_final.parquet')

# --- 1. Distribuicao do target ---
counts = df['top10'].value_counts().sort_index()
labels = ['Fora do top10 (0)', 'Top10 (1)']
colors = ['#d9534f', '#5cb85c']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(labels, counts.values, color=colors, edgecolor='white', width=0.5)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
axes[0].set_title('Contagem por classe', fontsize=13)
axes[0].set_ylabel('Corridas')
axes[1].pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, wedgeprops=dict(edgecolor='white'))
axes[1].set_title('Proporcao das classes', fontsize=13)
plt.suptitle('1. Distribuicao do target top10', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/img/01_target.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [OK] 01_target.png')

# --- 2. Taxa de top10 por grid ---
grid_valido = df[df['grid'].between(1, 20)].copy()
taxa_grid = grid_valido.groupby('grid')['top10'].mean().reset_index()
taxa_grid.columns = ['grid', 'taxa_top10']

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(taxa_grid['grid'], taxa_grid['taxa_top10'],
              color=['#5cb85c' if t >= 0.5 else '#d9534f' for t in taxa_grid['taxa_top10']],
              edgecolor='white')
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='50%')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel('Posicao de largada (grid)', fontsize=12)
ax.set_ylabel('Taxa de top10', fontsize=12)
ax.set_title('2. Taxa de top10 por posicao de grid\n(verde = >50%, vermelho = <50%)', fontsize=13)
ax.set_xticks(taxa_grid['grid'])
ax.legend()
plt.tight_layout()
plt.savefig('notebooks/img/02_grid.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [OK] 02_grid.png')

# --- 3. Taxa de top10 por construtor ---
taxa_constructor = (
    df.groupby('constructorId')['top10']
    .agg(taxa='mean', corridas='count')
    .reset_index()
    .sort_values('taxa', ascending=True)
)
fig, ax = plt.subplots(figsize=(10, 7))
colors_c = ['#5cb85c' if t >= 0.5 else '#d9534f' for t in taxa_constructor['taxa']]
bars = ax.barh(taxa_constructor['constructorId'].astype(str),
               taxa_constructor['taxa'], color=colors_c, edgecolor='white')
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
for bar, (_, row) in zip(bars, taxa_constructor.iterrows()):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f"{row['corridas']} corridas", va='center', fontsize=8, color='#555')
ax.set_xlabel('Taxa de top10', fontsize=12)
ax.set_ylabel('constructorId', fontsize=12)
ax.set_title('3. Taxa de top10 por construtor (2021-2023)', fontsize=13)
plt.tight_layout()
plt.savefig('notebooks/img/03_construtor.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [OK] 03_construtor.png')

# --- 4. Distribuicao features climaticas ---
clima_cols = ['TrackTemp', 'AirTemp', 'Humidity', 'Rainfall', 'WindSpeed']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(clima_cols):
    ax = axes[i]
    for top10_val, label, color in [(0, 'Fora top10', '#d9534f'), (1, 'Top10', '#5cb85c')]:
        subset = df[df['top10'] == top10_val][col]
        ax.hist(subset, bins=25, alpha=0.55, label=label, color=color, edgecolor='white')
    ax.set_title(col, fontsize=12)
    ax.set_xlabel('Valor medio na corrida')
    ax.set_ylabel('Frequencia')
    ax.legend(fontsize=9)
axes[-1].set_visible(False)
plt.suptitle('4. Distribuicao das features climaticas por classe', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/img/04_clima.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [OK] 04_clima.png')

# --- 5. Heatmap correlacao ---
num_cols = ['grid', 'round', 'year', 'TrackTemp', 'AirTemp', 'Humidity', 'Rainfall', 'WindSpeed', 'top10']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            square=True, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('5. Correlacao entre features numericas e target', fontsize=13)
plt.tight_layout()
plt.savefig('notebooks/img/05_correlacao.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [OK] 05_correlacao.png')

# --- 6. Taxa de top10 por circuito ---
taxa_circuit = (
    df.groupby('circuitId')['top10']
    .agg(taxa='mean', corridas='count')
    .reset_index()
    .sort_values('taxa', ascending=True)
)
fig, ax = plt.subplots(figsize=(10, 9))
colors_circ = ['#5cb85c' if t >= 0.5 else '#d9534f' for t in taxa_circuit['taxa']]
ax.barh(taxa_circuit['circuitId'].astype(str), taxa_circuit['taxa'],
        color=colors_circ, edgecolor='white')
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlabel('Taxa de top10', fontsize=12)
ax.set_ylabel('circuitId', fontsize=12)
ax.set_title('6. Taxa de top10 por circuito (2021-2023)', fontsize=13)
plt.tight_layout()
plt.savefig('notebooks/img/06_circuito.png', dpi=120, bbox_inches='tight')
plt.close()
print('  [OK] 06_circuito.png')

print('\nTodos os graficos salvos em notebooks/img/')
