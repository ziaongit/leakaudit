"""Figures for reframed Paper 2 (leakage audit)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8a85', 'axes.labelcolor': '#333', 'xtick.color': '#555', 'ytick.color': '#555'})
BLUE, ORANGE, AQUA, GRAY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#b9b8b0', '#222'
R = {}
for f in ['A', 'Anocharge', 'B', 'rule5']: R.update(json.load(open(f'results/audit_{f}.json')))
short = {'Average Submitted Charge Amount': 'Avg. Submitted Charge', 'Average Medicare Allowed Amount': 'Avg. Medicare Allowed Amt',
         'Credentials of the Provider': 'Provider Credentials', 'Charge_Allowed_Ratio': 'Charge_Allowed_Ratio'}

# ---------- Fig 1: audit flow ----------
fig, ax = plt.subplots(figsize=(7.0, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(1.4, 10); ax.axis('off')
steps = [('Step 1  Attribution concentration', 'SHAP share of top feature(s)', 'One feature dominates in every run?'),
         ('Step 2  Suspect-only model', 'Retrain on suspect features only', 'Matches the full model?  → decisive leak test'),
         ('Step 3  Removal & proxy check', 'Drop suspects, retrain, re-run SHAP', 'Large F1 drop?  Related feature takes over?'),
         ('Step 4  Global stability', 'Spearman / Kendall / top-5 Jaccard', 'Read together with Steps 1–3'),
         ('Step 5  Local check (LIME)', '30 records × 5 repeats', 'Same top-1? How stable is the rest?')]
y = 9.0
for i, (t, m, q) in enumerate(steps):
    hi = i == 1
    ax.add_patch(FancyBboxPatch((0.3, y - 0.65), 5.2, 1.3, boxstyle='round,pad=0.02,rounding_size=0.15',
                                fc='#e8f1fc' if hi else '#f4f4f1', ec=BLUE if hi else '#9a9990', lw=1.4 if hi else 0.8))
    ax.text(0.55, y + 0.22, t, fontsize=9.5, fontweight='bold', color=INK, va='center')
    ax.text(0.55, y - 0.28, m, fontsize=8.5, color='#444', va='center')
    ax.annotate('', xy=(6.0, y), xytext=(5.55, y), arrowprops=dict(arrowstyle='->', color='#777', lw=0.9))
    ax.text(6.1, y, q, fontsize=8.5, color=INK, va='center', wrap=True)
    if i < 4: ax.annotate('', xy=(2.9, y - 1.05), xytext=(2.9, y - 0.68), arrowprops=dict(arrowstyle='->', color='#777', lw=0.9))
    y -= 1.75
fig.tight_layout(); fig.savefig('figs/p2_fig1_audit.png', dpi=300); fig.savefig('figs/p2_fig1_audit.pdf'); plt.close(fig)

# ---------- Fig 2: SHAP shares A vs B ----------
def mean_imp(key):
    return pd.DataFrame([r['imp'] for r in R[key]['runs']]).mean().sort_values(ascending=False)
iA, iB = mean_imp('A_full').head(6), mean_imp('B').head(6)
rule = {'Average Submitted Charge Amount', 'Provider Type'}; derived = {'PCR', 'Charge_Allowed_Ratio'}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True)
for ax, imp, title in [(axs[0], iA, '(a) Dataset A: rule-based labels'), (axs[1], iB, '(b) Dataset B: supplied labels')]:
    names = [short.get(n, n) for n in imp.index][::-1]; vals = imp.values[::-1]
    cols = [BLUE if n in rule else (ORANGE if n in derived else GRAY) for n in imp.index][::-1]
    ax.barh(names, vals, color=cols, height=0.62, edgecolor='white', linewidth=1)
    for yy, v in enumerate(vals): ax.text(v + 0.008, yy, f'{v:.2f}', va='center', fontsize=8, color=INK)
    ax.set_title(title, fontsize=9.5, loc='left', color=INK); ax.set_xlim(0, 0.62)
    ax.xaxis.grid(True, color='#e6e6e1', lw=0.6); ax.set_axisbelow(True); ax.tick_params(axis='y', length=0)
axs[0].set_xlabel('Mean share of |SHAP| (15 runs)'); axs[1].set_xlabel('Mean share of |SHAP| (15 runs)')
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BLUE, label='Label-defining feature'), Patch(color=ORANGE, label='Derived from charge'), Patch(color=GRAY, label='Other feature')],
           loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig('figs/p2_fig2_shap.png', dpi=300); fig.savefig('figs/p2_fig2_shap.pdf'); plt.close(fig)

# ---------- Fig 3: F1 by feature set ----------
def f1(key):
    d = pd.DataFrame(R[key]['runs']); g = d.groupby('model').f1; return g.mean(), g.std()
mods = ['LGBM', 'XGB', 'RF']; lab = {'LGBM': 'LightGBM', 'XGB': 'XGBoost', 'RF': 'Random Forest'}
sets = [('A_full', 'All features', BLUE, ''), ('A_ruleonly', 'Suspect features only', ORANGE, '///'), ('A_nocharge', 'Suspects removed', AQUA, '..')]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={'width_ratios': [3, 1.25]}, sharey=True)
w = 0.26; x = np.arange(3)
for j, (k, name, c, h) in enumerate(sets):
    m, s = f1(k); mv = [m[q] for q in mods]; sv = [s[q] for q in mods]
    b = axs[0].bar(x + (j - 1) * w, mv, w - 0.03, yerr=sv, color=c, hatch=h, edgecolor='white', lw=0.8, capsize=2, error_kw=dict(lw=0.8, ecolor='#555'), label=name)
    for xx, v, sd in zip(x + (j - 1) * w, mv, sv): axs[0].text(xx, v + sd + 0.02, f'{v:.2f}', ha='center', fontsize=7.2, color=INK)
axs[0].set_xticks(x, [lab[q] for q in mods]); axs[0].set_ylabel('F1 (mean ± SD, 5 splits)'); axs[0].set_ylim(0, 1.1)
axs[0].set_title('(a) Dataset A', fontsize=9.5, loc='left', color=INK)
# Dataset B, LightGBM (b_single.py): all / ReimbSum only / ReimbSum removed
bm, bs = [0.650, 0.614, 0.654], [0.035, 0.035, 0.027]
for j, (v, s, c, h) in enumerate(zip(bm, bs, [BLUE, ORANGE, AQUA], ['', '///', '..'])):
    axs[1].bar((j - 1) * w, v, w - 0.03, yerr=s, color=c, hatch=h, edgecolor='white', lw=0.8, capsize=2, error_kw=dict(lw=0.8, ecolor='#555'))
    axs[1].text((j - 1) * w, v + s + 0.02, f'{v:.2f}', ha='center', fontsize=7.2, color=INK)
axs[1].set_xticks([0], ['LightGBM']); axs[1].set_xlim(-0.6, 0.6)
axs[1].set_title('(b) Dataset B\n(suspect = ReimbSum)', fontsize=9.5, loc='left', color=INK)
for ax in axs: ax.yaxis.grid(True, color='#e6e6e1', lw=0.6); ax.set_axisbelow(True)
fig.legend(loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig('figs/p2_fig3_f1.png', dpi=300); fig.savefig('figs/p2_fig3_f1.pdf'); plt.close(fig)
print('ok')
