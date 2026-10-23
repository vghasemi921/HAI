# -*- coding: utf-8 -*-
"""
==========================================================
Comprehensive QC Report Generator for HAI Dataset
Produces: Excel tables + PNG figures + console summary
==========================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# =========================================================
# 0) SETTINGS
# =========================================================
CSV_FILE   = "Hospital_infection_data.csv"
OUTPUT_DIR = "QC_Output"
FIGS_DIR   = os.path.join(OUTPUT_DIR, "figures")
EXCEL_FILE = os.path.join(OUTPUT_DIR, "QC_Tables.xlsx")
DPI        = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi']    = 100
plt.rcParams['savefig.dpi']   = DPI
plt.rcParams['savefig.bbox']  = 'tight'
plt.rcParams['font.family']   = 'DejaVu Sans'

# =========================================================
# 1) IRANIAN LAB REFERENCE RANGES
# =========================================================
# These are the ranges you said you'd verify later.
# Plausibility = physiologically possible at all
# Normal = clinical reference range (Iranian standard)

REFERENCE = {
    # var: (plaus_lo, plaus_hi, norm_lo, norm_hi, unit)
    'Age':       (0,   120,   0,   120,  'years'),
    'SPo2':      (50,  100,   94,  100,  '%'),
    'BS_Triage': (20,  800,   70,  200,  'mg/dL'),
    'BPMin':     (30,  150,   60,  90,   'mmHg'),
    'BPMax':     (50,  260,   90,  140,  'mmHg'),
    'T':         (32,  42,    36.1, 37.2,'°C'),
    'PR':        (30,  220,   60,  100,  'bpm'),
    'RR':        (8,   60,    12,  20,   'breaths/min'),
    'WBC':       (0.5, 50,    4.5, 11.0, '×10³/μL'),
    'RBC':       (2, 7, 4.1, 5.9, '×10⁶/μL'),
    'HB':        (5,   22,    12.0,17.5, 'g/dL'),
    'HCT':       (15,  65,    36.0,52.0, '%'),
    'PLT':       (10,  1000,  150, 450,  '×10³/μL'),
    'ESR':       (0,   150,   0,   20,   'mm/hr'),
    'UREA':      (5,   250,   15,  45,   'mg/dL'),
    'CR':        (0.2, 15,    0.6, 1.4,  'mg/dL'),
    'NA':        (100, 180,   135, 145,  'mEq/L'),
    'K':         (1.5, 8,     3.5, 5.1,  'mEq/L'),
    'PT':        (8,   60,    11,  13.5, 'seconds'),
    'BS_Lab':    (20,  800,   70,  100,  'mg/dL'),
}

# =========================================================
# 2) LOAD DATA
# =========================================================
def load_data(path):
    for enc in ['utf-8-sig', 'utf-8', 'cp1256', 'latin1']:
        try:
            d = pd.read_csv(path, encoding=enc)
            print(f"✔ Loaded with encoding: {enc}")
            return d
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Could not load CSV.")

df = load_data(CSV_FILE)

# Drop the leading row-number column
for c in ['#', 'Unnamed: 0']:
    if c in df.columns:
        df = df.drop(columns=[c])

# Clean column names
df.columns = [c.strip() for c in df.columns]

print(f"✔ Shape: {df.shape}")
print(f"✔ Columns: {list(df.columns)}")

# =========================================================
# 3) BASIC SETUP
# =========================================================
LABEL_COL   = 'Label'
NUMERIC_COLS = ['Age','SPo2','BS_Triage','BPMin','BPMax','T','PR','RR',
                'WBC','RBC','HB','HCT','PLT','ESR','UREA','CR','NA','K',
                'PT','BS_Lab']
CATEG_COLS   = ['Gender','Ward','Triage_Level','CRP','Diagnosis']

# Coerce numerics
for c in NUMERIC_COLS:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors='coerce').fillna(0).astype(int)

N_TOTAL = len(df)
N_HAI   = int((df[LABEL_COL] == 1).sum())
N_NON   = int((df[LABEL_COL] == 0).sum())

print(f"\n=== Dataset summary ===")
print(f"Total     : {N_TOTAL:,}")
print(f"HAI (+)   : {N_HAI:,}  ({100*N_HAI/N_TOTAL:.3f}%)")
print(f"non-HAI   : {N_NON:,}  ({100*N_NON/N_TOTAL:.3f}%)")
print(f"Imbalance : 1 : {N_NON/N_HAI:.1f}")

# Storage for Excel sheets
SHEETS = {}

# =========================================================
# TABLE 1 — Dataset summary
# =========================================================
SHEETS['Table1_Summary'] = pd.DataFrame({
    'Metric': ['Total samples','HAI cases','Non-HAI cases','HAI prevalence (%)',
               'Imbalance ratio (non-HAI : HAI)'],
    'Value' : [N_TOTAL, N_HAI, N_NON,
               round(100*N_HAI/N_TOTAL, 3),
               round(N_NON/N_HAI, 2)]
})

# =========================================================
# TABLE 2 — Variable characteristics (paper table 2 replacement)
# =========================================================
rows = []
for c in df.columns:
    s = df[c]
    miss = 100 * s.isna().sum() / len(s)
    if pd.api.types.is_numeric_dtype(s):
        vtype = 'Numeric'
        uniq = s.nunique(dropna=True)
    else:
        vtype = 'Categorical'
        uniq = s.nunique(dropna=True)
    rows.append({
        'Feature'     : c,
        'Type'        : vtype,
        'Unique'      : int(uniq),
        'Missing (%)' : round(miss, 2)
    })
SHEETS['Table2_Variables'] = pd.DataFrame(rows)

# =========================================================
# TABLE 3 — Preprocessing cascade
# =========================================================
cascade = pd.DataFrame({
    'Stage': [
        'Initial HIS extraction — Admission',
        'Initial HIS extraction — Triage',
        'Initial HIS extraction — Lab',
        'Records present in all three reports',
        'After removing medication-renewal episodes',
        'After removing records without vital signs',
        'After removing burn patients',
        'After removing implausible vital signs',
        'After removing laboratory consistency failures',
        'After removing records with >70% missing',
        'Final dataset',
        '— HAI positive',
        '— HAI negative',
    ],
    'Records_removed': ['—','—','—','—',3302,406,17557,87,93,312,'—','—','—'],
    'Records_remaining': [269805,242150,187777,141500,138198,137792,120235,
                          120148,120055,119743,119743,1660,118083]
})
SHEETS['Table3_Cascade'] = cascade

# =========================================================
# TABLE 4 — Descriptive statistics (numeric)
# =========================================================
def desc_table(df_in, cols):
    out = []
    for c in cols:
        if c not in df_in.columns:
            continue
        s = df_in[c].dropna()
        if len(s) == 0:
            continue
        out.append({
            'Variable'   : c,
            'N'          : int(len(s)),
            'Missing_%'  : round(100*(len(df_in)-len(s))/len(df_in), 2),
            'Mean'       : round(s.mean(), 3),
            'Std'        : round(s.std(), 3),
            'Median'     : round(s.median(), 3),
            'Q1'         : round(s.quantile(0.25), 3),
            'Q3'         : round(s.quantile(0.75), 3),
            'IQR'        : round(s.quantile(0.75)-s.quantile(0.25), 3),
            'Min'        : round(s.min(), 3),
            'Max'        : round(s.max(), 3),
        })
    return pd.DataFrame(out)

SHEETS['Table4_Descriptive'] = desc_table(df, NUMERIC_COLS)

# =========================================================
# TABLE 5 — Categorical distributions
# =========================================================
def cat_dist(df_in, cols, top_n=15):
    out = []
    for c in cols:
        if c not in df_in.columns:
            continue
        vc = df_in[c].value_counts(dropna=True)
        vc = vc.head(top_n)
        total = df_in[c].notna().sum()
        for k, v in vc.items():
            out.append({
                'Variable': c,
                'Category': str(k)[:60],
                'Count'   : int(v),
                'Percent' : round(100*v/total, 2)
            })
    return pd.DataFrame(out)

SHEETS['Table5_Categorical'] = cat_dist(df, CATEG_COLS)

# =========================================================
# TABLE 6 — Range / plausibility checks
# =========================================================
def range_checks(df_in):
    out = []
    for c, (plo, phi, nlo, nhi, unit) in REFERENCE.items():
        if c not in df_in.columns:
            continue
        s = df_in[c].dropna()
        if len(s) == 0:
            continue
        n_plaus_viol = int(((s < plo) | (s > phi)).sum())
        n_below_norm = int((s < nlo).sum())
        n_above_norm = int((s > nhi).sum())
        out.append({
            'Variable'        : c,
            'Unit'            : unit,
            'Plausible_range' : f"{plo} – {phi}",
            'Normal_range'    : f"{nlo} – {nhi}",
            'N_valid'         : int(len(s)),
            'Plaus_violations': n_plaus_viol,
            'Plaus_viol_%'    : round(100*n_plaus_viol/len(s), 3),
            'Below_normal'    : n_below_norm,
            'Below_normal_%'  : round(100*n_below_norm/len(s), 2),
            'Above_normal'    : n_above_norm,
            'Above_normal_%'  : round(100*n_above_norm/len(s), 2),
        })
    return pd.DataFrame(out)

SHEETS['Table6_RangeChecks'] = range_checks(df)

# =========================================================
# TABLE 7 — Internal consistency checks
# =========================================================
consistency_rows = []

def add_check(name, mask, n_total, description):
    n_fail = int((~mask).sum())
    consistency_rows.append({
        'Check'      : name,
        'Description': description,
        'N_evaluated': int(n_total),
        'N_failed'   : n_fail,
        'Fail_%'     : round(100*n_fail/n_total, 3) if n_total else 0,
        'Pass_%'     : round(100*(1-n_fail/n_total), 3) if n_total else 0,
    })

# 1) Diastolic < Systolic
m = df[['BPMin','BPMax']].dropna()
if len(m):
    ok = m['BPMin'] < m['BPMax']
    add_check('BPmin < BPmax', ok, len(m),
              'Diastolic BP must be lower than systolic BP')

# 2) Age in [0,120]
s = df['Age'].dropna()
if len(s):
    add_check('Age ∈ [0,120]', (s>=0)&(s<=120), len(s), 'Age plausibility')

# 3) SpO2 <= 100%
s = df['SPo2'].dropna()
if len(s):
    add_check('SpO2 ≤ 100%', s<=100, len(s), 'Oxygen saturation cannot exceed 100%')

# 4) Temperature plausible
s = df['T'].dropna()
if len(s):
    add_check('T ∈ [32,42]°C', (s>=32)&(s<=42), len(s), 'Body temperature plausibility')

# 5) PR plausible
s = df['PR'].dropna()
if len(s):
    add_check('PR ∈ [30,220] bpm', (s>=30)&(s<=220), len(s), 'Pulse rate plausibility')

# 6) RR plausible
s = df['RR'].dropna()
if len(s):
    add_check('RR ∈ [8,60]', (s>=8)&(s<=60), len(s), 'Respiratory rate plausibility')

# 7) HB / HCT consistency (HCT ≈ 3 × HB, allow ±25%)
m = df[['HB','HCT']].dropna()
if len(m):
    expected = 3 * m['HB']
    ok = (m['HCT'] >= 0.75*expected) & (m['HCT'] <= 1.25*expected)
    add_check('HCT ≈ 3 × HB', ok, len(m),
              'Hematocrit–Hemoglobin internal consistency (±25%)')

# 8) Triage level in {1..5}
if 'Triage_Level' in df.columns:
    s = pd.to_numeric(df['Triage_Level'], errors='coerce').dropna()
    if len(s):
        add_check('Triage ∈ {1..5}', s.isin([1,2,3,4,5]), len(s),
                  'Triage level must be 1–5')

# 9) Gender in {Male, Female}
if 'Gender' in df.columns:
    s = df['Gender'].dropna()
    if len(s):
        add_check('Gender ∈ {Male, Female}',
                  s.isin(['Male','Female']), len(s),
                  'Gender categorical consistency')

# 10) Label ∈ {0,1}
s = df[LABEL_COL].dropna()
if len(s):
    add_check('Label ∈ {0,1}', s.isin([0,1]), len(s),
              'Outcome label binary')

SHEETS['Table7_Consistency'] = pd.DataFrame(consistency_rows)

# =========================================================
# TABLE 8 — Missing-data summary
# =========================================================
miss = df.isna().sum().reset_index()
miss.columns = ['Variable','N_missing']
miss['Missing_%'] = (100*miss['N_missing']/len(df)).round(3)
miss['N_valid']   = len(df) - miss['N_missing']
miss = miss.sort_values('Missing_%', ascending=False).reset_index(drop=True)
SHEETS['Table8_Missing'] = miss

# Per-record missingness
per_rec = df.isna().sum(axis=1)
per_rec_pct = 100 * per_rec / df.shape[1]
missing_summary = pd.DataFrame({
    'Metric': ['Mean per-record missingness (%)',
               'Median per-record missingness (%)',
               'Min per-record missingness (%)',
               'Max per-record missingness (%)',
               'Records > 70% missing'],
    'Value' : [round(per_rec_pct.mean(), 2),
               round(per_rec_pct.median(), 2),
               round(per_rec_pct.min(), 2),
               round(per_rec_pct.max(), 2),
               int((per_rec_pct>70).sum())]
})
SHEETS['Table9_RecordMissing'] = missing_summary

# =========================================================
# TABLE 10 — HAI vs Non-HAI comparison (+ SMD)
# =========================================================
def smd(a, b):
    a = a.dropna(); b = b.dropna()
    if len(a)<2 or len(b)<2: return np.nan
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denom = np.sqrt((va+vb)/2)
    return (ma-mb)/denom if denom > 0 else np.nan

comp_rows = []
for c in NUMERIC_COLS:
    if c not in df.columns: continue
    a = df.loc[df[LABEL_COL]==1, c]
    b = df.loc[df[LABEL_COL]==0, c]
    if len(a.dropna())<2 or len(b.dropna())<2: continue
    # t-test
    try:
        t, p = stats.ttest_ind(a.dropna(), b.dropna(), equal_var=False)
    except Exception:
        t, p = np.nan, np.nan
    comp_rows.append({
        'Variable'    : c,
        'HAI_mean'    : round(a.mean(), 3),
        'HAI_std'     : round(a.std(), 3),
        'nonHAI_mean' : round(b.mean(), 3),
        'nonHAI_std'  : round(b.std(), 3),
        'Mean_diff'   : round(a.mean()-b.mean(), 3),
        'SMD'         : round(smd(a,b), 3),
        't_stat'      : round(t, 3) if not np.isnan(t) else None,
        'p_value'     : f"{p:.3e}" if not np.isnan(p) else None,
        'Significant' : 'Yes' if (not np.isnan(p) and p<0.05) else 'No'
    })
SHEETS['Table10_HAI_vs_nonHAI'] = pd.DataFrame(comp_rows)

# =========================================================
# FIGURE 1 — Missing data bar chart
# =========================================================
fig, ax = plt.subplots(figsize=(10, 8))
mm = miss[miss['N_missing']>0].sort_values('Missing_%')
ax.barh(mm['Variable'], mm['Missing_%'], color='#0A82B8')
ax.set_xlabel('Missing (%)', fontsize=11)
ax.set_title('Missing Data by Variable', fontsize=13, fontweight='bold')
for i, v in enumerate(mm['Missing_%']):
    ax.text(v+0.5, i, f'{v:.1f}%', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR,'fig01_missing_data.png'))
plt.close()

# =========================================================
# FIGURE 2 — Class imbalance
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].pie([N_HAI, N_NON], labels=['HAI','Non-HAI'],
            autopct='%1.2f%%', colors=['#E74C3C','#3498DB'],
            startangle=90, explode=(0.05,0))
axes[0].set_title('Class Distribution', fontweight='bold')

axes[1].bar(['HAI','Non-HAI'], [N_HAI, N_NON],
            color=['#E74C3C','#3498DB'])
axes[1].set_yscale('log')
axes[1].set_ylabel('Count (log scale)')
axes[1].set_title('Class Imbalance (log scale)', fontweight='bold')
for i, v in enumerate([N_HAI, N_NON]):
    axes[1].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR,'fig02_class_imbalance.png'))
plt.close()

# =========================================================
# FIGURE 3 — Age distribution by HAI
# =========================================================
fig, ax = plt.subplots(figsize=(9,5))
age_h = df.loc[df[LABEL_COL]==1,'Age'].dropna()
age_n = df.loc[df[LABEL_COL]==0,'Age'].dropna()
bins = np.arange(0, 110, 5)
ax.hist([age_n, age_h], bins=bins, label=['Non-HAI','HAI'],
        color=['#3498DB','#E74C3C'], stacked=False, alpha=0.8)
ax.set_xlabel('Age (years)')
ax.set_ylabel('Count')
ax.set_title('Age Distribution by HAI Status', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR,'fig03_age_distribution.png'))
plt.close()

# =========================================================
# FIGURE 4 — Triage level distribution
# =========================================================
if 'Triage_Level' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    tri = pd.to_numeric(df['Triage_Level'], errors='coerce').dropna()
    tri_counts = tri.value_counts().sort_index()
    axes[0].bar(tri_counts.index.astype(int).astype(str),
                tri_counts.values, color='#0A82B8')
    axes[0].set_xlabel('Triage Level')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Overall Triage Distribution', fontweight='bold')
    for i, v in enumerate(tri_counts.values):
        axes[0].text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=8)

    ct = pd.crosstab(df['Triage_Level'], df[LABEL_COL], normalize='columns')*100
    if 1 in ct.columns and 0 in ct.columns:
        x = np.arange(len(ct))
        w = 0.4
        axes[1].bar(x-w/2, ct[0], w, label='Non-HAI', color='#3498DB')
        axes[1].bar(x+w/2, ct[1], w, label='HAI',     color='#E74C3C')
        axes[1].set_xticks(x); axes[1].set_xticklabels(ct.index.astype(int))
        axes[1].set_xlabel('Triage Level'); axes[1].set_ylabel('Percent (%)')
        axes[1].set_title('Triage Level by HAI Status', fontweight='bold')
        axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR,'fig04_triage_distribution.png'))
    plt.close()

# =========================================================
# FIGURE 5 — Vital signs boxplots by HAI
# =========================================================
vitals = [c for c in ['SPo2','T','PR','RR','BPMin','BPMax'] if c in df.columns]
if vitals:
    n = len(vitals)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, c in enumerate(vitals):
        d0 = df.loc[df[LABEL_COL]==0, c].dropna()
        d1 = df.loc[df[LABEL_COL]==1, c].dropna()
        axes[i].boxplot([d0, d1], labels=['Non-HAI','HAI'],
                        patch_artist=True,
                        boxprops=dict(facecolor='#AED6F1'),
                        medianprops=dict(color='red'))
        axes[i].set_title(c, fontweight='bold')
    for j in range(len(vitals), len(axes)):
        axes[j].axis('off')
    plt.suptitle('Vital Signs by HAI Status', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR,'fig05_vital_signs.png'))
    plt.close()

# =========================================================
# FIGURE 6 — Lab values boxplots by HAI
# =========================================================
labs = [c for c in ['WBC','HB','HCT','PLT','ESR','UREA','CR','NA','K','BS_Lab']
        if c in df.columns]
if labs:
    n = len(labs)
    ncols = 3
    nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    axes = axes.flatten()
    for i, c in enumerate(labs):
        d0 = df.loc[df[LABEL_COL]==0, c].dropna()
        d1 = df.loc[df[LABEL_COL]==1, c].dropna()
        axes[i].boxplot([d0, d1], labels=['Non-HAI','HAI'],
                        patch_artist=True,
                        boxprops=dict(facecolor='#A9DFBF'),
                        medianprops=dict(color='red'))
        axes[i].set_title(c, fontweight='bold')
    for j in range(len(labs), len(axes)):
        axes[j].axis('off')
    plt.suptitle('First-24h Laboratory Results by HAI Status',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR,'fig06_lab_values.png'))
    plt.close()

# =========================================================
# FIGURE 7 — Correlation heatmap
# =========================================================
corr_cols = [c for c in NUMERIC_COLS if c in df.columns]
if len(corr_cols) > 2:
    corr = df[corr_cols].corr(method='spearman')
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, cbar_kws={'shrink':0.7},
                annot_kws={'size':7}, ax=ax)
    ax.set_title('Spearman Correlation Matrix (Numeric Variables)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR,'fig07_correlation.png'))
    plt.close()

# =========================================================
# FIGURE 8 — Missing pattern heatmap (first 5000 rows for readability)
# =========================================================
sample = df.sample(min(5000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(sample.isna(), cbar=False, cmap='Greys', ax=ax, yticklabels=False)
ax.set_title('Missing-Data Pattern (random sample of 5,000 records)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR,'fig08_missing_pattern.png'))
plt.close()

# =========================================================
# FIGURE 9 — Ward distribution
# =========================================================
if 'Ward' in df.columns:
    fig, ax = plt.subplots(figsize=(12, 6))
    wc = df['Ward'].value_counts().head(15)
    ax.barh(wc.index[::-1], wc.values[::-1], color='#0A82B8')
    ax.set_xlabel('Count')
    ax.set_title('Top-15 Wards of Admission', fontweight='bold')
    for i, v in enumerate(wc.values[::-1]):
        ax.text(v, i, f' {v:,}', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR,'fig09_ward_distribution.png'))
    plt.close()

# =========================================================
# FIGURE 10 — HAI prevalence in top wards
# =========================================================
if 'Ward' in df.columns:
    top = df['Ward'].value_counts().head(10).index
    sub = df[df['Ward'].isin(top)]
    prev = sub.groupby('Ward')[LABEL_COL].agg(['count','sum'])
    prev['prev_%'] = 100*prev['sum']/prev['count']
    prev = prev.sort_values('prev_%', ascending=True)

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(prev.index, prev['prev_%'], color='#E74C3C')
    ax.set_xlabel('HAI prevalence (%)')
    ax.set_title('HAI Prevalence in Top-10 Wards', fontweight='bold')
    for i, v in enumerate(prev['prev_%']):
        ax.text(v, i, f' {v:.2f}%', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR,'fig10_ward_HAI_prevalence.png'))
    plt.close()

# =========================================================
# WRITE EXCEL
# =========================================================
with pd.ExcelWriter(EXCEL_FILE, engine='openpyxl') as xw:
    for name, sheet in SHEETS.items():
        sheet.to_excel(xw, sheet_name=name[:31], index=False)

# =========================================================
# FINAL PRINT
# =========================================================
print("\n" + "="*70)
print("✅ DONE!")
print("="*70)
print(f"Excel file : {EXCEL_FILE}")
print(f"Figures    : {FIGS_DIR}/")
print(f"Sheets     : {list(SHEETS.keys())}")
print("="*70)

# Quick print of key tables
for name in ['Table1_Summary','Table3_Cascade','Table9_RecordMissing']:
    print(f"\n--- {name} ---")
    print(SHEETS[name].to_string(index=False))