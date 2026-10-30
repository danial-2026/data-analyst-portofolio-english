# -*- coding: utf-8 -*-
"""Revised analysis pipeline for the Asia-Africa green industrial leapfrogging panel.

Modules: descriptives/correlations; policy event-study and optional R gsynth; Global Moran/LISA
and spatial-model selection diagnostics; FE mediation/path analysis; RF/XGBoost/LightGBM + SHAP
using only AIC, DT, SI and Institution as features; controls remain in econometric models only. This script does NOT infer policy treatment dates from outcomes.

Install: pip install pandas numpy scipy openpyxl matplotlib seaborn statsmodels scikit-learn
         linearmodels libpysal esda spreg xgboost lightgbm shap
Optional generalized synthetic control: R + R package 'gsynth', and a complete policy registry.
"""
from __future__ import annotations
import os, json, warnings, subprocess, textwrap, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.environ.get('RESEARCH_READY_FILE', str(BASE_DIR / 'Analysis_Ready_Dataset.xlsx')))
DATA_SHEET = 'AnalysisData'
OUTPUT_DIR = Path(os.environ.get('RESEARCH_RESULTS_DIR', str(BASE_DIR / 'Results_FINAL')))
POLICY_FILE = Path(os.environ.get('RESEARCH_POLICY_REGISTRY_FILE', str(BASE_DIR / 'Policy_Registry_AI_Strategy_Audit_88.xlsx')))
SPATIAL_DIR = Path(os.environ.get('RESEARCH_SPATIAL_DIR', str(BASE_DIR / 'Spatial_Preparation')))
# Prefer intended binary matrices; fall back to row-standardized matrices shipped beside the script.
QUEEN_W_CANDIDATES = [SPATIAL_DIR / 'spatial_W_queen_binary.csv', BASE_DIR / 'spatial_W_queen_binary.csv', BASE_DIR / 'spatial_W_queen_row_standardized.csv']
KNN_W_CANDIDATES = [SPATIAL_DIR / 'spatial_W_knn5_binary_symmetrized.csv', BASE_DIR / 'spatial_W_knn5_binary_symmetrized.csv', BASE_DIR / 'spatial_W_knn5_row_standardized.csv']
def first_existing(paths):
    return next((p for p in paths if p.exists()), paths[0])
QUEEN_W = first_existing(QUEEN_W_CANDIDATES)
KNN_W = first_existing(KNN_W_CANDIDATES)
Y, X_MAIN = 'Y', ['AIC','DT','SI']
CONTROLS_CANDIDATES = ['ED_log','HC','FDI_ihs','GI','FD']  # ASI excluded: not available in this dataset
ID, YEAR = 'ID','Year'
SEED=42; PERMUTATIONS=999; N_ESTIMATORS=400

for d in ['Tables','Figures','Logs','Models','Data']:
    (OUTPUT_DIR/d).mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR/'Logs'/'RUN_LOCATION.txt').write_text(
    f'Script: {Path(__file__).resolve()}\nData: {DATA_FILE.resolve()}\nPolicy registry: {POLICY_FILE.resolve()}\n'
    f'Output directory: {OUTPUT_DIR.resolve()}\nSpatial directory: {SPATIAL_DIR.resolve()}\n', encoding='utf-8')
def log(s): print(s, flush=True)
def save_status(name, s): (OUTPUT_DIR/'Logs'/name).write_text(str(s), encoding='utf-8')
def formula(y, xs): return f'{y} ~ ' + ' + '.join(xs) + ' + C(ID) + C(Year)'
def clustered_ols(data, y, xs):
    d=data[[y,ID,YEAR]+xs].dropna().copy()
    if d[ID].nunique()<10: raise ValueError('Insufficient country clusters after listwise deletion.')
    return smf.ols(formula(y,xs),data=d).fit(cov_type='cluster',cov_kwds={'groups':d[ID], 'use_correction':True})
def coef_table(res, label):
    return pd.DataFrame({'Model':label,'Term':res.params.index,'Coef':res.params.values,'SE':res.bse.values,'P_value':res.pvalues.values,
                         'CI_low':res.conf_int()[0].values,'CI_high':res.conf_int()[1].values,'N':int(res.nobs)})
def read_W(path, ids):
    if not path.exists(): return None
    w=pd.read_csv(path,index_col=0); w.index=w.index.astype(str).str.upper(); w.columns=w.columns.astype(str).str.upper()
    ids=list(ids)
    if not set(ids).issubset(w.index) or not set(ids).issubset(w.columns):
        raise ValueError(f'{path.name} does not contain all analysis IDs. Missing={sorted(set(ids)-set(w.index))[:20]}')
    w=w.loc[ids,ids].astype(float).to_numpy()
    np.fill_diagonal(w,0)
    return w

def descriptive(df, controls):
    vars_=[Y,*X_MAIN,'Institution',*controls]
    desc=df[vars_].describe(percentiles=[.25,.5,.75]).T
    desc['Missing']=df[vars_].isna().sum(); desc['Missing_pct']=df[vars_].isna().mean()*100
    desc.to_excel(OUTPUT_DIR/'Tables'/'Table_1_Descriptive_Statistics.xlsx')
    df.groupby('Year')[Y].agg(['count','mean','std','median']).to_excel(OUTPUT_DIR/'Tables'/'Y_by_Year.xlsx')
    for method in ['pearson','spearman']:
        df[vars_].corr(method=method,min_periods=20).to_excel(OUTPUT_DIR/'Tables'/f'Correlation_{method.title()}.xlsx')
    corr=df[vars_].corr(min_periods=20)
    fig,ax=plt.subplots(figsize=(max(8,len(vars_)*.8),max(6,len(vars_)*.7)))
    sns.heatmap(corr,annot=True,fmt='.2f',center=0,cmap='RdBu_r',ax=ax); fig.tight_layout()
    fig.savefig(OUTPUT_DIR/'Figures'/'Figure_1_Correlation_Heatmap.png',dpi=250); plt.close(fig)

def policy_event_study(df):
    """Read the existing CountryRegistry. Candidate dates are exploratory unless verified."""
    if not POLICY_FILE.exists():
        msg=f'NOT ESTIMATED: policy registry not found at {POLICY_FILE}.'
        save_status('Policy_Event_Study_Status.txt',msg); save_status('GSynth_Status.txt',msg); return
    try:
        xl=pd.ExcelFile(POLICY_FILE)
        if 'CountryRegistry' not in xl.sheet_names: raise ValueError(f'CountryRegistry sheet missing: {xl.sheet_names}')
        reg=pd.read_excel(POLICY_FILE,sheet_name='CountryRegistry'); reg.columns=[str(c).strip() for c in reg.columns]
        if ID not in reg or 'AI_Treatment_Year' not in reg: raise ValueError(f'Expected ID and AI_Treatment_Year, found {reg.columns.tolist()}')
        reg[ID]=reg[ID].astype(str).str.strip().str.upper(); reg['treatment_year']=pd.to_numeric(reg['AI_Treatment_Year'],errors='coerce')
        if 'AI_Treatment_Status' not in reg: reg['AI_Treatment_Status']='unknown_not_researched'
        reg['AI_Treatment_Status']=reg['AI_Treatment_Status'].astype(str).str.strip()
        reg=reg[[ID,'treatment_year','AI_Treatment_Status']].drop_duplicates(ID)
        reg=reg[reg[ID].isin(df[ID].unique())].copy()
        # Preserve verified never-treated rows with missing treatment year; filter dated events separately.
        verified=reg[reg.AI_Treatment_Status.eq('treated_verified') & reg.treatment_year.between(2017,2024, inclusive='both')].copy()
        candidate=verified.copy()
        never=reg[reg.AI_Treatment_Status.eq('never_treated_verified')]
        counts=reg.AI_Treatment_Status.value_counts(dropna=False).to_dict()
        if len(candidate)<3:
            msg=f'NOT ESTIMATED: fewer than 3 dated candidate treated countries. Registry status counts={counts}'
            save_status('Policy_Event_Study_Status.txt',msg); save_status('GSynth_Status.txt',msg); return
        # Exploratory event study is still run, but clearly labelled non-confirmatory because most dates/statuses are unverified.
        d=df.merge(candidate[[ID,'treatment_year']],on=ID,how='left'); d['ever_treated']=d.treatment_year.notna().astype(int)
        d['event_time']=d[YEAR]-d.treatment_year; d['event_bin']=d.event_time.clip(-5,5); terms=[]
        for k in range(-5,6):
            if k==-1: continue
            nm=f'event_m{abs(k)}' if k<0 else f'event_p{k}'
            d[nm]=((d.ever_treated==1)&(d.event_bin==k)).astype(int); terms.append(nm)
        controls=[c for c in CONTROLS_CANDIDATES if c in d.columns]
        try:
            res=clustered_ols(d,Y,terms+controls)
            event_tbl=coef_table(res,'Exploratory_TWFE_event_study')
            event_tbl.to_excel(OUTPUT_DIR/'Tables'/'Event_Study_Coefficients.xlsx',index=False)
            # Figure 2: event-time coefficients and 95% confidence intervals; omitted t=-1 is the reference.
            event_rows=[]
            for k in range(-5,6):
                if k == -1: continue
                nm=f'event_m{abs(k)}' if k<0 else f'event_p{k}'
                if nm in res.params.index:
                    event_rows.append({'event_time':k,'coef':float(res.params[nm]),'lo':float(res.conf_int().loc[nm,0]),'hi':float(res.conf_int().loc[nm,1])})
            if event_rows:
                et=pd.DataFrame(event_rows).sort_values('event_time')
                fig,ax=plt.subplots(figsize=(8,5)); ax.errorbar(et.event_time,et.coef,yerr=[et.coef-et.lo,et.hi-et.coef],fmt='o',capsize=3)
                ax.axhline(0,color='black',linewidth=1); ax.axvline(-0.5,color='grey',linestyle='--',linewidth=1)
                ax.set(xlabel='Years relative to AI strategy adoption',ylabel='Coefficient on Y',title='Event study (exploratory; reference period t = -1)')
                fig.tight_layout(); fig.savefig(OUTPUT_DIR/'Figures'/'Figure_2_Event_Study.png',dpi=250); plt.close(fig)
            save_status('Policy_Event_Study_Status.txt',f'ESTIMATED EXPLORATORILY: dated candidates={len(candidate)}, treated_verified={len(verified)}, verified never-treated={len(never)}; status counts={counts}. NOT causal evidence until dates and control status are verified. TWFE may be biased under staggered heterogeneous effects.\n{res.summary().as_text()}')
        except Exception as e: save_status('Policy_Event_Study_Status.txt',f'EXPLORATORY EVENT STUDY FAILED: {e}\n{traceback.format_exc()}')
        # Cohort-specific GSynth: exact adoption years remain intact within each cohort. AIC, DT, SI enter as covariates (X1, X2, X3).
        # Unknown/candidate countries are never silently used as controls.
        cohort_defs = {
            'Cohort_1_2017_2019': (2017, 2019),
            'Cohort_2_2020_2021': (2020, 2021),
            'Cohort_3_2022_2024': (2022, 2024),
        }
        if len(never) < 3:
            save_status('GSynth_Status.txt',f'NOT ESTIMATED: fewer than 3 verified never-treated controls (n={len(never)}). Cohort outputs cannot be represented as valid GSynth estimates. Status counts={counts}.')
            return
        try:
            import shutil
            # Resolve Rscript from explicit env var, PATH, and common Windows install locations.
            candidates = [os.environ.get('RSCRIPT_EXE', '').strip(), shutil.which('Rscript') or '']
            candidates += [
                r'C:\Program Files\R\R-4.4.1\bin\x64\Rscript.exe',
                r'C:\Program Files\R\R-4.4.1\bin\Rscript.exe',
            ]
            rscript = next((c for c in candidates if c and (shutil.which(c) or Path(c).is_file())), 'Rscript')
            if shutil.which(rscript) is None and not Path(rscript).is_file():
                cohort_status=[]
                for cohort, (lo, hi) in cohort_defs.items():
                    tr=verified[verified.treatment_year.between(lo,hi,inclusive='both')]
                    cohort_status.append({'Cohort':cohort,'Years':f'{lo}-{hi}','N_treated':len(tr),'N_control':len(never),
                        'Status':f'NOT RUN: Rscript executable not found ({rscript}); install R and package gsynth'})
                pd.DataFrame(cohort_status).to_excel(OUTPUT_DIR/'Tables'/'GSynth_Cohort_Status.xlsx',index=False)
                save_status('GSynth_Status.txt',f'NOT RUN: Rscript executable not found ({rscript}). No cohort can be estimated until R and package gsynth are installed. Verified treated={len(verified)}; verified controls={len(never)}; status counts={counts}.\n'+pd.DataFrame(cohort_status).to_string(index=False))
                return
            cohort_status=[]
            for cohort, (lo, hi) in cohort_defs.items():
                tr=verified[verified.treatment_year.between(lo,hi,inclusive='both')].copy()
                if tr.empty:
                    cohort_status.append({'Cohort':cohort,'Years':f'{lo}-{hi}','N_treated':0,'N_control':len(never),'Status':'NOT ESTIMATED: no treated_verified countries in this adoption-year cohort'})
                    continue
                cohort_units=pd.concat([tr,never]).drop_duplicates(ID)
                gd=df[df[ID].isin(cohort_units[ID])].merge(cohort_units[[ID,'treatment_year','AI_Treatment_Status']],on=ID,how='left')
                gd['D']=((gd['AI_Treatment_Status'].eq('treated_verified'))&(gd[YEAR]>=gd['treatment_year'])).astype(int)
                gd=gd[[ID,YEAR,Y,'D','AIC','DT','SI']].dropna()
                # Require at least 3 controls and >=1 treated unit; sparse cohorts are flagged in outputs.
                if gd[ID].nunique() < 4 or gd.loc[gd.D.eq(1),ID].nunique() < 1:
                    cohort_status.append({'Cohort':cohort,'Years':f'{lo}-{hi}','N_treated':len(tr),'N_control':len(never),'Status':'NOT ESTIMATED: insufficient usable panel units'})
                    continue
                safe=cohort.replace('-','_')
                panel_csv=OUTPUT_DIR/'Data'/f'gsynth_panel_{safe}.csv'; gd.to_csv(panel_csv,index=False)
                rfile=OUTPUT_DIR/'Data'/f'run_gsynth_{safe}.R'
                attfile=OUTPUT_DIR/'Tables'/f'GSynth_ATT_{safe}.csv'
                figfile=OUTPUT_DIR/'Figures'/f'Figure_3_GSynth_{safe}.png'
                sumfile=OUTPUT_DIR/'Logs'/f'GSynth_summary_{safe}.txt'
                modelrds=OUTPUT_DIR/'Models'/f'gsynth_fit_{safe}.rds'
                lines=[
                    'suppressPackageStartupMessages(library(gsynth))',
                    f'd <- read.csv({json.dumps(str(panel_csv))}, stringsAsFactors=FALSE)',
                    'd$ID <- as.character(d$ID); d$Year <- as.integer(d$Year)',
                    'fit <- gsynth(Y ~ D + AIC + DT + SI, data=d, index=c("ID","Year"), force="two-way", CV=TRUE, r=c(0,3), se=FALSE, min.T0=3)',
                    f'sink({json.dumps(str(sumfile))}); print(summary(fit)); sink()',
                    f'saveRDS(fit, {json.dumps(str(modelrds))})',
                    f'if (!is.null(fit$est.att)) write.csv(data.frame(ATT=as.numeric(fit$est.att)), {json.dumps(str(attfile))}, row.names=FALSE)',
                    f'png({json.dumps(str(figfile))}, width=1800, height=1200, res=200); plot(fit, type="gap", main={json.dumps(cohort)}); dev.off()'
                ]
                rfile.write_text('\n'.join(lines),encoding='utf-8')
                proc=subprocess.run([rscript,str(rfile)],capture_output=True,text=True,timeout=3600)
                (OUTPUT_DIR/'Logs'/f'GSynth_console_{safe}.stdout.txt').write_text(proc.stdout or '',encoding='utf-8')
                (OUTPUT_DIR/'Logs'/f'GSynth_console_{safe}.stderr.txt').write_text(proc.stderr or '',encoding='utf-8')
                # A zero return code alone is not enough: require summary and ATT output.
                ok=proc.returncode==0 and sumfile.exists() and sumfile.stat().st_size>0 and attfile.exists() and attfile.stat().st_size>0
                cohort_status.append({'Cohort':cohort,'Years':f'{lo}-{hi}','N_treated':len(tr),'N_control':len(never),'Rscript':str(rscript),'Specification':'Y ~ D + AIC(X1) + DT(X2) + SI(X3); se=FALSE','Inference':'Point estimate only; no bootstrap CI','Return_Code':proc.returncode,'ATT_File_Exists':attfile.exists(),'Summary_File_Exists':sumfile.exists(),'Status':'POINT ESTIMATE ONLY (se=FALSE); inspect diagnostics' if ok else f'FAILED / NO VALID ATT (return code {proc.returncode})','R_stdout':proc.stdout[-1500:],'R_stderr':proc.stderr[-1500:]})
            pd.DataFrame(cohort_status).to_excel(OUTPUT_DIR/'Tables'/'GSynth_Cohort_Status.xlsx',index=False)
            save_status('GSynth_Status.txt','Cohort-specific GSynth attempts completed. Each cohort uses exact verified adoption years and verified never-treated controls only. Formula: Y ~ D + AIC(X1) + DT(X2) + SI(X3). se=FALSE avoids the installed gsynth 1.4.0/f ect wrapper bootstrap error; therefore ATT is a point estimate without bootstrap confidence intervals. Check GSynth_Cohort_Status.xlsx and per-cohort logs/ATT files. Missing cohort estimates are not imputed or fabricated.\n'+pd.DataFrame(cohort_status).to_string(index=False))
        except Exception as e:
            save_status('GSynth_Status.txt',f'GSYNTH COHORT RUN FAILED: {e}\n{traceback.format_exc()}')
    except Exception as e:
        save_status('Policy_Event_Study_Status.txt',f'FAILED TO READ POLICY REGISTRY: {e}\n{traceback.format_exc()}'); save_status('GSynth_Status.txt',f'FAILED TO READ POLICY REGISTRY: {e}')

def spatial_analysis(df, controls):
    """Moran/LISA plus spatial panel FE (SAR/SEM) and latest-year diagnostics."""
    try:
        from libpysal.weights import W
        from esda import Moran, Moran_Local
        from spreg import OLS, ML_Lag, ML_Error
    except Exception as e:
        msg = (f'NOT ESTIMATED: spatial Python dependencies could not be imported. Install in the SAME Python environment running this script: python -m pip install libpysal esda spreg.\n'
               f'Import error: {type(e).__name__}: {e}\nScript: {Path(__file__).resolve()}\nOutput directory: {OUTPUT_DIR.resolve()}')
        save_status('Spatial_Status.txt', msg)
        return
    needed=[Y,*X_MAIN,'Institution',*controls]; d=df.dropna(subset=needed).copy()
    complete=d.groupby(ID)[YEAR].nunique(); ids=sorted(complete[complete==df[YEAR].nunique()].index.astype(str)); d=d[d[ID].isin(ids)].copy()
    if len(ids)<30: save_status('Spatial_Status.txt',f'NOT ESTIMATED: only {len(ids)} countries in balanced complete-case sample.'); return
    moran_rows=[]; model_rows=[]; notes=[]
    for wname,path in [('Queen_binary',QUEEN_W),('KNN5_binary',KNN_W)]:
        try:
            arr=read_W(path,ids)
            if arr is None: notes.append(f'{wname}: missing matrix {path}'); continue
            arr=(arr>0).astype(float); np.fill_diagonal(arr,0)
            nb={ids[i]:[ids[j] for j in np.where(arr[i]>0)[0]] for i in range(len(ids))}
            w=W(nb,id_order=ids,silence_warnings=True); w.transform='r'
            for year,g0 in d.groupby(YEAR):
                g=g0.set_index(ID).reindex(ids)
                if g[Y].isna().any(): continue
                mi=Moran(g[Y].to_numpy(float),w,permutations=PERMUTATIONS)
                moran_rows.append({'W':wname,'Year':int(year),'N':len(ids),'Moran_I':mi.I,'Expected_I':mi.EI,'p_sim':mi.p_sim,'z_sim':mi.z_sim})
                if int(year)==int(d[YEAR].max()):
                    li=Moran_Local(g[Y].to_numpy(float),w,permutations=PERMUTATIONS,seed=SEED)
                    pd.DataFrame({'W':wname,'Year':int(year),'ID':ids,'Y':g[Y].to_numpy(float),'Local_I':li.Is,'p_sim':li.p_sim,'Quadrant':li.q,'Significant_5pct':li.p_sim<.05}).to_excel(OUTPUT_DIR/'Tables'/f'LISA_{wname}_latest_year.xlsx',index=False)
            # latest-year complete case; subset W to precisely the same IDs before constructing PySAL W.
            g=d[d[YEAR]==d[YEAR].max()].set_index(ID).reindex(ids); xvars=['AIC','DT','SI','Institution',*controls]
            good=g[[Y,*xvars]].notna().all(axis=1); model_ids=list(g.index[good]); gg=g.loc[model_ids]
            pos=[ids.index(i) for i in model_ids]; sub=arr[np.ix_(pos,pos)]
            # Exclude isolates from cross-sectional likelihood estimation; keep them in Moran/LISA outputs.
            keep=sub.sum(axis=1)>0; model_ids=list(np.asarray(model_ids)[keep]); gg=gg.loc[model_ids]; sub=sub[np.ix_(keep,keep)]
            if len(model_ids)<=len(xvars)+5: notes.append(f'{wname}: too few aligned non-isolates ({len(model_ids)})'); continue
            nb2={model_ids[i]:[model_ids[j] for j in np.where(sub[i]>0)[0]] for i in range(len(model_ids))}
            wm=W(nb2,id_order=model_ids,silence_warnings=True); wm.transform='r'
            X=gg[xvars].to_numpy(float); y=gg[Y].to_numpy(float).reshape(-1,1)
            ols=OLS(y,X,w=wm,name_y=Y,name_x=xvars,spat_diag=True,moran=True)
            def attr(obj,*names):
                for n in names:
                    v=getattr(obj,n,None)
                    if v is not None:
                        try:return float(np.asarray(v).reshape(-1)[0])
                        except Exception:pass
                return np.nan
            model_rows.append({'W':wname,'Year':int(d[YEAR].max()),'Model':'OLS_LM_diagnostics','N':len(model_ids),'R2':attr(ols,'r2'),'LogLik':attr(ols,'logll'),'AIC':attr(ols,'aic'),'LM_lag':attr(ols,'lm_lag'),'LM_lag_p':attr(ols,'lm_lag_pvalue'),'Robust_LM_lag':attr(ols,'rlm_lag'),'Robust_LM_lag_p':attr(ols,'rlm_lag_pvalue'),'LM_error':attr(ols,'lm_error'),'LM_error_p':attr(ols,'lm_error_pvalue'),'Robust_LM_error':attr(ols,'rlm_error'),'Robust_LM_error_p':attr(ols,'rlm_error_pvalue')})
            for mn,fn,kw in [('SAR',ML_Lag,{}),('SDM',ML_Lag,{'slx_lags':1}),('SEM',ML_Error,{})]:
                try:
                    fit=fn(y,X,w=wm,name_y=Y,name_x=xvars,**kw)
                    model_rows.append({'W':wname,'Year':int(d[YEAR].max()),'Model':mn,'N':len(model_ids),'R2':attr(fit,'pr2','r2'),'LogLik':attr(fit,'logll'),'AIC':attr(fit,'aic'),'Sigma2':attr(fit,'sig2')})
                except Exception as e: notes.append(f'{wname} {mn} failed: {e}')
        except Exception as e: notes.append(f'{wname} failed: {e}\n{traceback.format_exc()}')
    if moran_rows: pd.DataFrame(moran_rows).to_excel(OUTPUT_DIR/'Tables'/'Table_2_Global_Morans_I.xlsx',index=False)
    if model_rows: pd.DataFrame(model_rows).to_excel(OUTPUT_DIR/'Tables'/'Table_3_Spatial_Model_Comparison.xlsx',index=False)
    # Spatial panel fixed-effects estimators across all years, using balanced country-major stack.
    panel_rows, panel_coefs = [], []
    try:
        import spreg
        from libpysal.weights import W as PySALW
        panel_need=[Y,ID,YEAR,'AIC','DT','SI','Institution',*controls]
        panel=d.dropna(subset=panel_need).copy()
        panel_ids=sorted(panel[ID].astype(str).unique())
        panel_years=sorted(panel[YEAR].unique())
        n_by_country=panel.groupby(ID)[YEAR].nunique()
        if len(panel_ids)>=30 and n_by_country.nunique()==1 and n_by_country.iloc[0]==len(panel_years):
            xnames=['AIC','DT','SI','Institution',*controls]
            for wname,path in [('Queen_binary',QUEEN_W),('KNN5_binary',KNN_W)]:
                try:
                    raw=read_W(path,panel_ids)
                    if raw is None:
                        notes.append(f'{wname} panel FE: W matrix missing: {path}')
                        continue
                    raw=(raw>0).astype(float); np.fill_diagonal(raw,0)
                    neigh={i:np.flatnonzero(raw[i]).tolist() for i in range(len(panel_ids))}
                    weights={i:raw[i,neigh[i]].tolist() for i in range(len(panel_ids))}
                    wp=PySALW(neigh,weights,id_order=list(range(len(panel_ids))),silence_warnings=True)
                    wp.transform='r'
                    stacked=panel.assign(**{ID:panel[ID].astype(str)}).set_index([ID,YEAR]).sort_index()
                    stacked=stacked.reindex(pd.MultiIndex.from_product([panel_ids,panel_years],names=[ID,YEAR]))
                    if stacked[[Y, *xnames]].isna().any().any():
                        notes.append(f'{wname} panel FE: missing values after alignment; skipped.')
                        continue
                    yy=stacked[Y].to_numpy(float).reshape(-1,1)
                    xx=stacked[xnames].to_numpy(float)
                    for model_name, cls_name in [('SAR_Panel_FE','Panel_FE_Lag'),('SEM_Panel_FE','Panel_FE_Error')]:
                        cls=getattr(spreg,cls_name,None)
                        if cls is None:
                            notes.append(f'{wname} {model_name}: {cls_name} unavailable in installed spreg.')
                            continue
                        try:
                            fit=cls(yy,xx,wp,name_y=Y,name_x=xnames,name_w=wname)
                            betas=np.asarray(fit.betas).reshape(-1)
                            names=list(getattr(fit,'name_x',xnames))
                            if len(names)!=len(betas): names=[f'parameter_{j+1}' for j in range(len(betas))]
                            se=np.asarray(getattr(fit,'std_err',[])).reshape(-1)
                            zstats=getattr(fit,'z_stat',[])
                            for j,nm in enumerate(names):
                                panel_coefs.append({'W':wname,'Model':model_name,'Term':nm,'Coefficient':betas[j],
                                    'Std_Error':se[j] if j<len(se) else np.nan,
                                    'z_stat':zstats[j][0] if j<len(zstats) else np.nan,
                                    'p_value':zstats[j][1] if j<len(zstats) else np.nan})
                            panel_rows.append({'W':wname,'Model':model_name,'N_countries':len(panel_ids),'T_years':len(panel_years),
                                'N_country_year':len(panel_ids)*len(panel_years),'LogLik':getattr(fit,'logll',np.nan),
                                'AIC':getattr(fit,'aic',np.nan),'Pseudo_R2':getattr(fit,'pr2',np.nan),'Status':'ESTIMATED'})
                            notes.append(f'{wname} {model_name}: estimated on {len(panel_ids)} countries x {len(panel_years)} years.')
                        except Exception as e:
                            notes.append(f'{wname} {model_name} FAILED: {type(e).__name__}: {e}')
                except Exception as e:
                    notes.append(f'{wname} spatial panel setup FAILED: {type(e).__name__}: {e}')
        else:
            notes.append(f'Spatial panel FE not estimated: complete-case panel unbalanced or fewer than 30 countries; countries={len(panel_ids)}, years={len(panel_years)}.')
    except Exception as e:
        notes.append(f'Spatial panel dependencies/setup failed: {type(e).__name__}: {e}')
    if panel_rows:
        pd.DataFrame(panel_rows).to_excel(OUTPUT_DIR/'Tables'/'Spatial_Panel_FE_Model_Comparison.xlsx',index=False)
    if panel_coefs:
        pd.DataFrame(panel_coefs).to_excel(OUTPUT_DIR/'Tables'/'Spatial_Panel_FE_Coefficients.xlsx',index=False)
    status_text=('Global Moran/LISA by year plus SAR/SEM spatial panel FE where supported. Latest-year SAR/SDM/SEM results remain cross-sectional diagnostics, not panel estimates. '
        'Panel estimators require balanced complete-case data and aligned W. Queen isolates are retained as zero rows; KNN5 is a robustness matrix. '
        'SDM panel is NOT claimed: installed spreg must provide a true spatial lag-of-X panel estimator before SDM can be reported.\\n'
        +'\\n'.join(notes)+f'\\nMoran rows={len(moran_rows)}; latest-year model rows={len(model_rows)}; spatial panel model rows={len(panel_rows)}; panel coefficient rows={len(panel_coefs)}.')
    save_status('Spatial_Status.txt',status_text)

def mediation(df, controls):
    """Sequential FE path models. Indirect AIC->DT->SI->Y is a product-of-coefficients, not causal absent assumptions."""
    rows=[]; models={}
    specs=[('DT',['AIC','Institution',*controls],'AIC_to_DT'),('SI',['AIC','DT','Institution',*controls],'DT_to_SI'),(Y,['AIC','DT','SI','Institution',*controls],'Y_direct_paths')]
    for outcome,xs,label in specs:
        xs=[x for x in xs if x in df.columns]
        try:
            res=clustered_ols(df,outcome,xs); models[label]=res; rows.append(coef_table(res,label))
        except Exception as e: save_status(f'Mediation_{label}_ERROR.txt',str(e))
    if rows: pd.concat(rows,ignore_index=True).to_excel(OUTPUT_DIR/'Tables'/'Table_4_Structural_Regression.xlsx',index=False)
    # VIF screen for main predictors + controls (cross-sectional pooled diagnostic; not a formal FE VIF)
    vars_=[x for x in ['AIC','DT','SI','Institution',*controls] if x in df.columns]
    z=df[vars_].dropna().astype(float)
    vif=[]
    if len(z)>len(vars_)+2:
        from statsmodels.tools.tools import add_constant
        zz=add_constant(z,has_constant='add')
        for i,v in enumerate(zz.columns):
            if v!='const':
                try: vif.append({'Variable':v,'VIF':float(variance_inflation_factor(zz.to_numpy(),i))})
                except Exception: vif.append({'Variable':v,'VIF':np.nan})
    pd.DataFrame(vif).to_excel(OUTPUT_DIR/'Tables'/'Multicollinearity_VIF.xlsx',index=False)
    # Report indirect products for main predictors only; controls never receive mediated-effect labels.
    try:
        r_a=models['AIC_to_DT']; r_b=models['DT_to_SI']; r_y=models['Y_direct_paths']
        a=float(r_a.params['AIC']); b=float(r_b.params['DT']); c=float(r_y.params['SI'])
        b_y=float(r_y.params['DT'])
        c_x1=float(r_y.params['AIC'])
        summary=pd.DataFrame([
          {'Path':'AIC -> DT -> Y','Product_of_coefficients':a*b_y,'Components':f'{a:.8g} × {b_y:.8g}'},
          {'Path':'DT -> SI -> Y','Product_of_coefficients':b*c,'Components':f'{b:.8g} × {c:.8g}'},
          {'Path':'AIC -> DT -> SI -> Y','Product_of_coefficients':a*b*c,'Components':f'{a:.8g} × {b:.8g} × {c:.8g}'},
        ])
        summary['N_countries']=int(df[ID].nunique())
        summary['Interpretation']='FE product-of-coefficients; associational decomposition, not causal mediation without temporal ordering and sequential ignorability. No indirect effects are computed for controls.'
        summary.to_excel(OUTPUT_DIR/'Tables'/'Indirect_Effects_Main_Xs.xlsx',index=False)
    except Exception as e: save_status('Indirect_Effect_Status.txt',f'Indirect products not calculated: required path coefficient unavailable. {e}')

def machine_learning(df, controls):
    try:
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        import shap
    except Exception as e:
        save_status('ML_Status.txt',f'ML/SHAP not estimated: install xgboost lightgbm shap. {e}'); return
    features=[*X_MAIN,'Institution']  # SHAP/ML intentionally excludes controls; focus on AIC, DT, SI, Institution
    d=df[['ID','Year',Y,*features]].dropna().copy()
    # Country-held-out outer split; avoid leakage across years of a country.
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED)
    tr_idx,te_idx=next(splitter.split(d[features],d[Y],groups=d[ID]))
    train,test=d.iloc[tr_idx],d.iloc[te_idx]
    models={
      'RandomForest':RandomForestRegressor(n_estimators=N_ESTIMATORS,max_depth=8,min_samples_leaf=3,max_features=1.0,random_state=SEED,n_jobs=-1),
      'XGBoost':XGBRegressor(n_estimators=N_ESTIMATORS,max_depth=4,learning_rate=.03,subsample=.8,colsample_bytree=.8,reg_lambda=1.0,objective='reg:squarederror',random_state=SEED,n_jobs=4),
      'LightGBM':LGBMRegressor(n_estimators=N_ESTIMATORS,max_depth=-1,learning_rate=.03,num_leaves=15,subsample=.8,colsample_bytree=.8,reg_lambda=1.0,random_state=SEED,n_jobs=4,verbosity=-1),
    }
    scores=[]
    for name,m in models.items():
        try:
            m.fit(train[features],train[Y]); pred=m.predict(test[features])
            scores.append({'Model':name,'N_train':len(train),'N_test':len(test),'Test_R2':r2_score(test[Y],pred),'Test_RMSE':mean_squared_error(test[Y],pred)**.5,'Test_MAE':mean_absolute_error(test[Y],pred)})
            import joblib; joblib.dump(m,OUTPUT_DIR/'Models'/f'{name}.joblib')
        except Exception as e: save_status(f'ML_{name}_ERROR.txt',traceback.format_exc())
    if not scores: save_status('ML_Status.txt','All ML models failed.'); return
    score_df=pd.DataFrame(scores).sort_values('Test_RMSE'); score_df.to_excel(OUTPUT_DIR/'Tables'/'Table_5_ML_Model_Comparison.xlsx',index=False)
    best_name=score_df.iloc[0]['Model']; best=models[best_name]
    # SHAP and scenarios are predictive model-implied changes, not causal effects.
    try:
        explainer=shap.TreeExplainer(best); sample=test[features].sample(min(300,len(test)),random_state=SEED)
        sv=explainer.shap_values(sample)
        pd.DataFrame({'Feature':features,'Mean_abs_SHAP':np.abs(np.asarray(sv)).mean(axis=0)}).sort_values('Mean_abs_SHAP',ascending=False).to_excel(OUTPUT_DIR/'Tables'/'SHAP_Importance.xlsx',index=False)
        plt.figure(); shap.summary_plot(sv,sample,show=False); plt.tight_layout(); plt.savefig(OUTPUT_DIR/'Figures'/'Figure_4_SHAP_Summary.png',dpi=250,bbox_inches='tight'); plt.close()
        # Scenario: increase/decrease each primary X by 5%, 10%, 20%, 50%, 75%, 95% around test observations; report mean delta Y and percent of baseline prediction.
        baseline=np.asarray(best.predict(test[features])); scen=[]
        for x in X_MAIN:
            for pct in [-.95,-.75,-.50,-.20,-.10,-.05,.05,.10,.20,.50,.75,.95]:
                altered=test[features].copy(); altered[x]=altered[x]*(1+pct)
                # If X is an index where zero/negative values occur, percentage scenarios can be misleading; note this explicitly.
                pred=np.asarray(best.predict(altered)); delta=pred-baseline
                denom=np.where(np.abs(baseline)>1e-8,np.abs(baseline),np.nan)
                scen.append({'Variable':x,'Shock_pct':pct*100,'Mean_predicted_delta_Y':float(np.nanmean(delta)),'Mean_delta_pct_abs_baseline_prediction':float(np.nanmean(delta/denom*100)),'N_test':len(test),'Best_model':best_name,'Interpretation':'Predictive scenario, not causal effect; multiplicative shock assumes ratio-scale X.'})
        pd.DataFrame(scen).to_excel(OUTPUT_DIR/'Tables'/'Table_6_ML_Scenarios.xlsx',index=False)
        save_status('ML_Status.txt',f'Best model by country-held-out test RMSE: {best_name}. SHAP and model-implied scenarios generated. These are predictive associations, not causal impacts.\n{score_df.to_string(index=False)}')
    except Exception as e: save_status('ML_SHAP_ERROR.txt',traceback.format_exc()); save_status('ML_Status.txt',f'Model comparison completed; SHAP/scenario failed: {e}')

def main():
    if not DATA_FILE.exists(): raise FileNotFoundError(f'Dataset not found: {DATA_FILE}')
    df=pd.read_excel(DATA_FILE,sheet_name=DATA_SHEET); df.columns=[str(c).strip() for c in df.columns]
    # Internal research names: map legacy dataset columns to the requested labels.
    rename_map={old:new for old,new in [('X1','AIC'),('X2','DT'),('X3','SI')] if old in df.columns and new not in df.columns}
    if rename_map: df=df.rename(columns=rename_map)
    need=['Country',ID,YEAR,Y,*X_MAIN,'Institution']; miss=[c for c in need if c not in df.columns]
    if miss: raise ValueError(f'Missing required columns: {miss}')
    df[ID]=df[ID].astype(str).str.strip().str.upper(); df[YEAR]=pd.to_numeric(df[YEAR],errors='coerce'); df=df[df[YEAR].between(2010,2024)].copy(); df[YEAR]=df[YEAR].astype(int)
    for c in [Y,*X_MAIN,'Institution',*CONTROLS_CANDIDATES]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
    if df.duplicated([ID,YEAR]).any(): raise ValueError('Duplicate country-year observations found.')
    controls=[c for c in CONTROLS_CANDIDATES if c in df.columns and df[c].notna().sum()>0]
    audit={'rows':len(df),'countries':df[ID].nunique(),'years':sorted(df[YEAR].unique().tolist()),'controls_used':controls,'ASI_excluded':'ASI is absent from current analysis-ready dataset','data_file':str(DATA_FILE)}
    save_status('Analysis_Audit.json',json.dumps(audit,indent=2))
    log(json.dumps(audit,indent=2))
    descriptive(df,controls)
    policy_event_study(df)
    spatial_analysis(df,controls)
    mediation(df,controls)
    machine_learning(df,controls)
    save_status('READ_ME_BEFORE_INTERPRETING.txt', '''INTERPRETATION RULES\n- Descriptives/correlations are descriptive.\n- Event-study coefficients are not automatically causal; verify policy timing and staggered-adoption assumptions.\n- GSynth is attempted separately for three adoption-year cohorts (2017-2019, 2020-2021, 2022-2024) using exact verified treatment years and verified never-treated controls only. Cohort estimates may be unavailable if treatment/control evidence or runtime dependencies are insufficient; no results are fabricated.\n- Moran/LISA diagnose spatial association; cross-sectional LM tests are model-selection aids, not definitive model-selection rules.\n- Queen isolates remain zero rows; KNN5 is a separate alternative.\n- FE path products are associational, not causal mediation proof. Controls enter equations but are not assigned indirect effects.\n- VIF is a pooled diagnostic; inspect correlations and coefficient stability as well.\n- ML evaluation holds out countries. SHAP uses only AIC, DT, SI and Institution; control variables are excluded from ML/SHAP. SHAP and model scenarios are predictive, not causal. Percent shocks require ratio-scale predictors; if index scaling is not ratio-scale, use absolute or SD shocks instead.\n''')
    log(f'DONE. Results: {OUTPUT_DIR}')
if __name__=='__main__': main()
