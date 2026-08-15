"""Check whether raw-scale and transformed-target forests have complementary errors."""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold
from sklearn.pipeline import Pipeline
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,ROOT)
from app.nf_features import TARGETS,MEMBRANE_COL,DENS_COLS,TIME_COLS,CHAR_COLS,OP_COLS,build_feature_matrix,build_synth_key,fwd_transform,inv_transform
def model(mf): return Pipeline([('imp',SimpleImputer(strategy='median')),('m',ExtraTreesRegressor(n_estimators=600,max_features=mf,random_state=42,n_jobs=1))])
p=os.path.join(ROOT,'_recovery_package','nf_webapp','data','nanofiltration_data.xlsx'); df=pd.read_excel(p,sheet_name='main'); df['synth_key']=build_synth_key(df); df=df.drop_duplicates(subset=['synth_key']+OP_COLS+list(TARGETS.values()),keep='first')
for c in DENS_COLS+TIME_COLS+CHAR_COLS+OP_COLS+list(TARGETS.values()): df[c]=pd.to_numeric(df[c],errors='coerce')
cats=sorted(df[MEMBRANE_COL].fillna('Unk').astype(str).unique()); m=df[TARGETS['Mg_rej']].notna(); X=build_feature_matrix(df.copy(),membrane_categories=cats).loc[m].values; yr=df.loc[m,TARGETS['Mg_rej']].values; yt=fwd_transform('Mg_rej',yr); groups=df.loc[m,'synth_key'].values
def oof(cv, grouped=False):
    pr=np.zeros(len(yr)); pl=np.zeros(len(yr))
    sp=cv.split(X,yt,groups) if grouped else cv.split(X,yt)
    for tr,te in sp:
        pr[te]=model(.7).fit(X[tr],yr[tr]).predict(X[te]); pl[te]=inv_transform('Mg_rej',.5*model(.5).fit(X[tr],yt[tr]).predict(X[te])+.5*model(.7).fit(X[tr],yt[tr]).predict(X[te]))
    return pr,pl
out={}
for name,cv,grouped in [('KFold_seed42',KFold(n_splits=5,shuffle=True,random_state=42),False),('GroupKFold',GroupKFold(n_splits=5),True)]:
    pr,pl=oof(cv,grouped); rows=[]
    for w in np.arange(0,1.01,.1):
        pred=w*pr+(1-w)*pl; rows.append({'raw_ET_weight':round(float(w),1),'R2_raw':round(float(r2_score(yr,pred)),4),'MAE_raw':round(float(mean_absolute_error(yr,pred)),4)})
    out[name]=sorted(rows,key=lambda x:-x['R2_raw'])
with open(os.path.join(ROOT,'results','mg_blend_check.json'),'w',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
