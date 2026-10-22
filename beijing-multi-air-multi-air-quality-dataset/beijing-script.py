from __future__ import annotations
import argparse, math, json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
COORD = {
 'Aotizhongxin':(116.397,39.982),'Changping':(116.230,40.217),
 'Dingling':(116.220,40.292),'Dongsi':(116.417,39.929),
 'Guanyuan':(116.339,39.929),'Gucheng':(116.184,39.914),
 'Huairou':(116.628,40.328),'Nongzhanguan':(116.461,39.937),
 'Shunyi':(116.655,40.127),'Tiantan':(116.407,39.886),
 'Wanliu':(116.287,39.987),'Wanshouxigong':(116.352,39.878)
}
SEED=20260909
RMAX=10.
LAMBDA=-math.log(.30)/math.log(3)
RF_KW=dict(n_estimators=200,min_samples_leaf=5,max_features=.70,max_depth=None)
COVARS=['TEMP','PRES','DEWP','RAIN','WSPM','sin_year','cos_year','sin_month','cos_month']
MODELS=['RF','RF-XYT','RFsp','PZ-D-RFsp']

def write(path, content):
    path.write_text(content,encoding='utf-8')

def get_data(folder):
    fs=sorted(folder.glob('PRSA_Data_*.csv'))
    if len(fs)!=12: raise ValueError(f'Expected 12 PRSA_Data_*.csv station files in {folder}; found {len(fs)}')
    chunks=[]; total=0
    for fp in fs:
        d=pd.read_csv(fp,low_memory=False)
        required={'year','month','day','hour','station','PM2.5','TEMP','PRES','DEWP','RAIN','WSPM'}
        if required-set(d.columns):raise ValueError(f'{fp.name} missing {required-set(d.columns)}')
        total+=len(d)
        d['station']=d['station'].astype(str).str.strip()
        if len(d.station.unique())!=1 or d.station.iloc[0] not in COORD:
            raise ValueError(f'Unrecognized/mixed station names in {fp.name}: {d.station.unique()}')
        use=['station','year','month','day','PM2.5','TEMP','PRES','DEWP','RAIN','WSPM']
        chunks.append(d[use])
    hourly=pd.concat(chunks,ignore_index=True)
    hourly['date']=pd.to_datetime(hourly[['year','month','day']],errors='coerce')
    for col in ['PM2.5','TEMP','PRES','DEWP','RAIN','WSPM']:
        hourly[col]=pd.to_numeric(hourly[col],errors='coerce')
    daily=hourly.groupby(['station','date'],observed=True).agg(
        **{'PM2.5':('PM2.5','mean'),'n_hourly_target':('PM2.5','count'),
           'TEMP':('TEMP','mean'),'PRES':('PRES','mean'),'DEWP':('DEWP','mean'),
           'RAIN':('RAIN','sum'),'WSPM':('WSPM','mean')}).reset_index()
    daily=daily[daily.n_hourly_target>=18].dropna(subset=['date','PM2.5']).copy()
    daily['lon']=daily.station.map(lambda s:COORD[s][0]);daily['lat']=daily.station.map(lambda s:COORD[s][1])
    day=daily.date.dt.dayofyear.to_numpy()
    month=daily.date.dt.month.to_numpy()
    daily['sin_year']=np.sin(2*np.pi*day/365.25);daily['cos_year']=np.cos(2*np.pi*day/365.25)
    daily['sin_month']=np.sin(2*np.pi*month/12.);daily['cos_month']=np.cos(2*np.pi*month/12.)
    lat0=np.mean([v[1] for v in COORD.values()]);lon0=np.mean([v[0] for v in COORD.values()])
    daily['km_x']=(daily.lon-lon0)*111.32*math.cos(math.radians(lat0))
    daily['km_y']=(daily.lat-lat0)*111.32
    stations=np.array([(lon-lon0)*111.32*math.cos(math.radians(lat0)) for lon,lat in COORD.values()])
    ys=np.array([(lat-lat0)*111.32 for lon,lat in COORD.values()])
    lo=np.array([stations.min(),ys.min()]);hi=np.array([stations.max(),ys.max()]);span=hi-lo
    daily['sx']=(daily.km_x-lo[0])/span[0];daily['sy']=(daily.km_y-lo[1])/span[1]
    scale=max(span)
    daily['sx']=(daily.km_x-lo[0])/scale;daily['sy']=(daily.km_y-lo[1])/scale
    return daily.reset_index(drop=True),total,scale

def distance(a,b,r=1.,phi=0.):
    diff=a[:,None,:]-b[None,:,:]
    if abs(r-1)<1e-12:return np.sqrt(np.sum(diff*diff,axis=2))
    angle=np.deg2rad(phi);u=diff[:,:,0]*np.cos(angle)+diff[:,:,1]*np.sin(angle)
    v=-diff[:,:,0]*np.sin(angle)+diff[:,:,1]*np.cos(angle)
    return np.sqrt(np.maximum(u*u/r+r*v*v,0.))

def params_from_z(z):
    k=float(np.hypot(*z));r=float(np.exp(k))
    return r, float((0.5*np.rad2deg(np.arctan2(z[1],z[0])))%180) if k>1e-10 else float('nan'),k

def reml(theta,loc,y,X,lam=LAMBDA,rmax=RMAX):
    logr,loge,z1,z2=theta
    r,phi,k=params_from_z((z1,z2))
    if k>np.log(rmax)+1e-10:return 1e9+1e8*(k-np.log(rmax))**2
    try:
        d=distance(loc,loc,r,0. if not np.isfinite(phi) else phi)
        K=np.exp(-d/np.exp(logr));K.flat[::len(K)+1]+=np.exp(loge)+1e-9
        cf=cho_factor(K,lower=True,check_finite=False)
        ki_y=cho_solve(cf,y,check_finite=False);ki_x=cho_solve(cf,X,check_finite=False)
        xx=X.T@ki_x;cx=cho_factor(xx,lower=True,check_finite=False)
        beta=cho_solve(cx,X.T@ki_y,check_finite=False)
        resid=y-X@beta;q=float(resid@cho_solve(cf,resid,check_finite=False))
        df=len(y)-X.shape[1]
        if df<=0 or q<=0:return 1e12
        return float(2*np.log(np.diag(cf[0])).sum()+2*np.log(np.diag(cx[0])).sum()+df*np.log(q/df)+lam*k)
    except (np.linalg.LinAlgError,ValueError,FloatingPointError,OverflowError):return 1e12

def z_start(loc,y):
    ii,jj=np.triu_indices(len(y),1)
    delta=loc[jj]-loc[ii];rr=np.hypot(delta[:,0],delta[:,1]);sel=(rr>1e-10)&(rr<=np.quantile(rr[rr>1e-10],.5))
    if np.sum(sel)<5:return (0.,0.)
    angle=(np.rad2deg(np.arctan2(delta[sel,1],delta[sel,0]))%180)
    gamma=(y[jj[sel]]-y[ii[sel]])**2
    centers=np.arange(0,180,15);sc=[]
    for c in centers:
        dd=np.abs((angle-c+90)%180-90);sc.append(np.median(gamma[dd<=20]) if (dd<=20).sum()>=2 else np.inf)
    if not np.any(np.isfinite(sc)):return (0.,0.)
    phi=float(centers[np.argmin(sc)]);k=np.log(1.35)
    return (k*np.cos(np.deg2rad(2*phi)),k*np.sin(np.deg2rad(2*phi)))

def pz_estimate(df, covars, rmax=RMAX):
    x=np.asarray(covars,float);y=df['PM2.5'].to_numpy(float)
    x=np.column_stack([np.ones(len(x)),x]); b=np.linalg.lstsq(x,y,rcond=None)[0]
    station_resid=df[['station','sx','sy']].copy();station_resid['resid']=y-x@b
    site=station_resid.groupby('station').agg(sx=('sx','first'),sy=('sy','first'),resid=('resid','mean'))
    loc=site[['sx','sy']].to_numpy();yy=site.resid.to_numpy();X=np.ones((len(yy),1))
    iso_bounds=[(np.log(.02),np.log(2.)),(np.log(1e-4),np.log(10.))]
    iso=minimize(lambda a: reml([a[0],a[1],0.,0.],loc,yy,X,0.,rmax),
                 [np.log(.25),np.log(.25)],method='L-BFGS-B',bounds=iso_bounds,options={'maxiter':80,'ftol':1e-8})
    a=iso.x; zg=z_start(loc,yy)
    starts=[[a[0],a[1],0.,0.],[a[0],a[1],*zg]]
    for phi in (0.,45.,90.,135.):
        k=np.log(1.35);starts.append([a[0],a[1],k*np.cos(np.deg2rad(2*phi)),k*np.sin(np.deg2rad(2*phi))])
    kmax=np.log(rmax);bounds=iso_bounds+[(-kmax,kmax)]*2
    fits=[minimize(reml,s, args=(loc,yy,X,LAMBDA,rmax),method='L-BFGS-B',bounds=bounds,
                   options={'maxiter':120,'ftol':1e-8}) for s in starts]
    fit=min(fits,key=lambda z:z.fun if np.isfinite(z.fun) else np.inf)
    r,phi,k=params_from_z(fit.x[2:]);r=min(r,rmax)
    return {'r':r,'phi':phi,'kappa':k,'success':bool(fit.success and np.isfinite(fit.fun)),
            'objective':float(fit.fun),'sites':len(site),'at_bound':bool(r>=.98*rmax)}

def impute(train,test):
    med=train[COVARS].median().fillna(0.)
    return train[COVARS].fillna(med).to_numpy(float),test[COVARS].fillna(med).to_numpy(float)

def model_predict(features_train, features_test, y, seed, threads):
    m=RandomForestRegressor(**RF_KW,random_state=seed,n_jobs=threads)
    m.fit(features_train,y)
    return m.predict(features_test)

def metrics(y,p):
    return {'rmse':float(np.sqrt(mean_squared_error(y,p))),
            'mae':float(mean_absolute_error(y,p)),
            'r2':float(r2_score(y,p))}

def fmt(v,n=4):
    return f'{v:.{n}f}' if isinstance(v,(float,np.floating)) and np.isfinite(v) else ('NA' if v is None or isinstance(v,(float,np.floating)) else str(v))

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)

def render(meta,full,folds,results,complete):
    s=['# Beijing Multi-Site Air Quality — D-RFsp Real-Data Results','',
       '> **Status:** '+('COMPLETE' if complete else 'IN PROGRESS — incomplete results; do not cite as final.'),'',
       '## Dataset and frozen design','',
       f'- Source files: {meta["files"]} PRSA station CSVs; raw hourly records: {meta["raw"]:,}.',
       f'- Station-day observations with at least 18 observed hourly PM2.5 measurements: {meta["daily"]:,}.',
       f'- Unique monitoring sites: {meta["sites"]}; dates: {meta["start"]} to {meta["end"]}.',
       '- Response: daily average PM2.5 (µg/m³). This is a DAILY aggregation of the supplied hourly data.',
       '- Common meteorological inputs: TEMP, PRES, DEWP, RAIN, WSPM; time harmonics (day-of-year and month).',
       '- Other pollutant concentrations (PM10, SO2, NO2, CO, O3) deliberately excluded to avoid pollutant-proxy effects.',
       '- Fold assignment: station-wise four-fold holdout (3 entire stations per fold); fixed alphabetic grouping.',
       '- Same RF engine for every arm: 200 trees, min_samples_leaf=5, max_features=0.70, fixed seed.',
       '- RF = meteorology/time only; RF-XYT = RF + projected station x,y; RFsp = RF + 25 isotropic reference distances; PZ-D-RFsp = identical RFsp features with PZ-estimated directional distances.',
       '- Geometry: determinant-normalized d² = u²/r + rv²; r≥1, axial φ∈[0°,180°).',
       '- PZ: isotropic REML start, geometry start, joint Z-REML, penalty λ||Z|| with λ=-log(.30)/log(3), r_max=10; spatial range [0.02,2], nugget ratio [1e-4,10].',
       '- Coordinates: published approximate station coordinates, locally projected into kilometers, scaled by a common spatial length.',
       '- Screening anisotropy on FULL data is descriptive only; each prediction fold estimates its geometry with its TRAINING stations and outcomes only.',
       '- Missing meteorological features are filled with training-only medians. Scoring uses observed PM2.5 targets only.',
       '', '## Full-data anisotropy screening (NOT used for held-out prediction)','',
       table(['n stations','r_hat','phi_hat (degrees)','optimizer success','at r_max'],[[full['sites'],fmt(full['r']),fmt(full['phi']),full['success'],full['at_bound']]]),
       '', '## Model comparison (pooled station-out-of-sample observations)','']
    if results:
        allrows=[]
        for name in MODELS:
            pred=np.concatenate([np.array(f['pred'][name]) for f in results]); yt=np.concatenate([np.array(f['truth']) for f in results]);m=metrics(yt,pred)
            allrows.append([name,fmt(m['rmse']),fmt(m['mae']),fmt(m['r2']),fmt(m['rmse']-metrics(yt,np.concatenate([np.array(f['pred']['RFsp']) for f in results]))['rmse'])])
        s.extend([table(['model','RMSE','MAE','R²','Delta RMSE vs RFsp (negative=better)'],allrows),''])
        frows=[]
        for f in results:
            rf=f['stats']['RFsp'];dr=f['stats']['PZ-D-RFsp']
            frows.append([f['fold'],', '.join(f['test_stations']),fmt(f['est']['r']),fmt(f['est']['phi']),f['est']['success'],len(f['truth']),fmt(rf['rmse']),fmt(dr['rmse']),fmt(dr['rmse']-rf['rmse'])])
        s.extend(['## Fold-specific anisotropy and paired predictive differences','',table(['fold','held-out stations','r_hat','phi_hat (°)','REML success','test n','RFsp RMSE','PZ-D-RFsp RMSE','paired Delta RMSE'],frows),''])
        s.extend(['## All fold-wise model metrics','',table(['fold','model','test n','RMSE','MAE','R²'],[
            [f['fold'],m,len(f['truth']),fmt(f['stats'][m]['rmse']),fmt(f['stats'][m]['mae']),fmt(f['stats'][m]['r2'])]
            for f in results for m in MODELS]),''])
        deltas=np.array([f['stats']['PZ-D-RFsp']['rmse']-f['stats']['RFsp']['rmse'] for f in results])
        s.append(f'Completed folds: {len(results)}/4. Fold wins (negative Delta): {int(np.sum(deltas<0))}/{len(deltas)}. Mean fold paired Delta RMSE: {fmt(float(deltas.mean()))}.')
    else:s.append('_No fold has completed yet._')
    s.extend(['','## Statistical and identifiability cautions','',
        '- ONLY TWELVE distinct spatial coordinates exist; no number of hourly rows increases the number of spatial locations.',
        '- PZ-REML uses adjusted STATION MEAN residuals (one number/site), after removing available covariates, to prevent pseudoreplication at duplicate site coordinates. This preprocessing adaptation differs from the simulation’s independent-point GP setting.',
        '- Fold PZ fitting has only 9 sites; r and φ are weakly identified and confidence claims should be cautious.',
        '- Residuals pooled over time are not a fully specified spatiotemporal likelihood. Global spatial anisotropy estimates are exploratory screening estimates, not precise physical measurements.',
        '- Fold RMSE differences are paired because RFsp and PZ-D-RFsp use exactly the same holdout observations; no single-fold superiority proves a general effect.',
        '- Reported full-data r/φ are never inserted into test-fold D-RFsp distance construction.',
        '- No anisotropy-driven tuning of hyperparameters or selection of folds occurs.',
        '', '## Provenance and reproducibility','',
        '- Dataset: UCI Beijing Multi-Site Air Quality (PRSA), https://archive.ics.uci.edu/dataset/501/beijing',
        '- Station coordinate reference: https://www.mdpi.com/2071-1050/14/9/5104 (Table 1).',
        '- Frozen simulation source for optimizer constants and features: https://github.com/brpuneet898/D-RFsp/blob/main/final_simulation.py',
        '- To reproduce: `python beijing-script.py` from the project folder; save CSVs in `dataset/`.',
        '- No synthetic model scores or manually supplied anisotropy parameters are used.',''])
    return '\n'.join(s)

def main():
    ap=argparse.ArgumentParser(description='Blocked Beijing PM2.5 experiment')
    ap.add_argument('--data-dir',type=Path,default=HERE/'dataset')
    ap.add_argument('--output',type=Path,default=HERE/'beijing-results.md')
    ap.add_argument('--jobs',type=int,default=-1)
    ap.add_argument('--smoke-test',action='store_true',help='Run one fold; report marked incomplete')
    args=ap.parse_args()
    print('BEIJING D-RFsp MANUSCRIPT REAL-DATA ANALYSIS',flush=True)
    data,raw,scale=get_data(args.data_dir)
    sts=sorted(data.station.unique())
    if len(sts)!=12:raise RuntimeError('Expected 12 sites with usable observations')
    meta={'files':12,'raw':raw,'daily':len(data),'sites':len(sts),
          'start':str(data.date.min().date()),'end':str(data.date.max().date())}
    print(f'{raw:,} hourly records => {len(data):,} station-day observations across {len(sts)} stations',flush=True)
    d=data.copy();d[COVARS]=d[COVARS].fillna(d[COVARS].median().fillna(0.))
    full=pz_estimate(d,d[COVARS].to_numpy())
    print(f'Full-data screening: r_hat={fmt(full["r"])} phi_hat={fmt(full["phi"])} degrees',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    results=[];write(args.output,render(meta,full,[],results,False))
    axis=np.linspace(0,1,5)
    refs=np.stack(np.meshgrid(axis,axis),-1).reshape(-1,2)
    for fold in range(4 if not args.smoke_test else 1):
        held=sts[fold*3:(fold+1)*3]
        train=data.loc[~data.station.isin(held)].copy()
        test=data.loc[data.station.isin(held)].copy()
        if len(train)<20 or len(test)<20:raise RuntimeError('Insufficient train or test observations')
        xtr,xte=impute(train,test)
        pzdf=train.copy();pzdf[COVARS]=xtr
        est=pz_estimate(pzdf,xtr)
        loc_tr=train[['sx','sy']].to_numpy();loc_te=test[['sx','sy']].to_numpy()
        ytr=train['PM2.5'].to_numpy();yte=test['PM2.5'].to_numpy()
        designs={
            'RF':(xtr,xte),
            'RF-XYT':(np.column_stack((xtr,loc_tr)),np.column_stack((xte,loc_te))),
            'RFsp':(np.column_stack((xtr,distance(loc_tr,refs))),np.column_stack((xte,distance(loc_te,refs)))),
            'PZ-D-RFsp':(np.column_stack((xtr,distance(loc_tr,refs,est['r'],0 if not np.isfinite(est['phi']) else est['phi']))),
                          np.column_stack((xte,distance(loc_te,refs,est['r'],0 if not np.isfinite(est['phi']) else est['phi']))))
        }
        preds={};stats={}
        for mi,name in enumerate(MODELS):
            a,b=designs[name];p=model_predict(a,b,ytr,SEED+fold*101, args.jobs)
            preds[name]=p.tolist();stats[name]=metrics(yte,p)
        record={'fold':fold+1,'test_stations':held,'est':est,'pred':preds,'stats':stats,'truth':yte.tolist()}
        results.append(record)
        write(args.output,render(meta,full,[],results,not args.smoke_test and len(results)==4))
        print(f'[{fold+1}/4] held out {", ".join(held)} | r={fmt(est["r"])} phi={fmt(est["phi"])} | RFsp RMSE={fmt(stats["RFsp"]["rmse"])} | D-RFsp RMSE={fmt(stats["PZ-D-RFsp"]["rmse"])}',flush=True)
    print(f'Saved report: {args.output.resolve()}',flush=True)
    if args.smoke_test:print('SMOKE TEST: report explicitly marked incomplete; run without --smoke-test for final results.',flush=True)

if __name__=='__main__':main()
