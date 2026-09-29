import argparse
import numpy as np, pandas as pd

def build_asof(input_csv, output_csv):
    df=pd.read_csv(input_csv,parse_dates=['date']).sort_values(['league','date','match_id']).reset_index(drop=True)
    rows=[]
    for league,g in df.groupby('league',sort=False):
      teams=pd.unique(pd.concat([g.home_team,g.away_team],ignore_index=True))
      state={t:{'elo':1500.,'gf':[],'ga':[],'pts':[],'home_gf':[],'home_ga':[],'away_gf':[],'away_ga':[],'last_date':None,'n':0} for t in teams}
      for _,r in g.sort_values(['date','match_id']).iterrows():
        h,a=r.home_team,r.away_team; hs,ass=state[h],state[a]
        eh=1/(1+10**((ass['elo']-hs['elo'])/400)); ea=1-eh
        def avg(x,default): return float(np.mean(x[-5:])) if x else default
        row=r.to_dict(); form_h=sum(hs['pts'][-5:])/max(1,min(5,len(hs['pts']))); form_a=sum(ass['pts'][-5:])/max(1,min(5,len(ass['pts'])))
        gd_h=avg([x-y for x,y in zip(hs['gf'][-5:],hs['ga'][-5:])],0); gd_a=avg([x-y for x,y in zip(ass['gf'][-5:],ass['ga'][-5:])],0)
        row.update({'elo_home_pre':hs['elo'],'elo_away_pre':ass['elo'],'elo_diff_pre':hs['elo']-ass['elo'],'elo_prob_home':eh,
          'home_gf_5':avg(hs['gf'],1.25),'home_ga_5':avg(hs['ga'],1.25),'away_gf_5':avg(ass['gf'],1.25),'away_ga_5':avg(ass['ga'],1.25),
          'home_ppg_5':avg(hs['pts'],1.0),'away_ppg_5':avg(ass['pts'],1.0),'home_home_gf_5':avg(hs['home_gf'],1.25),'home_home_ga_5':avg(hs['home_ga'],1.25),
          'away_away_gf_5':avg(ass['away_gf'],1.25),'away_away_ga_5':avg(ass['away_ga'],1.25),'home_matches_before':hs['n'],'away_matches_before':ass['n'],
          'home_rest_days':((r['date']-hs['last_date']).days if hs['last_date'] is not None else np.nan),'away_rest_days':((r['date']-ass['last_date']).days if ass['last_date'] is not None else np.nan),
          'form_diff_5':form_h-form_a,'goal_diff_form_5':gd_h-gd_a,'rest_diff':((r['date']-hs['last_date']).days if hs['last_date'] is not None else 14)-((r['date']-ass['last_date']).days if ass['last_date'] is not None else 14),'experience_diff':hs['n']-ass['n']})
        rows.append(row)
        if pd.notna(r['home_goals']) and pd.notna(r['away_goals']) and r['result'] in ('H','D','A'):
          hg,ag=float(r['home_goals']),float(r['away_goals']); hp,ap={'H':(3,0),'D':(1,1),'A':(0,3)}[r['result']]
          margin=max(1,abs(hg-ag)); k=20*(1+.25*np.log1p(margin)); hs['elo']+=k*(hp/3-eh); ass['elo']+=k*(ap/3-ea)
          hs['gf'].append(hg); hs['ga'].append(ag); hs['pts'].append(hp); ass['gf'].append(ag); ass['ga'].append(hg); ass['pts'].append(ap)
          hs['home_gf'].append(hg); hs['home_ga'].append(ag); ass['away_gf'].append(ag); ass['away_ga'].append(hg); hs['last_date']=r['date']; ass['last_date']=r['date']; hs['n']+=1; ass['n']+=1
    out=pd.DataFrame(rows); out.to_csv(output_csv,index=False); return out
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='data/processed/all_matches.csv'); ap.add_argument('--output',default='data/processed/asof_features.csv'); a=ap.parse_args(); build_asof(a.input,a.output)
