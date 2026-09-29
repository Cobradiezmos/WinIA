from pathlib import Path
import pandas as pd
import argparse

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--out', default='data/processed/all_matches.csv')
    a=ap.parse_args()
    frames=[]
    for p in Path(a.raw).glob('*.csv'):
        try:
            x=pd.read_csv(p, encoding='utf-8')
            required={'Date','HomeTeam','AwayTeam'}
            if not required.issubset(x.columns): continue
            x['source_file']=p.name
            x['league']=p.name.split('_')[0]
            x['season']=p.stem.split('_')[-1]
            x['date']=pd.to_datetime(x['Date'], dayfirst=True, errors='coerce')
            x['home_team']=x['HomeTeam'].astype(str).str.strip()
            x['away_team']=x['AwayTeam'].astype(str).str.strip()
            x['home_goals']=pd.to_numeric(x.get('FTHG'), errors='coerce')
            x['away_goals']=pd.to_numeric(x.get('FTAG'), errors='coerce')
            x['result']=x.get('FTR', pd.Series(index=x.index, dtype='object')).map({'H':'H','D':'D','A':'A'})
            x['match_id']=(x['league'].astype(str)+'_'+x['season'].astype(str)+'_'+x['date'].dt.strftime('%Y%m%d').fillna('nodate')+'_'+x['home_team']+'_'+x['away_team']).str.replace(r'\s+','_',regex=True)
            frames.append(x)
        except Exception as e: print('skip',p,e)
    if not frames: raise SystemExit('No compatible raw CSV files found.')
    out=pd.concat(frames,ignore_index=True)
    out=out.dropna(subset=['date','home_team','away_team'])
    # Prefer the latest source row if duplicate files contain a refreshed current season.
    out=out.sort_values(['date','league','match_id','source_file']).drop_duplicates('match_id', keep='last')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.out,index=False)
    print('matches:',len(out),'completed:',out.home_goals.notna().sum(),'upcoming:',out.home_goals.isna().sum())
if __name__=='__main__': main()
