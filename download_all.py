import argparse, io, time
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime
BASE='https://www.football-data.co.uk/mmz4281/{season}/{league}.csv'
def codes(start,end):
    for y in range(start,end+1): yield f'{y%100:02d}{(y+1)%100:02d}'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start',type=int,default=2010); ap.add_argument('--end',type=int,default=(datetime.now().year if datetime.now().month>=7 else datetime.now().year-1)); ap.add_argument('--leagues',default=None); ap.add_argument('--out',default='data'); ap.add_argument('--refresh-current',action='store_true')
    args=ap.parse_args(); catalog=pd.read_csv('leagues.csv')
    if args.leagues: catalog=catalog[catalog.league_code.isin({x.strip() for x in args.leagues.split(',')})]
    raw=Path(args.out)/'raw'; raw.mkdir(parents=True,exist_ok=True); log=[]
    season_start=datetime.now().year if datetime.now().month>=7 else datetime.now().year-1
    current=f'{season_start%100:02d}{(season_start+1)%100:02d}'
    s=requests.Session(); s.headers['User-Agent']='football-value-model/3.0'
    for _,row in catalog.iterrows():
      league=row.league_code
      for season in codes(args.start,args.end):
        path=raw/f'{league}_{season}.csv'; url=BASE.format(season=season,league=league)
        if path.exists() and not (args.refresh_current and season==current): log.append([league,season,'cached',str(path)]); continue
        try:
          r=s.get(url,timeout=30)
          if r.status_code==404: log.append([league,season,'not_found',url]); continue
          r.raise_for_status(); df=pd.read_csv(io.BytesIO(r.content),encoding='latin1')
          if len(df)<2: log.append([league,season,'too_small',url]); continue
          df.to_csv(path,index=False,encoding='utf-8'); log.append([league,season,'downloaded',url]); print('OK',league,season,len(df)); time.sleep(.12)
        except Exception as e: log.append([league,season,'error',str(e)])
    pd.DataFrame(log,columns=['league','season','status','info']).to_csv(Path(args.out)/'download_log.csv',index=False)
if __name__=='__main__': main()
# Añadir en download_all.py para obtener partidos futuros de los próximos días
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

try:
    r = requests.get(FIXTURES_URL, headers=headers)
    if r.status_code == 200:
        with open("data/raw/fixtures.csv", "wb") as f:
            f.write(r.content)
        print("Fixtures de próximos partidos descargados correctamente.")
except Exception as e:
    print(f"No se pudieron descargar fixtures: {e}")
