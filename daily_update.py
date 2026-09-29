import argparse, subprocess, sys, json, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

def run(cmd):
    print('\n$', ' '.join(cmd)); return subprocess.run(cmd,check=True).returncode

def main():
    ap=argparse.ArgumentParser(description='Actualización diaria completa: descarga -> merge -> as-of -> entrenamiento -> value scan -> log.')
    ap.add_argument('--start',type=int,default=2010); ap.add_argument('--leagues',default=None); ap.add_argument('--refresh-seasons',type=int,default=2); ap.add_argument('--min-edge',type=float,default=.025); ap.add_argument('--min-ev',type=float,default=.02); ap.add_argument('--no-download',action='store_true'); a=ap.parse_args()
    root=Path(__file__).resolve().parent; py=sys.executable; now=datetime.now(timezone.utc).isoformat()
    log={'started_at':now,'status':'started'}
    try:
      if not a.no_download:
        end=datetime.now().year if datetime.now().month>=7 else datetime.now().year-1; start=max(a.start,end-a.refresh_seasons+1)
        run([py,str(root/'download_all.py'),'--start',str(a.start),'--end',str(end),'--refresh-current'] + (['--leagues',a.leagues] if a.leagues else []))
      run([py,str(root/'combine.py')]); run([py,str(root/'build_asof.py')]); run([py,str(root/'pipeline_de_modelado_de_f_tbol.py')]); run([py,str(root/'value_scan.py'),'--min-edge',str(a.min_edge),'--min-ev',str(a.min_ev)])
      pred=pd.read_csv(root/'data/processed/predictions_ensemble.csv'); vals=pd.read_csv(root/'data/processed/value_candidates.csv')
      log.update({'finished_at':datetime.now(timezone.utc).isoformat(),'status':'ok','matches':int(len(pred)),'completed':int(pred.home_goals.notna().sum()),'upcoming':int(pred.home_goals.isna().sum()),'value_candidates':int(len(vals))})
    except Exception as e:
      log.update({'finished_at':datetime.now(timezone.utc).isoformat(),'status':'error','error':repr(e)}); raise
    lp=root/'data/logs/daily_sync_log.csv'; lp.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame([log]).to_csv(lp,index=False,mode='a',header=not lp.exists())
    (root/'data/logs/last_sync.json').write_text(json.dumps(log,indent=2),encoding='utf-8'); print(json.dumps(log,indent=2))
if __name__=='__main__': main()
