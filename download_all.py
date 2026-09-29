import argparse, io, time
from pathlib import Path
import pandas as pd
import requests
from datetime import datetime

BASE = 'https://www.football-data.co.uk/mmz4281/{season}/{league}.csv'
FIXTURES_URL = 'https://www.football-data.co.uk/fixtures.csv'

def codes(start, end):
    for y in range(start, end + 1):
        yield f'{y%100:02d}{(y+1)%100:02d}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=2010)
    ap.add_argument('--end', type=int, default=(datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1))
    ap.add_argument('--leagues', default=None)
    ap.add_argument('--out', default='data')
    ap.add_argument('--refresh-current', action='store_true')
    args = ap.parse_args()

    catalog = pd.read_csv('leagues.csv')
    if args.leagues:
        catalog = catalog[catalog.league_code.isin({x.strip() for x in args.leagues.split(',')})]

    raw = Path(args.out) / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    
    season_start = datetime.now().year if datetime.now().month >= 7 else datetime.now().year - 1
    current = f'{season_start%100:02d}{(season_start+1)%100:02d}'

    s = requests.Session()
    s.headers['User-Agent'] = 'football-value-model/3.0'

    # 1. Descargar histórico de ligas
    for _, row in catalog.iterrows():
        league = row.league_code
        for season in codes(args.start, args.end):
            path = raw / f'{league}_{season}.csv'
            url = BASE.format(season=season, league=league)
            if path.exists() and not (args.refresh_current and season == current):
                continue
            try:
                res = s.get(url, timeout=10)
                if res.status_code == 200 and len(res.content) > 100:
                    with open(path, 'wb') as f:
                        f.write(res.content)
            except Exception as e:
                print(f"Error descargando {league} {season}: {e}")

    # 2. DESCARGAR PARTIDOS FUTUROS (FIXTURES)
    fixtures_path = raw / 'fixtures.csv'
    try:
        res = s.get(FIXTURES_URL, timeout=10)
        if res.status_code == 200 and len(res.content) > 100:
            with open(fixtures_path, 'wb') as f:
                f.write(res.content)
            print("✓ Descargados fixtures de próximos partidos correctamente.")
    except Exception as e:
        print(f"Error descargando fixtures: {e}")

if __name__ == '__main__':
    main()
