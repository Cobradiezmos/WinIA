import argparse, math
import numpy as np
import pandas as pd
from scipy.stats import poisson

def poisson_1x2(lam_h, lam_a, max_goals=10):
    probs = np.outer(
        poisson.pmf(np.arange(max_goals+1), lam_h),
        poisson.pmf(np.arange(max_goals+1), lam_a)
    )
    ph = np.tril(probs, -1).sum()       # fila home > columna away
    pd = np.trace(probs)
    pa = np.triu(probs, 1).sum()
    z = ph + pd + pa
    return ph/z, pd/z, pa/z

def fit_attack_defence(df, shrink=0.25):
    # Estimación simple y robusta: medias EWMA/as-of + regresión a media.
    # No usa el resultado del propio partido.
    league_mean = df[["home_gf_5","away_gf_5","home_ga_5","away_ga_5"]].stack().mean()
    rows = []
    for _, r in df.iterrows():
        # Ataque local contra defensa visitante.
        h_attack = (r.home_gf_5 + (r.away_ga_5 if pd.notna(r.away_ga_5) else league_mean))/2
        a_attack = (r.away_gf_5 + (r.home_ga_5 if pd.notna(r.home_ga_5) else league_mean))/2
        h_attack = (1-shrink)*h_attack + shrink*league_mean
        a_attack = (1-shrink)*a_attack + shrink*league_mean

        # Ajuste pequeño por Elo.
        elo_factor = np.clip((r.elo_diff_pre/400), -0.75, 0.75)
        lam_h = max(0.05, h_attack * math.exp(0.18*elo_factor))
        lam_a = max(0.05, a_attack * math.exp(-0.18*elo_factor))

        ph,pd,pa = poisson_1x2(lam_h, lam_a)
        rows.append([lam_h,lam_a,ph,pd,pa])

    x = pd.DataFrame(rows, columns=["lambda_home","lambda_away",
                                     "p_poisson_home","p_poisson_draw","p_poisson_away"])
    return pd.concat([df.reset_index(drop=True), x], axis=1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/asof_features.csv")
    ap.add_argument("--output", default="data/processed/predictions_elo_poisson.csv")
    a=ap.parse_args()
    df=pd.read_csv(a.input)
    out=fit_attack_defence(df)
    # Elo + Poisson: por ahora una mezcla conservadora; la calibración posterior
    # determinará el peso óptimo en validación temporal.
    out["p_model_home"] = 0.75*out.p_poisson_home + 0.25*out.elo_prob_home
    out["p_model_draw"] = 0.75*out.p_poisson_draw + 0.25*(1-out.elo_prob_home)*0.35
    out["p_model_away"] = 1-out.p_model_home-out.p_model_draw
    for side in ["home","draw","away"]:
        out[f"p_model_{side}"] = out[f"p_model_{side}"].clip(0.001,0.998)
    out.to_csv(a.output,index=False)
    print(out[["date","home_team","away_team","p_model_home","p_model_draw","p_model_away"]].tail())

if __name__=="__main__":
    main()
