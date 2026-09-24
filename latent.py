"""NBA 2025-26 player latent space.

    python latent.py                                  # default embedding -> results/*.json
    python latent.py --list                           # show every feature key and group
    python latent.py --groups shot_diet creation      # embed only these feature groups
    python latent.py --hold USG% FTr                  # drop individual features
    python latent.py --give efficiency                # add an optional group (e.g. shooting accuracy)
    python latent.py --k 3 --min-mp 500 --out results/k3

Needs: pip install jax optax numpy pandas

What it does
  1. Builds one row per player from data/player-stats (season row; traded players'
     per-team rows are kept for team rosters and the "style follows the player" check).
  2. Fits a weighted, masked probabilistic PCA in JAX: every stat is weighted by
     the sample size behind it (shots or minutes) and missing values
     (e.g. no corner threes) are simply left out.
  3. Runs the recovery analyses on things the embedding never sees
     (positions, awards, +/-, playoff minutes) and writes JSON for the website.

Answer keys (position estimate, listed position, awards, +/-, playoff minutes)
are never used as embedding features, whatever --groups/--give say.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

jax.config.update("jax_enable_x64", False)
DATA = Path(__file__).resolve().parent / "data"
POSITIONS = ["PG", "SG", "SF", "PF", "C"]


# Loading

def read_bbref(path: Path) -> pd.DataFrame:
    """Read a Basketball-Reference CSV (one- or two-row header). Adds `pid`, drops League Average."""
    first = path.read_text(encoding="utf-8").splitlines()[0]
    two_rows = first.startswith(",") or "Rk" not in first.split(",")[:2]
    df = pd.read_csv(path, header=1 if two_rows else 0)
    df = df.rename(columns={df.columns[-1]: "pid", "Tm": "Team"})
    df = df[df["pid"].notna() & (df["pid"] != "-9999")]
    df = df[~df["Player"].astype(str).str.contains("League Average")] if "Player" in df else df
    return df.reset_index(drop=True)


def load_players():
    """Return (season, stints): one row per player, and one row per player-team stint."""
    t = read_bbref(DATA / "player-stats/totals.csv")
    adv = read_bbref(DATA / "player-stats/advanced.csv")
    sh = read_bbref(DATA / "player-stats/shooting.csv")
    pbp = read_bbref(DATA / "player-stats/play-by-play.csv")
    p100 = read_bbref(DATA / "player-stats/per-100-poss.csv")

    # Rows are in the same order in every table, so a within-player row counter
    # lines them up (row 0 = season / combined row, rows 1.. = team stints).
    frames = []
    for name, df in [("t", t), ("adv", adv), ("sh", sh), ("pbp", pbp), ("p100", p100)]:
        df = df.copy()
        df["row"] = df.groupby("pid").cumcount()
        frames.append(df.set_index(["pid", "row"]).add_prefix(f"{name}:"))
    raw = pd.concat(frames, axis=1, join="inner")
    for c in raw.columns:
        num = pd.to_numeric(raw[c], errors="coerce")
        if num.notna().sum() >= raw[c].notna().sum() * 0.9:   # numeric column
            raw[c] = num

    is_combined = raw["t:Team"].astype(str).str.match(r"\dTM")
    season = raw.xs(0, level="row").copy()   # combined row for traded players
    stints = raw[~is_combined].copy()        # one row per player-team
    return season, stints


AWARD_RE = re.compile(r"(MVP|DPOY|ROY|MIP|6MOY|CPOY)-(\d+?)(?=6MOY|[A-Z]|$)|NBA([123])|DEF([12])|(AS)")


def parse_awards(s) -> dict:
    """'MVP-3DPOY-1ASNBA1DEF1' -> {'votes': {'MVP': 3, 'DPOY': 1}, 'all_star': True, 'all_nba': 1, 'all_def': 1}."""
    out = {"votes": {}, "all_star": False, "all_nba": None, "all_def": None, "all_rookie": None}
    if not isinstance(s, str):
        return out
    for m in AWARD_RE.finditer(s):
        if m.group(1):
            out["votes"][m.group(1)] = int(m.group(2))
        elif m.group(3):
            out["all_nba"] = int(m.group(3))
        elif m.group(4):
            out["all_def"] = int(m.group(4))
        elif m.group(5):
            out["all_star"] = True
    return out


def load_all_rookie() -> dict[str, int]:
    df = pd.read_csv(DATA / "awards/all-rookie-teams.csv", header=None, skiprows=1)
    return {name: int(r[2][0]) for _, r in df.iterrows() for name in r[4:] if isinstance(name, str)}


def load_teams(season: pd.DataFrame):
    """Team table keyed by abbreviation. The name<->abbreviation map is derived from the
    data itself: each team's summed player rows equal exactly one row of team totals."""
    tt = read_bbref_team(DATA / "team-stats/total-stats.csv")
    adv = read_bbref_team(DATA / "team-stats/advanced-stats.csv")
    pt = read_bbref(DATA / "player-stats/totals.csv")
    pt = pt[~pt["Team"].astype(str).str.match(r"\dTM")]
    sums = pt.groupby("Team")[["PTS", "FGA", "AST", "TRB"]].sum()
    abbr = {}
    for ab, r in sums.iterrows():
        hit = tt[(tt.PTS == r.PTS) & (tt.FGA == r.FGA) & (tt.AST == r.AST) & (tt.TRB == r.TRB)]
        if len(hit) == 1:
            abbr[ab] = hit.index[0]
    teams = pd.DataFrame({"abbr": list(abbr), "name": list(abbr.values())}).set_index("abbr")
    for c in ("W", "L", "NRtg", "ORtg", "DRtg", "Pace", "SRS"):
        teams[c] = [float(adv.loc[n, c]) for n in teams["name"]]
    conf = {}
    for side, label in (("east", "East"), ("west", "West")):
        df = pd.read_csv(DATA / f"team-stats/division-standings-{side}.csv")
        div = None
        for _, r in df.iterrows():
            nm = str(r.iloc[0])
            if nm.endswith("Division"):
                div = nm.replace(" Division", "")
            else:
                conf[nm.replace("*", "")] = (label, div)
    teams["conference"] = [conf.get(n, (None, None))[0] for n in teams["name"]]
    teams["division"] = [conf.get(n, (None, None))[1] for n in teams["name"]]
    return teams


def read_bbref_team(path: Path) -> pd.DataFrame:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    df = pd.read_csv(path, header=1 if first.startswith(",") else 0)
    df = df.rename(columns={"Tm": "Team"})
    df = df[df["Team"].notna() & ~df["Team"].astype(str).str.contains("League Average")]
    df["Team"] = df["Team"].str.replace("*", "", regex=False)
    return df.set_index("Team").apply(pd.to_numeric, errors="coerce")


def load_playoff_minutes() -> pd.Series:
    po = read_bbref(DATA / "playoff-stats/player-stats/totals.csv")
    return po.groupby("pid")["MP"].sum()


def load_playoff_players(columns) -> pd.DataFrame:
    """Playoff player rows laid out like `season` (same column names) so the same features
    and encoder apply. The playoff tables have no shooting or play-by-play splits, so those
    columns stay NaN and the encoder simply ignores them."""
    out = None
    for name, f in [("t", "totals"), ("adv", "advanced"), ("p100", "per-100-poss")]:
        df = read_bbref(DATA / f"playoff-stats/player-stats/{f}.csv").groupby("pid").first()
        df = df.add_prefix(f"{name}:")
        out = df if out is None else out.join(df, how="inner", rsuffix="_dup")
    out = out.reindex(columns=columns)
    for c in out.columns:
        num = pd.to_numeric(out[c], errors="coerce")
        if num.notna().any():
            out[c] = num
    return out


# Features: add / remove freely. Each feature = (group, key, fn(df)->(value, n)).
#    value is the stat, n is the sample size behind it (used as its weight).
#    transform: "logit" for shares/percentages in [0,1], "sqrt" for rates, None otherwise.

@dataclass
class Feature:
    group: str
    key: str
    label: str
    fn: callable
    transform: str | None = None


def _share(col, n_col):
    return lambda d: (d[col], d[n_col])


def _pct(col, n_col):          # Basketball-Reference % columns on a 0-100 scale
    return lambda d: (d[col] / 100.0, d[n_col])


def _per36(col):
    return lambda d: (d[col] / d["t:MP"] * 36.0, d["t:MP"])


def _zone(z):
    return lambda d: (d[f"sh:{z}"], d["t:FGA"])


FEATURES = [
    # where you shoot from
    Feature("shot_diet", "dist", "Avg shot distance (ft)", lambda d: (d["sh:Dist."], d["t:FGA"])),
    *[Feature("shot_diet", f"fga_{z}", f"Share of FGA {z}", _zone(z), "logit")
      for z in ["0-3", "3-10", "10-16", "16-3P", "3P"]],
    Feature("shot_diet", "dunk_share", "Dunks / FGA", _share("sh:%FGA", "t:FGA"), "logit"),
    Feature("shot_diet", "corner3_share", "Corner 3s / 3PA", _share("sh:%3PA", "t:3PA"), "logit"),
    # how shots get created
    Feature("creation", "ast2", "% of 2P makes assisted", _share("sh:2P.2", "t:2P"), "logit"),
    Feature("creation", "ast3", "% of 3P makes assisted", _share("sh:3P.2", "t:3P"), "logit"),
    Feature("creation", "AST%", "Assist %", _pct("adv:AST%", "t:MP"), "logit"),
    Feature("creation", "USG%", "Usage %", _pct("adv:USG%", "t:MP"), "logit"),
    Feature("creation", "pga", "Points generated by assists / 36", _per36("pbp:PGA"), "sqrt"),
    # glass and defence
    Feature("glass_defense", "ORB%", "Off. rebound %", _pct("adv:ORB%", "t:MP"), "logit"),
    Feature("glass_defense", "DRB%", "Def. rebound %", _pct("adv:DRB%", "t:MP"), "logit"),
    Feature("glass_defense", "STL%", "Steal %", _pct("adv:STL%", "t:MP"), "logit"),
    Feature("glass_defense", "BLK%", "Block %", _pct("adv:BLK%", "t:MP"), "logit"),
    Feature("glass_defense", "pf100", "Fouls / 100 poss", lambda d: (d["p100:PF"], d["t:MP"]), "sqrt"),
    # "Fouls Drawn: Off." = offensive fouls this player drew, i.e. charges taken on defence
    Feature("glass_defense", "charges", "Charges drawn / 36", _per36("pbp:Off..1"), "sqrt"),
    # contact
    Feature("contact", "FTr", "FT rate (FTA/FGA)", lambda d: (d["adv:FTr"], d["t:FGA"]), "sqrt"),
    Feature("contact", "shoot_fouls_drawn", "Shooting fouls drawn / 36", _per36("pbp:Shoot.1"), "sqrt"),
    Feature("contact", "and1", "And-ones / 36", _per36("pbp:And1"), "sqrt"),
    Feature("contact", "blkd", "Shots blocked / 36", _per36("pbp:Blkd"), "sqrt"),
    # OPTIONAL (not embedded unless --give): how WELL you shoot
    *[Feature("efficiency", f"fg_{z}", f"FG% {z}",
              (lambda z: lambda d: (d[f"sh:{z}.1"], d["t:FGA"] * d[f"sh:{z}"]))(z), "logit")
      for z in ["0-3", "3-10", "10-16", "16-3P", "3P"]],
    Feature("efficiency", "TS%", "True shooting %", lambda d: (d["adv:TS%"], d["t:FGA"]), "logit"),
    Feature("efficiency", "FT%", "Free throw %", lambda d: (d["t:FT%"], d["t:FTA"]), "logit"),
    # OPTIONAL: ball security. Turnovers are execution (one of the Four Factors), not style:
    # with them embedded, the turnover-heavy axes correlated with team net rating.
    Feature("ball_security", "TOV%", "Turnover %", _pct("adv:TOV%", "t:MP"), "logit"),
    Feature("ball_security", "bad_pass", "Bad-pass TOs / 36", _per36("pbp:BadPass"), "sqrt"),
    Feature("ball_security", "lost_ball", "Lost-ball TOs / 36", _per36("pbp:LostBall"), "sqrt"),
    # OPTIONAL: volume
    Feature("volume", "pts100", "Points / 100 poss", lambda d: (d["p100:PTS"], d["t:MP"]), "sqrt"),
    Feature("volume", "fga100", "FGA / 100 poss", lambda d: (d["p100:FGA"], d["t:MP"]), "sqrt"),
]
DEFAULT_GROUPS = ["shot_diet", "creation", "glass_defense", "contact"]


def build_matrix(df: pd.DataFrame, feats: list[Feature]):
    """-> X (n,d) transformed values, N (n,d) sample sizes, mask (n,d)."""
    X, N = [], []
    for f in feats:
        v, n = f.fn(df)
        v = pd.to_numeric(v, errors="coerce").to_numpy(float, copy=True)
        n = pd.to_numeric(n, errors="coerce").fillna(0).to_numpy(float, copy=True)
        if f.transform == "logit":
            eps = 0.5 / np.maximum(n, 1.0)
            v = np.clip(v, eps, 1 - eps)
            v = np.log(v / (1 - v))
        elif f.transform == "sqrt":
            v = np.sqrt(np.maximum(v, 0))
        v[n <= 0] = np.nan
        X.append(v)
        N.append(n)
    X, N = np.stack(X, 1), np.stack(N, 1)
    return X, N, ~np.isnan(X)


# The model: weighted, masked probabilistic PCA (marginal-likelihood fit in JAX)
#    x_ij = mu_j + z_i . w_j + noise,  noise var = s2_j / weight_ij,  z_i ~ N(0, I)

@dataclass
class Fit:
    mu: np.ndarray       # (d,)
    W: np.ndarray        # (d,k)
    s2: np.ndarray       # (d,)
    center: np.ndarray   # (d,) standardisation
    scale: np.ndarray    # (d,)
    wref: np.ndarray     # (d,) weight normaliser


def standardize(X, M, N):
    w = np.where(M, N, 0.0)
    center = np.nansum(np.where(M, X, 0) * w, 0) / w.sum(0)
    var = np.nansum(np.where(M, (X - center) ** 2, 0) * w, 0) / w.sum(0)
    scale = np.sqrt(np.maximum(var, 1e-8))
    wref = np.array([np.median(N[M[:, j], j]) for j in range(X.shape[1])])
    return center, scale, wref


def _weights(N, M, wref):
    return np.where(M, np.clip(N / wref, 0.02, 5.0), 0.0)


S2_FLOOR = 0.05   # minimum noise variance per feature (standardised units)


def fit_model(X, M, N, k, steps=2500, lr=0.02, seed=0) -> Fit:
    """Fit W, mu, s2 by the marginal likelihood (z integrated out).

    Fitting z jointly (MAP) overfits once k >= 6: the extra axes memorise noise
    and held-out error jumps. Integrating z out avoids that. Per player the
    covariance is W W^T + diag(noise), handled in k x k form (Woodbury), the
    same algebra as encode(). Missing entries have zero precision, so they drop out.
    """
    center, scale, wref = standardize(X, M, N)
    Xs = np.where(M, (X - center) / scale, 0.0)
    Wt = _weights(N, M, wref)
    n, d = X.shape
    # Initialise from SVD of the mean-imputed matrix (plus a little noise when seed != 0).
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    W0 = Vt[:k].T * S[:k] / np.sqrt(n)
    if seed:
        W0 = W0 + 0.05 * np.random.default_rng(seed).normal(size=W0.shape)
    params = {"W": jnp.asarray(W0), "mu": jnp.zeros(d), "log_s2": jnp.full(d, np.log(0.3))}
    Xj, Wj = jnp.asarray(Xs), jnp.asarray(Wt)
    Mj = jnp.asarray(M, dtype=jnp.float32)
    eye = jnp.eye(k)

    def loss(p):
        # Noise floor: without it some features' noise goes to 0 (fit perfectly)
        # and every player's uncertainty collapses.
        s2 = S2_FLOOR + jnp.exp(p["log_s2"])
        P = Mj * Wj / s2                                     # per-entry precision, 0 if missing
        r = Mj * (Xj - p["mu"])
        A = jnp.einsum("nd,dk,dl->nkl", P, p["W"], p["W"]) + eye
        b = jnp.einsum("nd,dk->nk", P * r, p["W"])
        La = jnp.linalg.cholesky(A)
        c = jax.scipy.linalg.solve_triangular(La, b[..., None], lower=True)[..., 0]
        quad = (P * r ** 2).sum(1) - (c ** 2).sum(1)
        logdet = -jnp.where(Mj > 0, jnp.log(jnp.maximum(P, 1e-12)), 0.0).sum(1) \
            + 2 * jnp.log(jnp.diagonal(La, axis1=1, axis2=2)).sum(1)
        return 0.5 * (quad + logdet).mean() + 1e-3 * (p["W"] ** 2).sum()

    opt = optax.adam(lr)

    @jax.jit
    def run(p):
        s = opt.init(p)

        def body(c, _):
            p, s = c
            g = jax.grad(loss)(p)
            u, s = opt.update(g, s)
            return (optax.apply_updates(p, u), s), None

        (p, _), _ = jax.lax.scan(body, (p, s), None, length=steps)
        return p

    p = run(params)
    return Fit(np.asarray(p["mu"]), np.asarray(p["W"]), S2_FLOOR + np.exp(np.asarray(p["log_s2"])),
               center, scale, wref)


def encode(fit: Fit, X, M, N):
    """Closed-form posterior mean and sd of z for each row (works for any rows, e.g. stints)."""
    Xs = np.where(M, (X - fit.center) / fit.scale, 0.0) - fit.mu
    prec_w = _weights(N, M, fit.wref) / fit.s2              # (n,d)
    k = fit.W.shape[1]
    A = np.einsum("nd,dk,dl->nkl", prec_w, fit.W, fit.W) + np.eye(k)
    b = np.einsum("nd,dk->nk", prec_w * Xs, fit.W)
    cov = np.linalg.inv(A)
    return np.einsum("nkl,nl->nk", cov, b), np.sqrt(np.diagonal(cov, axis1=1, axis2=2))


def decode(fit: Fit, Z):
    return Z @ fit.W.T + fit.mu   # standardised units


def orient(fit: Fit, Z, keys):
    """Rotate so axes are ordered by the data variance they explain (eigenvectors of W^T W);
    flip signs so axis 1 points toward rim-heavy play."""
    evals, R = np.linalg.eigh(fit.W.T @ fit.W)
    R = R[:, ::-1]
    W = fit.W @ R
    anchor = keys.index("fga_0-3") if "fga_0-3" in keys else 0
    signs = np.sign(W[anchor])
    signs[signs == 0] = 1
    for j in range(1, len(signs)):   # other axes: largest loading positive
        signs[j] = np.sign(W[np.argmax(np.abs(W[:, j])), j]) or 1
    R = R * signs
    fit.W = fit.W @ R
    return Z @ R, evals[::-1] / evals.sum()


K_TOLERANCE = 0.02   # choose the smallest k within 2% of the best held-out error

# Hand-written names for the default embedding (DEFAULT_GROUPS, auto k = 8), (negative end, positive end).
# Only attached when that configuration is used; features.json also lists each axis's extreme
# players so the names can be checked against the data. Axes 6-8 are small (2-3% each).
AXIS_NAMES = [
    ("perimeter shooter", "rim finisher"),
    ("off-ball spacer", "on-ball downhill scorer"),
    ("shot-maker", "pass-first pest"),
    ("slashing wing", "skilled big"),
    ("mid-range scorer", "connector"),
    ("foul magnet", "low-contact role player"),
    ("veteran polish", "young slasher"),
    ("ground-bound", "long and athletic"),
]


def choose_k(X, M, N, kmax=8, frac=0.1, seed=0, steps=1500, masks=3):
    """Held-out reconstruction error (weighted, standardised) for k = 1..kmax, averaged
    over several random hide-masks so the choice doesn't hinge on one draw."""
    hides = [M & (np.random.default_rng(seed + m).random(M.shape) < frac) for m in range(masks)]
    errs = {}
    for k in range(1, kmax + 1):
        e = []
        for hide in hides:
            f = fit_model(X, M & ~hide, N, k, steps=steps, seed=seed)
            Z, _ = encode(f, X, M & ~hide, N)
            Xs = (X - f.center) / f.scale
            e.append(float(np.sqrt(np.mean(((decode(f, Z) - Xs)[hide]) ** 2))))
        errs[k] = float(np.mean(e))
    best = min(errs.values())
    return min(kk for kk, e in errs.items() if e <= best * (1 + K_TOLERANCE)), errs


def resample_fits(X, M, N, k, Z_ref, runs=10, frac=0.8, steps=1500, seed=0):
    """Refit on random 80% subsets of players, encode everyone, and rotate each
    run onto Z_ref (orthogonal Procrustes). Returns a list of (n,k) arrays."""
    rng = np.random.default_rng(seed + 1000)
    n = len(X)
    out = []
    for b in range(runs):
        sub = rng.choice(n, int(frac * n), replace=False)
        f = fit_model(X[sub], M[sub], N[sub], k, steps=steps, seed=seed + b + 1)
        Zb = encode(f, X, M, N)[0]
        U, _, Vt = np.linalg.svd(Zb.T @ Z_ref)
        out.append(Zb @ (U @ Vt))
    return out


# Analyses (each only reads Z plus an answer key the model never saw)

def loo_linear(Z, y):
    """Leave-one-out predictions of y from a linear readout of Z (exact hat-matrix shortcut)."""
    A = np.c_[Z, np.ones(len(Z))]
    H = A @ np.linalg.pinv(A)
    resid = y - H @ y
    return y - resid / (1 - np.diag(H))


def misfit_set(Z, pos_num, eligible, top=12):
    """Players whose recovered position differs most from the play-by-play one. The linear
    readout is clipped to the 1-5 scale so extrapolation (e.g. "0.7") can't inflate a gap."""
    gap = np.clip(loo_linear(Z, pos_num), 1, 5) - pos_num
    order = [i for i in np.argsort(-np.abs(gap)) if eligible[i]]
    return set(order[:top]), gap


def knn(Z, k=10):
    D = ((Z[:, None] - Z[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, 1)[:, :k]


def analyze_positions(Z, pos_pct, listed):
    num = pos_pct @ np.arange(1, 6)
    pred = loo_linear(Z, num)
    r2 = 1 - ((num - pred) ** 2).sum() / ((num - num.mean()) ** 2).sum()
    nn = knn(Z)
    knn_primary = pos_pct[nn].mean(1).argmax(1)
    truth = pos_pct.argmax(1)
    listed_idx = np.array([POSITIONS.index(p) if p in POSITIONS else -1 for p in listed])
    return {
        "axis_corr": [float(np.corrcoef(Z[:, j], num)[0, 1]) for j in range(Z.shape[1])],
        "loo_r2": float(r2),
        "knn_accuracy": float((knn_primary == truth).mean()),
        "majority_baseline": float(np.bincount(truth).max() / len(truth)),
        "listed_vs_pbp_agreement": float((listed_idx == truth).mean()),
    }, num, pred, knn_primary


def analyze_all_defense(Z, is_def, dpoy_rank):
    idx = np.where(is_def)[0]
    ranks = {}
    for i in idx:
        others = [j for j in idx if j != i]
        direction = Z[others].mean(0) - Z.mean(0)
        score = Z @ direction
        ranks[int(i)] = int((score > score[i]).sum() + 1)
    direction = Z[idx].mean(0) - Z.mean(0)
    score = Z @ direction
    voted = ~np.isnan(dpoy_rank)
    rho = float(pd.Series(-score[voted]).corr(pd.Series(dpoy_rank[voted]), method="spearman")) if voted.sum() > 3 else None
    return {"heldout_ranks": ranks, "median_heldout_rank": float(np.median(list(ranks.values()))),
            "n_players": len(Z), "random_median_rank": len(Z) / 2,
            "dpoy_vote_spearman": rho}, score, direction


def percentile_within(score, groups):
    """Percentile (0-100) of each score among players in the same group."""
    out = np.zeros(len(score))
    for g in np.unique(groups):
        ix = np.where(groups == g)[0]
        ranks = np.argsort(np.argsort(score[ix]))
        out[ix] = 100.0 * ranks / max(len(ix) - 1, 1)
    return out


def remove_direction(Z, y):
    """Project out the direction in Z that best predicts y (e.g. position), so what is
    left can't simply re-find y."""
    Zc = Z - Z.mean(0)
    beta = np.linalg.lstsq(Zc, y - y.mean(), rcond=None)[0]
    d = beta / np.linalg.norm(beta)
    return Z - np.outer(Zc @ d, d)


def shrink_by_minutes(v, mp):
    """Empirical-Bayes shrinkage toward the mean for a stat whose noise variance ~ c / minutes.
    Method of moments: E[(v - mean)^2] = tau2 + c / mp, fitted by least squares on 1/mp."""
    ok = ~np.isnan(v)
    dev2 = (v[ok] - v[ok].mean()) ** 2
    A = np.c_[np.ones(ok.sum()), 1.0 / mp[ok]]
    tau2, c = np.linalg.lstsq(A, dev2, rcond=None)[0]
    tau2, c = max(tau2, 1e-6), max(c, 0.0)
    factor = tau2 / (tau2 + c / mp)
    return v[ok].mean() + factor * (v - v[ok].mean()), factor, {"tau2": float(tau2), "c": float(c)}


def corr_with_p(x, y, n_perm=5000, seed=0):
    """Pearson r and a two-sided permutation p-value."""
    r_ = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([abs(np.corrcoef(x, rng.permutation(y))[0, 1]) for _ in range(n_perm)])
    return r_, float((1 + (null >= abs(r_)).sum()) / (n_perm + 1))


def playoff_rotation(Z, pid_to_i, all_reg, po_tot, n_perm=5000, seed=0):
    """Which styles gain playoff minutes? For every embedded player on a playoff team:
    delta = playoff share of team minutes - regular-season share (both shares over the
    whole roster). Partial correlation of delta with each axis, controlling for
    regular-season share (so "stars play more" can't explain it), with a permutation p-value."""
    rows = []
    for ab in po_tot["Team"].dropna().unique():
        rs = all_reg[all_reg["t:Team"] == ab].groupby("pid")["t:MP"].sum()
        po = po_tot[po_tot["Team"] == ab].groupby("pid")["MP"].sum()
        for p in rs.index:
            if p in pid_to_i:
                rows.append((pid_to_i[p], rs[p] / rs.sum(), po.get(p, 0.0) / po.sum()))
    idx = np.array([i for i, _, _ in rows])
    rs_share = np.array([a for _, a, _ in rows])
    delta = np.array([b for _, _, b in rows]) - rs_share
    A = np.c_[np.ones(len(rows)), rs_share]
    resid = lambda y: y - A @ np.linalg.lstsq(A, y, rcond=None)[0]  # noqa: E731
    d_res = resid(delta)
    out = []
    for j in range(Z.shape[1]):
        c, p = corr_with_p(resid(Z[idx, j]), d_res, n_perm=n_perm, seed=seed)
        out.append({"partial_r": c, "p": p})
    return {"n_players": len(rows), "axes": out,
            "rs_share_vs_delta_r": float(np.corrcoef(rs_share, delta)[0, 1])}, idx, delta


def playoff_style(fit, season, pids, feats, min_po_mp=150, n_perm=5000, seed=0):
    """Place playoff players with the fitted model and compare with their regular season,
    re-encoded from the same restricted feature set so the comparison is like for like.
    Returns (summary, per-player dict pid -> (z_playoff, z_sd))."""
    po = load_playoff_players(season.columns)
    po = po[po["t:MP"] >= min_po_mp]
    po = po[po.index.isin(pids)]
    Xp, Np, Mp = build_matrix(po, feats)
    avail = Mp.any(0)                                        # features the playoff tables supply
    Mp &= avail
    reg = season.loc[po.index]
    Xr, Nr, Mr = build_matrix(reg, feats)
    Mr &= avail
    Zp, Zp_sd = encode(fit, Xp, Mp, Np)
    Zr, _ = encode(fit, Xr, Mr, Nr)
    # how faithful is the restricted placement? compare restricted vs full regular-season z
    Xf, Nf, Mf = build_matrix(reg, feats)
    Zf, _ = encode(fit, Xf, Mf, Nf)
    faithful = [float(np.corrcoef(Zr[:, j], Zf[:, j])[0, 1]) for j in range(Zr.shape[1])]
    diff = Zp - Zr
    same = np.linalg.norm(diff, axis=1)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Zp))
    rand = np.linalg.norm(Zp - Zr[perm], axis=1)
    # paired sign-flip test per axis: does the whole playoff population shift?
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    axes = []
    for j in range(diff.shape[1]):
        obs = diff[:, j].mean()
        null = np.abs((flips * diff[:, j]).mean(1))
        axes.append({"mean_shift": float(obs), "p": float((1 + (null >= abs(obs)).sum()) / (n_perm + 1))})
    summary = {"n_players": len(po), "min_playoff_mp": min_po_mp,
               "features_used": [f.key for f, a in zip(feats, avail) if a],
               "restricted_vs_full_axis_corr": faithful,
               "median_distance_same_player": float(np.median(same)),
               "median_distance_random_pair": float(np.median(rand)),
               "axes": axes}
    per = {p: (Zp[i], Zp_sd[i], Zr[i]) for i, p in enumerate(po.index)}
    return summary, per


def analyze_age(Z, age, is_rookie, n_perm=5000, seed=0):
    """Does any axis track age (never an input), and where do All-Rookie players sit?
    Rookie test: difference in mean z vs everyone else, permutation p-value."""
    rng = np.random.default_rng(seed)
    out = {"age_corr": [], "rookie_mean_diff": []}
    idx = np.where(is_rookie)[0]
    for j in range(Z.shape[1]):
        c, p = corr_with_p(Z[:, j], age, n_perm=n_perm, seed=seed)
        out["age_corr"].append({"r": c, "p": p})
        diff = Z[idx, j].mean() - np.delete(Z[:, j], idx).mean()
        null = []
        for _ in range(n_perm):
            s = rng.choice(len(Z), len(idx), replace=False)
            null.append(abs(Z[s, j].mean() - np.delete(Z[:, j], s).mean()))
        out["rookie_mean_diff"].append({"diff": float(diff),
                                        "p": float((1 + (np.array(null) >= abs(diff)).sum()) / (n_perm + 1))})
    return out


def analyze_dispersion(Z, members, pool, n_perm=5000, seed=0):
    def disp(ix):
        P = Z[ix]
        return float(np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)).sum() / (len(ix) * (len(ix) - 1)))
    obs = disp(members)
    rng = np.random.default_rng(seed)
    null = np.array([disp(rng.choice(pool, len(members), replace=False)) for _ in range(n_perm)])
    return {"observed": obs, "null_mean": float(null.mean()),
            "p_more_spread": float((1 + (null >= obs).sum()) / (n_perm + 1)),
            "p_more_clustered": float((1 + (null <= obs).sum()) / (n_perm + 1))}


def masked_recovery(X, M, N, k, frac=0.2, seed=0, steps=2000, groups=None):
    rng = np.random.default_rng(seed)
    hide = M & (rng.random(M.shape) < frac)
    f = fit_model(X, M & ~hide, N, k, steps=steps, seed=seed)
    Z, _ = encode(f, X, M & ~hide, N)
    Xs = (X - f.center) / f.scale
    err = decode(f, Z) - Xs
    out = {"model_rmse": float(np.sqrt(np.mean(err[hide] ** 2))),
           "column_mean_rmse": float(np.sqrt(np.mean(Xs[hide] ** 2)))}
    if groups is not None:
        out["by_group"] = {}
        for g in sorted(set(groups)):
            sel = hide & (np.array(groups)[None, :] == g)
            if sel.any():
                out["by_group"][g] = {"model": float(np.sqrt(np.mean(err[sel] ** 2))),
                                      "column_mean": float(np.sqrt(np.mean(Xs[sel] ** 2)))}
    return out


# Main

def r(x, nd=4):
    """Round for JSON; NaN -> None."""
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS, help="feature groups to embed")
    ap.add_argument("--give", nargs="*", default=[], help="extra groups or feature keys to add")
    ap.add_argument("--hold", nargs="*", default=[], help="feature keys or groups to leave out")
    ap.add_argument("--k", default="auto", help="latent dimensions, or 'auto'")
    ap.add_argument("--min-mp", type=float, default=300, help="minutes needed to be embedded")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=10, help="resampled refits for stability (0 = skip)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--list", action="store_true", help="list feature keys and exit")
    args = ap.parse_args(argv)

    if args.list:
        for g in dict.fromkeys(f.group for f in FEATURES):
            tag = "" if g in DEFAULT_GROUPS else "  (optional, use --give)"
            print(f"{g}{tag}")
            for f in FEATURES:
                if f.group == g:
                    print(f"   {f.key:<18} {f.label}")
        return

    t0 = time.time()
    wanted = set(args.groups) | set(args.give)
    feats = [f for f in FEATURES
             if (f.group in wanted or f.key in wanted) and f.key not in args.hold and f.group not in args.hold]
    keys = [f.key for f in feats]
    print(f"embedding {len(feats)} features: {', '.join(keys)}")

    season, stints = load_players()
    season = season[season["t:MP"] >= args.min_mp].copy()
    X, N, M = build_matrix(season, feats)
    n = len(season)
    print(f"{n} players with >= {args.min_mp:g} MP; {100 * (~M).mean():.1f}% of entries missing")

    if args.k == "auto":
        k, kerr = choose_k(X, M, N, seed=args.seed)
        print(f"k chosen by held-out error: {k}  ({', '.join(f'{kk}:{e:.3f}' for kk, e in kerr.items())})")
    else:
        k, kerr = int(args.k), {}

    fit = fit_model(X, M, N, k, steps=args.steps, seed=args.seed)
    Z, Zsd = encode(fit, X, M, N)
    Z, var_share = orient(fit, Z, keys)
    Zsd = encode(fit, X, M, N)[1]   # sd on the rotated axes

    # answer keys
    pids = list(season.index)
    names = season["t:Player"].tolist()
    team = season["t:Team"].astype(str).tolist()
    listed = season["t:Pos"].astype(str).tolist()
    pos_pct = season[[f"pbp:{p}%" for p in POSITIONS]].to_numpy(float) / 100.0
    awards = [parse_awards(a) for a in season["t:Awards"]]
    rookies = load_all_rookie()
    for i, nm in enumerate(names):
        awards[i]["all_rookie"] = rookies.get(nm)
    on_court = pd.to_numeric(season["pbp:OnCourt"], errors="coerce").to_numpy(float)
    on_off = pd.to_numeric(season["pbp:On-Off"], errors="coerce").to_numpy(float)
    mp = season["t:MP"].to_numpy(float)
    po_mp = load_playoff_minutes().reindex(pids).fillna(0).to_numpy(float)

    # traded players: team with most minutes, for colouring
    st = stints.reset_index()
    main_team = st[st["pid"].isin(pids)].sort_values("t:MP").groupby("pid")["t:Team"].last()
    team = [main_team.get(p, tm) if tm.endswith("TM") else tm for p, tm in zip(pids, team)]

    # analyses
    print("analysing ...")
    pos_res, pos_num, pos_rec, knn_primary = analyze_positions(Z, pos_pct, listed)
    pos_rec = np.clip(pos_rec, 1, 5)   # R2 above uses the raw readout; displays/misfits use the 1-5 scale
    is_def = np.array([a["all_def"] is not None for a in awards])
    dpoy = np.array([a["votes"].get("DPOY", np.nan) for a in awards], float)
    def_res, def_score, _ = analyze_all_defense(Z, is_def, dpoy)
    # the raw defence direction is mostly "big man"; take the position direction out first
    Z_pc = remove_direction(Z, pos_num)
    def_pc_res, def_pc_score, _ = analyze_all_defense(Z_pc, is_def, dpoy)
    # comparison: an embedding built from the glass/defence features alone (same position control)
    dfeats = [f for f in FEATURES if f.group == "glass_defense" and f.key not in args.hold]
    Xd, Nd, Md = build_matrix(season, dfeats)
    kd, _ = choose_k(Xd, Md, Nd, kmax=min(4, len(dfeats) - 1), seed=args.seed)
    kd = max(kd, 2)   # removing the position direction from a 1-D space would leave nothing
    Zd = encode(fit_model(Xd, Md, Nd, kd, steps=args.steps, seed=args.seed), Xd, Md, Nd)[0]
    def_only_res, def_only_score, _ = analyze_all_defense(remove_direction(Zd, pos_num), is_def, dpoy)
    # display field: defence score ranked only among players with the same primary position
    primary_pos = pos_pct.argmax(1)
    def_pct_pos = percentile_within(def_score, primary_pos)
    is_nba = np.array([a["all_nba"] is not None for a in awards])
    nba_idx = np.where(is_nba)[0]
    pool = np.where(mp >= mp[nba_idx].min())[0]
    nba_res = analyze_dispersion(Z, nba_idx, pool, seed=args.seed)
    non_star = np.array([not a["all_star"] and a["all_nba"] is None for a in awards])

    def twins_of(Zx):
        order = np.argsort(((Zx[:, None] - Zx[None]) ** 2).sum(-1), 1)
        tw = {i: [int(j) for j in order[i, 1:6]] for i in range(n)}
        star = {int(i): int(next(j for j in order[i] if non_star[j] and j != i)) for i in nba_idx}
        return tw, star

    twins, star_twin = twins_of(Z)

    # impact beyond style: on-off (already relative to the player's own team) minus what a
    # linear readout of his style predicts (leave-one-out), then shrunk by minutes since
    # on-off is noisy. How much style explains at all is itself a headline number.
    has_oo = ~np.isnan(on_off)
    on_off_expected = np.full(n, np.nan)
    on_off_expected[has_oo] = loo_linear(Z[has_oo], on_off[has_oo])
    oo = on_off[has_oo]
    style_r2_on_off = float(1 - ((oo - on_off_expected[has_oo]) ** 2).sum() / ((oo - oo.mean()) ** 2).sum())
    beyond_raw = on_off - on_off_expected
    beyond, shrink_factor, shrink_par = shrink_by_minutes(beyond_raw, mp)
    IMPACT_MIN_MP = 1500

    # stability: refit on resampled players and count how often each headline result reappears
    eligible = mp >= 1000
    misfits_main, gap_main = misfit_set(Z, pos_pct @ np.arange(1, 6), eligible)
    stab = None
    if args.runs:
        print(f"stability: {args.runs} refits on 80% of players ...")
        boots = resample_fits(X, M, N, k, Z, runs=args.runs, seed=args.seed)
        num = pos_pct @ np.arange(1, 6)
        misfit_count = np.zeros(n)
        gaps = []
        twin_hit = np.zeros(n)
        star_hit = {i: 0 for i in star_twin}
        run_r2, run_def, run_def_pc = [], [], []
        for Zb in boots:
            ms, g = misfit_set(Zb, num, eligible)
            misfit_count[list(ms)] += 1
            gaps.append(g)
            tw, sb = twins_of(Zb)
            for i in range(n):
                twin_hit[i] += twins[i][0] in tw[i]
            for i in star_twin:
                star_hit[i] += sb[i] == star_twin[i]
            run_r2.append(analyze_positions(Zb, pos_pct, listed)[0]["loo_r2"])
            run_def.append(analyze_all_defense(Zb, is_def, dpoy)[0]["median_heldout_rank"])
            run_def_pc.append(analyze_all_defense(remove_direction(Zb, num), is_def, dpoy)[0]["median_heldout_rank"])
        R = len(boots)
        stab = {"runs": R,
                "z_sd": np.std(np.stack(boots), axis=0),
                "misfit_freq": misfit_count / R,
                "gap_mean": np.mean(gaps, axis=0), "gap_sd": np.std(gaps, axis=0),
                "twin_freq": twin_hit / R,
                "star_twin_freq": {i: c / R for i, c in star_hit.items()},
                "pos_r2": (float(np.mean(run_r2)), float(np.std(run_r2))),
                "def_rank": (float(np.mean(run_def)), float(np.std(run_def))),
                "def_rank_pc": (float(np.mean(run_def_pc)), float(np.std(run_def_pc)))}
    groups_of = [f.group for f in feats]
    mask_res = masked_recovery(X, M, N, k, seed=args.seed, groups=groups_of)

    # traded players: does style follow the player? (encode each stint with the fitted model)
    traded = st[st["pid"].isin(pids) & (st["row"] > 0) & (st["t:MP"] >= 150)]
    trade_res = None
    if len(traded) > 4:
        Xs_, Ns_, Ms_ = build_matrix(traded.set_index("pid"), feats)
        Zs_, _ = encode(fit, Xs_, Ms_, Ns_)   # fit.W is already rotated, so same axes as Z
        same, rand = [], []
        by = traded["pid"].to_numpy()
        rng = np.random.default_rng(args.seed)
        for p in np.unique(by):
            ix = np.where(by == p)[0]
            if len(ix) >= 2:
                same.append(np.linalg.norm(Zs_[ix[0]] - Zs_[ix[1]]))
        allpairs = rng.integers(0, len(Zs_), (2000, 2))
        rand = [np.linalg.norm(Zs_[a] - Zs_[b]) for a, b in allpairs if by[a] != by[b]]
        trade_res = {"n_players": len(same), "median_same_player": float(np.median(same)),
                     "median_random_pair": float(np.median(rand))}

    # teams: minutes-weighted centroids (regular season and playoffs)
    teams = load_teams(season)
    pid_to_i = {p: i for i, p in enumerate(pids)}
    reg = st[st["pid"].isin(pids) & ~st["t:Team"].astype(str).str.match(r"\dTM")]
    po_tot = read_bbref(DATA / "playoff-stats/player-stats/totals.csv")
    team_json = []
    for ab, trow in teams.iterrows():
        ros = reg[reg["t:Team"] == ab]
        w = ros["t:MP"].to_numpy(float)
        zi = np.array([Z[pid_to_i[p]] for p in ros["pid"]])
        cen = (w[:, None] * zi).sum(0) / w.sum()
        entry = {"team": ab, "name": trow["name"], "conference": trow["conference"],
                 "division": trow["division"], "W": r(trow["W"]), "L": r(trow["L"]),
                 "NRtg": r(trow["NRtg"]), "SRS": r(trow["SRS"]),
                 "centroid": r(cen),
                 "roster": [{"pid": p, "share": r(m / w.sum())} for p, m in zip(ros["pid"], w)]}
        pr = po_tot[(po_tot["Team"] == ab) & po_tot["pid"].isin(pids)]
        if len(pr):
            pw = pr["MP"].to_numpy(float)
            pz = np.array([Z[pid_to_i[p]] for p in pr["pid"]])
            entry["playoff_centroid"] = r((pw[:, None] * pz).sum(0) / pw.sum())
        team_json.append(entry)
    nrtg = np.array([t["NRtg"] for t in team_json], float)
    cents = np.array([t["centroid"] for t in team_json], float)
    team_style = [corr_with_p(cents[:, j], nrtg, seed=args.seed) for j in range(k)]
    team_style_corr = [c for c, _ in team_style]
    # the direct test: roster minute-weighted turnover stats (never embedded by default) vs net rating
    ball_security = {}
    for f in FEATURES:
        if f.group != "ball_security":
            continue
        v = pd.to_numeric(f.fn(season)[0], errors="coerce").to_numpy(float)
        tv = []
        for t in team_json:
            ros = reg[reg["t:Team"] == t["team"]]
            w = ros["t:MP"].to_numpy(float)
            vi = np.array([v[pid_to_i[p]] for p in ros["pid"]])
            ok = ~np.isnan(vi)
            tv.append((w[ok] * vi[ok]).sum() / w[ok].sum())
        ball_security[f.key] = corr_with_p(np.array(tv), nrtg, seed=args.seed)
    shift = np.array([np.array(t["playoff_centroid"]) - np.array(t["centroid"])
                      for t in team_json if "playoff_centroid" in t])
    all_reg = st[~st["t:Team"].astype(str).str.match(r"\dTM")]   # every player, for roster totals
    rot_res, rot_idx, rot_delta = playoff_rotation(Z, pid_to_i, all_reg, po_tot, seed=args.seed)
    po_share_delta = np.full(n, np.nan)
    po_share_delta[rot_idx] = rot_delta
    pstyle_res, pstyle = playoff_style(fit, season, pids, feats, seed=args.seed)
    age = season["t:Age"].to_numpy(float)
    is_rookie = np.array([a["all_rookie"] is not None for a in awards])
    age_res = analyze_age(Z, age, is_rookie, n_perm=2000, seed=args.seed)

    # JSON
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    jitter = rng.uniform(-0.4, 0.4, n)
    players = []
    for i, p in enumerate(pids):
        a = awards[i]
        players.append({
            "pid": p, "name": names[i], "team": team[i], "age": r(season["t:Age"].iloc[i]),
            "mp": r(mp[i]), "playoff_mp": r(po_mp[i]), "playoff_share_change": r(po_share_delta[i]),
            "pos_listed": listed[i],
            "pos_pbp": {q: r(100 * pos_pct[i, j], 1) for j, q in enumerate(POSITIONS)},
            "pos_num_pbp": r(pos_num[i], 3), "pos_num_latent": r(pos_rec[i], 3),
            "pos_gap": r(pos_rec[i] - pos_num[i], 3),
            "pos_primary_recovered": POSITIONS[knn_primary[i]],
            "z": r(Z[i]), "z_sd": r(Zsd[i]),
            "columns_xy": [POSITIONS.index(listed[i]) if listed[i] in POSITIONS else 2, r(jitter[i], 3)],
            "awards": a,
            "def_score": r(def_score[i]), "def_rank": int((def_score > def_score[i]).sum() + 1),
            "def_score_pc": r(def_pc_score[i]), "def_rank_pc": int((def_pc_score > def_pc_score[i]).sum() + 1),
            "def_pct_in_position": r(def_pct_pos[i], 1),
            "twins": [pids[j] for j in twins[i]],
            "impact": {"on_court": r(on_court[i]), "on_off": r(on_off[i]),
                       "on_off_expected_from_style": r(on_off_expected[i]),
                       "beyond_style": r(beyond[i]), "beyond_style_raw": r(beyond_raw[i]),
                       "shrink": r(shrink_factor[i], 3)},
        })
        if p in pstyle:
            zp, zp_sd, zr = pstyle[p]
            # compare playoff_z with regular_z_same_features, not with z (different feature sets)
            players[-1]["playoff_style"] = {"z": r(zp), "z_sd": r(zp_sd), "regular_z_same_features": r(zr)}
        if stab:
            players[-1]["stability"] = {"z_sd": r(stab["z_sd"][i]), "twin_freq": r(stab["twin_freq"][i], 2),
                                        "misfit_freq": r(stab["misfit_freq"][i], 2)}
    feature_json = [{"key": f.key, "label": f.label, "group": f.group, "loadings": r(fit.W[j])}
                    for j, f in enumerate(feats)]
    named = (not args.give and not args.hold and set(args.groups) == set(DEFAULT_GROUPS)
             and k == len(AXIS_NAMES))
    regulars = np.where(mp >= 1500)[0]
    axes = []
    for j in range(k):
        order = np.argsort(fit.W[:, j])
        by_z = regulars[np.argsort(Z[regulars, j])]
        axes.append({"id": j, "var_share": r(var_share[j]),
                     "name_neg": AXIS_NAMES[j][0] if named else None,
                     "name_pos": AXIS_NAMES[j][1] if named else None,
                     "top_pos": [[keys[o], r(fit.W[o, j])] for o in order[::-1][:4]],
                     "top_neg": [[keys[o], r(fit.W[o, j])] for o in order[:4]],
                     "players_neg": [pids[i] for i in by_z[:4]],
                     "players_pos": [pids[i] for i in by_z[::-1][:4]]})
    # misfits: with resampling, keep players who are misfits in at least half the refits
    if stab:
        cand = [i for i in range(n) if eligible[i] and stab["misfit_freq"][i] >= 0.5]
        misfit_ids = sorted(cand, key=lambda i: (-stab["misfit_freq"][i], -abs(stab["gap_mean"][i])))[:12]
    else:
        misfit_ids = sorted(misfits_main, key=lambda i: -abs(gap_main[i]))
    analysis = {
        "positions": {**pos_res, **({"loo_r2_resampled": r(stab["pos_r2"])} if stab else {})},
        "misfits": [{"pid": pids[i], "name": names[i], "listed": listed[i], "pbp": r(pos_num[i], 2),
                     "latent": r(pos_rec[i], 2),
                     **({"freq": r(stab["misfit_freq"][i], 2), "gap_sd": r(stab["gap_sd"][i], 2)} if stab else {})}
                    for i in misfit_ids],
        "all_defense": {**def_res,
                        "heldout_ranks": {pids[i]: v for i, v in def_res["heldout_ranks"].items()},
                        "latent_top15": [pids[i] for i in np.argsort(-def_score)[:15]],
                        **({"median_heldout_rank_resampled": r(stab["def_rank"])} if stab else {}),
                        "position_controlled": {
                            **def_pc_res,
                            "heldout_ranks": {pids[i]: v for i, v in def_pc_res["heldout_ranks"].items()},
                            "latent_top15": [pids[i] for i in np.argsort(-def_pc_score)[:15]],
                            **({"median_heldout_rank_resampled": r(stab["def_rank_pc"])} if stab else {})},
                        "within_position": {
                            "honoree_pct": {pids[i]: r(def_pct_pos[i], 1) for i in np.where(is_def)[0]},
                            "median_honoree_pct": r(float(np.median(def_pct_pos[is_def])), 1),
                            "corr_score_with_position": r(float(np.corrcoef(def_score, pos_num)[0, 1]))},
                        "defense_only_embedding": {
                            **def_only_res, "k": kd, "features": [f.key for f in dfeats],
                            "heldout_ranks": {pids[i]: v for i, v in def_only_res["heldout_ranks"].items()},
                            "latent_top15": [pids[i] for i in np.argsort(-def_only_score)[:15]]}},
        "all_nba": {**nba_res, "star_twins": {pids[i]: pids[j] for i, j in star_twin.items()},
                    **({"star_twin_freq": {pids[i]: r(f, 2) for i, f in stab["star_twin_freq"].items()}}
                       if stab else {})},
        "masked_recovery": mask_res,
        "impact": {
            "style_r2_on_off": r(style_r2_on_off),
            "basis": ("on-off minus a leave-one-out linear readout of on-off from the style coordinates, "
                      "shrunk toward the mean by minutes (empirical Bayes)"),
            "shrinkage": {**{kk: r(v) for kk, v in shrink_par.items()}, "min_mp": IMPACT_MIN_MP},
            "top": [{"pid": pids[i], "name": names[i], "team": team[i], "mp": r(mp[i]),
                     "on_off": r(on_off[i]), "expected": r(on_off_expected[i]),
                     "beyond_style": r(beyond[i]), "beyond_style_raw": r(beyond_raw[i]),
                     "shrink": r(shrink_factor[i], 3)}
                    for i in np.argsort(-np.nan_to_num(beyond, nan=-99)) if mp[i] >= IMPACT_MIN_MP][:12]},
        "trades": trade_res,
        "teams": {"centroid_vs_nrtg_corr": r(team_style_corr),
                  "centroid_vs_nrtg_p": r([p for _, p in team_style]),
                  "ball_security_vs_nrtg": {kk: {"r": r(c), "p": r(p)} for kk, (c, p) in ball_security.items()},
                  "mean_playoff_shift": r(shift.mean(0)) if len(shift) else None,
                  "playoff_shift_teams_up": [int((shift[:, j] > 0).sum()) for j in range(k)] if len(shift) else None},
        "age_rookies": {"age_corr": [{kk: r(v) for kk, v in a.items()} for a in age_res["age_corr"]],
                        "all_rookie_mean_diff": [{kk: r(v) for kk, v in a.items()}
                                                 for a in age_res["rookie_mean_diff"]],
                        "all_rookie": [pids[i] for i in np.where(is_rookie)[0]]},
        "playoff_style": {**pstyle_res,
                          "restricted_vs_full_axis_corr": r(pstyle_res["restricted_vs_full_axis_corr"]),
                          "axes": [{kk: r(v) for kk, v in a.items()} for a in pstyle_res["axes"]]},
        "playoff_rotation": {**rot_res, "axes": [{kk: r(v) for kk, v in a.items()} for a in rot_res["axes"]]},
    }
    manifest = {"season": "2025-26", "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "k": k, "k_errors": {str(kk): r(e) for kk, e in kerr.items()},
                "min_mp": args.min_mp, "n_players": n, "features": keys,
                "stability_runs": stab["runs"] if stab else 0,
                "files": ["players.json", "features.json", "teams.json", "analysis.json"]}
    for name, obj in [("manifest", manifest), ("players", players),
                      ("features", {"features": feature_json, "axes": axes}),
                      ("teams", team_json), ("analysis", analysis)]:
        (out / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=r))

    # summary
    print(f"\nk = {k}; variance share per axis: {', '.join(f'{v:.0%}' for v in var_share)}")
    pid_name = dict(zip(pids, names))
    for a in axes:
        label = f"{a['name_neg']} <-> {a['name_pos']}" if a["name_neg"] else "(unnamed)"
        print(f"  axis {a['id'] + 1} {label}: "
              f"{', '.join(pid_name[p] for p in a['players_neg'][:3])} <-> "
              f"{', '.join(pid_name[p] for p in a['players_pos'][:3])}")
    print(f"positions: axis corr {', '.join(f'{c:+.2f}' for c in pos_res['axis_corr'])} | "
          f"LOO R2 {pos_res['loo_r2']:.2f} | kNN {pos_res['knn_accuracy']:.0%} "
          f"(majority {pos_res['majority_baseline']:.0%}; listed-vs-pbp {pos_res['listed_vs_pbp_agreement']:.0%})")
    if stab:
        print(f"  resampled ({stab['runs']} refits): LOO R2 {stab['pos_r2'][0]:.2f} +/- {stab['pos_r2'][1]:.2f}")
    print("misfits:", ", ".join(f"{m['name']} ({m['listed']} {m['pbp']}->{m['latent']}"
                                + (f", {m['freq']:.0%} of refits" if stab else "") + ")"
                                for m in analysis["misfits"][:8]))
    print(f"All-Defensive: median held-out rank {def_res['median_heldout_rank']:.0f} of {n} "
          f"(random ~{n // 2}); DPOY vote spearman {def_res['dpoy_vote_spearman']}")
    if stab:
        print(f"  resampled: median held-out rank {stab['def_rank'][0]:.0f} +/- {stab['def_rank'][1]:.0f}")
    print("latent All-Defensive top 10:", ", ".join(names[i] for i in np.argsort(-def_score)[:10]))
    print(f"  position-controlled: median held-out rank {def_pc_res['median_heldout_rank']:.0f}"
          + (f" (resampled {stab['def_rank_pc'][0]:.0f} +/- {stab['def_rank_pc'][1]:.0f})" if stab else "")
          + f"; DPOY spearman {def_pc_res['dpoy_vote_spearman']:.2f}")
    print("  position-controlled top 10:", ", ".join(names[i] for i in np.argsort(-def_pc_score)[:10]))
    print(f"  held-out ranks: " + ", ".join(f"{names[i]} {v}" for i, v in def_pc_res["heldout_ranks"].items()))
    print(f"  within position: defence score vs position r = {np.corrcoef(def_score, pos_num)[0, 1]:+.2f}; "
          f"honorees' percentile among same position: "
          + ", ".join(f"{names[i]} {def_pct_pos[i]:.0f}" for i in np.where(is_def)[0])
          + f" (median {np.median(def_pct_pos[is_def]):.0f})")
    print(f"  defence-only embedding (k={kd}, position-controlled): median held-out rank "
          f"{def_only_res['median_heldout_rank']:.0f}; top 10: "
          + ", ".join(names[i] for i in np.argsort(-def_only_score)[:10]))
    print(f"All-NBA dispersion {nba_res['observed']:.2f} vs null {nba_res['null_mean']:.2f} "
          f"(p spread {nba_res['p_more_spread']:.3f}, p clustered {nba_res['p_more_clustered']:.3f})")
    print("star twins:", ", ".join(f"{names[i]}->{names[j]}"
                                   + (f" ({stab['star_twin_freq'][i]:.0%})" if stab else "")
                                   for i, j in list(star_twin.items())[:8]))
    print(f"impact: style explains {style_r2_on_off:.1%} of on-off (leave-one-out R2)")
    print(f"  beyond style (shrunk, >= {IMPACT_MIN_MP} MP):",
          ", ".join(f"{h['name']} {h['team']} {h['beyond_style']:+.1f} (raw {h['beyond_style_raw']:+.1f})"
                    for h in analysis["impact"]["top"][:8]))
    print(f"masked recovery RMSE {mask_res['model_rmse']:.2f} vs column mean {mask_res['column_mean_rmse']:.2f}")
    if trade_res:
        print(f"traded players ({trade_res['n_players']}): same-player stint distance "
              f"{trade_res['median_same_player']:.2f} vs random {trade_res['median_random_pair']:.2f}")
    print("team centroid vs NRtg per axis: " + ", ".join(f"{c:+.2f} (p={p:.3f})" for c, p in team_style))
    print(f"playoff rotation ({rot_res['n_players']} players): minutes-share change vs axis, "
          "controlling for regular-season share: "
          + ", ".join(f"ax{j + 1} {a['partial_r']:+.2f} (p={a['p']:.3f})" for j, a in enumerate(rot_res["axes"])))
    print("age (never an input) vs axis: "
          + ", ".join(f"ax{j + 1} {a['r']:+.2f} (p={a['p']:.3f})" for j, a in enumerate(age_res["age_corr"])))
    print(f"All-Rookie ({is_rookie.sum()}) mean z minus everyone else: "
          + ", ".join(f"ax{j + 1} {a['diff']:+.2f} (p={a['p']:.3f})" for j, a in enumerate(age_res["rookie_mean_diff"])))
    ps = pstyle_res
    print(f"playoff style ({ps['n_players']} players >= {ps['min_playoff_mp']} playoff MP, "
          f"{len(ps['features_used'])} features): same-player distance {ps['median_distance_same_player']:.2f} "
          f"vs random {ps['median_distance_random_pair']:.2f}; mean shift per axis "
          + ", ".join(f"ax{j + 1} {a['mean_shift']:+.2f} (p={a['p']:.3f})" for j, a in enumerate(ps["axes"]))
          + "; restricted-vs-full axis corr " + ", ".join(f"{c:.2f}" for c in ps["restricted_vs_full_axis_corr"]))
    print("ball security (roster-weighted) vs NRtg: "
          + ", ".join(f"{kk} {c:+.2f} (p={p:.3f})" for kk, (c, p) in ball_security.items()))
    print(f"wrote {out}/ in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
