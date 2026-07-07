# -*- coding: utf-8 -*-
"""ALM 계산 모듈: 시장 모형(Vasicek/GBM) 시뮬레이션과 포트폴리오 유틸.
Streamlit UI에 의존하지 않는 순수 함수만 둔다.
"""
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.optimize import minimize


def millions_formatter(x, pos):
    return f'{x / 1_000_000:,.0f}'

# ---------- Vasicek Calibration (AR(1) 등가 OLS) ----------
def vasicek_calibrate(rates: pd.Series, dt=1/12):
    r = np.asarray(rates, dtype=float)
    x = r[:-1]
    y = r[1:]
    # y = a + b*x + e  → b = exp(-kappa*dt), a = theta*(1 - b)
    X = np.column_stack([np.ones_like(x), x])  # [1, x]
    beta, _, _, _ = lstsq(X, y, rcond=None)
    a_hat, b_hat = beta
    # 방어: b_hat이 (0,1) 밖이면 클리핑
    b_hat = np.clip(b_hat, 1e-6, 1 - 1e-6)
    kappa = -np.log(b_hat) / dt
    theta = a_hat / (1 - b_hat)
    # 잔차로 sigma 추정
    resid = y - (a_hat + b_hat * x)
    sigma_eps = np.std(resid, ddof=1)
    # eps ~ N(0, sigma_eps^2) 이고, 연속시간 sigma는:
    sigma = sigma_eps * np.sqrt(2 * kappa / (1 - np.exp(-2 * kappa * dt)))
    r0 = r[-1]
    return kappa, theta, sigma, r0


# ---------- Vasicek Simulation (Euler) ----------
def vasicek_simul(r0, kappa, theta, sigma, simulation_times, T, dt, epsilon_var):
    # --- scalarize (핵심) ---
    r0    = float(np.asarray(r0).squeeze())
    kappa = float(np.asarray(kappa).squeeze())
    theta = float(np.asarray(theta).squeeze())   # ★ 여기 때문에 에러 났던 것
    sigma = float(np.asarray(sigma).squeeze())
    dt    = float(np.asarray(dt).squeeze())

    eps = np.asarray(epsilon_var, dtype=np.float64)
    if eps.shape != (simulation_times, T):
        raise ValueError(f"epsilon_var shape mismatch: expected {(simulation_times, T)}, got {eps.shape}")

    out = np.zeros((simulation_times, T), dtype=np.float64)
    out[:, 0] = r0
    sdt = np.sqrt(dt)

    for i in range(simulation_times):
        for j in range(1, T):
            dW = float(eps[i, j])  # 스칼라 보장
            out[i, j] = out[i, j-1] + kappa*(theta - out[i, j-1])*dt + sigma*sdt*dW

    return out


# ---------- Vasicek Objective (평균레벨 맞추기) ----------
def vasicek_objective(kappa, theta, sigma, r0, simulation_times, T, dt, epsilon_var, target_level):
    kappa = float(np.maximum(kappa[0], 1e-6))
    sims = vasicek_simul(r0, kappa, theta, sigma, simulation_times, T, dt, epsilon_var)
    # 방법 A: 마지막 시점 평균레벨을 타깃으로
    mean_terminal = sims[:, -1].mean()
    return (mean_terminal - target_level)**2


# ---------- GBM Simulation ----------
def gbm_simul(S0, mu, sigma, simulation_times, T, dt, epsilon_var):
    """
    dS/S = mu*dt + sigma*dW  → S_t = S0 * exp( (mu - 0.5*sigma^2)*t + sigma*W_t )
    epsilon_var: (simulation_times x T) dW ~ N(0,1)
    """
    eps = np.asarray(epsilon_var, dtype=float)  # shape: (simulation_times, T)
    assert eps.shape == (simulation_times, T), "epsilon_var shape mismatch"

    out = np.zeros((simulation_times, T), dtype=float)
    out[:, 0] = S0
    sdt = np.sqrt(dt)
    drift = (mu - 0.5 * sigma**2) * dt

    for i in range(simulation_times):
        logS = np.log(S0)
        for j in range(1, T):
            dW = eps[i, j] * sdt
            logS = logS + drift + sigma * dW
            out[i, j] = np.exp(logS)
    return out


# ---------- GBM Objective (목표 수익률 매칭) ----------
def gbm_objective(mu, target_annual_return, S0, sigma, simulation_times, T, dt, epsilon_var):
    mu = float(mu[0])  # 월 로그수익률
    sims = gbm_simul(S0, mu, sigma, simulation_times, T, dt, epsilon_var)
    mean_terminal = sims[:, -1].mean()

    # 목표지수 수준: 연 수익률 target → 월 로그수익률로 변환
    monthly_target_return = (1 + target_annual_return) ** (1/12) - 1
    target_log_ret = np.log(1 + monthly_target_return)
    target_level = S0 * np.exp(target_log_ret * T)

    return (mean_terminal - target_level)**2


# ---------- Wrapper: find_optimal_mu ----------
def find_optimal_mu(target_annual_return, S0, sigma, simulation_times, T, dt, epsilon_var, init_mu=None):
    if init_mu is None:
        init_mu = np.log(1 + target_annual_return) / 12  # 초기 월 로그수익률 추정
    res = minimize(
        gbm_objective, x0=init_mu,
        args=(target_annual_return, S0, sigma, simulation_times, T, dt, epsilon_var),
        method="Nelder-Mead"
    )
    return float(res.x[0])


def month_offset(base_dt: pd.Timestamp, liab_dt: pd.Timestamp) -> int:
    return (liab_dt.year - base_dt.year) * 12 + (liab_dt.month - base_dt.month)

def year_end_cols(W: int, o: int) -> np.ndarray:
    if o > W - 1:
        raise ValueError(f"offset {o}가 범위를 벗어남(W={W})")
    K = 1 + (W - 1 - o) // 12
    return np.array([o + 12*k for k in range(K)], dtype=int)

def build_new_ALM_DB(ALM_DB_sim: dict):
    s_keys = sorted(ALM_DB_sim.keys())
    t_keys = sorted(ALM_DB_sim[s_keys[0]].keys())
    S = len(s_keys); K = len(t_keys)
    DBO_mat = np.empty((K, S), dtype=float)
    NC_mat  = np.empty((K, S), dtype=float)
    EBP_mat = np.empty((K, S), dtype=float)
    for ti, t in enumerate(t_keys):
        for si, s in enumerate(s_keys):
            rec = ALM_DB_sim[s][t]
            DBO_mat[ti, si] = float(rec['DBO'])
            NC_mat[ti,  si] = float(rec['NC'])
            EBP_mat[ti, si] = float(rec['EBP'])
    return {'DBO': DBO_mat, 'NC': NC_mat, 'EBP': EBP_mat}


def make_ef_for_year(R: np.ndarray,
                    bounds: list,
                    n_target: int = 20,
                    w0: np.ndarray = None):
    S, A = R.shape
    mu   = R.mean(axis=0)              # (A,)
    Sig  = np.cov(R, rowvar=False)     # (A,A)
    mu_min, mu_max = np.quantile(mu, 0.05), np.quantile(mu, 0.95)
    if mu_min == mu_max:
        _mn, _mx = float(mu.min()), float(mu.max())
        mu_min, mu_max = (_mn, _mx) if _mx > _mn else (_mn, _mn + 0.02)
    targets = np.linspace(mu_min, mu_max, n_target)
    if w0 is None:
        w0 = np.ones(A) / A
    cons_sum1 = ({'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)},)
    ws, tg = [], []
    for tt in targets:
        obj  = lambda w: float(w @ Sig @ w)
        cons = cons_sum1 + ({'type': 'eq', 'fun': lambda w, ttar=tt: float(np.dot(w, mu) - ttar)},)
        res  = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons,
                        options={"ftol":1e-10,"maxiter":400})
        if res.success:
            ws.append(res.x); tg.append(tt)
    return np.array(ws), np.array(tg), mu, Sig, targets

def liability_year_labels(liab_dt: pd.Timestamp, o_raw: int, col_idx: np.ndarray) -> pd.DatetimeIndex:
    dates = []
    for c in col_idx:
        delta = c - o_raw
        y = liab_dt.year + (liab_dt.month - 1 + delta) // 12
        m = (liab_dt.month - 1 + delta) % 12 + 1
        dates.append(pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(0))
    return pd.DatetimeIndex(dates)
