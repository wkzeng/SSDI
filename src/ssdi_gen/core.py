from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd


EPS = 1e-12
LCD_TYPES = ("client", "class", "joint")
LDS_TYPES = ("client", "special", "lowrank")


@dataclass
class SSDIMetrics:
    LCD: float
    LDS: float
    SSDI: float
    DSR: float
    missing_rate: float


@dataclass
class GenerationOutput:
    n_ck: np.ndarray
    matrix_df: pd.DataFrame
    metrics: SSDIMetrics
    success: bool
    iter_used: int
    actual_alpha: float
    actual_beta: float
    alpha: float
    beta: float
    lcd_type: str
    lds_type: str
    lcd_params: Dict[str, Any]
    lds_params: Dict[str, Any]
    target_ssdi: float
    C: int
    K: int
    N: int
    generator_variant: str = "v2"

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        record.update(asdict(self.metrics))
        record["actual_ssdi"] = self.metrics.SSDI
        record["n_ck"] = self.n_ck
        record["matrix_df"] = self.matrix_df
        return record


def theoretical_vmax(C: int, K: int, eps: float = EPS) -> float:
    m = min(int(C), int(K))
    if m <= 1:
        return eps
    return float(np.sqrt(2.0 * (1.0 - 1.0 / m)) + eps)


def compute_ssdi_metrics(n_ck: np.ndarray, eps: float = EPS) -> Dict[str, float]:
    n_ck = np.asarray(n_ck, dtype=float)
    if n_ck.ndim != 2:
        raise ValueError("n_ck must be a 2D array with shape (C, K).")

    C, K = n_ck.shape
    N = float(n_ck.sum())
    if N <= eps:
        return dict(LCD=0.0, LDS=0.0, SSDI=0.0, DSR=0.0, missing_rate=1.0)

    n_k = n_ck.sum(axis=0) + eps
    N_c = n_ck.sum(axis=1) + eps

    P = n_ck / N
    Q = np.outer(N_c, n_k) / (N ** 2)

    D = P - Q
    M = (n_ck <= 0).astype(float)
    D_LCD = M * (-Q)
    D_LDS = (1.0 - M) * D

    W = np.sqrt((N / N_c)[:, None] + (N / n_k)[None, :])
    Vmax = theoretical_vmax(C, K, eps=eps)

    LCD = float(np.linalg.norm(W * D_LCD, ord="fro") / Vmax)
    LDS = float(np.linalg.norm(W * D_LDS, ord="fro") / Vmax)
    SSDI = float(np.sqrt(LCD ** 2 + LDS ** 2))
    DSR = float(LCD / (LDS + eps))

    return dict(LCD=LCD, LDS=LDS, SSDI=SSDI, DSR=DSR, missing_rate=float(M.mean()))


def estimate_pareto_alpha(n_ck: np.ndarray, min_samples: int = 5) -> float:
    client_totals = np.asarray(n_ck, dtype=float).sum(axis=0)
    positive = client_totals[client_totals > 0]
    if len(positive) < min_samples:
        return float("nan")
    xm = positive.min()
    denom = np.sum(np.log(positive / xm))
    if denom <= EPS:
        return float("nan")
    return float(len(positive) / denom)


def estimate_zipf_beta(n_ck: np.ndarray, min_labels: int = 3) -> float:
    class_totals = np.asarray(n_ck, dtype=float).sum(axis=1)
    positive = class_totals[class_totals > 0]
    if len(positive) < min_labels:
        return float("nan")
    freqs = np.sort(positive)[::-1]
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    beta_hat = -np.polyfit(np.log(ranks), np.log(freqs), 1)[0]
    return float(beta_hat)


def truncated_pareto(K: int, alpha: float, mean_size: float, rng: np.random.Generator) -> np.ndarray:
    alpha = max(float(alpha), 1.05)
    x = rng.pareto(alpha, size=K) + 1.0
    upper = np.quantile(x, 0.95)
    x = np.clip(x, 0.0, upper)
    x = x / x.mean() * mean_size
    return x / x.sum()


def smooth_zipf(C: int, beta: float, smooth: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    del rng
    ranks = np.arange(1, C + 1, dtype=float)
    beta = max(float(beta), 0.01)
    smooth = max(float(smooth), 0.0)
    values = 1.0 / np.power(ranks + smooth, beta)
    values = np.maximum(values, EPS)
    return values / values.sum()


def largest_remainder_rounding(P: np.ndarray, N: int) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if P.sum() <= 0:
        raise ValueError("Probability matrix must have positive mass.")
    P = P / P.sum()
    raw = P * int(N)
    floored = np.floor(raw).astype(int)
    remainder = int(N) - int(floored.sum())
    if remainder > 0:
        frac = (raw - floored).ravel()
        idx = np.argsort(-frac)[:remainder]
        floored.ravel()[idx] += 1
    return floored


def ensure_nonempty(n_ck: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_ck = np.asarray(n_ck, dtype=int).copy()
    C, K = n_ck.shape

    for _ in range(3):
        col_sums = n_ck.sum(axis=0)
        row_sums = n_ck.sum(axis=1)

        for k in np.where(col_sums == 0)[0]:
            donor_col = int(np.argmax(col_sums))
            donor_rows = np.where(n_ck[:, donor_col] > 1)[0]
            if donor_rows.size == 0:
                donor_rows = np.where(row_sums > 1)[0]
                donor_row = int(donor_rows[0] if donor_rows.size else rng.integers(0, C))
                donor_col = int(np.argmax(n_ck[donor_row]))
            else:
                donor_row = int(donor_rows[np.argmax(n_ck[donor_rows, donor_col])])
            n_ck[donor_row, donor_col] -= 1
            n_ck[donor_row, k] += 1

        col_sums = n_ck.sum(axis=0)
        row_sums = n_ck.sum(axis=1)
        for c in np.where(row_sums == 0)[0]:
            donor_row = int(np.argmax(row_sums))
            donor_cols = np.where(n_ck[donor_row, :] > 1)[0]
            donor_col = int(donor_cols[np.argmax(n_ck[donor_row, donor_cols])] if donor_cols.size else np.argmax(col_sums))
            n_ck[donor_row, donor_col] -= 1
            n_ck[c, donor_col] += 1

        if np.all(n_ck.sum(axis=0) > 0) and np.all(n_ck.sum(axis=1) > 0):
            break

    return n_ck


def generate_support_mask(
    class_w: np.ndarray,
    client_w: np.ndarray,
    lcd_type: str,
    lcd_params: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    C = len(class_w)
    K = len(client_w)
    miss = float(np.clip(lcd_params.get("missing_rate", 0.0), 0.0, 0.98))
    if miss <= 0:
        return np.ones((C, K), dtype=int)

    tau = float(lcd_params.get("tau", 1.0))
    gamma = float(lcd_params.get("gamma", 1.0))
    a = float(lcd_params.get("a", 1.0))
    b = float(lcd_params.get("b", 1.0))

    if lcd_type == "client":
        scores = np.power(client_w[None, :] + EPS, -tau)
        scores = np.repeat(scores, C, axis=0)
    elif lcd_type == "class":
        scores = np.power(class_w[:, None] + EPS, -gamma)
        scores = np.repeat(scores, K, axis=1)
    elif lcd_type == "joint":
        scores = np.power(class_w[:, None] + EPS, -b) * np.power(client_w[None, :] + EPS, -a)
    else:
        raise ValueError(f"Unknown lcd_type: {lcd_type}")

    scores = np.maximum(scores, EPS)
    flat_scores = scores.ravel().astype(float)
    flat_scores /= flat_scores.sum()
    num_zero = int(round(miss * C * K))
    num_zero = min(num_zero, C * K - max(C, K))
    mask = np.ones(C * K, dtype=int)
    if num_zero > 0:
        idx = rng.choice(C * K, size=num_zero, replace=False, p=flat_scores)
        mask[idx] = 0
    return structured_zero_adjustment(mask.reshape(C, K), rng)


def structured_zero_adjustment(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = np.asarray(mask, dtype=int).copy()
    C, K = mask.shape
    for _ in range(5):
        row_sums = mask.sum(axis=1)
        col_sums = mask.sum(axis=0)
        if np.all(row_sums > 0) and np.all(col_sums > 0):
            break
        for c in np.where(row_sums == 0)[0]:
            k = int(rng.integers(0, K))
            mask[c, k] = 1
        for k in np.where(col_sums == 0)[0]:
            c = int(rng.integers(0, C))
            mask[c, k] = 1
    return mask


def generate_lds_bias(
    C: int,
    K: int,
    lds_type: str,
    lds_params: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    strength = float(lds_params.get("strength", 0.0))
    if strength <= 0:
        return np.zeros((C, K), dtype=float)

    if lds_type == "client":
        u = rng.normal(0.0, 1.0, size=K)
        G = np.repeat(u[None, :], C, axis=0)
    elif lds_type == "special":
        num_special = int(max(1, lds_params.get("num_special", max(1, K // 4))))
        G = np.zeros((C, K), dtype=float)
        for c in range(C):
            idx = rng.choice(K, size=min(num_special, K), replace=False)
            G[c, idx] = rng.normal(1.0, 0.5, size=len(idx))
        G += rng.normal(0.0, 0.15, size=(C, K))
    elif lds_type == "lowrank":
        rank = int(max(1, lds_params.get("rank", max(1, int(np.sqrt(min(C, K)))))))
        U = rng.normal(0.0, 1.0, size=(C, rank))
        V = rng.normal(0.0, 1.0, size=(rank, K))
        G = U @ V
    else:
        raise ValueError(f"Unknown lds_type: {lds_type}")

    G = G - G.mean()
    scale = np.std(G)
    if scale > EPS:
        G = G / scale
    return strength * G


def make_matrix_df(n_ck: np.ndarray) -> pd.DataFrame:
    C, K = n_ck.shape
    return pd.DataFrame(
        n_ck,
        index=[f"class_{i}" for i in range(C)],
        columns=[f"client_{j}" for j in range(K)],
    )


def _clip_params_inplace(lcd_params: Dict[str, Any], lds_params: Dict[str, Any], C: int, K: int) -> None:
    if 'missing_rate' in lcd_params:
        lcd_params['missing_rate'] = float(np.clip(lcd_params['missing_rate'], 0.0, 0.98))
    for key in ('tau', 'gamma', 'a', 'b'):
        if key in lcd_params:
            lcd_params[key] = float(np.clip(lcd_params[key], 0.01, 6.0))
    if 'strength' in lds_params:
        lds_params['strength'] = float(np.clip(lds_params['strength'], 0.0, 15.0))
    if 'rank' in lds_params:
        lds_params['rank'] = int(np.clip(int(lds_params['rank']), 1, max(1, min(C, K))))
    if 'num_special' in lds_params:
        lds_params['num_special'] = int(np.clip(int(lds_params['num_special']), 1, max(1, K)))


def _is_structured_targeted(lcd_params: Dict[str, Any], lds_params: Dict[str, Any]) -> bool:
    """Whether current params come from structured target projection."""
    return bool(
        lcd_params.get("_structured_targeted", False)
        or lcd_params.get("_structured_missing_locked", False)
        or ("_target_missing_rate" in lcd_params)
        or ("_target_theta" in lcd_params)
        or lds_params.get("_structured_targeted", False)
    )


def _get_structured_missing_spec(lcd_params: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], bool]:
    """Return (target_missing_rate, missing_tol, locked)."""
    target_missing = lcd_params.get("_target_missing_rate", None)
    missing_tol = lcd_params.get("_missing_tol", None)
    locked = bool(lcd_params.get("_structured_missing_locked", False))

    if target_missing is None:
        return None, None, locked

    target_missing = float(np.clip(float(target_missing), 0.0, 0.98))
    if missing_tol is None:
        missing_tol = 0.02
    missing_tol = float(np.clip(float(missing_tol), 0.0, 0.08))
    return target_missing, missing_tol, locked

def _compute_theta_from_lcd_lds(lcd: float, lds: float, eps: float = EPS) -> float:
    return float(np.arctan(float(lcd) / (float(lds) + eps)))

def _compute_target_geometry_from_params(
    lcd_params: Dict[str, Any],
    lds_params: Dict[str, Any],
    target_ssdi: float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (target_lcd, target_lds, target_theta) if available."""
    target_theta = lcd_params.get("_target_theta", None)
    target_lcd = lcd_params.get("_target_lcd", None)
    target_lds = lds_params.get("_target_lds", None)

    if target_theta is not None:
        target_theta = float(np.clip(float(target_theta), 0.0, np.pi / 2.0))

    if target_lcd is None and target_theta is not None:
        target_lcd = float(np.clip(float(target_ssdi) * np.sin(target_theta), 0.0, 1.0))
    elif target_lcd is not None:
        target_lcd = float(np.clip(float(target_lcd), 0.0, 1.0))

    if target_lds is None and target_theta is not None:
        target_lds = float(np.clip(float(target_ssdi) * np.cos(target_theta), 0.0, 1.0))
    elif target_lds is not None:
        target_lds = float(np.clip(float(target_lds), 0.0, 1.0))

    return target_lcd, target_lds, target_theta


def _get_structured_bias_and_targets(
    lcd_params: Dict[str, Any],
    lds_params: Dict[str, Any],
    target_ssdi: float,
) -> Tuple[float, Optional[float], Optional[float], Optional[float]]:
    """Return (structure_bias, target_lcd, target_lds, target_theta)."""
    structure_bias = float(lcd_params.get("_domain_bias", lds_params.get("_domain_bias", 0.0)))
    structure_bias = float(np.clip(structure_bias, -1.0, 1.0))
    target_lcd, target_lds, target_theta = _compute_target_geometry_from_params(
        lcd_params, lds_params, target_ssdi
    )
    return structure_bias, target_lcd, target_lds, target_theta


def _apply_structured_probability_floor(
    P: np.ndarray,
    mask: np.ndarray,
    datasize_hint: float,
    structure_bias: float,
    target_ssdi: float,
    target_theta: Optional[float],
) -> np.ndarray:
    """Apply a small probability floor to suppress implicit LCD leakage.

    设计原则：
    - 只在 structured 且偏 LDS 的情形下启用；
    - 只保护 mask=1 的格子；
    - floor 与 datasize 相关，本质上是避免大量非 mask 格子在 rounding 时掉成 0；
    - bias 越偏 LDS、target_ssdi 越高，floor 稍强；
    - 不追求把每个格子都拉大，只是避免尾部整体塌陷。
    """
    P = np.asarray(P, dtype=float)
    mask = np.asarray(mask, dtype=float)

    if datasize_hint <= 0:
        return P

    # 只在偏 LDS 时启用
    if structure_bias > -0.15:
        return P

    # 若目标角本身不偏左，也不启用
    if target_theta is not None and target_theta > np.deg2rad(35.0):
        return P

    active = mask > 0
    if not np.any(active):
        return P

    # floor_mass 的通俗意义：
    # 希望保留下来的格子，期望计数至少不要长期远低于 1。
    # bias 越负、SSDI 越高，允许略高一点的 floor。
    left_strength = float(np.clip(-structure_bias, 0.0, 1.0))
    high_ssdi_strength = float(np.clip((target_ssdi - 0.45) / 0.55, 0.0, 1.0))

    floor_mass = (
        0.30
        + 0.70 * left_strength
        + 0.80 * high_ssdi_strength
        + 0.45 * left_strength * high_ssdi_strength
    )
    floor_mass = float(np.clip(floor_mass, 0.20, 1.80))

    # floor probability ~ floor_mass / N
    p_floor = float(floor_mass / max(float(datasize_hint), 1.0))

    # 不让 floor 过强，防止把分布洗平
    active_vals = P[active]
    if active_vals.size > 0:
        q20 = float(np.quantile(active_vals, 0.20))
        p_floor = min(p_floor, 0.55 * q20 if q20 > 0 else p_floor)

    P_new = P.copy()
    P_new[active] = np.maximum(P_new[active], p_floor)

    if P_new.sum() <= EPS:
        return P
    P_new /= P_new.sum()
    return P_new

def _local_param_correction(
    lcd_params: Dict[str, Any],
    lds_params: Dict[str, Any],
    metrics: SSDIMetrics,
    target_ssdi: float,
    lcd_type: str,
    lds_type: str,
    C: int,
    K: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Local correction with structured-aware direction control.

    设计原则：
    - 非 structured 路径：保留旧式“总强度”修正风格；
    - structured 路径：优先根据 theta / LCD / LDS 误差修正；
    - 如果 missing 被 structured 锁定，则不再直接修改 missing_rate 中心值；
    - 高 LDS / 高 SSDI 区域若半径不足，应优先补 LDS，而不是默认抬高 missing。
    """
    lcd_new = dict(lcd_params)
    lds_new = dict(lds_params)

    err_r = float(target_ssdi - metrics.SSDI)
    mag = min(0.25, 0.05 + 0.8 * abs(err_r))
    direction = 1.0 if err_r > 0 else -1.0

    structured = _is_structured_targeted(lcd_new, lds_new)
    target_missing, missing_tol, missing_locked = _get_structured_missing_spec(lcd_new)
    target_lcd, target_lds, target_theta = _compute_target_geometry_from_params(lcd_new, lds_new, target_ssdi)

    # --------- fallback: old generic behavior ----------
    if not structured or target_theta is None or target_lcd is None or target_lds is None:
        if ('missing_rate' in lcd_new) and (not missing_locked):
            lcd_new['missing_rate'] = float(lcd_new.get('missing_rate', 0.0) + direction * 0.6 * mag)

        if lcd_type == 'client' and 'tau' in lcd_new:
            lcd_new['tau'] = float(lcd_new['tau'] * np.exp(direction * 0.45 * mag))
        elif lcd_type == 'class' and 'gamma' in lcd_new:
            lcd_new['gamma'] = float(lcd_new['gamma'] * np.exp(direction * 0.45 * mag))
        elif lcd_type == 'joint':
            if 'a' in lcd_new:
                lcd_new['a'] = float(lcd_new['a'] * np.exp(direction * 0.35 * mag))
            if 'b' in lcd_new:
                lcd_new['b'] = float(lcd_new['b'] * np.exp(direction * 0.35 * mag))

        if 'strength' in lds_new:
            lds_new['strength'] = float(lds_new['strength'] * np.exp(direction * 0.70 * mag))
        if lds_type == 'lowrank' and 'rank' in lds_new and abs(err_r) > 0.04:
            lds_new['rank'] = int(lds_new['rank'] + (1 if err_r > 0 else -1))
        if lds_type == 'special' and 'num_special' in lds_new and abs(err_r) > 0.04:
            lds_new['num_special'] = int(lds_new['num_special'] + (1 if err_r > 0 else -1))

        _clip_params_inplace(lcd_new, lds_new, C, K)
        return lcd_new, lds_new

    # --------- structured-aware correction ----------
    actual_theta = _compute_theta_from_lcd_lds(metrics.LCD, metrics.LDS)
    err_theta = float(actual_theta - target_theta)
    err_lcd = float(metrics.LCD - target_lcd)
    err_lds = float(metrics.LDS - target_lds)
    theta_tol = float(np.deg2rad(4.0))

    # A. 方向偏右：LCD 过强，需要往左压
    if err_theta > theta_tol:
        if ('missing_rate' in lcd_new) and (not missing_locked):
            step = min(0.10, 0.55 * mag)
            new_missing = float(lcd_new.get('missing_rate', 0.0) - step)
            if target_missing is not None and missing_tol is not None:
                low = max(0.0, target_missing - missing_tol)
                high = min(0.98, target_missing + missing_tol)
                new_missing = float(np.clip(new_missing, low, high))
            lcd_new['missing_rate'] = new_missing

        if lcd_type == 'client' and 'tau' in lcd_new:
            lcd_new['tau'] = float(lcd_new['tau'] * np.exp(-0.40 * mag))
        elif lcd_type == 'class' and 'gamma' in lcd_new:
            lcd_new['gamma'] = float(lcd_new['gamma'] * np.exp(-0.40 * mag))
        elif lcd_type == 'joint':
            if 'a' in lcd_new:
                lcd_new['a'] = float(lcd_new['a'] * np.exp(-0.32 * mag))
            if 'b' in lcd_new:
                lcd_new['b'] = float(lcd_new['b'] * np.exp(-0.32 * mag))

        # 若 LDS 仍偏低，同时适当补 LDS
        if err_lds < -0.01:
            if 'strength' in lds_new:
                lds_new['strength'] = float(lds_new['strength'] * np.exp(0.45 * mag))
            if lds_type == 'lowrank' and 'rank' in lds_new and abs(err_lds) > 0.03:
                lds_new['rank'] = int(lds_new['rank'] + 1)
            if lds_type == 'special' and 'num_special' in lds_new and abs(err_lds) > 0.03:
                lds_new['num_special'] = int(lds_new['num_special'] + 1)

    # B. 方向偏左：LCD 不够，需要往右补
    elif err_theta < -theta_tol:
        if ('missing_rate' in lcd_new) and (not missing_locked):
            step = min(0.10, 0.55 * mag)
            new_missing = float(lcd_new.get('missing_rate', 0.0) + step)
            if target_missing is not None and missing_tol is not None:
                low = max(0.0, target_missing - missing_tol)
                high = min(0.98, target_missing + missing_tol)
                new_missing = float(np.clip(new_missing, low, high))
            lcd_new['missing_rate'] = new_missing

        if lcd_type == 'client' and 'tau' in lcd_new:
            lcd_new['tau'] = float(lcd_new['tau'] * np.exp(0.38 * mag))
        elif lcd_type == 'class' and 'gamma' in lcd_new:
            lcd_new['gamma'] = float(lcd_new['gamma'] * np.exp(0.38 * mag))
        elif lcd_type == 'joint':
            if 'a' in lcd_new:
                lcd_new['a'] = float(lcd_new['a'] * np.exp(0.30 * mag))
            if 'b' in lcd_new:
                lcd_new['b'] = float(lcd_new['b'] * np.exp(0.30 * mag))

        # 为避免继续往 y 轴上堆，可轻微回收 LDS
        if err_lds > 0.04 and 'strength' in lds_new:
            lds_new['strength'] = float(lds_new['strength'] * np.exp(-0.18 * mag))

    # C. 方向基本对，但半径不够 / 过强：按缺哪边补哪边
    else:
        if err_r > 0:
            # 半径不足：优先补缺口更大的那一侧
            need_lcd = max(0.0, -err_lcd)
            need_lds = max(0.0, -err_lds)

            if need_lds >= need_lcd:
                if 'strength' in lds_new:
                    lds_new['strength'] = float(lds_new['strength'] * np.exp(0.55 * mag))
                if lds_type == 'lowrank' and 'rank' in lds_new and need_lds > 0.03:
                    lds_new['rank'] = int(lds_new['rank'] + 1)
                if lds_type == 'special' and 'num_special' in lds_new and need_lds > 0.03:
                    lds_new['num_special'] = int(lds_new['num_special'] + 1)

                if need_lcd > 0.02:
                    if ('missing_rate' in lcd_new) and (not missing_locked):
                        step = min(0.05, 0.25 * mag)
                        new_missing = float(lcd_new.get('missing_rate', 0.0) + step)
                        if target_missing is not None and missing_tol is not None:
                            low = max(0.0, target_missing - missing_tol)
                            high = min(0.98, target_missing + missing_tol)
                            new_missing = float(np.clip(new_missing, low, high))
                        lcd_new['missing_rate'] = new_missing
            else:
                if ('missing_rate' in lcd_new) and (not missing_locked):
                    step = min(0.08, 0.45 * mag)
                    new_missing = float(lcd_new.get('missing_rate', 0.0) + step)
                    if target_missing is not None and missing_tol is not None:
                        low = max(0.0, target_missing - missing_tol)
                        high = min(0.98, target_missing + missing_tol)
                        new_missing = float(np.clip(new_missing, low, high))
                    lcd_new['missing_rate'] = new_missing

                if lcd_type == 'client' and 'tau' in lcd_new:
                    lcd_new['tau'] = float(lcd_new['tau'] * np.exp(0.22 * mag))
                elif lcd_type == 'class' and 'gamma' in lcd_new:
                    lcd_new['gamma'] = float(lcd_new['gamma'] * np.exp(0.22 * mag))
                elif lcd_type == 'joint':
                    if 'a' in lcd_new:
                        lcd_new['a'] = float(lcd_new['a'] * np.exp(0.18 * mag))
                    if 'b' in lcd_new:
                        lcd_new['b'] = float(lcd_new['b'] * np.exp(0.18 * mag))

                if need_lds > 0.02 and 'strength' in lds_new:
                    lds_new['strength'] = float(lds_new['strength'] * np.exp(0.20 * mag))
        else:
            # 半径过强：按过强来源适度回收
            over_lcd = max(0.0, err_lcd)
            over_lds = max(0.0, err_lds)

            if over_lcd >= over_lds:
                if ('missing_rate' in lcd_new) and (not missing_locked):
                    step = min(0.08, 0.35 * mag)
                    new_missing = float(lcd_new.get('missing_rate', 0.0) - step)
                    if target_missing is not None and missing_tol is not None:
                        low = max(0.0, target_missing - missing_tol)
                        high = min(0.98, target_missing + missing_tol)
                        new_missing = float(np.clip(new_missing, low, high))
                    lcd_new['missing_rate'] = new_missing

                if lcd_type == 'client' and 'tau' in lcd_new:
                    lcd_new['tau'] = float(lcd_new['tau'] * np.exp(-0.18 * mag))
                elif lcd_type == 'class' and 'gamma' in lcd_new:
                    lcd_new['gamma'] = float(lcd_new['gamma'] * np.exp(-0.18 * mag))
                elif lcd_type == 'joint':
                    if 'a' in lcd_new:
                        lcd_new['a'] = float(lcd_new['a'] * np.exp(-0.15 * mag))
                    if 'b' in lcd_new:
                        lcd_new['b'] = float(lcd_new['b'] * np.exp(-0.15 * mag))
            else:
                if 'strength' in lds_new:
                    lds_new['strength'] = float(lds_new['strength'] * np.exp(-0.30 * mag))
                if lds_type == 'lowrank' and 'rank' in lds_new and over_lds > 0.03:
                    lds_new['rank'] = int(lds_new['rank'] - 1)
                if lds_type == 'special' and 'num_special' in lds_new and over_lds > 0.03:
                    lds_new['num_special'] = int(lds_new['num_special'] - 1)

    # structured-missing lock：最后再锁回目标带
    if ('missing_rate' in lcd_new) and (target_missing is not None) and (missing_tol is not None):
        low = max(0.0, target_missing - missing_tol)
        high = min(0.98, target_missing + missing_tol)
        lcd_new['missing_rate'] = float(np.clip(float(lcd_new['missing_rate']), low, high))

    _clip_params_inplace(lcd_new, lds_new, C, K)
    return lcd_new, lds_new


def generate_probability_matrix(
    C: int,
    K: int,
    target_ssdi: float,
    lcd_type: str,
    lds_type: str,
    lcd_params: Dict[str, Any],
    lds_params: Dict[str, Any],
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    *,
    zipf_smooth: float = 1.5,
    attempt_index: int = 1,
    lcd_scale: float = 1.0,
    lds_scale: float = 1.0,
    power_scale: float = 1.0,
    variant: str = "v2",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = float(np.clip(target_ssdi, 0.0, 0.999))

    client_w = truncated_pareto(K, alpha, mean_size=1.0, rng=rng)
    class_w = smooth_zipf(C, beta, smooth=zipf_smooth)

    local_lcd = dict(lcd_params)
    local_lds = dict(lds_params)

    structured = _is_structured_targeted(local_lcd, local_lds)
    target_missing, missing_tol, missing_locked = _get_structured_missing_spec(local_lcd)
    structure_bias, target_lcd, target_lds, target_theta = _get_structured_bias_and_targets(
        local_lcd, local_lds, target_ssdi
    )

    # --------- missing-rate handling ----------
    miss = float(local_lcd.get("missing_rate", 0.0))
    if structured and target_missing is not None:
        # structured 路径下，missing 中心值由 generate 层给定；
        # core 这里只允许小范围扰动，不再按高 SSDI 逻辑重标定。
        jitter_scale = 0.20 if missing_locked else 0.45
        base_tol = max(missing_tol or 0.01, 0.005)

        # 偏 LDS 区域进一步收紧抖动
        if structure_bias <= -0.75:
            jitter_scale *= 0.55
        elif structure_bias <= -0.50:
            jitter_scale *= 0.72

        if variant == "v1":
            jitter = rng.normal(0.0, jitter_scale * base_tol)
        else:
            jitter = rng.normal(0.0, 1.10 * jitter_scale * base_tol)

        miss = float(target_missing + jitter)

        if missing_tol is not None:
            low = max(0.0, target_missing - missing_tol)
            high = min(0.995, target_missing + missing_tol)
            miss = float(np.clip(miss, low, high))
        else:
            miss = float(np.clip(miss, 0.0, 0.995))
    else:
        # 旧 generic 逻辑保留
        if variant == 'v1':
            miss *= lcd_scale * (0.95 + 0.12 * rng.random() + 0.06 * s + 0.02 * np.log1p(attempt_index))
        else:
            miss *= lcd_scale * (0.92 + 0.18 * rng.random() + 0.10 * s + 0.04 * np.log1p(attempt_index))
        miss = float(np.clip(miss, 0.0, 0.995))

    local_lcd["missing_rate"] = miss

    # --------- LDS-strength handling ----------
    strength = float(local_lds.get("strength", 0.0))
    if structured:
        # structured 下保留一定缩放，但不让其随 attempt / ssdi 失控
        if variant == 'v1':
            strength *= lds_scale * (0.97 + 0.08 * rng.random() + 0.05 * s + 0.01 * np.log1p(attempt_index))
        else:
            strength *= lds_scale * (0.95 + 0.12 * rng.random() + 0.08 * s + 0.015 * np.log1p(attempt_index))

        # 偏 LDS 时可适度增强，但不要靠极端尖化来做半径
        if structure_bias <= -0.75:
            strength *= 1.06
        elif structure_bias <= -0.50:
            strength *= 1.03
    else:
        if variant == 'v1':
            strength *= lds_scale * (0.94 + 0.15 * rng.random() + 0.10 * s + 0.02 * np.log1p(attempt_index))
        else:
            strength *= lds_scale * (0.90 + 0.25 * rng.random() + 0.15 * s + 0.03 * np.log1p(attempt_index))
    local_lds["strength"] = float(max(0.0, strength))

    mask = generate_support_mask(class_w, client_w, lcd_type, local_lcd, rng)
    base = np.outer(class_w, client_w) * mask
    if base.sum() <= EPS:
        base = structured_zero_adjustment(mask, rng) * (np.outer(class_w, client_w) + EPS)
    base = base / base.sum()

    G = generate_lds_bias(C, K, lds_type, local_lds, rng)
    G = np.clip(G, -6.0, 6.0)

    P = base * np.exp(G) * mask
    P = np.maximum(P, 0.0)
    if (not np.isfinite(P).all()) or P.sum() <= EPS:
        P = base.copy()
    P /= max(P.sum(), EPS)

    # --------- power handling ----------
    if structured:
        if variant == 'v1':
            power = power_scale * (1.0 + 0.55 * s + 1.40 * (s ** 2) + 0.02 * np.log1p(attempt_index))
        else:
            power = power_scale * (1.0 + 1.10 * s + 2.80 * (s ** 2) + 0.05 * np.log1p(attempt_index))

        # 关键修复：
        # 偏 LDS 时，不允许通过过强尖化去“硬做大半径”，否则会出现大量隐式 0。
        if structure_bias <= -0.90:
            power *= 0.58
        elif structure_bias <= -0.75:
            power *= 0.66
        elif structure_bias <= -0.50:
            power *= 0.78
        elif structure_bias <= -0.25:
            power *= 0.88
    else:
        if variant == 'v1':
            power = power_scale * (1.0 + 0.8 * s + 2.4 * (s ** 2) + 0.05 * np.log1p(attempt_index))
        else:
            power = power_scale * (1.0 + 2.0 * s + 6.0 * (s ** 2) + 0.12 * np.log1p(attempt_index))

    power = float(np.clip(power, 0.65, 12.0))

    P = np.power(np.maximum(P, EPS), power) * mask
    P = np.maximum(P, 0.0)
    if (not np.isfinite(P).all()) or P.sum() <= EPS:
        P = base.copy()
    P /= max(P.sum(), EPS)

    # --------- structured probability floor ----------
    # 关键修复：
    # 偏 LDS 时，对非 mask 格子加一个与 datasize 相关的概率下界，
    # 避免很多“小于 1 个样本期望值”的格子在 rounding 后掉成 0，
    # 从而引发隐式 LCD 泄漏。
    datasize_hint = float(local_lcd.get("_datasize_hint", 0.0))
    if structured and datasize_hint > 0:
        P = _apply_structured_probability_floor(
            P=P,
            mask=mask,
            datasize_hint=datasize_hint,
            structure_bias=structure_bias,
            target_ssdi=target_ssdi,
            target_theta=target_theta,
        )

    return P, class_w, client_w, mask, G


def generate_ssdi_matrix_array(
    *,
    client: int,
    label: int,
    datasize: int,
    ssdi: float,
    lcdtype: str = "client",
    ldstype: str = "client",
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    lcd_params: Optional[Dict[str, Any]] = None,
    lds_params: Optional[Dict[str, Any]] = None,
    ssdi_error: float = 0.025,
    seed: Optional[int] = None,
    max_iters: int = 100,
    zipf_smooth: float = 1.5,
    force_nonempty: bool = True,
    get_default_params_fn=None,
    preferred_variant: Optional[str] = None,
) -> GenerationOutput:
    if get_default_params_fn is None:
        raise ValueError("get_default_params_fn must be provided.")

    C, K, N = int(label), int(client), int(datasize)
    if lcdtype not in LCD_TYPES:
        raise ValueError(f"lcdtype must be one of {LCD_TYPES}")
    if ldstype not in LDS_TYPES:
        raise ValueError(f"ldstype must be one of {LDS_TYPES}")

    rng = np.random.default_rng(seed)
    mechanism_lcd = f"lcd_{lcdtype}"
    mechanism_lds = f"lds_{ldstype}"
    lcd_defaults = get_default_params_fn(ssdi, mechanism_lcd, C=C, K=K, seed=seed)
    lds_defaults = get_default_params_fn(ssdi, mechanism_lds, C=C, K=K, seed=None if seed is None else seed + 1)

    merged_lcd = dict(lcd_defaults)
    merged_lds = dict(lds_defaults)
    if lcd_params:
        merged_lcd.update(lcd_params)
    if lds_params:
        merged_lds.update(lds_params)

    alpha = float(alpha if alpha is not None else merged_lds.get("alpha", merged_lcd.get("alpha")))
    beta = float(beta if beta is not None else merged_lds.get("beta", merged_lcd.get("beta")))

    lcd_params_current = {k: v for k, v in merged_lcd.items() if k not in {"alpha", "beta", "_randomized", "mechanism", "_default_version"}}
    lds_params_current = {k: v for k, v in merged_lds.items() if k not in {"alpha", "beta", "_randomized", "mechanism", "_default_version"}}



    lcd_params_current["_datasize_hint"] = float(N)

    lds_params_current["_datasize_hint"] = float(N)


    best_output: Optional[GenerationOutput] = None
    best_gap = np.inf
    lcd_scale = 1.0 + 0.9 * (ssdi ** 1.5)
    lds_scale = 1.0 + 1.2 * (ssdi ** 1.5)
    power_scale = 1.0 + 1.5 * (ssdi ** 1.8)

    if preferred_variant is None:
        preferred_variant = "v2" if ssdi <= 0.7 else ("v1" if rng.random() < 0.7 else "v2")
    variant_order: List[str] = [preferred_variant, "v1" if preferred_variant == "v2" else "v2"]

    iter_counter = 0
    correction_rounds = max(1, max_iters // max(1, len(variant_order)))
    last_metrics: Optional[SSDIMetrics] = None

    structured = _is_structured_targeted(lcd_params_current, lds_params_current)

    for round_idx in range(1, correction_rounds + 1):
        for variant in variant_order:
            iter_counter += 1
            P, _, _, _, _ = generate_probability_matrix(
                C, K, ssdi, lcdtype, ldstype, lcd_params_current, lds_params_current,
                alpha, beta, rng, zipf_smooth=zipf_smooth, attempt_index=iter_counter,
                lcd_scale=lcd_scale, lds_scale=lds_scale, power_scale=power_scale,
                variant=variant,
            )
            n_ck = largest_remainder_rounding(P, N)
            if force_nonempty:
                n_ck = ensure_nonempty(n_ck, rng)

            metrics_dict = compute_ssdi_metrics(n_ck)
            metrics = SSDIMetrics(**metrics_dict)
            last_metrics = metrics
            gap = abs(metrics.SSDI - ssdi)

            output = GenerationOutput(
                n_ck=n_ck,
                matrix_df=make_matrix_df(n_ck),
                metrics=metrics,
                success=gap <= ssdi_error,
                iter_used=iter_counter,
                actual_alpha=estimate_pareto_alpha(n_ck),
                actual_beta=estimate_zipf_beta(n_ck),
                alpha=alpha,
                beta=beta,
                lcd_type=lcdtype,
                lds_type=ldstype,
                lcd_params=dict(lcd_params_current),
                lds_params=dict(lds_params_current),
                target_ssdi=float(ssdi),
                C=C,
                K=K,
                N=N,
                generator_variant=variant,
            )

            if gap < best_gap:
                best_gap = gap
                best_output = output
            if output.success:
                return output

            if metrics.SSDI < ssdi:
                ratio = (ssdi + EPS) / (metrics.SSDI + EPS)
                if structured:
                    # structured 下：半径不够时，优先增强 LDS 和整体集中度；
                    # 不让 lcd_scale 持续上冲，避免把低缺失目标拉成高 LCD。
                    lcd_scale *= min(1.015, 1.0 + 0.010 * np.log1p(ratio))
                    lds_scale *= min(1.14, 1.0 + 0.090 * np.log1p(ratio))
                    power_scale *= min(1.08, 1.0 + 0.045 * np.log1p(ratio))
                else:
                    lcd_scale *= min(1.08, 1.0 + 0.05 * np.log1p(ratio))
                    lds_scale *= min(1.12, 1.0 + 0.08 * np.log1p(ratio))
                    power_scale *= min(1.08, 1.0 + 0.05 * np.log1p(ratio))
            else:
                ratio = (metrics.SSDI + EPS) / (ssdi + EPS)
                if structured:
                    lcd_scale *= max(0.985, 1.0 - 0.010 * np.log1p(ratio))
                    lds_scale *= max(0.90, 1.0 - 0.070 * np.log1p(ratio))
                    power_scale *= max(0.95, 1.0 - 0.035 * np.log1p(ratio))
                else:
                    lcd_scale *= max(0.93, 1.0 - 0.04 * np.log1p(ratio))
                    lds_scale *= max(0.90, 1.0 - 0.07 * np.log1p(ratio))
                    power_scale *= max(0.94, 1.0 - 0.04 * np.log1p(ratio))

        if last_metrics is not None:
            lcd_params_current, lds_params_current = _local_param_correction(
                lcd_params_current, lds_params_current, last_metrics, ssdi, lcdtype, ldstype, C, K
            )
            structured = _is_structured_targeted(lcd_params_current, lds_params_current)

    assert best_output is not None
    return best_output


def count_empty_clients_and_labels(n_ck: np.ndarray) -> Tuple[int, int]:
    """Count empty clients (columns) and empty labels (rows)."""
    arr = np.asarray(n_ck)
    empty_clients = int(np.sum(arr.sum(axis=0) <= 0))
    empty_labels = int(np.sum(arr.sum(axis=1) <= 0))
    return empty_clients, empty_labels


def ensure_nonempty_labels(n_ck: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ensure every label/row has at least one sample by moving one sample from the largest cell."""
    arr = np.asarray(n_ck, dtype=int).copy()
    row_sums = arr.sum(axis=1)
    for i, s in enumerate(row_sums):
        if s > 0:
            continue
        donor_idx = np.unravel_index(np.argmax(arr), arr.shape)
        if arr[donor_idx] <= 1:
            continue
        arr[donor_idx] -= 1
        j = int(rng.integers(0, arr.shape[1]))
        arr[i, j] += 1
    return arr


def validate_support_constraints(
    n_ck: np.ndarray,
    *,
    allow_empty_label: bool = True,
    max_empty_clients: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate simple support constraints used by generate.py."""
    ec, el = count_empty_clients_and_labels(n_ck)
    ok = True
    violations = []
    if not allow_empty_label and el > 0:
        ok = False
        violations.append('empty_label')
    if max_empty_clients is not None and ec > int(max_empty_clients):
        ok = False
        violations.append('empty_client_cap')
    return {'ok': ok, 'empty_client_count': ec, 'empty_label_count': el, 'violations': violations}
