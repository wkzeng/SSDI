from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import numpy as np
import pandas as pd

from .core import (
    GenerationOutput,
    compute_ssdi_metrics,
    estimate_pareto_alpha,
    estimate_zipf_beta,
    generate_ssdi_matrix_array,
    largest_remainder_rounding,
    theoretical_vmax,
)
from .defaults import get_default_params

__all__ = [
    "compute_ssdi_metrics",
    "estimate_pareto_alpha",
    "estimate_zipf_beta",
    "largest_remainder_rounding",
    "theoretical_vmax",
    "get_combo_params",
    "generate_ssdi_matrix",
    "generate_ssdi_matrix_structured",
    "generate_9_methods_and_analyse",
    "generate_9_methods_and_analyse_structured",
]

EPS = 1e-12
LCD_TYPES = ("client", "class", "joint")
LDS_TYPES = ("client", "special", "lowrank")



def _normalize_seed(seed, default=42):
    return int(default if seed is None else seed)



def get_combo_params(
    target_ssdi: float,
    lcd_type: str,
    lds_type: str,
    param_manager=None,
    C: int = 10,
    K: int = 20,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Return merged LCD/LDS default parameter packs.

    param_manager is accepted for backward compatibility and ignored.
    """
    del param_manager
    lcd_params = get_default_params(target_ssdi, f"lcd_{lcd_type}", C=C, K=K, seed=2*seed)
    lds_params = get_default_params(target_ssdi, f"lds_{lds_type}", C=C, K=K, seed=2*seed+1)

    alpha_value = float(lcd_params.get("alpha", lds_params.get("alpha", 2.0)))
    beta_value = float(lcd_params.get("beta", lds_params.get("beta", 1.0)))

    lcd_specific = {k: v for k, v in lcd_params.items() if k not in {"alpha", "beta", "_randomized", "mechanism", "_default_version"}}
    lds_specific = {k: v for k, v in lds_params.items() if k not in {"alpha", "beta", "_randomized", "mechanism", "_default_version"}}

    return {
        "alpha": alpha_value,
        "beta": beta_value,
        "lcd_params": lcd_specific,
        "lds_params": lds_specific,
    }


def _candidate_ssdi_list(target_ssdi: float) -> List[float]:
    grid = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]
    t = float(target_ssdi)
    return [x for x in sorted(grid, key=lambda x: (abs(x - t), x)) if round(x, 3) != round(t, 3)]


def _stage_code(stage: Optional[str]) -> Optional[str]:
    mapping = {
        "current target parameters": "normal",
        "borrowed SSDI parameters": "borrowed",
        "random fallback": "random",
        "structured random fallback": "structured_random",
    }
    if stage is None:
        return None
    return mapping.get(stage, stage)


def _stage_label_cn(stage: Optional[str]) -> Optional[str]:
    mapping = {
        "current target parameters": "当前目标参数",
        "borrowed SSDI parameters": "借用邻近 SSDI 参数",
        "random fallback": "随机兜底",
        "structured random fallback": "结构化随机兜底",
        "normal": "当前目标参数",
        "borrowed": "借用邻近 SSDI 参数",
        "random": "随机兜底",
        "structured_random": "结构化随机兜底",
    }
    if stage is None:
        return None
    return mapping.get(stage, stage)


def _single_output_basename(client: int, label: int, datasize: int, ssdi: float, lcdtype: str, ldstype: str) -> str:
    ssdi_str = f"{float(ssdi):.3f}".rstrip("0").rstrip(".")
    return f"c{label}_k{client}_n{datasize}_ssdi{ssdi_str}_{lcdtype}_{ldstype}"




def _print_single_generation_summary(details: Dict[str, Any], saved_paths: Optional[Dict[str, Any]], verbose: bool = True):
    if not verbose:
        return

    print("已生成矩阵.")
    print()
    print("基本信息:")
    print(f"- Clients: {details.get('K')}")
    print(f"- Labels: {details.get('C')}")
    print(f"- Total size: {details.get('N')}")
    print(f"- Target SSDI: {float(details.get('target_ssdi', np.nan)):.3f}")
    print(f"- Actual SSDI: {float(details.get('SSDI', np.nan)):.3f}")
    print(f"- LCD: {float(details.get('LCD', np.nan)):.3f}")
    print(f"- LDS: {float(details.get('LDS', np.nan)):.3f}")
    print(f"- DSR: {float(details.get('DSR', np.nan)):.3f}")
    print(f"- Actual theta (deg): {float(np.degrees(details.get('actual_theta', 0.0))):.1f}")
    print(f"- Target LCD: {details.get('target_lcd')}")
    print(f"- Target LDS: {details.get('target_lds')}")
    print(f"- Target theta (deg): {float(np.degrees(details.get('target_theta', 0.0))):.1f}")

    print()
    print("机制:")
    print(f"- LCD type: {details.get('lcd_type')}")
    print(f"- LDS type: {details.get('lds_type')}")
    print(f"- Structure mode: {details.get('structure_mode')}")

    print()
    print("生成结果:")
    status = "成功命中目标范围" if bool(details.get("success", False)) else "返回最接近目标的 best-effort"
    print(f"- 状态: {status}")
    print(f"- 使用生成器: {details.get('generator_variant')}")
    print(f"- 搜索来源: {details.get('source_stage')}")
    print(f"- 偏差: {float(details.get('ssdi_gap', np.nan)):.3f}")
    print(f"- Alpha used: {details.get('alpha_used')}")
    print(f"- Beta used: {details.get('beta_used')}")
    print(f"- Estimated alpha: {details.get('actual_alpha')}")
    print(f"- Estimated beta: {details.get('actual_beta')}")
    print(f"- Missing rate: {details.get('missing_rate')}")
    print(f"- Iterations used: {details.get('iter_used')}")
    print(f"- Total budget used: {details.get('total_budget_used')}")
    print(f"- Returned from: {details.get('returned_from')}")
    print(f"- Dispatch case: {details.get('dispatch_case')}")
    print(f"- Attempt stage/path: {details.get('attempt_stage')} | {details.get('attempt_path')}")
    print(f"- Time used (s): {details.get('time_used', details.get('time_elapsed'))}")
    print(f"- LCD params: {details.get('lcd_params')}")
    print(f"- LDS params: {details.get('lds_params')}")
    print(f"- Structure score: {details.get('structure_score')}")
    print(f"- Structure success: {details.get('structure_success')}")
    print(f"- Dominance target: {details.get('dominance_target')}")
    print(f"- Dominance actual: {details.get('dominance_actual')}")
    print(f"- Domain stage: {details.get('domain_stage')}")

    if saved_paths:
        print()
        print("文件已保存到:")
        print(saved_paths.get("output_dir"))

        existing = []
        for key in ["npy", "txt", "metrics_csv", "metrics_json", "matrix_csv"]:
            path = saved_paths.get(key)
            if path:
                existing.append(path)

        if existing:
            print()
            print("已保存文件:")
            for path in existing:
                print(f"- {os.path.basename(path)}")

def _make_matrix_df_local(n_ck: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        n_ck,
        index=[f"class_{i}" for i in range(n_ck.shape[0])],
        columns=[f"client_{j}" for j in range(n_ck.shape[1])],
    )


def _count_empty_clients_and_labels_local(n_ck: np.ndarray) -> Tuple[int, int]:
    return int(np.sum(n_ck.sum(axis=0) <= 0)), int(np.sum(n_ck.sum(axis=1) <= 0))


def _compute_ssdi0_local(C: int, K: int, N: int, lambda_: float = 1.2) -> float:
    """Engineering routing threshold, not a theoretical boundary."""
    del N, lambda_
    m = max(1, min(int(C), int(K)))
    mismatch = abs(int(C) - int(K)) / max(1, m)
    return float(np.clip(max(0.75, 1.0 - 0.08 * mismatch), 0.0, 1.0))


def _truncated_rank_weights(n: int, exponent: float) -> np.ndarray:
    exp = float(max(0.1, exponent))
    ranks = np.arange(1, n + 1, dtype=float)
    w = 1.0 / np.power(ranks, exp)
    w /= w.sum()
    return w


def _allocate_targets(total: int, n: int, exponent: float, minimum: int = 0) -> np.ndarray:
    total = int(total)
    n = int(n)
    minimum = int(minimum)

    base = np.full(n, minimum, dtype=int)
    rem = total - minimum * n
    if rem <= 0:
        return base

    w = _truncated_rank_weights(n, exponent)
    extra = largest_remainder_rounding(w, rem)
    return base + extra


def _dominant_label_order_for_mechanism(K: int, C: int, ldstype: str) -> np.ndarray:
    if ldstype == 'client':
        return np.arange(K) % C
    if ldstype == 'lowrank':
        block = max(1, K // max(1, C))
        return np.array([(j // block) % C for j in range(K)], dtype=int)
    # special / fallback
    return np.arange(K) % C

def _fmt_mean_std(series: pd.Series, decimals: int) -> str:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) == 0:
        return "N/A"
    mean = series.mean()
    std = series.std()
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"

def _default_main_out_dir(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"./{prefix}_{timestamp}"
    os.makedirs(path, exist_ok=True)
    return path

def _output_to_record(
    output: GenerationOutput,
    *,
    source_stage: str,
    source_ssdi: Optional[float],
) -> Dict[str, Any]:
    d = output.to_record()
    record = {
        "success": bool(output.success),
        "SSDI": float(output.metrics.SSDI),
        "LCD": float(output.metrics.LCD),
        "LDS": float(output.metrics.LDS),
        "DSR": float(output.metrics.DSR),
        "missing_rate": float(output.metrics.missing_rate),
        "actual_alpha": float(output.actual_alpha),
        "actual_beta": float(output.actual_beta),
        "iter_used": int(output.iter_used),
        "time_elapsed": np.nan,
        "time_used": np.nan,
        "n_ck": output.n_ck,
        "matrix_df": output.matrix_df,
        "lcd_type": output.lcd_type,
        "lds_type": output.lds_type,
        "alpha_used": float(output.alpha),
        "beta_used": float(output.beta),
        "lcd_params": dict(output.lcd_params),
        "lds_params": dict(output.lds_params),
        "source_stage": source_stage,
        "search_source": _stage_code(source_stage),
        "source_ssdi": None if source_ssdi is None else float(source_ssdi),
        "generator_variant": output.generator_variant,
        "target_ssdi": float(output.target_ssdi),
        "actual_ssdi": float(output.metrics.SSDI),
        "C": int(output.C),
        "K": int(output.K),
        "N": int(output.N),
        "guaranteed_result": True,
        # structured-related keys always present for schema stability
        "structure_mode": None,
        "structure_score": None,
        "structure_success": None,
        "dominance_target": None,
        "dominance_actual": None,
        "target_lcd": None,
        "target_lds": None,
        "target_dsr": None,
        "domain_stage": None,
        "is_best_effort": None,
        # ===== 新增 =====
        "total_budget_used": None,
        "returned_from": None,
    }
    return record

def _merge_param_pack(
    target_ssdi: float,
    lcdtype: str,
    ldstype: str,
    C: int,
    K: int,
    alpha: Optional[float],
    beta: Optional[float],
    lcd_params: Optional[Dict[str, Any]],
    lds_params: Optional[Dict[str, Any]],
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    combo = get_combo_params(target_ssdi, lcdtype, ldstype, C=C, K=K, seed=seed)
    if alpha is not None:
        combo["alpha"] = float(alpha)
    if beta is not None:
        combo["beta"] = float(beta)
    if lcd_params:
        combo["lcd_params"].update(lcd_params)
    if lds_params:
        combo["lds_params"].update(lds_params)
    return combo

def _run_core_with_pack(
    *,
    client: int,
    label: int,
    datasize: int,
    ssdi: float,
    lcdtype: str,
    ldstype: str,
    pack: Dict[str, Any],
    ssdi_error: float,
    seed: Optional[int],
    max_iters: int,
    source_stage: str,
    source_ssdi: Optional[float],
    preferred_variant: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    output = generate_ssdi_matrix_array(
        client=client,
        label=label,
        datasize=datasize,
        ssdi=ssdi,
        lcdtype=lcdtype,
        ldstype=ldstype,
        alpha=pack["alpha"],
        beta=pack["beta"],
        lcd_params=pack["lcd_params"],
        lds_params=pack["lds_params"],
        ssdi_error=ssdi_error,
        seed=seed,
        max_iters=max_iters,
        get_default_params_fn=get_default_params,
        preferred_variant=preferred_variant,
    )
    record = _output_to_record(output, source_stage=source_stage, source_ssdi=source_ssdi)
    record["time_elapsed"] = float(time.time() - t0)
    record["time_used"] = record["time_elapsed"]
    record["ssdi_gap"] = abs(float(record["SSDI"]) - float(ssdi))
    return record


def generate_ssdi_matrix(
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
    ssdi_error: float = 0.02,
    seed: Optional[int] = 42,
    max_iters: int = 160,
    return_details: bool = False,
    save: bool = True,
    save_metrics: bool = False,
    save_csv: bool = False,
    output_dir: str = "./single_outputs",
    verbose: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    start_time = time.time()
    seed = _normalize_seed(seed)
    C, K, N = int(label), int(client), int(datasize)
    target_ssdi = float(ssdi)
    rng = np.random.default_rng(seed)

    best_details: Optional[Dict[str, Any]] = None
    best_gap = float("inf")

    def consider(record: Dict[str, Any]) -> None:
        nonlocal best_details, best_gap
        gap = abs(float(record["SSDI"]) - target_ssdi)
        record["ssdi_gap"] = float(gap)
        record["target_ssdi"] = target_ssdi
        record["C"] = C
        record["K"] = K
        record["N"] = N
        record["guaranteed_result"] = True
        if gap < best_gap:
            best_gap = gap
            best_details = record

    base_pack = _merge_param_pack(target_ssdi, lcdtype, ldstype, C, K, alpha, beta, lcd_params, lds_params,seed=seed)
    preferred_variant = "v2" if target_ssdi <= 0.7 else ("v1" if rng.random() < 0.7 else "v2")
    consider(
        _run_core_with_pack(
            client=client,
            label=label,
            datasize=datasize,
            ssdi=target_ssdi,
            lcdtype=lcdtype,
            ldstype=ldstype,
            pack=base_pack,
            ssdi_error=ssdi_error,
            seed=seed,
            max_iters=max_iters,
            source_stage="current target parameters",
            source_ssdi=target_ssdi,
            preferred_variant=preferred_variant,
        )
    )

    if best_gap > ssdi_error:
        for idx, cand in enumerate(_candidate_ssdi_list(target_ssdi)):
            cand_seed = _stable_derive_seed(seed, "borrowed", idx, cand)
            cand_pack = get_combo_params(cand, lcdtype, ldstype, C=C, K=K, seed=cand_seed)
            rec = _run_core_with_pack(
                client=client,
                label=label,
                datasize=datasize,
                ssdi=target_ssdi,
                lcdtype=lcdtype,
                ldstype=ldstype,
                pack=cand_pack,
                ssdi_error=ssdi_error,
                seed=cand_seed,
                max_iters=max_iters,
                source_stage="borrowed SSDI parameters",
                source_ssdi=cand,
            )
            consider(rec)
            if best_gap <= ssdi_error:
                break

    if best_gap > ssdi_error:
        for idx in range(8):
            rand_seed = _stable_derive_seed(seed, "random_fallback", idx)
            rand_alpha, rand_beta, rand_lcd, rand_lds = _sample_target_aware_random_params(
                target_ssdi, lcdtype, ldstype, C, K, rng
            )
            rec = _run_core_with_pack(
                client=client,
                label=label,
                datasize=datasize,
                ssdi=target_ssdi,
                lcdtype=lcdtype,
                ldstype=ldstype,
                pack={"alpha": rand_alpha, "beta": rand_beta, "lcd_params": rand_lcd, "lds_params": rand_lds},
                ssdi_error=ssdi_error,
                seed=rand_seed,
                max_iters=max_iters,
                source_stage="random fallback",
                source_ssdi=target_ssdi,
            )
            consider(rec)
            if best_gap <= ssdi_error:
                break

    if best_details is None:
        raise RuntimeError("Internal error: no candidate matrix was generated.")

    best_details["is_best_effort"] = not bool(best_details.get("success", False))
    best_details["time_used"] = float(time.time() - start_time)
    best_details["time_elapsed"] = best_details["time_used"]

    saved_paths = None
    if save:
        saved_paths = _save_single_matrix_outputs(
            best_details["n_ck"],
            best_details,
            client=client,
            label=label,
            datasize=datasize,
            ssdi=ssdi,
            lcdtype=lcdtype,
            ldstype=ldstype,
            output_dir=output_dir,
            save_metrics=save_metrics,
            save_csv=save_csv,
        )
        best_details["saved_paths"] = saved_paths
        best_details["output_dir"] = saved_paths["output_dir"]
        best_details["npy_path"] = saved_paths["npy"]
        best_details["metrics_csv_path"] = saved_paths["metrics_csv"]
        best_details["matrix_csv_path"] = saved_paths.get("matrix_csv")

    _print_single_generation_summary(best_details, saved_paths, verbose=verbose)

    if return_details:
        return best_details["matrix_df"], best_details
    return best_details["matrix_df"]





def _resolve_mode_value_local(structure_bias: float) -> int:
    b = float(np.clip(structure_bias, -1.0, 1.0))

    # -1 这边仍然对0.98端点触发
    if b < -0.97:
        return -1

    # +1 这边改成 > 0.89 就触发 mode=1
    if b > 0.89:
        return 1

    return 0



def _zipf_pareto_refill_zeros(
    n_ck: np.ndarray,
    *,
    rng: np.random.Generator,
    alpha: float = 1.2,
    beta: float = 1.2,
    max_fill_each: int = 3,
) -> np.ndarray:
    """
    Fill zero cells with small positive counts using a Zipf×Pareto-like preference.

    作用：
    - 给 mode=-1 / near1-left 一个真正可执行的“去 0”动作；
    - donor 取自较大值位置，尽量少破坏整体结构。
    """
    n = n_ck.copy().astype(int)
    zeros = np.argwhere(n <= 0)
    if zeros.size == 0:
        return n

    row_mass = n.sum(axis=1).astype(float) + 1.0
    col_mass = n.sum(axis=0).astype(float) + 1.0

    row_rank = np.argsort(np.argsort(-row_mass)) + 1
    col_rank = np.argsort(np.argsort(-col_mass)) + 1

    row_pref = 1.0 / np.power(row_rank.astype(float), max(beta, 1e-6))
    col_pref = 1.0 / np.power(col_rank.astype(float), max(alpha, 1e-6))

    cell_pref = np.array([row_pref[i] * col_pref[j] for i, j in zeros], dtype=float)
    if np.all(cell_pref <= 0):
        cell_pref = np.ones(len(zeros), dtype=float)
    cell_pref = cell_pref / cell_pref.sum()

    order = rng.choice(len(zeros), size=len(zeros), replace=False, p=cell_pref)

    for idx in order:
        i, j = zeros[idx]
        i = int(i)
        j = int(j)

        donor_candidates = np.argwhere(n > 1)
        if donor_candidates.size == 0:
            break

        donor_vals = np.array([n[r, c] for r, c in donor_candidates], dtype=float)
        donor_prob = donor_vals / donor_vals.sum()
        d_idx = int(rng.choice(len(donor_candidates), p=donor_prob))
        di, dj = donor_candidates[d_idx]
        di = int(di)
        dj = int(dj)

        if n[di, dj] <= 1:
            continue

        amt = int(min(max_fill_each, max(1, n[di, dj] // 20)))
        amt = min(amt, int(n[di, dj] - 1))
        if amt <= 0:
            continue

        n[di, dj] -= amt
        n[i, j] += amt

    return n


def _mode_minus_one_repair(n_ck: np.ndarray) -> np.ndarray:
    """
    Deterministic repair for mode=-1:
    if any cell is zero, borrow 1 count from the largest cell in the same column,
    otherwise from the largest cell in the same row.
    """
    n = n_ck.copy().astype(int)
    C, K = n.shape
    zeros = np.argwhere(n <= 0)

    for i, j in zeros:
        i = int(i)
        j = int(j)

        donor_row = int(np.argmax(n[:, j]))
        if n[donor_row, j] > 1:
            n[donor_row, j] -= 1
            n[i, j] += 1
            continue

        donor_col = int(np.argmax(n[i, :]))
        if n[i, donor_col] > 1:
            n[i, donor_col] -= 1
            n[i, j] += 1

    return n


def _near1_left_controlled_lds_up_move(
    n_ck: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Very small backward correction for LDS:
    - choose one diagonal peak, larger peaks more likely
    - collect 1 count each from 2~4 existing small nonzero supports
    - add them back to that peak
    This is only for overshoot correction, so the step is intentionally tiny.
    """
    out = n_ck.copy().astype(int)
    C, K = out.shape
    m = min(C, K)

    diag_vals = np.array([int(out[i, i]) for i in range(m)], dtype=float)
    if np.all(diag_vals <= 0):
        return out

    peak_prob = diag_vals / diag_vals.sum()
    peak_idx = int(rng.choice(np.arange(m), p=peak_prob))

    rows_desc, cols_desc = _largest_rows_cols(out)
    top_rows = set(rows_desc[: max(1, int(np.ceil(0.2 * C)))])
    top_cols = set(cols_desc[: max(1, int(np.ceil(0.2 * K)))])

    donors = []
    donor_scores = []
    for i in range(C):
        for j in range(K):
            v = int(out[i, j])
            if v <= 1:
                continue
            if i == j:
                continue
            if (i in top_rows) or (j in top_cols):
                donors.append((i, j, v))
                donor_scores.append(1.0 / max(1.0, float(v)))  # smaller supports easier to shave

    if not donors:
        return out

    donor_scores = np.asarray(donor_scores, dtype=float)
    donor_scores = donor_scores / donor_scores.sum()

    split_n = min(len(donors), int(rng.integers(2, 5)))  # tiny step
    chosen_idx = rng.choice(
        np.arange(len(donors)),
        size=split_n,
        replace=False,
        p=donor_scores,
    )

    moved = 0
    for idx in chosen_idx:
        di, dj, dv = donors[int(idx)]
        di, dj = int(di), int(dj)
        if out[di, dj] <= 1:
            continue
        out[di, dj] -= 1
        moved += 1

    if moved > 0:
        out[peak_idx, peak_idx] += int(moved)

    return out

def _near1_left_controlled_lcd_down_move(
    n_ck: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Very small backward correction for LCD:
    - choose ONE high-value zero
    - take 1 count each from 2~4 existing small nonzero cells
    - fill that zero into a tiny positive support
    This is only for overshoot correction, so the step is intentionally tiny.
    """
    out = n_ck.copy().astype(int)
    C, K = out.shape

    zeros = _top_zero_positions_by_value(out, topk=48)
    if not zeros:
        return out

    # choose one high-value zero, but not necessarily the absolute first one
    z_keep = min(len(zeros), 8)
    zi, zj = zeros[int(rng.integers(0, z_keep))]
    zi, zj = int(zi), int(zj)

    rows_desc, cols_desc = _largest_rows_cols(out)
    top_rows = set(rows_desc[: max(1, int(np.ceil(0.2 * C)))])
    top_cols = set(cols_desc[: max(1, int(np.ceil(0.2 * K)))])

    donors = []
    donor_scores = []
    for i in range(C):
        for j in range(K):
            v = int(out[i, j])
            if v <= 1:
                continue
            if i == j:
                continue
            if (i in top_rows) or (j in top_cols):
                donors.append((i, j, v))
                donor_scores.append(1.0 / max(1.0, float(v)))  # small supports preferred

    if not donors:
        return out

    donor_scores = np.asarray(donor_scores, dtype=float)
    donor_scores = donor_scores / donor_scores.sum()

    split_n = min(len(donors), int(rng.integers(2, 5)))  # tiny step: only 2~4 cells contribute
    chosen_idx = rng.choice(
        np.arange(len(donors)),
        size=split_n,
        replace=False,
        p=donor_scores,
    )

    moved = 0
    for idx in chosen_idx:
        di, dj, dv = donors[int(idx)]
        di, dj = int(di), int(dj)
        if out[di, dj] <= 1:
            continue
        out[di, dj] -= 1
        moved += 1

    if moved > 0:
        out[zi, zj] += int(moved)

    return out



def _near1_left_lcd_gap_profile(abs_d_lcd: float) -> int:
    """
    [MOD] 左侧 LCD gap 分层：
    gap 越大，一次做的“挖空/填充”格子数越多。
    """
    if abs_d_lcd > 0.18:
        return 6
    if abs_d_lcd > 0.10:
        return 4
    if abs_d_lcd > 0.05:
        return 2
    return 1


def _near1_left_lds_gap_profile(abs_d_lds: float) -> int:
    """
    [MOD] 左侧 LDS gap 分层：
    gap 越大，一次做的“削峰/补峰”次数越多。
    """
    if abs_d_lds > 0.08:
        return 3
    if abs_d_lds > 0.04:
        return 2
    return 1





def _build_sparse_extremal_seed(C: int, K: int, N: int, alpha: float, beta: float, ldstype: str) -> np.ndarray:
    m = min(C, K)
    n_ck = np.zeros((C, K), dtype=int)
    active_cols = list(range(m))
    row_targets = _allocate_targets(N, C, exponent=beta, minimum=1)
    col_targets_active = _allocate_targets(N, m, exponent=alpha, minimum=1)
    for j in range(m):
        c = j % C
        n_ck[c, j] = col_targets_active[j]
    # rebalance rows by moving active-column mass among active columns only
    row_now = n_ck.sum(axis=1)
    deficits = row_targets - row_now
    if np.any(deficits > 0):
        for i in np.where(deficits > 0)[0]:
            rem = int(deficits[i])
            cols = np.argsort(-n_ck.sum(axis=0))
            for j in cols:
                if rem <= 0:
                    break
                donor_row = int(np.argmax(n_ck[:, j]))
                if donor_row == i or n_ck[donor_row, j] <= 1:
                    continue
                take = min(rem, n_ck[donor_row, j] - 1)
                n_ck[donor_row, j] -= take
                n_ck[i, j] += take
                rem -= take
    return n_ck


def _matrix_to_record_like(seed_record: Dict[str, Any], n_ck: np.ndarray) -> Dict[str, Any]:
    metrics = compute_ssdi_metrics(n_ck)
    row_totals = n_ck.sum(axis=1)
    col_totals = n_ck.sum(axis=0)
    rec = dict(seed_record)
    rec['n_ck'] = n_ck.astype(int, copy=True)
    rec['matrix_df'] = _make_matrix_df_local(n_ck)
    rec['SSDI'] = float(metrics['SSDI'])
    rec['LCD'] = float(metrics['LCD'])
    rec['LDS'] = float(metrics['LDS'])
    rec['DSR'] = float(metrics['DSR'])
    rec['missing_rate'] = float(metrics['missing_rate'])
    rec['actual_alpha'] = float(estimate_pareto_alpha(n_ck))
    rec['actual_beta'] = float(estimate_zipf_beta(n_ck))
    rec['iter_used'] = int(rec.get('iter_used', 0))
    rec['time_elapsed'] = float(rec.get('time_elapsed', 0.0))
    rec['time_used'] = float(rec.get('time_used', rec['time_elapsed']))
    rec['search_source'] = _stage_code(rec.get('source_stage'))
    rec.setdefault('success', False)
    rec.setdefault("total_budget_used", None)
    rec.setdefault("returned_from", None)
    rec.setdefault('C', int(n_ck.shape[0]))
    rec.setdefault('K', int(n_ck.shape[1]))
    rec.setdefault('N', int(n_ck.sum()))
    ec, el = _count_empty_clients_and_labels_local(n_ck)
    rec['empty_client_count'] = ec
    rec['empty_label_count'] = el
    rec['ssdi_gap'] = abs(float(rec.get('target_ssdi', rec.get('SSDI', 0.0))) - rec['SSDI'])
    return rec


def _construct_exact_one_record(*, client: int, label: int, datasize: int, ssdi: float, lcdtype: str, ldstype: str, alpha: Optional[float], beta: Optional[float], lcd_params: Optional[Dict[str, Any]], lds_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    C, K, N = int(label), int(client), int(datasize)
    m = min(C, K)
    n_ck = np.zeros((C, K), dtype=int)
    base = N // m
    rem = N % m
    for i in range(m):
        n_ck[i, i] = base + (1 if i < rem else 0)
    return _matrix_to_record_like({
        'target_ssdi': float(ssdi), 'lcd_type': lcdtype, 'lds_type': ldstype,
        'alpha_used': alpha, 'beta_used': beta, 'lcd_params': lcd_params or {}, 'lds_params': lds_params or {},
        'source_stage': 'direct exact extremal', 'generator_variant': 'direct_exact_one', 'C': C, 'K': K, 'N': N,
    }, n_ck)


def _construct_exact_zero_record(
    *,
    client: int,
    label: int,
    datasize: int,
    ssdi: float,
    lcdtype: str,
    ldstype: str,
    alpha: Optional[float],
    beta: Optional[float],
    lcd_params: Optional[Dict[str, Any]],
    lds_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    C, K, N = int(label), int(client), int(datasize)

    row_targets = _allocate_targets(
        N, C,
        exponent=beta if beta is not None else 1.0,
        minimum=1,
    )
    col_targets = _allocate_targets(
        N, K,
        exponent=alpha if alpha is not None else 1.0,
        minimum=1,
    )

    outer = np.outer(row_targets, col_targets) / max(1, N)

    # 关键修复:largest_remainder_rounding 需要传 (P, N)
    n_ck = largest_remainder_rounding(outer, N)


    # 守恒修补行列和：只做“搬运”，不改变总和
    max_repair_rounds = C * K * 4

    for _ in range(max_repair_rounds):
        row_sum = n_ck.sum(axis=1)
        col_sum = n_ck.sum(axis=0)

        row_diff = row_targets - row_sum   # >0 means row deficit
        col_diff = col_targets - col_sum   # >0 means col deficit

        if np.all(row_diff == 0) and np.all(col_diff == 0):
            break

        deficit_rows = np.where(row_diff > 0)[0]
        deficit_cols = np.where(col_diff > 0)[0]
        surplus_rows = np.where(row_diff < 0)[0]
        surplus_cols = np.where(col_diff < 0)[0]

        if (
            len(deficit_rows) == 0 or len(deficit_cols) == 0
            or len(surplus_rows) == 0 or len(surplus_cols) == 0
        ):
            break

        moved_any = False

        # 从“行超+列超”的格子挪到“行缺+列缺”的格子
        for i_to in deficit_rows:
            if row_diff[i_to] <= 0:
                continue
            for j_to in deficit_cols:
                if row_diff[i_to] <= 0 or col_diff[j_to] <= 0:
                    continue

                # 优先从最富余的 donor 位置搬
                donor_candidates = []
                donor_scores = []
                for i_from in surplus_rows:
                    if -row_diff[i_from] <= 0:
                        continue
                    for j_from in surplus_cols:
                        if -col_diff[j_from] <= 0:
                            continue
                        if n_ck[i_from, j_from] <= 0:
                            continue
                        donor_candidates.append((i_from, j_from))
                        donor_scores.append(float(n_ck[i_from, j_from]))

                if not donor_candidates:
                    continue

                donor_scores = np.asarray(donor_scores, dtype=float)
                donor_idx = int(np.argmax(donor_scores))
                i_from, j_from = donor_candidates[donor_idx]

                take = min(
                    int(row_diff[i_to]),
                    int(col_diff[j_to]),
                    int(-row_diff[i_from]),
                    int(-col_diff[j_from]),
                    int(n_ck[i_from, j_from]),
                )

                if take <= 0:
                    continue

                n_ck[i_from, j_from] -= take
                n_ck[i_to, j_to] += take

                moved_any = True

                # 立即刷新 diff，避免用旧状态继续搬
                row_sum = n_ck.sum(axis=1)
                col_sum = n_ck.sum(axis=0)
                row_diff = row_targets - row_sum
                col_diff = col_targets - col_sum

        if not moved_any:
            break



    return _matrix_to_record_like(
        {
            'target_ssdi': float(ssdi),
            'lcd_type': lcdtype,
            'lds_type': ldstype,
            'alpha_used': alpha,
            'beta_used': beta,
            'lcd_params': lcd_params or {},
            'lds_params': lds_params or {},
            'source_stage': 'direct exact iid',
            'generator_variant': 'direct_exact_zero',
            'C': C,
            'K': K,
            'N': N,
        },
        n_ck,
    )



def _build_full_support_extreme_seed(C: int, K: int, N: int, alpha: float, beta: float, ldstype: str) -> np.ndarray:
    col_targets = _allocate_targets(N, K, exponent=alpha, minimum=C)
    row_targets = _allocate_targets(N, C, exponent=beta, minimum=K)
    n_ck = np.ones((C, K), dtype=int)
    row_rem = row_targets - K
    dom = _dominant_label_order_for_mechanism(K, C, ldstype)
    for j in np.argsort(-col_targets):
        rem = int(col_targets[j] - C)
        if rem <= 0:
            continue
        c = int(dom[j])
        give = min(rem, int(max(0, row_rem[c])))
        if give > 0:
            n_ck[c, j] += give
            row_rem[c] -= give
            rem -= give
        while rem > 0:
            i = int(np.argmax(row_rem))
            if row_rem[i] <= 0:
                i = c
            take = min(rem, max(1, int(max(0, row_rem[i])))) if row_rem[i] > 0 else rem
            n_ck[i, j] += take
            row_rem[i] -= take
            rem -= take
    # distribute any leftover row deficits
    for i in np.where(row_rem > 0)[0]:
        rem = int(row_rem[i])
        cols = np.argsort(-col_targets)
        t = 0
        while rem > 0:
            j = int(cols[t % K])
            n_ck[i, j] += 1
            rem -= 1
            t += 1
    return n_ck



def _build_combined_row(details: Dict[str, Any], rep: int, structured: bool) -> Dict[str, Any]:
    row = {
        "C": details.get("C"),
        "K": details.get("K"),
        "N": details.get("N"),
        "rep": rep,
        "target_SSDI": details.get("target_ssdi"),
        "target_ssdi": details.get("target_ssdi"),
        "SSDI": details.get("SSDI"),
        "actual_ssdi": details.get("SSDI"),
        "LCD": details.get("LCD"),
        "lcd": details.get("LCD"),
        "LDS": details.get("LDS"),
        "lds": details.get("LDS"),
        "DSR": details.get("DSR"),
        "dsr": details.get("DSR"),
        "missing_rate": details.get("missing_rate"),
        "actual_alpha": details.get("actual_alpha"),
        "actual_beta": details.get("actual_beta"),
        "iter_used": details.get("iter_used"),
        "time_elapsed": details.get("time_used", details.get("time_elapsed")),
        "time_used": details.get("time_used", details.get("time_elapsed")),
        "success": details.get("success"),
        "lcd_type": details.get("lcd_type"),
        "lds_type": details.get("lds_type"),
        "lcd_params": json.dumps(details.get("lcd_params", {}), ensure_ascii=False),
        "lds_params": json.dumps(details.get("lds_params", {}), ensure_ascii=False),
        "alpha_used": details.get("alpha_used"),
        "beta_used": details.get("beta_used"),
        "generator_variant": details.get("generator_variant"),
        "search_source": details.get("search_source"),
        "source_stage": details.get("source_stage"),
        "source_ssdi": details.get("source_ssdi"),
        "ssdi_gap": details.get("ssdi_gap"),
        "is_best_effort": details.get("is_best_effort"),
    }
    if structured:
        row.update({
            "structure_mode": details.get("structure_mode"),
            "structure_bias": details.get("structure_bias"),
            "structure_score": details.get("structure_score"),
            "structure_success": details.get("structure_success"),
            "dominance_target": details.get("dominance_target"),
            "dominance_actual": details.get("dominance_actual"),
            "domain_stage": details.get("domain_stage"),
            "target_lcd": details.get("target_lcd"),
            "target_lds": details.get("target_lds"),
            "target_dsr": details.get("target_dsr"),
            "target_theta": details.get("target_theta"),
            "actual_theta": details.get("actual_theta"),
            "theta_gap": details.get("theta_gap"),
            "target_missing_rate": details.get("target_missing_rate"),
            "missing_tol": details.get("missing_tol"),
            "missing_rate_gap": details.get("missing_rate_gap"),
            "missing_rate_overflow": details.get("missing_rate_overflow"),

            # ===== 新增:失败原因字段,写入 combined_results.csv =====
            # 备注:
            # 这几个字段前面在 structured score / success 诊断里已经算出来了,
            # 之前只是没有导出到批量结果表中.
            "failure_primary": details.get("failure_primary"),
            "failure_structure_detail": details.get("failure_structure_detail"),
            "optimization_hint": details.get("optimization_hint"),
            "failure_note": details.get("failure_note"),
            # ===== 新增结束 =====




        })
    else:
        row.update({
            "structure_mode": None,
            "structure_bias": None,
            "structure_score": None,
            "structure_success": None,
            "dominance_target": None,
            "dominance_actual": None,
            "domain_stage": None,
            "target_lcd": None,
            "target_lds": None,
            "target_dsr": None,
            "target_theta": None,
            "actual_theta": None,
            "theta_gap": None,
            "target_missing_rate": None,
            "missing_tol": None,
            "missing_rate_gap": None,
            "missing_rate_overflow": None,

            # ===== 新增:保持 schema 稳定,非 structured 时也保留这些列 =====
            "failure_primary": None,
            "failure_structure_detail": None,
            "optimization_hint": None,
            "failure_note": None,
            # ===== 新增结束 =====


        })
    return row

def _build_mechanism_stats(combined_df: pd.DataFrame, *, structured: bool) -> pd.DataFrame:
    if combined_df.empty:
        return pd.DataFrame()

    success_col = "structure_success" if structured and "structure_success" in combined_df.columns else "success"
    work = combined_df.copy()
    work[success_col] = pd.to_numeric(work[success_col], errors="coerce").fillna(0.0).astype(float)

    group_cols = ["lcd_type", "lds_type"]
    include_structure_bias = (
        structured
        and ("structure_bias" in work.columns)
        and (pd.to_numeric(work["structure_bias"], errors="coerce").dropna().nunique() > 1)
    )
    include_structure_mode = (
        (not include_structure_bias)
        and structured
        and ("structure_mode" in work.columns)
        and (work["structure_mode"].dropna().astype(str).nunique() > 1)
    )

    if include_structure_bias:
        group_cols = ["structure_bias"] + group_cols
    elif include_structure_mode:
        group_cols = ["structure_mode"] + group_cols

    agg_dict = dict(
        total_trials=(success_col, "size"),
        success_count=(success_col, "sum"),
        success_rate=(success_col, "mean"),
        avg_iterations=("iter_used", "mean"),
        avg_ssdi=("SSDI", "mean"),
        avg_lcd=("LCD", "mean"),
        avg_lds=("LDS", "mean"),
        avg_missing_rate=("missing_rate", "mean"),
        avg_alpha=("actual_alpha", "mean"),
        avg_beta=("actual_beta", "mean"),
    )
    if structured and "target_missing_rate" in work.columns:
        agg_dict["avg_target_missing_rate"] = ("target_missing_rate", "mean")
    if structured and "missing_tol" in work.columns:
        agg_dict["avg_missing_tol"] = ("missing_tol", "mean")
    if structured and "missing_rate_gap" in work.columns:
        agg_dict["avg_missing_rate_gap"] = ("missing_rate_gap", "mean")
    if structured and "theta_gap" in work.columns:
        agg_dict["avg_theta_gap"] = ("theta_gap", "mean")

    out = work.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()
    out["success_rate"] *= 100.0
    return out.sort_values("success_rate", ascending=False).reset_index(drop=True)



def _build_detailed_stats(combined_df: pd.DataFrame, *, structured: bool) -> pd.DataFrame:
    if combined_df.empty:
        return pd.DataFrame()
    success_col = "structure_success" if structured and "structure_success" in combined_df.columns else "success"
    rows: List[Dict[str, Any]] = []
    combos = [(lcd, lds) for lcd in LCD_TYPES for lds in LDS_TYPES]

    group_cols = ["C", "K", "N", "target_SSDI"]
    include_structure_bias = structured and ("structure_bias" in combined_df.columns) and (pd.to_numeric(combined_df["structure_bias"], errors="coerce").dropna().nunique() > 1)
    include_structure_mode = (not include_structure_bias) and structured and ("structure_mode" in combined_df.columns) and (combined_df["structure_mode"].dropna().astype(str).nunique() > 1)
    if include_structure_bias:
        group_cols = ["structure_bias"] + group_cols
    elif include_structure_mode:
        group_cols = ["structure_mode"] + group_cols

    for group_key, ssdi_data in combined_df.groupby(group_cols, dropna=False):
        if include_structure_bias:
            structure_bias, C, K, N, target_ssdi = group_key
            structure_mode = _bias_to_structure_mode(float(structure_bias)) if pd.notna(structure_bias) else None
        elif include_structure_mode:
            structure_mode, C, K, N, target_ssdi = group_key
            structure_bias = None
        else:
            C, K, N, target_ssdi = group_key
            structure_mode = None
            structure_bias = None
        for idx, (lcd_type, lds_type) in enumerate(combos, start=1):
            mech = ssdi_data[(ssdi_data["lcd_type"] == lcd_type) & (ssdi_data["lds_type"] == lds_type)].copy()
            if mech.empty:
                continue
            succ = mech[pd.to_numeric(mech[success_col], errors="coerce").fillna(0).gt(0.5)]
            total_count = len(mech)
            success_count = len(succ)
            sample = mech.iloc[0]
            row = {
                "C": C,
                "K": K,
                "N": N,
                "LCD_params": sample.get("lcd_params", "{}"),
                "LDS_params": sample.get("lds_params", "{}"),
                "SSDI": f"{float(target_ssdi):.3f}",
                "Index": idx,
                "LCD_Type": lcd_type,
                "LDS_Type": lds_type,
                "Success_Total": f"{success_count}/{total_count}",
                "Success_Rate(%)": f"{(100.0 * success_count / total_count):.1f}%" if total_count else "0.0%",
                "Iterations_Range_Success": _fmt_mean_std(succ["iter_used"], 1),
                "SSDI_Range_Success": _fmt_mean_std(succ["SSDI"], 3),
                "LCD_Range_Success": _fmt_mean_std(succ["LCD"], 3),
                "LDS_Range_Success": _fmt_mean_std(succ["LDS"], 3),
                "Missing_Rate_Range_Success": _fmt_mean_std(succ["missing_rate"], 3),
                "Alpha_Range_Success": _fmt_mean_std(succ["actual_alpha"], 3),
                "Beta_Range_Success": _fmt_mean_std(succ["actual_beta"], 3),
                "Time_Range_Success": _fmt_mean_std(succ["time_elapsed"], 2),
                "SSDI_Range_Global": _fmt_mean_std(mech["SSDI"], 3),
                "LCD_Range_Global": _fmt_mean_std(mech["LCD"], 3),
                "LDS_Range_Global": _fmt_mean_std(mech["LDS"], 3),
                "Missing_Rate_Range_Global": _fmt_mean_std(mech["missing_rate"], 3),
                "Alpha_Range_Global": _fmt_mean_std(mech["actual_alpha"], 3),
                "Beta_Range_Global": _fmt_mean_std(mech["actual_beta"], 3),
                "Time_Range_Global": _fmt_mean_std(mech["time_elapsed"], 2),
            }
            if structured and "target_missing_rate" in mech.columns:
                row["Target_Missing_Rate"] = _fmt_mean_std(mech["target_missing_rate"], 3)
            if structured and "missing_tol" in mech.columns:
                row["Missing_Tol"] = _fmt_mean_std(mech["missing_tol"], 3)
            if structured and "missing_rate_gap" in mech.columns:
                row["Missing_Rate_Gap"] = _fmt_mean_std(mech["missing_rate_gap"], 3)
            if structured and "theta_gap" in mech.columns:
                row["Theta_Gap"] = _fmt_mean_std(mech["theta_gap"], 3)
            if include_structure_bias:
                row["Structure_Bias"] = float(structure_bias)
                if structure_mode is not None:
                    row["Structure_Mode"] = structure_mode
            elif include_structure_mode:
                row["Structure_Mode"] = structure_mode
            rows.append(row)

    return pd.DataFrame(rows)



def _run_batch(
    *,
    structured: bool,
    structure_mode: Union[str, List[str], None],
    C_list: List[int],
    K_list: List[int],
    N_list: List[int],
    SSDI_list: List[float],
    repeats: int,
    ssdi_error: float,
    seed: int,
    max_iters: int,
    show_progress: bool,
    show_result: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    main_out_dir = _default_main_out_dir("SSDI_Results_structured" if structured else "SSDI_Results")
    rows: List[Dict[str, Any]] = []

    # 2026-03 multi-mode support: allow None/list and default to all three structured modes
    if structured:
        if structure_mode is None:
            structure_modes =["mixed", "coverage", "skew"]
        elif isinstance(structure_mode, str):
            structure_modes = [structure_mode]
        else:
            structure_modes = list(structure_mode)
        # de-duplicate while preserving order
        seen = set()
        structure_modes = [m for m in structure_modes if not (m in seen or seen.add(m))]
    else:
        structure_modes = ["balanced"]

    for mode in structure_modes:
        for C in C_list:
            for K in K_list:
                for N in N_list:
                    for target_ssdi in SSDI_list:
                        for lcd_type in LCD_TYPES:
                            for lds_type in LDS_TYPES:
                                if show_result:
                                    if structured:
                                        print(f"Running structured: mode={mode}, C={C}, K={K}, N={N}, SSDI={target_ssdi:.3f}, {lcd_type}×{lds_type}")
                                    else:
                                        print(f"Running: C={C}, K={K}, N={N}, SSDI={target_ssdi:.3f}, {lcd_type}×{lds_type}")
                                for rep in range(repeats):
                                    rep_seed = _stable_derive_seed(    seed,    C, K, N, float(target_ssdi),    lcd_type, lds_type,    mode if structured else "plain",    rep,)
                                    if structured:
                                        _, details = generate_ssdi_matrix_structured(
                                            client=K,
                                            label=C,
                                            datasize=N,
                                            ssdi=target_ssdi,
                                            structure_mode=mode,
                                            lcdtype=lcd_type,
                                            ldstype=lds_type,
                                            ssdi_error=ssdi_error,
                                            seed=rep_seed,
                                            max_iters=max_iters,
                                            return_details=True,
                                            save=False,
                                            save_metrics=False,
                                            save_csv=False,
                                            output_dir=main_out_dir,
                                            verbose=False,
                                        )
                                    else:
                                        _, details = generate_ssdi_matrix(
                                            client=K,
                                            label=C,
                                            datasize=N,
                                            ssdi=target_ssdi,
                                            lcdtype=lcd_type,
                                            ldstype=lds_type,
                                            ssdi_error=ssdi_error,
                                            seed=rep_seed,
                                            max_iters=max_iters,
                                            return_details=True,
                                            save=False,
                                            save_metrics=False,
                                            save_csv=False,
                                            output_dir=main_out_dir,
                                            verbose=False,
                                        )
                                    rows.append(_build_combined_row(details, rep=rep, structured=structured))
                                    if show_progress:
                                        print(".", end="", flush=True)
                                if show_progress:
                                    print()

    combined_df = pd.DataFrame(rows)
    detailed_stats_df = _build_detailed_stats(combined_df, structured=structured)
    mechanism_stats_df = _build_mechanism_stats(combined_df, structured=structured)

    combined_path = os.path.join(main_out_dir, "combined_results.csv")
    detailed_path = os.path.join(main_out_dir, "detailed_statistics.csv")
    mechanism_path = os.path.join(main_out_dir, "mechanism_statistics.csv")
    combined_df.to_csv(combined_path, index=False, encoding="utf-8")
    detailed_stats_df.to_csv(detailed_path, index=False, encoding="utf-8")
    mechanism_stats_df.to_csv(mechanism_path, index=False, encoding="utf-8")

    return combined_df, detailed_stats_df, mechanism_stats_df, main_out_dir


def generate_9_methods_and_analyse(
    C_list: List[int],
    K_list: List[int],
    N_list: List[int],
    SSDI_list: List[float],
    repeats: int = 5,
    repeats_factor: int = 3,
    alpha: float = 2.0,
    beta: float = 1.0,
    ssdi_error: float = 0.025,
    seed: int = 42,
    max_iters: int = 120,
    show_progress: bool = False,
    show_result: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    del repeats_factor, alpha, beta
    return _run_batch(
        structured=False,
        structure_mode="balanced",
        C_list=C_list,
        K_list=K_list,
        N_list=N_list,
        SSDI_list=SSDI_list,
        repeats=repeats,
        ssdi_error=ssdi_error,
        seed=seed,
        max_iters=max_iters,
        show_progress=show_progress,
        show_result=show_result,
    )


def generate_9_methods_and_analyse_structured(
    C_list: List[int],
    K_list: List[int],
    N_list: List[int],
    SSDI_list: List[float],
    structure_mode: Union[str, List[str], None] = None,
    repeats: int = 5,
    repeats_factor: int = 3,
    alpha: float = 2.0,
    beta: float = 1.0,
    ssdi_error: float = 0.025,
    seed: int = 42,
    max_iters: int = 120,
    show_progress: bool = False,
    show_result: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    del repeats_factor, alpha, beta
    return _run_batch(
        structured=True,
        structure_mode=structure_mode,
        C_list=C_list,
        K_list=K_list,
        N_list=N_list,
        SSDI_list=SSDI_list,
        repeats=repeats,
        ssdi_error=ssdi_error,
        seed=seed,
        max_iters=max_iters,
        show_progress=show_progress,
        show_result=show_result,
    )



# =========================
# Rewritten structured generation layer
# =========================

def _compute_theta_from_lcd_lds(lcd: float, lds: float) -> float:
    """
    Angle w.r.t. y-axis.
    - theta = 0   : pure y-axis (LCD=0)
    - theta -> pi/2 : pure x-axis (LDS=0)
    """
    x = float(max(0.0, lcd))
    y = float(max(EPS, lds))
    return float(np.arctan2(x, y))


def _bias_to_structure_mode(bias: float) -> str:
    b = float(np.clip(bias, -1.0, 1.0))
    if b <= -0.33:
        return "skew"
    if b >= 0.33:
        return "coverage"
    return "mixed"

def _coerce_structure_mode_value(structure_mode: Any) -> Any:
    """
    Compatibility helper:
    - single mode string / float / int -> return as is
    - list/tuple/set with one item -> unwrap
    - list/tuple/set with multiple items -> keep as list (for batch caller)
    """
    if isinstance(structure_mode, (list, tuple, set)):
        values = list(structure_mode)
        if len(values) == 0:
            return "mixed"
        if len(values) == 1:
            return values[0]
        return values
    return structure_mode

def _structure_mode_to_bias(structure_mode: Union[str, float, int, None]) -> float:
    if structure_mode is None:
        return 0.0

    structure_mode = _coerce_structure_mode_value(structure_mode)

    # batch callers may still pass multiple modes; for single-call compatibility,
    # just take the first one
    if isinstance(structure_mode, list):
        structure_mode = structure_mode[0]

    if isinstance(structure_mode, (int, float, np.floating)):
        return float(np.clip(structure_mode, -1.0, 1.0))

    s = str(structure_mode).strip().lower()

    mapping = {
        # ===== 新接口主名字 =====
        "skew": -0.75,
        "mixed": 0.0,
        "coverage": 0.75,

        # ===== 旧接口兼容 =====
        "lds_dominant": -0.75,
        "lds": -0.75,

        "balanced": 0.0,
        "balance": 0.0,

        "lcd_dominant": 0.75,
        "lcd": 0.75,
    }
    if s in mapping:
        return float(mapping[s])

    try:
        return float(np.clip(float(s), -1.0, 1.0))
    except Exception:
        return 0.0

def _resolve_generation_phase(target_ssdi: float, C: int, K: int, N: int) -> str:
    """
    Routing phases:
    - exact_zero : target SSDI extremely close to 0
    - near_zero  : small heterogeneity
    - middle     : ordinary region
    - near_one   : high heterogeneity, routed by ssdi0 instead of fixed 0.90
    - exact_one  : target SSDI extremely close to 1
    """
    s = float(np.clip(target_ssdi, 0.0, 1.0))
    ssdi0_local = float(_compute_ssdi0_local(C, K, N))

    if s <= 1e-12:
        return "exact_zero"
    if s >= 1.0 - 1e-12:
        return "exact_one"
    if s < 0.09:
        return "near_zero"
    if s > ssdi0_local:
        return "near_one"
    return "middle"


def _resolve_stage_ssdi_error(target_ssdi: float, phase: str, base_error: float) -> float:
    """
    Layered SSDI tolerance.
    Global principle from your rule:
    every non-exact point must stay within phase-aware SSDI tolerance.
    """
    s = float(target_ssdi)
    tol = float(base_error)

    if phase == "exact_zero":
        return min(tol, 0.005)
    if phase == "exact_one":
        return min(tol, 0.005)
    if phase == "near_zero":
        return max(0.010, min(tol, 0.020))
    if phase == "near_one":
        return max(0.010, min(tol, 0.020))
    if s < 0.20 or s > 0.80:
        return max(0.012, min(tol, 0.022))
    return max(0.015, tol)


def _target_missing_rate_from_target_geometry(target_lcd: float, target_lds: float) -> float:
    """
    Lightweight engineering proxy only.
    Keep for reporting/stats compatibility.
    """
    r = float(np.hypot(target_lcd, target_lds))
    if r <= EPS:
        return 0.0
    # missing rate should correlate more with LCD than LDS
    return float(np.clip((target_lcd / max(r, EPS)) * r, 0.0, 1.0))


def _dominance_value(lcd: float, lds: float) -> float:
    return float(lcd - lds)


def _safe_record_from_matrix(
    n_ck: np.ndarray,
    *,
    target_ssdi: float,
    lcdtype: str,
    ldstype: str,
    alpha: Optional[float],
    beta: Optional[float],
    lcd_params: Optional[Dict[str, Any]],
    lds_params: Optional[Dict[str, Any]],
    source_stage: str,
    generator_variant: str,
) -> Dict[str, Any]:
    return _matrix_to_record_like(
        {
            "target_ssdi": float(target_ssdi),
            "lcd_type": lcdtype,
            "lds_type": ldstype,
            "alpha_used": alpha,
            "beta_used": beta,
            "lcd_params": lcd_params or {},
            "lds_params": lds_params or {},
            "source_stage": source_stage,
            "generator_variant": generator_variant,
            "C": int(n_ck.shape[0]),
            "K": int(n_ck.shape[1]),
            "N": int(n_ck.sum()),
        },
        n_ck.astype(int, copy=True),
    )


def _construct_lcdmax_record(
    *,
    client: int,
    label: int,
    datasize: int,
    ssdi: float,
    lcdtype: str,
    ldstype: str,
    alpha: Optional[float],
    beta: Optional[float],
    lcd_params: Optional[Dict[str, Any]],
    lds_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Deterministic 'lcd-max' anchor:
    not exact1, but a high-LCD / reduced-LDS diagonal-binaryized anchor.

    Construction logic:
    - active support stays on diagonal skeleton of size m=min(C,K)
    - two dominant diagonal cells take almost all mass
    - remaining active diagonal cells keep tiny residual mass
    """
    C, K, N = int(label), int(client), int(datasize)
    m = max(1, min(C, K))
    n_ck = np.zeros((C, K), dtype=int)

    if m == 1:
        n_ck[0, 0] = N
        return _safe_record_from_matrix(
            n_ck,
            target_ssdi=ssdi,
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
            source_stage="direct lcd-max anchor",
            generator_variant="direct_lcdmax",
        )

    # keep a tiny residual on all active diagonal cells
    residual = np.ones(m, dtype=int)
    residual_sum = int(residual.sum())
    rem = int(N - residual_sum)
    rem = max(rem, 0)

    # two dominant diagonal values
    major1 = int(0.65 * rem)
    major2 = rem - major1
    diag = residual.copy()
    diag[0] += major1
    diag[1] += major2

    for i in range(m):
        n_ck[i, i] = int(diag[i])

    return _safe_record_from_matrix(
        n_ck,
        target_ssdi=ssdi,
        lcdtype=lcdtype,
        ldstype=ldstype,
        alpha=alpha,
        beta=beta,
        lcd_params=lcd_params,
        lds_params=lds_params,
        source_stage="direct lcd-max anchor",
        generator_variant="direct_lcdmax",
    )


def _circle_segment_intersection(
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Return the point on segment a->b whose norm is radius.
    If no exact in-segment solution exists, return the closer endpoint-scaled proxy.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    r = float(max(0.0, radius))
    d = b - a

    A = float(np.dot(d, d))
    B = float(2.0 * np.dot(a, d))
    C = float(np.dot(a, a) - r * r)

    if A <= EPS:
        n = float(np.linalg.norm(a))
        if n <= EPS:
            return np.zeros_like(a)
        return a * (r / n)

    disc = B * B - 4.0 * A * C
    if disc >= 0:
        sqrt_disc = float(np.sqrt(max(0.0, disc)))
        sols = [(-B - sqrt_disc) / (2.0 * A), (-B + sqrt_disc) / (2.0 * A)]
        valid = [t for t in sols if -1e-9 <= t <= 1.0 + 1e-9]
        if valid:
            # choose the one farther along the segment (more "rightward" on that branch)
            t = float(np.clip(max(valid), 0.0, 1.0))
            return a + t * d

    # engineering fallback:
    # choose the point on the line segment with closest norm, then scale if needed.
    cand = [a, b, 0.5 * (a + b)]
    best = min(cand, key=lambda p: abs(np.linalg.norm(p) - r))
    n = float(np.linalg.norm(best))
    if n <= EPS:
        return np.zeros_like(best)
    return best * (r / n)

def _clip_local(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _estimate_region_params(C: int, K: int, N: int) -> Dict[str, float]:
    """
    Empirical strict-region geometry parameters.

    Returns a right-end angle theta_R_deg that is adaptive to C/K mismatch
    and finite-sample density CK/N.

    Notes
    -----
    - C: number of labels
    - K: number of clients
    - N: total sample size
    - theta_R_deg is the right-boundary endpoint angle measured w.r.t. y-axis
    """
    C = max(1, int(C))
    K = max(1, int(K))
    N = max(1, int(N))

    m = min(C, K)
    mu = abs(C - K) / max(1, m)
    eta = (C * K) / max(1, N)

    # right-end direction:
    # larger mismatch / larger CK/N => smaller theta_R => endpoint shifts left/up
    alpha_x_deg = _clip_local(
        30.0
        + 10.0 * math.tanh(0.15 * mu)
        + 8.0 * math.sqrt(max(eta, 0.0) / 0.01),
        20.0,
        70.0,
    )
    theta_R_deg = 90.0 - alpha_x_deg

    # keep a few extra fields for future use / plotting compatibility
    r_y = _clip_local(
        0.58 - 0.004 * mu - 0.06 * math.sqrt(10.0 * eta),
        0.35,
        0.70,
    )
    r_p = _clip_local(
        0.95 - 0.010 * mu - 0.10 * math.sqrt(10.0 * eta),
        0.55,
        0.95,
    )
    r_r = _clip_local(
        r_p - 0.05 - 0.02 * math.sqrt(max(mu, 0.0)),
        0.45,
        0.90,
    )
    theta_p_deg = 0.6 * theta_R_deg

    return {
        "m": float(m),
        "mu": float(mu),
        "eta": float(eta),
        "alpha_x_deg": float(alpha_x_deg),
        "theta_R_deg": float(theta_R_deg),
        "r_y": float(r_y),
        "r_p": float(r_p),
        "r_r": float(r_r),
        "theta_p_deg": float(theta_p_deg),
    }
def _near_zero_right_boundary_point(
    *,
    client: int,
    label: int,
    datasize: int,
    r: float,
    theta_R: float,
) -> np.ndarray:
    """
    Empirical right boundary for SSDI in [0, 0.1], only for the near-zero branch.

    Design:
    - keep the same 0.1 endpoint direction theta_R
    - but for smaller radii, bend the branch BELOW the straight ray
      (i.e. larger angle than theta_R, hence larger LCD and smaller LDS)
    - never touch the x-axis except at the origin
    - make the sinking stronger when |C-K|/min(C,K) is larger,
      and mildly stronger when CK/N is larger

    Parameters
    ----------
    client, label, datasize:
        current K, C, N
    r:
        target SSDI radius, clipped into [0, 0.1]
    theta_R:
        angle w.r.t. y-axis of the 0.1 endpoint / right boundary endpoint
    """
    r = float(np.clip(r, 0.0, 0.1))
    if r <= 1e-12:
        return np.array([0.0, 0.0], dtype=float)

    C = int(label)
    K = int(client)
    N = int(datasize)

    m = max(1, min(C, K))
    mu = abs(C - K) / max(1, m)
    eta = (C * K) / max(1, N)

    # ---------------------------------------------------------
    # sinking amplitude:
    # larger mismatch -> stronger sink below the straight starter
    # larger CK/N     -> mildly stronger sink
    # ---------------------------------------------------------
    delta_deg = _clip_local(
        10.0
        + 12.0 * math.tanh(0.9 * mu)
        + 6.0 * math.sqrt(10.0 * eta),
        8.0,
        28.0,
    )

    # ---------------------------------------------------------
    # shape:
    # larger a => more curvature concentrated near the origin,
    # then smoothly return to theta_R as r -> 0.1
    # ---------------------------------------------------------
    shape_a = _clip_local(
        1.7
        + 0.5 * math.tanh(0.8 * mu)
        + 0.2 * math.sqrt(10.0 * eta),
        1.3,
        2.6,
    )

    t = r / 0.1
    theta_R_deg = float(np.degrees(theta_R))

    # below-the-line curved branch:
    # small r uses a larger angle than theta_R,
    # then smoothly relaxes back to theta_R at r=0.1
    phi_deg = theta_R_deg + delta_deg * ((1.0 - t) ** shape_a)
    phi_deg = min(phi_deg, 89.0)

    phi = math.radians(phi_deg)
    x = float(r * math.sin(phi))
    y = float(r * math.cos(phi))
    return np.array([x, y], dtype=float)

def _compute_geometry_bundle(
    *,
    client: int,
    label: int,
    datasize: int,
    target_ssdi: Optional[float] = None,
    ref_ssdi: Optional[float] = None,
    lcdtype: str = "client",
    ldstype: str = "client",
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    lcd_params: Optional[Dict[str, Any]] = None,
    lds_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Backward-compatible geometry helper.

    Compatible with both call styles:
    - new:
        _compute_geometry_bundle(..., target_ssdi=0.95, lcdtype=..., ldstype=..., ...)
    - old plotting helper:
        _compute_geometry_bundle(..., ref_ssdi=0.95)
        or even only client/label/datasize

    Returns BOTH:
    1) current structured-generator fields
    2) old plot_ssdi-friendly aliases
    """

    # ---------------------------------------------------------
    # 0) compatibility: resolve reference SSDI
    # ---------------------------------------------------------
    if target_ssdi is None:
        if ref_ssdi is not None:
            target_ssdi = float(ref_ssdi)
        else:
            target_ssdi = 0.95
    target_ssdi = float(np.clip(target_ssdi, 0.0, 1.0))

    # keep defaults stable
    lcd_params = dict(lcd_params or {})
    lds_params = dict(lds_params or {})

    # ---------------------------------------------------------
    # 1) exact1 anchor
    # ---------------------------------------------------------
    exact1 = _construct_exact_one_record(
        client=client,
        label=label,
        datasize=datasize,
        ssdi=1.0,
        lcdtype=lcdtype,
        ldstype=ldstype,
        alpha=alpha,
        beta=beta,
        lcd_params=lcd_params,
        lds_params=lds_params,
    )

    # ---------------------------------------------------------
    # 2) lcd-max anchor
    # ---------------------------------------------------------
    lcdmax = _construct_lcdmax_record(
        client=client,
        label=label,
        datasize=datasize,
        ssdi=target_ssdi,
        lcdtype=lcdtype,
        ldstype=ldstype,
        alpha=alpha,
        beta=beta,
        lcd_params=lcd_params,
        lds_params=lds_params,
    )

    exact1_pt = np.array([float(exact1["LCD"]), float(exact1["LDS"])], dtype=float)
    lcdmax_pt = np.array([float(lcdmax["LCD"]), float(lcdmax["LDS"])], dtype=float)

    # near-zero right-branch anchor requested by your plotting logic
    # (the "0.1 -> lcdmax" line)
    low_x_anchor = np.array([0.1, 0.0], dtype=float)

    ssdi_lcdmax = float(np.linalg.norm(lcdmax_pt))
    ssdi_exact1 = float(np.linalg.norm(exact1_pt))

    # engineering guard: lcdmax should stay inside exact1 radius
    if ssdi_lcdmax >= ssdi_exact1:
        lcdmax_pt = 0.92 * exact1_pt
        ssdi_lcdmax = float(np.linalg.norm(lcdmax_pt))

    theta_lcdmax = float(_compute_theta_from_lcd_lds(lcdmax_pt[0], lcdmax_pt[1]))
    theta_exact1 = float(_compute_theta_from_lcd_lds(exact1_pt[0], exact1_pt[1]))

    # ---------------------------------------------------------
    # 3) background curve for plot_ssdi
    #    piece A: from low_x_anchor to lcdmax
    #    piece B: from lcdmax to exact1
    # ---------------------------------------------------------
    n1 = 120
    n2 = 180

    seg1_t = np.linspace(0.0, 1.0, n1)
    seg1 = np.stack(
        [
            low_x_anchor[0] + seg1_t * (lcdmax_pt[0] - low_x_anchor[0]),
            low_x_anchor[1] + seg1_t * (lcdmax_pt[1] - low_x_anchor[1]),
        ],
        axis=1,
    )

    seg2_t = np.linspace(0.0, 1.0, n2)
    # keep your current "convex-up" style branch:
    x0 = float(lcdmax_pt[0])
    y0 = float(lcdmax_pt[1])
    x1 = float(exact1_pt[0])
    y1 = float(exact1_pt[1])

    seg2_x = x1 + (x0 - x1) * (1.0 - seg2_t ** 2)
    seg2_y = np.sqrt(np.maximum(0.0, (y0 + seg2_t * (y1 - y0)) ** 2))
    seg2 = np.stack([seg2_x, seg2_y], axis=1)

    curve = np.vstack([seg1, seg2[1:]])
    curve_lcd = curve[:, 0].astype(float).tolist()
    curve_lds = curve[:, 1].astype(float).tolist()
    curve_ssdi = np.sqrt(curve[:, 0] ** 2 + curve[:, 1] ** 2).astype(float).tolist()
    curve_theta_deg = np.degrees(
        np.array([_compute_theta_from_lcd_lds(x, y) for x, y in curve], dtype=float)
    ).astype(float).tolist()

    # ---------------------------------------------------------
    # 4) rich point dicts for old plot code
    # ---------------------------------------------------------
    exact1_dict = {
        "lcd": float(exact1_pt[0]),
        "lds": float(exact1_pt[1]),
        "ssdi": float(ssdi_exact1),
        "theta": float(theta_exact1),
        "theta_deg": float(np.degrees(theta_exact1)),
    }

    lcdmax_dict = {
        "lcd": float(lcdmax_pt[0]),
        "lds": float(lcdmax_pt[1]),
        "ssdi": float(ssdi_lcdmax),
        "theta": float(theta_lcdmax),
        "theta_deg": float(np.degrees(theta_lcdmax)),
    }

    low_x_anchor_dict = {
        "lcd": float(low_x_anchor[0]),
        "lds": float(low_x_anchor[1]),
        "ssdi": float(np.linalg.norm(low_x_anchor)),
        "theta": float(_compute_theta_from_lcd_lds(low_x_anchor[0], max(low_x_anchor[1], EPS))),
        "theta_deg": float(np.degrees(_compute_theta_from_lcd_lds(low_x_anchor[0], max(low_x_anchor[1], EPS)))),
    }

    # ---------------------------------------------------------
    # 5) return both new + old fields
    # ---------------------------------------------------------
    return {
        # ===== current structured-generator fields =====
        "exact1_record": exact1,
        "lcdmax_record": lcdmax,
        "exact1_point": exact1_pt,
        "lcdmax_point": lcdmax_pt,
        "low_x_anchor": low_x_anchor,
        "ssdi_lcdmax": float(ssdi_lcdmax),
        "ssdi_exact1": float(ssdi_exact1),
        "theta_lcdmax": float(theta_lcdmax),
        "theta_exact1": float(theta_exact1),

        # ===== old / plotting-friendly dict aliases =====
        "exact1": exact1_dict,
        "lcdmax": lcdmax_dict,
        "low_x_anchor_dict": low_x_anchor_dict,

        # some old plot code may expect low_x_anchor itself to be a dict
        # but current structured code needs ndarray. so both are exposed:
        "low_x_anchor_point": low_x_anchor,
        "low_x_anchor_plot": low_x_anchor_dict,

        # flattened aliases
        "exact1_lcd": float(exact1_pt[0]),
        "exact1_lds": float(exact1_pt[1]),
        "lcdmax_lcd": float(lcdmax_pt[0]),
        "lcdmax_lds": float(lcdmax_pt[1]),
        "low_x_anchor_lcd": float(low_x_anchor[0]),
        "low_x_anchor_lds": float(low_x_anchor[1]),

        # curve for background plotting
        "curve": {
            "lcd": curve_lcd,
            "lds": curve_lds,
            "ssdi": curve_ssdi,
            "theta_deg": curve_theta_deg,
        },
        "curve_lcd": curve_lcd,
        "curve_lds": curve_lds,
        "curve_ssdi": curve_ssdi,
        "curve_theta_deg": curve_theta_deg,

        # compatibility bookkeeping
        "target_ssdi": float(target_ssdi),
        "ref_ssdi": float(target_ssdi),

        # [MOD] carry current geometry case info for near-zero curved branch
        "client": int(client),
        "label": int(label),
        "datasize": int(datasize),
    }
    

def _structured_target_components(
    target_ssdi: float,
    structure_bias: float,
    geometry: Dict[str, Any],
) -> Dict[str, Any]:
    """
    New target geometry:
    - exact0: no auxiliary point
    - exact1: fixed point
    - if ssdi <= ssdi_lcdmax:
        use the branch determined by line (0.1, 0) -> lcdmax
      for very small r < 0.1, use the NEW curved near-zero right boundary
      instead of the old origin->lcdmax straight continuation
    - if ssdi > ssdi_lcdmax:
        use the branch determined by line lcdmax -> exact1

    bias in [-1, 1] uniformly partitions the admissible angle interval [0, theta_max(r)].
    """
    r = float(np.clip(target_ssdi, 0.0, 1.0))
    b = float(np.clip(structure_bias, -1.0, 1.0))

    lcdmax_pt = np.asarray(geometry["lcdmax_point"], dtype=float)
    exact1_pt = np.asarray(geometry["exact1_point"], dtype=float)
    low_anchor = np.asarray(geometry["low_x_anchor"], dtype=float)
    ssdi_lcdmax = float(geometry["ssdi_lcdmax"])

    if r <= 1e-12:
        return {
            "target_lcd": 0.0,
            "target_lds": 0.0,
            "target_theta": 0.0,
            "theta_max": 0.0,
            "domain_stage": "exact_zero",
            "target_dsr": 0.0,
        }

    if r >= 1.0 - 1e-12:
        target_lcd = float(exact1_pt[0])
        target_lds = float(exact1_pt[1])
        target_theta = _compute_theta_from_lcd_lds(target_lcd, target_lds)
        return {
            "target_lcd": target_lcd,
            "target_lds": target_lds,
            "target_theta": target_theta,
            "theta_max": target_theta,
            "domain_stage": "exact_one",
            "target_dsr": float(target_lcd / max(target_lds, EPS)),
        }

    # ---------------------------------------------------------
    # stage 1: below lcdmax radius
    # ---------------------------------------------------------
    if r <= ssdi_lcdmax + 1e-12:
            
        if r < 0.1:
            # [FIX]
            # Do NOT infer theta_R from low_anchor.
            # low_anchor in current geometry may still be the old plotting anchor (0.1, 0),
            # which would incorrectly force theta_R ~= 90 deg and make mode=1 collapse to x-axis.
            client_cur = int(geometry.get("K", geometry.get("client", 0)) or 0)
            label_cur = int(geometry.get("C", geometry.get("label", 0)) or 0)
            datasize_cur = int(geometry.get("N", geometry.get("datasize", 0)) or 0)

            if client_cur > 0 and label_cur > 0 and datasize_cur > 0:
                params = _estimate_region_params(label_cur, client_cur, datasize_cur)
                theta_R_deg = float(params["theta_R_deg"])
                theta_R = math.radians(theta_R_deg)

                base_pt = _near_zero_right_boundary_point(
                    client=client_cur,
                    label=label_cur,
                    datasize=datasize_cur,
                    r=r,
                    theta_R=theta_R,
                )
            else:
                # fallback only if C/K/N were really not carried into geometry
                # keep a conservative, non-x-axis curved branch
                theta_R = float(_compute_theta_from_lcd_lds(lcdmax_pt[0], max(lcdmax_pt[1], EPS)))
                t = r / 0.1
                theta_R_deg = float(np.degrees(theta_R))
                phi_deg = theta_R_deg + 12.0 * ((1.0 - t) ** 1.8)
                phi_deg = min(phi_deg, 89.0)
                phi = math.radians(phi_deg)
                base_pt = np.array(
                    [float(r * math.sin(phi)), float(r * math.cos(phi))],
                    dtype=float,
                )


            # if geometry dict did not carry C/K/N, fall back to using the current
            # right-end direction extracted from low_anchor itself, but still keep
            # the same curved formula with a neutral mismatch/density effect.
            if int(geometry.get("K", geometry.get("client", 0)) or 0) <= 0 or \
               int(geometry.get("C", geometry.get("label", 0)) or 0) <= 0 or \
               int(geometry.get("N", geometry.get("datasize", 0)) or 0) <= 0:
                # neutral fallback: no extra C/K/N-dependent sinking, but still curved
                t = r / 0.1
                theta_R_deg = float(np.degrees(theta_R))
                phi_deg = theta_R_deg + 12.0 * ((1.0 - t) ** 1.8)
                phi_deg = min(phi_deg, 89.0)
                phi = math.radians(phi_deg)
                base_pt = np.array(
                    [float(r * math.sin(phi)), float(r * math.cos(phi))],
                    dtype=float,
                )
        else:
            base_pt = _circle_segment_intersection(low_anchor, lcdmax_pt, r)

        theta_max = _compute_theta_from_lcd_lds(base_pt[0], base_pt[1])
        target_theta = ((b + 1.0) / 2.0) * theta_max

        target_lcd = float(r * np.sin(target_theta))
        target_lds = float(r * np.cos(target_theta))
        return {
            "target_lcd": target_lcd,
            "target_lds": target_lds,
            "target_theta": target_theta,
            "theta_max": theta_max,
            "domain_stage": "below_lcdmax",
            "target_dsr": float(target_lcd / max(target_lds, EPS)),
        }

    # ---------------------------------------------------------
    # stage 2: above lcdmax radius
    # ---------------------------------------------------------
    r0 = float(ssdi_lcdmax)
    r1 = float(geometry["ssdi_exact1"])
    x0 = float(lcdmax_pt[0])
    x1 = float(exact1_pt[0])

    if r1 <= r0 + EPS:
        base_pt = exact1_pt.copy()
    else:
        # normalized radius position in [0, 1]
        t = float(np.clip((r - r0) / max(r1 - r0, EPS), 0.0, 1.0))

        # quadratic outward curve: q = 2
        x_max = float(x1 + (x0 - x1) * (1.0 - t ** 2))

        # numerical safety: x cannot exceed radius
        x_max = min(x_max, r)

        y_max = float(np.sqrt(max(0.0, r * r - x_max * x_max)))
        base_pt = np.array([x_max, y_max], dtype=float)

    theta_max = _compute_theta_from_lcd_lds(base_pt[0], base_pt[1])
    target_theta = ((b + 1.0) / 2.0) * theta_max

    target_lcd = float(r * np.sin(target_theta))
    target_lds = float(r * np.cos(target_theta))
    return {
        "target_lcd": target_lcd,
        "target_lds": target_lds,
        "target_theta": target_theta,
        "theta_max": theta_max,
        "domain_stage": "above_lcdmax",
        "target_dsr": float(target_lcd / max(target_lds, EPS)),
    }



def _theta_tolerance_by_phase(phase: str, theta_max: float) -> float:
    if phase in {"exact_zero", "exact_one"}:
        return 0.0

    if phase == "near_one":
        # [MOD] near1 proposal 改细之后，角度控制会更平滑；
        # 这里把最低容差从 3° 略放宽到 4°，避免只差一点角度也被判 miss。
        return float(max(np.deg2rad(4.0), 0.10 * max(theta_max, EPS)))

    if phase == "near_zero":
        return float(max(np.deg2rad(6.0), 0.18 * max(theta_max, EPS)))

    return float(max(np.deg2rad(4.0), 0.12 * max(theta_max, EPS)))





def _collapse_same_col_tail_to_peak(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    *,
    steps: int = 1,
) -> Tuple[np.ndarray, bool]:
    """
    [MOD] near1 左侧（偏 lds）专用：
    在同一 client 列内，把非主峰的小尾巴收回该列主峰。

    作用：
    - 提高 LDS
    - 压低 theta（更靠近 y 轴）
    - 不扩 support，不新增非零位置

    [MOD] 现在额外返回 changed，供外层判断这一步是不是 no-op。
    """
    out = n_ck.astype(int, copy=True)
    changed = False
    C, K = out.shape

    for _ in range(int(max(1, steps))):
        valid_cols = np.where(out.sum(axis=0) > 0)[0]
        if valid_cols.size == 0:
            break

        col_scores = []
        for j in valid_cols:
            col = out[:, j]
            nz = np.where(col > 0)[0]
            if nz.size <= 1:
                col_scores.append(0.0)
                continue
            peak = float(col.max())
            total = float(col.sum())
            tail = total - peak
            col_scores.append(max(0.0, tail))

        col_scores = np.asarray(col_scores, dtype=float)
        if np.all(col_scores <= 0):
            break

        probs = col_scores / col_scores.sum()
        j = int(rng.choice(valid_cols, p=probs))

        col = out[:, j]
        peak_row = int(np.argmax(col))
        tail_rows = np.where((col > 0) & (np.arange(C) != peak_row))[0]
        if tail_rows.size == 0:
            continue

        tail_vals = col[tail_rows].astype(float)
        tail_probs = 1.0 / np.maximum(tail_vals, 1.0)
        tail_probs = tail_probs / tail_probs.sum()
        src_row = int(rng.choice(tail_rows, p=tail_probs))

        src_val = int(out[src_row, j])
        if src_val <= 0:
            continue

        delta = int(min(max(1, src_val // 2), src_val))
        if delta <= 0:
            continue

        out[src_row, j] -= delta
        out[peak_row, j] += delta
        changed = True

    return out, changed



def _open_tiny_same_row_support_from_diag_left(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    *,
    opens: int = 1,
    amount_frac: float = 0.01,
    min_amount: int = 1,
) -> np.ndarray:
    """
    [MOD] near1 左侧（偏 lds）专用：
    当当前仍是 exact1-like 骨架、几乎没有尾巴可收时，
    从对角主峰拿极小质量，在同一 row 的新列上开一个很小的 support。

    作用：
    - 降低 LCD（因为减少缺失）
    - 尽量保持高 LDS（比 same_col_split 更温和）
    - 让 very-low-theta 目标不再因为“无尾巴可收”而原地踏步
    """
    out = n_ck.astype(int, copy=True)
    C, K = out.shape
    m = min(C, K)

    for _ in range(int(max(1, opens))):
        diag_candidates = [i for i in range(m) if out[i, i] > 1]
        if not diag_candidates:
            break

        donor_weights = np.array([max(1.0, float(out[i, i])) for i in diag_candidates], dtype=float)
        donor_weights = donor_weights / donor_weights.sum()
        i = int(rng.choice(diag_candidates, p=donor_weights))

        # 优先空列；否则找同一 row 的高价值 0
        zero_cols = np.where(out.sum(axis=0) == 0)[0]
        if zero_cols.size > 0:
            j_new = int(rng.choice(zero_cols))
        else:
            zeros_same_row = [(r, c) for (r, c) in _top_zero_positions_by_value(out, topk=64) if r == i]
            if not zeros_same_row:
                break
            _, j_new = zeros_same_row[int(rng.integers(0, len(zeros_same_row)))]

        src_val = int(out[i, i])
        if src_val <= 1:
            continue

        delta = int(min(max(min_amount, src_val * amount_frac), src_val - 1))
        if delta <= 0:
            continue

        out = _move_mass(out, (i, i), (i, j_new), delta)

    return out


def _activate_sparse_same_row_support_from_diag(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    *,
    opens: int = 1,
    amount_frac: float = 0.004,
    min_amount: int = 1,
) -> np.ndarray:
    """
    [MOD] near1 右侧（偏 lcd）专用：
    从 diagonal donor 拿出极少量质量，放到同一 row 的新 client 上，
    用来打破“始终锁死在 exact1 最大缺失骨架”的问题。

    作用：
    - 给 near1 右侧引入中间骨架
    - 仍保持高异质
    - 避免所有偏 lcd 的目标都塌到最大缺失边界
    """
    out = n_ck.astype(int, copy=True)
    C, K = out.shape
    m = min(C, K)

    for _ in range(int(max(1, opens))):
        # 优先找完全空列；如果没有，再找高价值 0
        zero_cols = np.where(out.sum(axis=0) == 0)[0]

        diag_candidates = [i for i in range(m) if out[i, i] > 1]
        if not diag_candidates:
            break

        # donor 尽量从“非 keeper”或较弱对角上来，避免破坏主二值骨架
        diag_vals = np.array([out[i, i] for i in range(m)], dtype=float)
        order = list(np.argsort(-diag_vals))
        keepers = set(order[:2])

        donor_pool = [i for i in diag_candidates if i not in keepers]
        if not donor_pool:
            donor_pool = diag_candidates

        donor_weights = np.array([max(1.0, float(out[i, i])) for i in donor_pool], dtype=float)
        donor_weights = donor_weights / donor_weights.sum()
        i = int(rng.choice(donor_pool, p=donor_weights))

        if zero_cols.size > 0:
            # 优先激活空列
            j_new = int(rng.choice(zero_cols))
            dst = (i, j_new)
        else:
            # 没有空列时，找同一 row 的高价值 0
            zeros = [(r, c) for (r, c) in _top_zero_positions_by_value(out, topk=64) if r == i]
            if not zeros:
                break
            dst = zeros[int(rng.integers(0, len(zeros)))]

        src = (i, i)
        src_val = int(out[src])
        if src_val <= 1:
            continue

        delta = int(min(max(min_amount, src_val * amount_frac), src_val - 1))
        if delta <= 0:
            continue

        out = _move_mass(out, src, dst, delta)

    return out





def _decorate_record_constraints(
    record: Dict[str, Any],
    *,
    structure_bias: float,
    structure_mode: str,
    target_spec: Dict[str, Any],
    phase: str,
    stage_ssdi_error: float,
) -> Dict[str, Any]:
    rec = dict(record)

    actual_theta = _compute_theta_from_lcd_lds(float(rec["LCD"]), float(rec["LDS"]))
    theta_gap = abs(actual_theta - float(target_spec["target_theta"]))
    missing_target = _target_missing_rate_from_target_geometry(
        float(target_spec["target_lcd"]),
        float(target_spec["target_lds"]),
    )
    missing_gap = abs(float(rec.get("missing_rate", 0.0)) - missing_target)

    rec["structure_bias"] = float(structure_bias)
    rec["structure_mode"] = structure_mode
    rec["target_lcd"] = float(target_spec["target_lcd"])
    rec["target_lds"] = float(target_spec["target_lds"])
    rec["target_dsr"] = float(target_spec["target_dsr"])
    rec["target_theta"] = float(target_spec["target_theta"])
    rec["actual_theta"] = float(actual_theta)
    rec["theta_gap"] = float(theta_gap)
    rec["target_missing_rate"] = float(missing_target)
    rec["missing_tol"] = float(max(0.03, 0.20 * missing_target))
    rec["missing_rate_gap"] = float(missing_gap)
    rec["missing_rate_overflow"] = float(max(0.0, missing_gap - rec["missing_tol"]))
    rec["dominance_target"] = _dominance_value(
        float(target_spec["target_lcd"]),
        float(target_spec["target_lds"]),
    )
    rec["dominance_actual"] = _dominance_value(
        float(rec["LCD"]),
        float(rec["LDS"]),
    )
    rec["domain_stage"] = str(target_spec["domain_stage"])
    rec["ssdi_gap"] = abs(float(rec["SSDI"]) - float(rec["target_ssdi"]))

    # ---------------------------------------------------------
    # ensure empty-count fields always exist
    # ---------------------------------------------------------
    if "empty_client_count" not in rec or "empty_label_count" not in rec:
        n_ck = rec.get("n_ck", None)
        if n_ck is not None:
            ec, el = _count_empty_clients_and_labels_local(np.asarray(n_ck))
            rec["empty_client_count"] = int(ec)
            rec["empty_label_count"] = int(el)
        else:
            rec["empty_client_count"] = int(rec.get("empty_client_count", 0) or 0)
            rec["empty_label_count"] = int(rec.get("empty_label_count", 0) or 0)

    # ---------------------------------------------------------
    # 本次修改：
    # 删除 below_lcdmax ordinary middle 的 strict-middle 修补层标记。
    # 这里统一关闭该 flag，仅保留字段以维持 schema 稳定。
    # ---------------------------------------------------------
    rec["strict_middle_nonempty_required"] = False

    success, failure_primary, failure_detail, hint = _structured_success(
        rec,
        phase=phase,
        stage_ssdi_error=stage_ssdi_error,
        theta_tol=_theta_tolerance_by_phase(phase, float(target_spec["theta_max"])),
    )
    rec["structure_success"] = bool(success)
    rec["structure_score"] = _structured_score(
        rec,
        phase=phase,
        stage_ssdi_error=stage_ssdi_error,
        theta_tol=_theta_tolerance_by_phase(phase, float(target_spec["theta_max"])),
    )
    rec["failure_primary"] = failure_primary
    rec["failure_structure_detail"] = failure_detail
    rec["optimization_hint"] = hint
    rec["failure_note"] = f"{failure_primary}: {failure_detail}" if failure_primary else None
    return rec








def _structured_success(
    record: Dict[str, Any],
    *,
    phase: str,
    stage_ssdi_error: float,
    theta_tol: float,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Global rule after removing the low-segment middle repair layer:
    - exact0 : absolute IID
    - exact1 : fixed exact point
    - near0  : radial band + point band
    - all other non-exact points: SSDI band + angle band
    - mode=-1 : must additionally satisfy LCD≈0 and full support (no zeros)
    """
    ssdi_gap = float(abs(record["SSDI"] - record["target_ssdi"]))
    theta_gap = float(record.get("theta_gap", 0.0))
    structure_bias = float(record.get("structure_bias", 0.0))
    mode_value = _resolve_mode_value_local(structure_bias)

    lcd_gap = float(record["LCD"] - record["target_lcd"])
    lds_gap = float(record["LDS"] - record["target_lds"])
    point_gap = float(np.hypot(lcd_gap, lds_gap))

    if phase == "exact_zero":
        ok = (
            abs(float(record["SSDI"])) <= stage_ssdi_error
            and abs(float(record["LCD"])) <= stage_ssdi_error
            and abs(float(record["LDS"])) <= stage_ssdi_error
            and float(record.get("missing_rate", 0.0)) <= 1e-12
        )
        if ok:
            return True, None, None, None
        return False, "exact_zero_miss", "not absolute iid enough", "increase full-support balancing"

    if phase == "exact_one":
        ok = ssdi_gap <= stage_ssdi_error
        if ok:
            return True, None, None, None
        return False, "exact_one_miss", "not close enough to exact1", "stay on extremal diagonal skeleton"

    if phase == "near_zero":
        if ssdi_gap > stage_ssdi_error:
            return False, "ssdi_miss", "outside layered ssdi tolerance", "first repair radial distance"

        point_tol = _near_zero_point_tolerance(record, stage_ssdi_error)
        if point_gap > point_tol:
            return False, "point_miss", "outside near0 target-point tolerance", "repair lcd/lds coordinates directly"

        if mode_value == -1:
            n_ck = record.get("n_ck")
            if n_ck is None:
                return False, "full_support_miss", "matrix missing for support check", "keep full support during leftward repair"

            if np.any(np.asarray(n_ck) <= 0):
                return False, "full_support_miss", "mode=-1 requires all cells > 0", "continue filling high-value zeros until no zeros remain"

            if float(record["LCD"]) > max(stage_ssdi_error, 1e-3):
                return False, "lcd_not_zero", "mode=-1 requires LCD≈0", "continue support filling before fine tuning"

        return True, None, None, None

    # ---------------------------------------------------------
    # all other non-exact phases: first radius, then angle
    # ---------------------------------------------------------
    if ssdi_gap > stage_ssdi_error:
        return False, "ssdi_miss", "outside layered ssdi tolerance", "first repair radial distance"

    if theta_gap > theta_tol:
        return False, "theta_miss", "outside angle tolerance", "repair direction before fine-tuning"

    # ---------------------------------------------------------
    # 本次修改：
    # 删除 strict-middle gate
    # 不再对 below_lcdmax ordinary middle 施加
    # strict_middle_support_miss / strict_middle_missing_overflow 这层成功门
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # mode=-1 extra hard rule
    # ---------------------------------------------------------
    if mode_value == -1:
        n_ck = record.get("n_ck")
        if n_ck is None:
            return False, "full_support_miss", "matrix missing for support check", "keep full support during leftward repair"

        if np.any(np.asarray(n_ck) <= 0):
            return False, "full_support_miss", "mode=-1 requires all cells > 0", "continue filling high-value zeros until no zeros remain"

        if float(record["LCD"]) > max(stage_ssdi_error, 1e-3):
            return False, "lcd_not_zero", "mode=-1 requires LCD≈0", "continue support filling before fine tuning"

    return True, None, None, None







def _structured_score(
    record: Dict[str, Any],
    *,
    phase: str,
    stage_ssdi_error: float,
    theta_tol: float,
) -> float:
    """
    Score priority after removing the low-segment middle repair layer:
    1) stay inside SSDI tolerance
    2) match target (LCD, LDS) point
    3) match angle
    4) mode=-1 strongly prefers full support and LCD≈0
    """
    ssdi_gap = float(abs(record["SSDI"] - record["target_ssdi"]))
    lcd_gap = float(record["LCD"] - record["target_lcd"])
    lds_gap = float(record["LDS"] - record["target_lds"])
    theta_gap = float(record.get("theta_gap", 0.0))
    missing_over = float(record.get("missing_rate_overflow", 0.0))

    structure_bias = float(record.get("structure_bias", 0.0))
    mode_value = _resolve_mode_value_local(structure_bias)

    point_gap = float(np.hypot(lcd_gap, lds_gap))

    if phase == "near_zero":
        point_tol = _near_zero_point_tolerance(record, stage_ssdi_error)

        huge = 5000.0 * max(0.0, ssdi_gap - stage_ssdi_error) ** 2
        radial = 180.0 * ssdi_gap
        point_ratio = point_gap / max(point_tol, 1e-12)
        point = 90.0 * (point_ratio ** 2)
        angle = 0.8 * (theta_gap / max(theta_tol, np.deg2rad(1.0))) ** 2

        lcd_zero_penalty = 0.0
        if float(record["target_lcd"]) > point_tol and float(record["LCD"]) <= 1e-15:
            lcd_zero_penalty = 15.0

        score = huge + radial + point + angle + 0.5 * (missing_over ** 2) + lcd_zero_penalty

        if mode_value == -1:
            n_ck = record.get("n_ck")
            zero_count = 0 if n_ck is None else int(np.sum(np.asarray(n_ck) <= 0))
            full_support_penalty = 80.0 * zero_count
            lcd_zero_penalty_left = 400.0 * float(record["LCD"]) ** 2
            score += full_support_penalty + lcd_zero_penalty_left

        return float(score)

    huge = 1200.0 * max(0.0, ssdi_gap - stage_ssdi_error)
    radial = 100.0 * ssdi_gap
    point = 18.0 * (lcd_gap * lcd_gap + lds_gap * lds_gap)
    angle = 10.0 * (theta_gap / max(theta_tol, np.deg2rad(1.0))) ** 2

    miss = 1.0 * (missing_over ** 2)
    full_support_penalty = 0.0
    lcd_zero_penalty = 0.0

    # ---------------------------------------------------------
    # 本次修改：
    # 删除 strict-middle below_lcdmax 的额外重罚
    # 不再因为 strict_middle_nonempty_required 为真而额外压制 sparse 候选
    # ---------------------------------------------------------

    if mode_value == -1:
        n_ck = record.get("n_ck")
        if n_ck is not None:
            zero_count = int(np.sum(np.asarray(n_ck) <= 0))
            full_support_penalty += 50.0 * zero_count
        lcd_zero_penalty = 200.0 * float(record["LCD"]) ** 2

    if phase == "near_one" and structure_bias >= 0.95:
        strict_ssdi_gap = max(0.0, ssdi_gap - 0.02)

        strict_ssdi_penalty = 5000.0 * (strict_ssdi_gap ** 2)
        radial_penalty = 300.0 * ssdi_gap

        in_strict_band = (ssdi_gap <= 0.02)
        if in_strict_band:
            lcd_reward = -120.0 * float(record["LCD"])
            theta_reward = -10.0 * float(record.get("actual_theta", 0.0))
        else:
            lcd_reward = 0.0
            theta_reward = 0.0

        return float(
            strict_ssdi_penalty
            + radial_penalty
            + 0.2 * miss
            + full_support_penalty
            + lcd_zero_penalty
            + lcd_reward
            + theta_reward
        )

    rightward_bonus = 0.0
    if phase == "near_one":
        in_band = ssdi_gap <= stage_ssdi_error
        if structure_bias >= 0.90 and in_band:
            rightward_bonus = -12.0 * float(record["LCD"]) - 2.0 * float(record.get("actual_theta", 0.0))
        elif float(record["target_lcd"]) >= float(record["target_lds"]):
            rightward_bonus = -0.15 * float(record["LCD"])

    return float(
        huge
        + radial
        + point
        + angle
        + miss
        + full_support_penalty
        + lcd_zero_penalty
        + rightward_bonus
    )




def _row_col_expected_mass(n_ck: np.ndarray) -> np.ndarray:
    row_tot = n_ck.sum(axis=1, keepdims=True).astype(float)
    col_tot = n_ck.sum(axis=0, keepdims=True).astype(float)
    N = float(max(1, int(n_ck.sum())))
    return (row_tot @ col_tot) / N


def _top_zero_positions_by_value(n_ck: np.ndarray, topk: int = 32) -> List[Tuple[int, int]]:
    q = _row_col_expected_mass(n_ck)
    zeros = np.argwhere(n_ck <= 0)
    if zeros.size == 0:
        return []
    scores = np.array([q[i, j] for i, j in zeros], dtype=float)
    order = np.argsort(-scores)
    return [tuple(map(int, zeros[idx])) for idx in order[:topk]]


def _top_nonzero_positions_by_value(n_ck: np.ndarray, topk: int = 32) -> List[Tuple[int, int]]:
    q = _row_col_expected_mass(n_ck)
    nz = np.argwhere(n_ck > 0)
    if nz.size == 0:
        return []
    scores = np.array([q[i, j] * n_ck[i, j] for i, j in nz], dtype=float)
    order = np.argsort(-scores)
    return [tuple(map(int, nz[idx])) for idx in order[:topk]]


def _top_nonzero_positions_by_mass(n_ck: np.ndarray, topk: int = 32) -> List[Tuple[int, int]]:
    nz = np.argwhere(n_ck > 0)
    if nz.size == 0:
        return []
    scores = np.array([n_ck[i, j] for i, j in nz], dtype=float)
    order = np.argsort(-scores)
    return [tuple(map(int, nz[idx])) for idx in order[:topk]]


def _move_mass(
    n_ck: np.ndarray,
    src: Tuple[int, int],
    dst: Tuple[int, int],
    amount: int,
) -> np.ndarray:
    out = n_ck.astype(int, copy=True)
    si, sj = map(int, src)
    di, dj = map(int, dst)
    amt = int(max(0, amount))
    if amt <= 0:
        return out
    if out[si, sj] <= 0:
        return out
    amt = min(amt, int(out[si, sj]))
    if amt <= 0:
        return out
    out[si, sj] -= amt
    out[di, dj] += amt
    return out


def _donor_from_heaviest(n_ck: np.ndarray, exclude: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    for pos in _top_nonzero_positions_by_mass(n_ck, topk=64):
        if exclude is not None and tuple(pos) == tuple(exclude):
            continue
        i, j = pos
        if n_ck[i, j] > 0:
            return (int(i), int(j))
    return None


def _receiver_existing_weak_support(n_ck: np.ndarray) -> Optional[Tuple[int, int]]:
    nz = np.argwhere(n_ck > 0)
    if nz.size == 0:
        return None
    scores = np.array([n_ck[i, j] for i, j in nz], dtype=float)
    idx = int(np.argmin(scores))
    return tuple(map(int, nz[idx]))


def _largest_rows_cols(n_ck: np.ndarray) -> Tuple[List[int], List[int]]:
    rows = list(np.argsort(-n_ck.sum(axis=1)))
    cols = list(np.argsort(-n_ck.sum(axis=0)))
    return rows, cols


def _apply_fill_high_value_zeros(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    steps: int = 1,
) -> np.ndarray:
    """
    Lower LCD:
    fill the highest-value zeros by moving mass from heavy donors.

    Fix:
    - stronger than the old 8% version
    - preferentially keeps filling high-value zeros
    - better suited for retreating from exact1 / sparse near1 skeleton
    """
    out = n_ck.astype(int, copy=True)

    for _ in range(int(max(1, steps))):
        targets = _top_zero_positions_by_value(out, topk=32)
        if not targets:
            break

        # Prefer one of the most valuable zeros
        top_pool = targets[: min(8, len(targets))]
        ti, tj = top_pool[int(rng.integers(0, len(top_pool)))]

        donor = _donor_from_heaviest(out, exclude=(ti, tj))
        if donor is None:
            break

        donor_mass = int(out[donor])
        if donor_mass <= 1:
            break

        # stronger fill than before:
        # old code used only 8%, which is too weak for sparse near1 seeds
        amount = max(1, int(donor_mass * 0.18))
        amount = min(amount, donor_mass)

        out = _move_mass(out, donor, (ti, tj), amount)

    return out


def _apply_create_high_value_zeros(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    steps: int = 1,
    *,
    target_ssdi: float = 0.5,
    structure_bias: float = 0.0,
) -> np.ndarray:
    """
    Middle 右移 / 增 LCD：

    新规则：
    - donor 不再优先选“高值 occupied cell”
    - 改成优先挖：小 client / 稀有 label / 小 cell
    - donor 的质量不再灌给一个峰，而是分散到多个 existing support
    - receiver 必须受 row/col/cell cap 约束
    """
    out = np.asarray(n_ck, dtype=int).copy()
    C, K = out.shape

    for _ in range(int(max(1, steps))):
        N = float(max(1, int(out.sum())))
        row_tot = out.sum(axis=1).astype(float)
        col_tot = out.sum(axis=0).astype(float)

        row_nz = np.sum(out > 0, axis=1)
        col_nz = np.sum(out > 0, axis=0)

        row_med = float(np.median(row_tot)) if C > 0 else 1.0
        col_med = float(np.median(col_tot)) if K > 0 else 1.0

        donors: List[Tuple[int, int, int]] = []
        donor_scores: List[float] = []

        for i in range(C):
            for j in range(K):
                v = int(out[i, j])
                if v <= 0:
                    continue

                # mild / moderate middle：避免一口气挖成空 row / 空 col
                if float(target_ssdi) <= 0.55:
                    if row_nz[i] <= 1 or col_nz[j] <= 1:
                        continue

                rare_label_pref = (row_med + 1.0) / (row_tot[i] + 1.0)
                small_client_pref = (col_med + 1.0) / (col_tot[j] + 1.0)
                weak_cell_pref = 1.0 / np.sqrt(1.0 + float(v))

                # 保留一点点“几何有效性”，但不再把高 Q 当主导
                q_soft = ((row_tot[i] * col_tot[j] / N) + 1.0) ** 0.15

                score = rare_label_pref * small_client_pref * weak_cell_pref * q_soft
                if score <= 0:
                    continue

                donors.append((i, j, v))
                donor_scores.append(float(score))

        if not donors:
            break

        donor_scores_arr = np.asarray(donor_scores, dtype=float)
        donor_scores_arr = donor_scores_arr / donor_scores_arr.sum()
        d_idx = int(rng.choice(np.arange(len(donors)), p=donor_scores_arr))
        si, sj, sv = donors[d_idx]

        receivers, recv_weights = _middle_existing_receiver_pool(
            out,
            src=(si, sj),
            target_ssdi=target_ssdi,
            structure_bias=structure_bias,
            axis_mix=0.35,
        )
        if not receivers:
            break

        # donor 整格清零，但要分散填出去
        # mild / moderate 时尽量分得更散
        num_receivers = int(min(
            len(receivers),
            max(3, min(6, sv))
        ))
        chosen = _middle_sample_diverse_receivers(
            receivers,
            recv_weights,
            rng,
            n_pick=num_receivers,
        )
        if not chosen:
            break

        out[si, sj] = 0
        remaining = int(sv)

        for k, (ri, rj, _) in enumerate(chosen):
            if remaining <= 0:
                break
            share = int(np.ceil(remaining / max(1, len(chosen) - k)))
            share = max(1, min(share, remaining))
            out[ri, rj] += int(share)
            remaining -= int(share)

    return out


def _apply_peak_reweight(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    steps: int = 1,
    *,
    target_ssdi: float = 0.5,
    structure_bias: float = 0.0,
) -> np.ndarray:
    """
    Middle 增 LDS：

    新规则：
    - 不再只往 top-3 rows × top-3 cols 交点灌
    - donor 从较弱 / 中等 existing support 来
    - receiver 在“已有 support + 容量未超 cap”的池里选
    - 允许略偏大 row/col，但仍强制分散，不能只灌一个点
    """
    out = np.asarray(n_ck, dtype=int).copy()

    for _ in range(int(max(1, steps))):
        positives = np.argwhere(out > 0)
        if positives.size == 0:
            break

        donor_cands = []
        donor_scores = []
        for i, j in positives:
            i, j = int(i), int(j)
            v = int(out[i, j])
            if v <= 1:
                continue
            donor_cands.append((i, j, v))
            donor_scores.append(1.0 / max(1.0, float(v)))  # 越弱越容易被抽

        if not donor_cands:
            break

        donor_scores_arr = np.asarray(donor_scores, dtype=float)
        donor_scores_arr = donor_scores_arr / donor_scores_arr.sum()
        d_idx = int(rng.choice(np.arange(len(donor_cands)), p=donor_scores_arr))
        si, sj, sv = donor_cands[d_idx]

        receivers, recv_weights = _middle_existing_receiver_pool(
            out,
            src=(si, sj),
            target_ssdi=target_ssdi,
            structure_bias=structure_bias,
            axis_mix=0.75,   # 可以偏向大 row/col，但不能失控
        )
        if not receivers:
            break

        chosen = _middle_sample_diverse_receivers(
            receivers,
            recv_weights,
            rng,
            n_pick=min(len(receivers), max(2, min(4, sv))),
        )
        if not chosen:
            break

        amt_total = max(1, int(sv * 0.30))
        amt_total = min(amt_total, int(out[si, sj]) - 1)
        if amt_total <= 0:
            continue

        out[si, sj] -= amt_total
        remaining = int(amt_total)

        for k, (ri, rj, _) in enumerate(chosen):
            if remaining <= 0:
                break
            share = int(np.ceil(remaining / max(1, len(chosen) - k)))
            share = max(1, min(share, remaining))
            out[ri, rj] += int(share)
            remaining -= int(share)

    return out

def _apply_flatten_reweight(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    steps: int = 1,
    *,
    target_ssdi: float = 0.5,
    structure_bias: float = 0.0,
) -> np.ndarray:
    """
    Middle 降 LDS：

    新规则：
    - donor 从最强 / 最拥挤的格子里来
    - 但 receiver 仍然走“已有 support + cap + 分散”
    - 不再允许把 donor 的大块质量灌给单个弱点
    """
    out = np.asarray(n_ck, dtype=int).copy()
    C, K = out.shape

    for _ in range(int(max(1, steps))):
        positives = np.argwhere(out > 0)
        if positives.size == 0:
            break

        N = float(max(1, int(out.sum())))
        row_tot = out.sum(axis=1).astype(float)
        col_tot = out.sum(axis=0).astype(float)
        row_share = row_tot / N
        col_share = col_tot / N

        donor_cands = []
        donor_scores = []

        for i, j in positives:
            i, j = int(i), int(j)
            v = int(out[i, j])
            if v <= 1:
                continue

            crowd = row_share[i] + col_share[j]
            score = float(v) * (1.0 + crowd)
            donor_cands.append((i, j, v))
            donor_scores.append(score)

        if not donor_cands:
            break

        donor_scores_arr = np.asarray(donor_scores, dtype=float)
        donor_scores_arr = donor_scores_arr / donor_scores_arr.sum()
        d_idx = int(rng.choice(np.arange(len(donor_cands)), p=donor_scores_arr))
        si, sj, sv = donor_cands[d_idx]

        receivers, recv_weights = _middle_existing_receiver_pool(
            out,
            src=(si, sj),
            target_ssdi=target_ssdi,
            structure_bias=structure_bias,
            axis_mix=0.10,   # flatten 更偏向弱 existing support
        )
        if not receivers:
            break

        chosen = _middle_sample_diverse_receivers(
            receivers,
            recv_weights,
            rng,
            n_pick=min(len(receivers), max(3, min(5, sv))),
        )
        if not chosen:
            break

        amt_total = max(1, int(sv * 0.12))
        amt_total = min(amt_total, int(out[si, sj]) - 1)
        if amt_total <= 0:
            continue

        out[si, sj] -= amt_total
        remaining = int(amt_total)

        for k, (ri, rj, _) in enumerate(chosen):
            if remaining <= 0:
                break
            share = int(np.ceil(remaining / max(1, len(chosen) - k)))
            share = max(1, min(share, remaining))
            out[ri, rj] += int(share)
            remaining -= int(share)

    return out

def _diag_floor_local(n_ck: np.ndarray) -> int:
    """
    Hard protection floor for diagonal cells:
    donor diagonal value may NOT be reduced below N / C / K / 2.
    Here N is total count in matrix.
    """
    N = int(np.sum(n_ck))
    C, K = n_ck.shape
    if C <= 0 or K <= 0:
        return 1
    return int(max(1, N // C // K // 2))


def _zipf_rank_probs_desc(values: List[float], tau: float = 1.2) -> np.ndarray:
    """
    Sort by value descending, then assign Zipf probabilities by rank.
    Larger value => higher rank => larger probability.
    """
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=float)

    order = np.argsort(-np.asarray(values, dtype=float))
    ranks = np.arange(1, n + 1, dtype=float)
    base = 1.0 / np.power(ranks, float(max(1e-8, tau)))

    probs = np.zeros(n, dtype=float)
    probs[order] = base
    probs_sum = float(probs.sum())
    if probs_sum <= 0:
        return np.full(n, 1.0 / n, dtype=float)
    return probs / probs_sum


def _sample_diag_donor_zipf(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    *,
    tail_frac: float = 0.8,
    tau: float = 1.2,
) -> Optional[int]:
    """
    Donor diagonal index:
    only sample from the weaker 80% diagonal cells (after descending sort),
    with Zipf probability inside that donor pool.

    Also enforces hard diagonal floor:
    donor diagonal may not go below N/C/K/2.
    """
    m = min(n_ck.shape[0], n_ck.shape[1])
    if m <= 0:
        return None

    floor_diag = _diag_floor_local(n_ck)

    diag_vals = np.array([int(n_ck[i, i]) for i in range(m)], dtype=int)
    order_desc = list(np.argsort(-diag_vals))  # large -> small

    # head 20% kept for receivers; donor pool is the remaining tail 80%
    head_n = max(1, int(np.ceil(0.2 * m)))
    donor_pool = order_desc[head_n:]

    # donor must still have movable mass above hard floor
    donor_pool = [i for i in donor_pool if int(n_ck[i, i]) > floor_diag]
    if len(donor_pool) == 0:
        return None

    donor_vals = [float(n_ck[i, i]) for i in donor_pool]
    probs = _zipf_rank_probs_desc(donor_vals, tau=tau)
    idx = int(rng.choice(np.arange(len(donor_pool)), p=probs))
    return int(donor_pool[idx])


def _sample_diag_receiver_head_random(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    *,
    head_frac: float = 0.2,
    exclude_idx: Optional[int] = None,
) -> Optional[int]:
    """
    Receiver diagonal index:
    uniformly random among the top 20% diagonal cells.
    NOT Zipf anymore, per your latest requirement.
    """
    m = min(n_ck.shape[0], n_ck.shape[1])
    if m <= 0:
        return None

    diag_vals = np.array([int(n_ck[i, i]) for i in range(m)], dtype=int)
    order_desc = list(np.argsort(-diag_vals))
    head_n = max(1, int(np.ceil(head_frac * m)))
    recv_pool = order_desc[:head_n]

    if exclude_idx is not None:
        recv_pool = [i for i in recv_pool if int(i) != int(exclude_idx)]

    if len(recv_pool) == 0:
        return None

    return int(recv_pool[int(rng.integers(0, len(recv_pool)))])


def _move_diag_mass_guarded(
    n_ck: np.ndarray,
    donor_idx: int,
    recv_idx: int,
    amount: int,
    *,
    recv_cap_frac: float = 0.08,
) -> np.ndarray:
    """
    Guarded diagonal-to-diagonal move.

    Rules:
    1) donor diagonal cannot go below N/C/K/2
    2) receiver cannot absorb too much in one move
    """
    out = n_ck.astype(int, copy=True)
    donor_idx = int(donor_idx)
    recv_idx = int(recv_idx)
    if donor_idx == recv_idx:
        return out

    floor_diag = _diag_floor_local(out)

    donor_val = int(out[donor_idx, donor_idx])
    recv_val = int(out[recv_idx, recv_idx])

    movable = donor_val - floor_diag
    if movable <= 0:
        return out

    amt = int(max(0, amount))
    if amt <= 0:
        return out

    # single-receiver overload protection
    recv_cap = max(1, int(recv_val * recv_cap_frac))
    amt = min(amt, movable, recv_cap)
    if amt <= 0:
        return out

    out[donor_idx, donor_idx] -= amt
    out[recv_idx, recv_idx] += amt
    return out

def _apply_near1_binaryize_diagonal(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    keep_min_diag: int = 1,
    steps: int = 1,
    move_frac: float = 0.12,
    soften_frac: float = 0.0,
) -> np.ndarray:
    """
    near1 right-down / right-side diagonal concentration:

    New rule:
    - donor: only from weaker 80% diagonal cells
    - donor sampling: Zipf within donor pool
    - receiver: uniformly random among top 20% diagonal cells
    - hard diagonal protection: donor cannot go below N/C/K/2
    - receiver overload protection: one move cannot add too much to one receiver

    This keeps the "tend to stronger diagonal peaks" idea,
    but avoids always draining/feeding the same single diagonal cell.
    """
    out = n_ck.astype(int, copy=True)
    m = min(out.shape[0], out.shape[1])
    if m <= 1:
        return out

    floor_diag = _diag_floor_local(out)

    for _ in range(int(max(1, steps))):
        donor_idx = _sample_diag_donor_zipf(
            out,
            rng,
            tail_frac=0.8,
            tau=1.2,
        )
        if donor_idx is None:
            break

        recv_idx = _sample_diag_receiver_head_random(
            out,
            rng,
            head_frac=0.2,
            exclude_idx=donor_idx,
        )
        if recv_idx is None:
            break

        donor_val = int(out[donor_idx, donor_idx])

        # still allow outer keep_min_diag, but hard floor is the real protection
        effective_floor = max(int(keep_min_diag), int(floor_diag))
        movable = donor_val - effective_floor
        if movable <= 0:
            continue

        move_amt = max(1, int(movable * move_frac))
        out = _move_diag_mass_guarded(
            out,
            donor_idx,
            recv_idx,
            move_amt,
            recv_cap_frac=0.08,
        )

        # optional soften: keep your old behavior, but do NOT touch diagonal protection logic
        if soften_frac > 0.0:
            src = _donor_from_heaviest(out)
            dst = _receiver_existing_weak_support(out)
            if src is not None and dst is not None and src != dst:
                soften = max(1, int(out[src] * soften_frac))
                out = _move_mass(out, src, dst, soften)

    return out

def _propose_move_near_one_mid(
    record: Dict[str, Any],
    target_spec: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    [MOD] near1 中段连续移动专用：
    只处理 near1 里的中段（大致 -0.3 ~ 0.3 这类区域），
    直接基于当前点做连续小步修正，不走：
    - 左侧 y轴基准重置
    - 右侧 diagonal binaryization

    核心目标：
    - 连续地调 LCD
    - 连续地调 LDS
    - 避免掉到 y轴盆地或 exact1 盆地
    """
    n = record["n_ck"].copy().astype(int)

    cur_lcd = float(record["LCD"])
    cur_lds = float(record["LDS"])
    target_lcd = float(target_spec["target_lcd"])
    target_lds = float(target_spec["target_lds"])

    d_lcd = target_lcd - cur_lcd
    d_lds = target_lds - cur_lds

    # [MOD] 中段只做连续小步，不做大步
    gap = abs(d_lcd) + abs(d_lds)
    if gap > 0.10:
        steps = 2
    else:
        steps = 1

    # =========================================================
    # 四象限连续移动：
    # 右上 : 增 LCD, 增 LDS -> create zeros + peak
    # 右下 : 增 LCD, 减 LDS -> create zeros + flatten
    # 左上 : 减 LCD, 增 LDS -> fill zeros + peak
    # 左下 : 减 LCD, 减 LDS -> fill zeros + flatten
    #
    # 注意：这里全部只用小步 steps，不允许 near1 极端骨架操作
    # =========================================================
    if d_lcd >= 0 and d_lds >= 0:
        out = _apply_create_high_value_zeros(n, rng, steps=steps)
        out = _apply_peak_reweight(out, rng, steps=steps)
        return out

    if d_lcd >= 0 and d_lds < 0:
        out = _apply_create_high_value_zeros(n, rng, steps=steps)
        out = _apply_flatten_reweight(out, rng, steps=steps)
        return out

    if d_lcd < 0 and d_lds >= 0:
        out = _apply_fill_high_value_zeros(n, rng, steps=steps)
        out = _apply_peak_reweight(out, rng, steps=steps)
        return out

    out = _apply_fill_high_value_zeros(n, rng, steps=steps)
    out = _apply_flatten_reweight(out, rng, steps=steps)
    return out



def _left_branch_target_distance(
    lcd: float,
    lds: float,
    target_lcd: float,
    target_lds: float,
) -> float:
    """
    Greedy score for near1-left branch:
    only care whether we are closer to the target (LCD, LDS) point.
    """
    dx = float(lcd) - float(target_lcd)
    dy = float(lds) - float(target_lds)
    return float(dx * dx + dy * dy)


def _same_matrix(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b))


def _build_near1_left_yaxis_anchor_record(
    seed_record: Dict[str, Any],
    rng: np.random.Generator,
    *,
    target_ssdi: float,
) -> Dict[str, Any]:
    """
    near1-left 的 y-axis 锚点：

    Step 1:
    - 先把矩阵推到 LCD=0 / full support

    Step 2:
    - 在 full support 约束下，把 LDS 校准到 target_ssdi 附近
    - 但“对角线峰间重分配”必须走统一规则：
        donor   : 后 80% 对角线里按 Zipf 采样
        receiver: 前 20% 对角线里随机选一个
        donor 不能跌破 N/C/K/2
    """
    n = seed_record["n_ck"].copy().astype(int)

    alpha_used = float(seed_record.get("alpha_used", 1.2) or 1.2)
    beta_used = float(seed_record.get("beta_used", 1.2) or 1.2)
    lcdtype = str(seed_record.get("lcd_type", "client"))
    ldstype = str(seed_record.get("lds_type", "client"))
    lcd_params = dict(seed_record.get("lcd_params", {}) or {})
    lds_params = dict(seed_record.get("lds_params", {}) or {})

    target_ssdi = float(np.clip(target_ssdi, 0.0, 1.0))

    # ---------------------------------------------------------
    # Step 1: 精确推到 LCD=0 / full support
    # ---------------------------------------------------------
    for _ in range(12):
        if np.any(n <= 0):
            n = _zipf_pareto_refill_zeros(
                n,
                rng=rng,
                alpha=alpha_used,
                beta=beta_used,
                max_fill_each=3,
            )
        n = _mode_minus_one_repair(n)

        if not np.any(n <= 0):
            cur = compute_ssdi_metrics(n)
            if float(cur["LCD"]) <= 1e-12:
                break

    # 双保险：若仍有 0，则逐个补成正
    if np.any(n <= 0):
        zeros = np.argwhere(n <= 0)
        for i, j in zeros:
            i, j = int(i), int(j)

            # 先尝试同列 donor，但 donor 也优先从对角 donor 规则来
            donor_idx = _sample_diag_donor_zipf(n, rng, tail_frac=0.8, tau=1.2)
            if donor_idx is not None and int(n[donor_idx, donor_idx]) > _diag_floor_local(n):
                if donor_idx == j:
                    # donor 正好在该列，最自然
                    n = _move_diag_mass_guarded(n, donor_idx, donor_idx, 0, recv_cap_frac=0.08)
                    if int(n[donor_idx, donor_idx]) > _diag_floor_local(n):
                        n[donor_idx, donor_idx] -= 1
                        n[i, j] += 1
                        continue

            # 否则退回同列最大 donor（这里只是补 0 的兜底，不是主峰重分配）
            donor_row = int(np.argmax(n[:, j]))
            if int(n[donor_row, j]) > 1:
                n[donor_row, j] -= 1
                n[i, j] += 1
                continue

            # 最后兜底全局最大 donor
            di, dj = np.unravel_index(np.argmax(n), n.shape)
            di, dj = int(di), int(dj)
            if int(n[di, dj]) > 1:
                n[di, dj] -= 1
                n[i, j] += 1

    # ---------------------------------------------------------
    # Step 2: 在 LCD=0 约束下校准 LDS≈target_ssdi
    #
    # 这里的关键修正：
    # - 不再从全矩阵任意大值 donor / 任意大值 receiver 里乱搬
    # - 先做“对角 donor -> 对角 receiver”的受保护重分配
    # - 必要时再做轻微非对角 flatten / peak
    # ---------------------------------------------------------
    def _axis_score(arr: np.ndarray) -> float:
        m = compute_ssdi_metrics(arr)
        lcd = float(m["LCD"])
        lds = float(m["LDS"])
        return float(1000.0 * abs(lcd) + abs(lds - target_ssdi))

    best_n = n.copy()
    best_score = _axis_score(best_n)

    for _ in range(120):
        improved = False
        cur = best_n.copy()
        cur_metrics = compute_ssdi_metrics(cur)
        cur_lds = float(cur_metrics["LDS"])

        candidates: List[np.ndarray] = []

        # ---------------------------------------------
        # 2A. 若 LDS 太高：先做对角 flatten
        # donor: 后80%对角线 Zipf
        # recv : 前20%对角线随机
        # ---------------------------------------------
        if cur_lds > target_ssdi + 1e-4:
            for _trial in range(10):
                donor_idx = _sample_diag_donor_zipf(cur, rng, tail_frac=0.8, tau=1.2)
                if donor_idx is None:
                    break

                recv_idx = _sample_diag_receiver_head_random(
                    cur, rng, head_frac=0.2, exclude_idx=donor_idx
                )
                if recv_idx is None:
                    break

                donor_val = int(cur[donor_idx, donor_idx])
                floor_diag = _diag_floor_local(cur)
                movable = donor_val - floor_diag
                if movable <= 0:
                    continue

                # 轻量 move，不要一步抽太多
                amt = max(1, int(movable * 0.06))
                cand = _move_diag_mass_guarded(
                    cur,
                    donor_idx,
                    recv_idx,
                    amt,
                    recv_cap_frac=0.08,
                )
                if np.any(cand <= 0):
                    continue
                candidates.append(cand)

            # 若纯对角 move 还不够，再叠加一点轻微 flatten
            nz = np.argwhere(cur > 0)
            weak_pool = []
            for i, j in nz:
                i, j = int(i), int(j)
                if int(cur[i, j]) > 1:
                    weak_pool.append((i, j, int(cur[i, j])))
            weak_pool.sort(key=lambda x: x[2])

            donors = _top_nonzero_positions_by_mass(cur, topk=16)
            for si, sj in donors[:6]:
                src_val = int(cur[si, sj])
                if src_val <= 1:
                    continue
                for ri, rj, rv in weak_pool[:8]:
                    if (ri, rj) == (si, sj):
                        continue
                    out = cur.copy()
                    amt = 1 if src_val <= 6 else 2
                    amt = min(amt, int(out[si, sj] - 1))
                    if amt <= 0:
                        continue
                    out[si, sj] -= amt
                    out[ri, rj] += amt
                    if np.any(out <= 0):
                        continue
                    candidates.append(out)

        # ---------------------------------------------
        # 2B. 若 LDS 太低：先做对角 peak
        # donor: 后80%对角线 Zipf
        # recv : 前20%对角线随机
        # ---------------------------------------------
        else:
            for _trial in range(10):
                donor_idx = _sample_diag_donor_zipf(cur, rng, tail_frac=0.8, tau=1.2)
                if donor_idx is None:
                    break

                recv_idx = _sample_diag_receiver_head_random(
                    cur, rng, head_frac=0.2, exclude_idx=donor_idx
                )
                if recv_idx is None:
                    break

                donor_val = int(cur[donor_idx, donor_idx])
                floor_diag = _diag_floor_local(cur)
                movable = donor_val - floor_diag
                if movable <= 0:
                    continue

                amt = max(1, int(movable * 0.04))
                cand = _move_diag_mass_guarded(
                    cur,
                    donor_idx,
                    recv_idx,
                    amt,
                    recv_cap_frac=0.08,
                )
                if np.any(cand <= 0):
                    continue
                candidates.append(cand)

            # 再允许少量 tail -> peak 的非对角轻补峰
            nz = np.argwhere(cur > 0)
            vals = [(int(i), int(j), int(cur[i, j])) for i, j in nz]
            vals_sorted_desc = sorted(vals, key=lambda x: -x[2])
            vals_sorted_asc = sorted(vals, key=lambda x: x[2])

            peak_targets = vals_sorted_desc[:6]
            tail_sources = vals_sorted_asc[:10]

            for si, sj, sv in tail_sources:
                if sv <= 1:
                    continue
                for di, dj, dv in peak_targets:
                    if (si, sj) == (di, dj):
                        continue
                    out = cur.copy()
                    amt = 1 if sv <= 6 else 2
                    amt = min(amt, int(out[si, sj] - 1))
                    if amt <= 0:
                        continue
                    out[si, sj] -= amt
                    out[di, dj] += amt
                    if np.any(out <= 0):
                        continue
                    candidates.append(out)

        # 贪心接受：只要更接近同半径 y 轴点就收
        for cand in candidates:
            if np.any(cand <= 0):
                continue
            sc = _axis_score(cand)
            if sc + 1e-15 < best_score:
                best_score = sc
                best_n = cand
                improved = True

        if not improved:
            break

    rec = _safe_record_from_matrix(
        best_n,
        target_ssdi=float(target_ssdi),
        lcdtype=lcdtype,
        ldstype=ldstype,
        alpha=alpha_used,
        beta=beta_used,
        lcd_params=lcd_params,
        lds_params=lds_params,
        source_stage="near_one left y-axis anchor",
        generator_variant="seed_near_one_left_yaxis_anchor",
    )

    rec["near_one_left_anchor_lcd"] = float(rec["LCD"])
    rec["near_one_left_anchor_lds"] = float(rec["LDS"])
    rec["near_one_left_anchor_theta"] = float(
        _compute_theta_from_lcd_lds(rec["LCD"], rec["LDS"])
    )
    return rec



def _near1_left_mode_minus_one_microstep(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    *,
    target_lcd: float,
    target_lds: float,
    refill_alpha: float = 1.2,
    refill_beta: float = 1.2,
    max_fill_each: int = 3,
    peak_soften_frac: float = 0.015,
) -> np.ndarray:
    """
    near1-left 唯一搜索器：

    当前点已在“同半径 y 轴锚点”附近。
    之后只做 4 类矫正候选，谁更接近 target 就接受谁。

    这次的关键修正：
    - 涉及“峰间搬运”的 donor / receiver 统一落实到左侧：
        donor   : 后80%对角线里按 Zipf 采样
        receiver: 前20%对角线里随机选一个
    - donor 对角线不能被抽到 N/C/K/2 以下
    - 不再允许把大块质量灌到奇怪的非对角热点
    """
    n = n_ck.copy().astype(int)
    C, K = n.shape

    def _score(arr: np.ndarray) -> float:
        m = compute_ssdi_metrics(arr)
        return _left_branch_target_distance(
            float(m["LCD"]),
            float(m["LDS"]),
            float(target_lcd),
            float(target_lds),
        )

    cur_metrics = compute_ssdi_metrics(n)
    cur_lcd = float(cur_metrics["LCD"])
    cur_lds = float(cur_metrics["LDS"])
    cur_score = _score(n)

    dx = float(target_lcd - cur_lcd)
    dy = float(target_lds - cur_lds)

    candidates: List[np.ndarray] = []

    # =========================================================
    # A. 增大 LDS，增大 LCD  (right-up correction)
    # 仍允许把小值整格挖掉制造 0，
    # 但“加强峰”时优先走对角 receiver 规则，不再总灌到任意非对角热点
    # =========================================================
    if dx > 0 and dy > 0:
        positives = np.argwhere(n > 0)
        for si, sj in positives:
            si, sj = int(si), int(sj)
            src_val = int(n[si, sj])
            if src_val <= 0:
                continue

            # receiver 优先用头部 20% 对角线里随机选一个
            recv_diag = _sample_diag_receiver_head_random(n, rng, head_frac=0.2, exclude_idx=None)

            out = n.copy()
            out[si, sj] = 0
            if recv_diag is not None:
                out[recv_diag, recv_diag] += src_val
            else:
                # 兜底：同列最大
                col_peak_row = int(np.argmax(n[:, sj]))
                if col_peak_row == si:
                    continue
                out[col_peak_row, sj] += src_val
            candidates.append(out)

    # =========================================================
    # B. 增大 LDS，减小 LCD  (left-up correction)
    # 从小值拿一点：
    # - 一小部分填一个高价值 0
    # - 剩余部分优先加到头部20%对角 receiver
    # =========================================================
    if dx < 0 and dy > 0:
        zeros = _top_zero_positions_by_value(n, topk=48)
        positives = np.argwhere(n > 1)

        for si, sj in positives:
            si, sj = int(si), int(sj)
            src_val = int(n[si, sj])
            if src_val <= 1:
                continue

            same_col_zeros = [(i, j) for (i, j) in zeros if int(j) == sj]
            if not same_col_zeros:
                continue

            zi, zj = same_col_zeros[0]
            recv_diag = _sample_diag_receiver_head_random(n, rng, head_frac=0.2, exclude_idx=None)

            out = n.copy()
            take = 2 if src_val >= 4 else 1
            take = min(take, int(out[si, sj] - 1))
            if take <= 0:
                continue

            fill_amt = 1
            peak_amt = take - fill_amt
            if peak_amt < 0:
                continue

            out[si, sj] -= take
            out[zi, zj] += fill_amt
            if peak_amt > 0:
                if recv_diag is not None:
                    out[recv_diag, recv_diag] += peak_amt
                else:
                    col_peak_row = int(np.argmax(n[:, sj]))
                    if col_peak_row != si:
                        out[col_peak_row, sj] += peak_amt
                    else:
                        out[si, sj] += peak_amt
            candidates.append(out)

    # =========================================================
    # C. 减小 LDS，减小 LCD  (left-down correction)
    # 从 donor 对角线（后80%，Zipf）里少量拿出，分散填到高价值 0
    # donor 受 N/C/K/2 保护
    # =========================================================
    if dx < 0 and dy < 0:
        zeros = _top_zero_positions_by_value(n, topk=48)

        if zeros:
            for _trial in range(10):
                donor_idx = _sample_diag_donor_zipf(n, rng, tail_frac=0.8, tau=1.2)
                if donor_idx is None:
                    break

                donor_val = int(n[donor_idx, donor_idx])
                floor_diag = _diag_floor_local(n)
                movable = donor_val - floor_diag
                if movable <= 0:
                    continue

                top_zeros = zeros[: min(3, len(zeros))]
                out = n.copy()

                remain = int(min(3, movable))
                if remain <= 0:
                    continue

                for zi, zj in top_zeros:
                    zi, zj = int(zi), int(zj)
                    if remain <= 0:
                        break
                    out[donor_idx, donor_idx] -= 1
                    out[zi, zj] += 1
                    remain -= 1

                candidates.append(out)

    # =========================================================
    # D. 减小 LDS，增大 LCD  (right-down main path)
    # 主路径：
    # - 优先从 donor 对角线（后80%，Zipf）里拿
    # - 但不能低于 N/C/K/2
    # - 然后在同列找中小接收者，不回主峰
    # =========================================================
    if dx > 0 and dy < 0:
        for _trial in range(12):
            donor_idx = _sample_diag_donor_zipf(n, rng, tail_frac=0.8, tau=1.2)
            if donor_idx is None:
                break

            sj = int(donor_idx)  # 这里主用 donor 对角线所在列
            donor_val = int(n[donor_idx, donor_idx])
            floor_diag = _diag_floor_local(n)
            movable = donor_val - floor_diag
            if movable <= 0:
                continue

            # 同列里找“非最大”的小/中值接收，而不是回主峰
            col_vals = [
                (i, int(n[i, sj]))
                for i in range(C)
                if i != donor_idx and int(n[i, sj]) > 0
            ]
            if not col_vals:
                continue

            col_vals_sorted = sorted(col_vals, key=lambda x: x[1])
            recv_row = int(col_vals_sorted[0][0])  # 最弱非零接收

            out = n.copy()
            # 这里不是整列抽空，而是一次拿少量，防止 donor 被快速吃穿
            amt = max(1, int(movable * 0.06))
            amt = min(amt, movable)
            if amt <= 0:
                continue

            out[donor_idx, donor_idx] -= amt
            out[recv_row, sj] += amt
            candidates.append(out)

        # 仍允许非对角小值 -> 同列中小值 的温和 right-down move，
        # 但不允许把对角 donor 抽到 1
        positives = np.argwhere(n > 0)
        for si, sj in positives:
            si, sj = int(si), int(sj)

            # 若是对角 donor，则必须受 floor 保护
            if si == sj:
                floor_diag = _diag_floor_local(n)
                if int(n[si, sj]) <= floor_diag:
                    continue

            src_val = int(n[si, sj])
            if src_val <= 0:
                continue

            col_vals = [(i, int(n[i, sj])) for i in range(C) if i != si and int(n[i, sj]) > 0]
            if not col_vals:
                continue

            col_vals_sorted = sorted(col_vals, key=lambda x: x[1])
            recv_row = int(col_vals_sorted[0][0])

            out = n.copy()
            if si == sj:
                movable = int(out[si, sj] - _diag_floor_local(out))
                if movable <= 0:
                    continue
                amt = max(1, int(movable * 0.04))
                amt = min(amt, movable)
                if amt <= 0:
                    continue
                out[si, sj] -= amt
                out[recv_row, sj] += amt
            else:
                out[si, sj] = 0
                out[recv_row, sj] += src_val
            candidates.append(out)

    # ---------------------------------------------------------
    # 贪心接受：只要更接近 target 就移动
    # ---------------------------------------------------------
    best = n
    best_score = cur_score
    for cand in candidates:
        sc = _score(cand)
        if sc + 1e-15 < best_score:
            best = cand
            best_score = sc

    return best

def _propose_move_near_one(
    record: Dict[str, Any],
    target_spec: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    near1 统一规则：

    - 左侧：必须先回到“同半径 y 轴锚点”，然后只用 near1-left 专用贪心搜索
    - 右侧：保持当前 exact1 / extremal 骨架右推逻辑
    """
    n = record["n_ck"].copy().astype(int)
    cur = compute_ssdi_metrics(n)

    cur_ssdi = float(cur["SSDI"])
    cur_lcd = float(cur["LCD"])
    cur_lds = float(cur["LDS"])
    cur_theta = float(_compute_theta_from_lcd_lds(cur_lcd, cur_lds))
    cur_missing = float(cur.get("missing_rate", 0.0))

    target_ssdi = float(target_spec["target_ssdi"])
    target_lcd = float(target_spec["target_lcd"])
    target_lds = float(target_spec["target_lds"])
    target_theta = float(target_spec["target_theta"])
    target_missing = float(_target_missing_rate_from_target_geometry(target_lcd, target_lds))

    seed_lcd = float(target_spec.get("near_one_seed_lcd", record.get("near_one_seed_lcd", cur_lcd)))
    seed_theta = float(target_spec.get("near_one_seed_theta", record.get("near_one_seed_theta", cur_theta)))
    lcd_margin = float(target_spec.get("near_one_lcd_margin", 1e-12))

    d_ssdi = target_ssdi - cur_ssdi
    d_lds = target_lds - cur_lds

    mode_value = _resolve_mode_value_local(float(record.get("structure_bias", 0.0)))
    lds_eps = 5e-4

    # mode=-1: 优先修零 / 补零
    if mode_value == -1 and np.any(n <= 0):
        n = _zipf_pareto_refill_zeros(
            n,
            rng=rng,
            alpha=1.2,
            beta=1.2,
            max_fill_each=2,
        )
        n = _mode_minus_one_repair(n)
        return n

    # =========================================================
    # 只按 exact1 左右分流
    # =========================================================
    is_left_of_exact1 = (target_lcd < seed_lcd - lcd_margin)
    is_right_of_exact1 = not is_left_of_exact1

    # =========================================================
    # A. LEFT SIDE
    # 只允许：
    # 1) 先切到同半径 y 轴锚点
    # 2) 再做 near1-left 专用贪心搜索
    # 没有其他 left 操作
    # =========================================================
    if is_left_of_exact1:
        left_anchor_n = target_spec.get("near_one_left_anchor_n_ck", None)
        left_anchor_lcd = float(target_spec.get("near_one_left_anchor_lcd", 0.0))
        left_anchor_lds = float(target_spec.get("near_one_left_anchor_lds", 0.0))

        if left_anchor_n is not None:
            cur_dist = _left_branch_target_distance(
                cur_lcd, cur_lds, target_lcd, target_lds
            )
            anchor_dist = _left_branch_target_distance(
                left_anchor_lcd, left_anchor_lds, target_lcd, target_lds
            )

            # 当前若不比锚点更好，直接回锚点
            if cur_dist >= anchor_dist - 1e-15 and not _same_matrix(n, left_anchor_n):
                return np.array(left_anchor_n, dtype=int, copy=True)

        # 已经在锚点附近 / 比锚点更近后，只做 left 专用贪心搜索
        return _near1_left_mode_minus_one_microstep(
            n,
            rng,
            target_lcd=target_lcd,
            target_lds=target_lds,
            refill_alpha=1.2,
            refill_beta=1.2,
            max_fill_each=3,
            peak_soften_frac=0.015,
        )

    # =========================================================
    # B. RIGHT SIDE
    # 统一从 exact1/extremal 骨架向右推进
    # =========================================================
    if is_right_of_exact1:
        theta_shift = target_theta - seed_theta

        # moderate right shift
        if theta_shift <= np.deg2rad(6.0):
            out = _apply_near1_binaryize_diagonal(
                n,
                rng,
                keep_min_diag=1,
                steps=1,
                move_frac=0.08,
                soften_frac=0.0,
            )

            if cur_missing > target_missing + 0.06:
                out = _activate_sparse_same_row_support_from_diag(
                    out,
                    rng,
                    opens=1,
                    amount_frac=0.003,
                    min_amount=1,
                )
            return out

        # stronger right shift
        out = _apply_near1_binaryize_diagonal(
            n,
            rng,
            keep_min_diag=1,
            steps=1,
            move_frac=0.16,
            soften_frac=0.0,
        )

        if d_lds < -lds_eps:
            out = _apply_flatten_reweight(out, rng, steps=1)

        if cur_missing > target_missing + 0.08 and mode_value != 1:
            out = _activate_sparse_same_row_support_from_diag(
                out,
                rng,
                opens=1,
                amount_frac=0.002,
                min_amount=1,
            )

        return out



    return n





def _near_zero_point_tolerance(record: Dict[str, Any], stage_ssdi_error: float) -> float:
    """
    near_zero 专属成功容差：
    - success 看的是到 target (lcd, lds) 点的二维距离
    - 容差按 target_ssdi 成比例缩放：target 越小，允许误差越小
    - 但设置一个绝对下限，避免 very small ssdi 因整数矩阵离散性而永远无法成功

    当前规则：
        tol = max(abs_floor, rel_ratio * target_ssdi)

    经验上：
    - abs_floor 不宜太小，否则 0.01/0.02 一带会几乎全失败
    - rel_ratio 不宜太大，否则 near0 会过宽，丢掉“精细贴点”的意义
    """
    target_ssdi = float(record.get("target_ssdi", 0.0))
    abs_floor = 0.0035
    rel_ratio = 0.12
    return float(max(abs_floor, rel_ratio * target_ssdi))


def _near0_stage_steps(
    record: Dict[str, Any],
    target_spec: Dict[str, Any],
) -> Tuple[int, int, int, float]:
    """
    near0 三阶段控制:
    - lcd_steps    : Phase A support/LCD 阶段步数
    - lds_steps    : Phase B fixed-support LDS 阶段步数
    - repair_steps : Phase C support-preserving 精修步数
    - lcd_tol      : 只有 LCD 进入这个容差带，才允许进入 LDS 阶段

    这版重点：
    - LCD 先做准
    - 一旦 LCD 进带，给 Phase B / C 更充足的 LDS 精修预算
    """
    target_ssdi = float(target_spec.get("target_ssdi", record.get("target_ssdi", 0.0)))
    target_lcd = float(target_spec["target_lcd"])
    target_lds = float(target_spec["target_lds"])

    cur_lcd = float(record["LCD"])
    cur_lds = float(record["LDS"])

    lcd_gap = abs(target_lcd - cur_lcd)
    lds_gap = abs(target_lds - cur_lds)

    denom = max(target_ssdi, 0.02)

    def _to_steps(g: float, hard_cap: int) -> int:
        rel = float(g) / denom
        if rel >= 1.20:
            s = hard_cap
        elif rel >= 0.80:
            s = max(3, hard_cap - 1)
        elif rel >= 0.40:
            s = 2
        elif rel >= 0.10:
            s = 1
        else:
            s = 0
        return int(min(hard_cap, s))

    # Phase A 还是以 LCD 为主，但不用无限大
    lcd_steps = _to_steps(lcd_gap, hard_cap=8)
    # Phase B 现在明显加强
    lds_steps = _to_steps(lds_gap, hard_cap=10)

    if target_lcd > 1e-12 and cur_lcd <= 1e-15:
        if target_ssdi <= 0.02:
            lcd_steps = max(lcd_steps, 5)
        elif target_ssdi <= 0.05:
            lcd_steps = max(lcd_steps, 6)
        else:
            lcd_steps = max(lcd_steps, 4)

    if target_ssdi <= 0.02:
        lcd_steps = min(lcd_steps, 6)
        lds_steps = min(max(lds_steps, 4), 8)
        repair_steps = 6
    elif target_ssdi <= 0.05:
        lcd_steps = min(lcd_steps, 7)
        lds_steps = min(max(lds_steps, 5), 10)
        repair_steps = 8
    else:
        lcd_steps = min(lcd_steps, 8)
        lds_steps = min(max(lds_steps, 6), 10)
        repair_steps = 10

    # strict LCD gate
    lcd_tol = max(0.0015, 0.22 * max(target_lcd, 0.0))
    if target_lcd <= 1e-12:
        lcd_tol = 1e-12

    # 如果 LCD 已经接近目标，明显加大 Phase B/Phase C 的 LDS 精修预算
    if abs(cur_lcd - target_lcd) <= 2.0 * lcd_tol:
        lds_steps = max(lds_steps, 6 if target_ssdi <= 0.05 else 8)
        repair_steps = max(repair_steps, 8 if target_ssdi <= 0.05 else 10)

    return int(lcd_steps), int(lds_steps), int(repair_steps), float(lcd_tol)

def _near0_open_support_once(
    n_ck: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Phase A / Phase C:
    做一次“开 support / 升 LCD”动作。

    新规则（按你刚刚要求重写）：
    - donor：仍然选“价值高但质量小”的正值格
    - 但 donor 的质量不再全部灌给 strongest support
    - 而是分散到多个“已有非0”的格子里
    - 优先分散到不同 row / col、较弱 existing support
    - 目标：升 LCD，但尽量少把 LDS 一起推高
    """
    out = n_ck.copy().astype(int)
    C, K = out.shape

    q = _row_col_expected_mass(out)

    positives = []
    donor_scores = []
    for i in range(C):
        for j in range(K):
            v = int(out[i, j])
            if v <= 0:
                continue
            positives.append((i, j, v))
            # 高 Q、低 v 的格子优先挖空
            donor_scores.append(float(q[i, j]) / max(1.0, float(v)))

    if not positives:
        return out

    donor_scores = np.asarray(donor_scores, dtype=float)
    donor_idx = int(np.argmax(donor_scores))
    si, sj, sv = positives[donor_idx]

    if sv <= 0:
        return out

    # ---------------------------------------------------------
    # receiver 候选：必须是已有非0格子，但不再选 strongest；
    # 改为优先“较弱 existing support + 不同 row/col + 分散”
    # ---------------------------------------------------------
    receivers = []
    recv_scores = []
    for i in range(C):
        for j in range(K):
            if i == si and j == sj:
                continue
            v = int(out[i, j])
            if v <= 0:
                continue

            # 想要的 receiver：
            # 1) 已有非0
            # 2) 值不要太大（避免全灌进主峰）
            # 3) 尽量与 donor 不同行不同列，增加分散性
            diff_bonus = 1.0
            if i != si:
                diff_bonus += 0.5
            if j != sj:
                diff_bonus += 0.5

            # 小 existing support 更优；但不能是 0
            # q 小一点也更适合承接，避免继续强化高价值峰
            score = diff_bonus / (1.0 + float(v)) / (1.0 + float(q[i, j]))

            receivers.append((i, j, v))
            recv_scores.append(score)

    if not receivers:
        return out

    recv_scores = np.asarray(recv_scores, dtype=float)
    order = np.argsort(-recv_scores)

    # 至少分散到 3 个点；若 donor 很大则可多一些
    num_receivers = int(min(len(order), max(3, min(8, sv))))
    chosen = [receivers[idx] for idx in order[:num_receivers]]

    # ---------------------------------------------------------
    # 把 donor 整格清零，但质量分散转移到多个已有非0格子
    # ---------------------------------------------------------
    out[si, sj] = 0

    remaining = int(sv)
    t = len(chosen)
    for idx, (ri, rj, _) in enumerate(chosen):
        if remaining <= 0:
            break

        # 尽量均匀分散，但前几个点略多一点，防止全是 1 太碎
        share = int(np.ceil(remaining / max(1, t - idx)))
        share = max(1, min(share, remaining))
        out[ri, rj] += share
        remaining -= share

    return out



def _near0_fill_support_once(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    *,
    amount: int = 1,
) -> np.ndarray:
    """
    Phase A / Phase C:
    做一次“填 support / 降 LCD”动作。

    规则：
    - 选一个高价值 0
    - 从已有最强 donor 拿出少量质量填进去
    """
    out = n_ck.copy().astype(int)

    zeros = _top_zero_positions_by_value(out, topk=32)
    if not zeros:
        return out

    zi, zj = zeros[0]

    donor = _donor_from_heaviest(out)
    if donor is None:
        return out

    di, dj = donor
    donor_val = int(out[di, dj])
    if donor_val <= 1:
        return out

    amt = int(max(1, amount))
    amt = min(amt, donor_val - 1)
    if amt <= 0:
        return out

    out[di, dj] -= amt
    out[zi, zj] += amt
    return out

def _build_near0_lcd_seed(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    *,
    target_lcd: float,
    target_ssdi: float,
    steps: int,
    lcd_tol: float,
    target_lds: Optional[float] = None,
) -> np.ndarray:
    """
    Phase A:
    只追 target_lcd，直到 LCD 进入容差带，才允许退出。

    这版按你的新要求重写：
    1) score 不再用 lcd_gap + ssdi_gap
       因为 LCD 与 SSDI 本身高度相关，重复了
    2) score 改成以 lcd_gap 为主、lds_gap 为辅
    3) 不再 no_improve >= 3 就提前停
    4) 必须“做够为止”，直到：
       - LCD 进入 lcd_tol 带内
       - 或者达到 500 次迭代上限
    """
    best = n_ck.copy().astype(int)

    # 没传 target_lds 时，退化成只看 LCD
    if target_lds is None:
        target_lds = 0.0

    def _metrics(arr: np.ndarray) -> Dict[str, float]:
        m = compute_ssdi_metrics(arr)
        return {
            "lcd": float(m["LCD"]),
            "lds": float(m["LDS"]),
            "ssdi": float(m["SSDI"]),
        }

    def _score(arr: np.ndarray) -> float:
        m = _metrics(arr)
        lcd_gap = abs(m["lcd"] - float(target_lcd))
        lds_gap = abs(m["lds"] - float(target_lds))

        # LCD 是绝对主目标，LDS 只是辅助 tie-break
        # 这里不用 SSDI gap，避免和 LCD 重复
        return float(lcd_gap * lcd_gap + 0.15 * lds_gap * lds_gap)

    if float(target_lcd) <= 1e-12:
        return best

    best_score = _score(best)

    # 你明确要求：做到够，或者到 500 次
    max_rounds = 500

    for _ in range(max_rounds):
        cur = _metrics(best)
        cur_lcd = cur["lcd"]
        lcd_gap = abs(cur_lcd - float(target_lcd))

        # 只有 LCD 进带才允许退出
        if lcd_gap <= float(lcd_tol):
            break

        cands = [best]

        if cur_lcd < target_lcd:
            # LCD 不够：连续更强 open
            c1 = _near0_open_support_once(best, rng)
            c2 = _near0_open_support_once(c1, rng)
            c3 = _near0_open_support_once(c2, rng)
            cands.extend([c1, c2, c3])
        else:
            # LCD 过高：连续 fill
            c1 = _near0_fill_support_once(best, rng, amount=1)
            c2 = _near0_fill_support_once(best, rng, amount=2)
            c3 = _near0_fill_support_once(best, rng, amount=3)
            cands.extend([c1, c2, c3])

        local_best = best
        local_score = best_score

        for cand in cands:
            sc = _score(cand)
            if sc + 1e-15 < local_score:
                local_best = cand
                local_score = sc

        # 不再因为“几轮没进步”就停；
        # 只要找到更好的就更新，否则就继续下一轮。
        if local_score + 1e-15 < best_score:
            best = local_best
            best_score = local_score

    return best

def _near0_support_preserving_lds_move(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    *,
    target_lds: float,
    steps: int,
) -> np.ndarray:
    """
    Phase B:
    固定 support，只在非0格子内调 LDS。

    新版：
    - donor > 1, receiver > 0
    - support 不变
    - 允许连续精修，不再只是很粗的 flatten
    - gap 大时可以搬 2~3
    - gap 小时自动退回 1，避免过冲
    """
    out = n_ck.copy().astype(int)

    def _cur_lds(arr: np.ndarray) -> float:
        return float(compute_ssdi_metrics(arr)["LDS"])

    for _ in range(int(max(1, steps))):
        cur_lds = _cur_lds(out)
        gap = float(target_lds - cur_lds)

        positives = [
            (i, j, int(out[i, j]))
            for i in range(out.shape[0])
            for j in range(out.shape[1])
            if int(out[i, j]) > 0
        ]
        donors = [(i, j, v) for (i, j, v) in positives if v > 1]
        if not positives or not donors:
            break

        if cur_lds < target_lds:
            # LDS 偏低：从较弱 nonzero 拿，补给较强 nonzero
            donor = min(donors, key=lambda x: (x[2], rng.random()))
            recv = max(positives, key=lambda x: (x[2], rng.random()))
        else:
            # LDS 偏高：从较强 nonzero 拿，补给较弱 nonzero
            donor = max(donors, key=lambda x: (x[2], rng.random()))
            recv = min(positives, key=lambda x: (x[2], rng.random()))

        di, dj, dv = donor
        ri, rj, _ = recv
        if di == ri and dj == rj:
            continue

        abs_gap = abs(gap)
        if abs_gap >= 0.015:
            amt = 3
        elif abs_gap >= 0.007:
            amt = 2
        else:
            amt = 1

        amt = min(int(amt), int(out[di, dj]) - 1)
        if amt <= 0:
            continue

        trial = out.copy()
        trial[di, dj] -= amt
        trial[ri, rj] += amt

        # 只接受“真的让 LDS 更接近 target”的 move
        if abs(_cur_lds(trial) - target_lds) + 1e-15 < abs(cur_lds - target_lds):
            out = trial

    return out

def _near0_final_tiny_repair(
    n_ck: np.ndarray,
    rng: np.random.Generator,
    *,
    target_lcd: float,
    target_lds: float,
    target_ssdi: float,
    steps: int,
) -> np.ndarray:
    """
    Phase C:
    最后 continuous tiny repair。

    新版重点：
    - 不再只是“每轮 1 步 LDS-only tiny move”
    - 每轮允许一个 small support correction + 一个连续的 LDS-only refine 候选
    - 谁更优收谁；没有提升再停
    """
    best = n_ck.copy().astype(int)

    def _score(arr: np.ndarray) -> float:
        m = compute_ssdi_metrics(arr)
        dlcd = float(m["LCD"]) - float(target_lcd)
        dlds = float(m["LDS"]) - float(target_lds)
        dssdi = float(m["SSDI"]) - float(target_ssdi)
        return float(dlcd * dlcd + dlds * dlds + 0.20 * dssdi * dssdi)

    best_score = _score(best)

    for _ in range(int(max(1, steps))):
        cur_metrics = compute_ssdi_metrics(best)
        cur_lcd = float(cur_metrics["LCD"])

        cands = [best]

        # very small support correction（仍保留）
        if cur_lcd < target_lcd - 1e-12:
            cands.append(_near0_open_support_once(best, rng))
        elif cur_lcd > target_lcd + 1e-12:
            cands.append(_near0_fill_support_once(best, rng, amount=1))

        # 关键改动：不再只做 1 步 LDS-only，
        # 改成小范围连续精修
        cands.append(
            _near0_support_preserving_lds_move(
                best,
                rng,
                target_lds=target_lds,
                steps=3,
            )
        )
        cands.append(
            _near0_support_preserving_lds_move(
                best,
                rng,
                target_lds=target_lds,
                steps=5,
            )
        )

        local_best = best
        local_score = best_score

        for cand in cands:
            sc = _score(cand)
            if sc + 1e-15 < local_score:
                local_best = cand
                local_score = sc

        if local_score + 1e-15 < best_score:
            best = local_best
            best_score = local_score
        else:
            break

    return best


def _propose_move_near_zero(
    record: Dict[str, Any],
    target_spec: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    near0 严格门控版三阶段 proposal:

    - mode=-1:
        保留原 full-support + LDS-only 逻辑
    - 其它 near0:
        Phase A: 先做 LCD-support seed
        Phase B: 只有 LCD 已进入容差带，才允许 fixed-support LDS
        Phase C: 最后 tiny repair

    核心新规则：
    LCD 未精准到带内 -> 不允许调 LDS
    """
    n_ck = record["n_ck"].copy().astype(int)
    structure_bias = float(record.get("structure_bias", 0.0))
    mode_value = _resolve_mode_value_local(structure_bias)

    target_ssdi = float(target_spec.get("target_ssdi", record.get("target_ssdi", 0.0)))
    target_lcd = float(target_spec["target_lcd"])
    target_lds = float(target_spec["target_lds"])

    lcd_gap = float(target_lcd - record["LCD"])
    lds_gap = float(target_lds - record["LDS"])

    lcd_steps, lds_steps, repair_steps, lcd_tol = _near0_stage_steps(record, target_spec)

    # ---------------------------------------------------------
    # mode=-1: near0 左极端保持原逻辑
    # ---------------------------------------------------------
    if mode_value == -1:
        if np.any(n_ck <= 0):
            out = _zipf_pareto_refill_zeros(
                n_ck,
                rng=rng,
                alpha=1.2,
                beta=1.2,
                max_fill_each=max(2, lcd_steps + 1),
            )
            out = _mode_minus_one_repair(out)
            return out

        if lds_gap > 0 or lds_gap < 0:
            return _near0_support_preserving_lds_move(
                n_ck,
                rng,
                target_lds=target_lds,
                steps=max(1, lds_steps),
            )
        return n_ck

    # ---------------------------------------------------------
    # Phase A: 先把 LCD 做准
    # ---------------------------------------------------------
    out = _build_near0_lcd_seed(
        n_ck,
        rng,
        target_lcd=target_lcd,
        target_ssdi=target_ssdi,
        steps=lcd_steps,
        lcd_tol=lcd_tol,
        target_lds=target_lds,
    )


    out_metrics = compute_ssdi_metrics(out)
    out_lcd = float(out_metrics["LCD"])
    lcd_ready = (abs(out_lcd - target_lcd) <= float(lcd_tol))

    # ---------------------------------------------------------
    # LCD 未进带：禁止进入 LDS 阶段
    # 直接返回当前 LCD-support seed，让下一轮继续 Phase A
    # ---------------------------------------------------------
    if not lcd_ready:
        return out

    # ---------------------------------------------------------
    # Phase B: 只有 LCD 准了，才允许固定 support 调 LDS
    # ---------------------------------------------------------
    if lds_steps > 0:
        out = _near0_support_preserving_lds_move(
            out,
            rng,
            target_lds=target_lds,
            steps=lds_steps,
        )

    # ---------------------------------------------------------
    # Phase C: 最后 tiny repair
    # 这里允许 very small 的补0 / 填充 / 微调
    # ---------------------------------------------------------
    if repair_steps > 0:
        out = _near0_final_tiny_repair(
            out,
            rng,
            target_lcd=target_lcd,
            target_lds=target_lds,
            target_ssdi=target_ssdi,
            steps=repair_steps,
        )

    return out



def _propose_move(
    record: Dict[str, Any],
    target_spec: Dict[str, Any],
    phase: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Unified proposal entrance.

    - near_zero : dedicated near0 proposal
    - middle    : dedicated middle proposal
    - near_one  : dedicated near1 proposal
    """
    mode_value = _resolve_mode_value_local(float(record.get("structure_bias", 0.0)))

    # mode=-1: keep repairing zeros during search
    if mode_value == -1:
        n0 = record["n_ck"].copy().astype(int)
        if np.any(n0 <= 0):
            n0 = _zipf_pareto_refill_zeros(
                n0,
                rng=rng,
                alpha=1.2,
                beta=1.2,
                max_fill_each=2,
            )
            n0 = _mode_minus_one_repair(n0)
            tmp_record = dict(record)
            tmp_record["n_ck"] = n0
            record = tmp_record

    if phase == "near_zero":
        return _propose_move_near_zero(record, target_spec, rng)

    if phase == "near_one":
        return _propose_move_near_one(record, target_spec, rng)

    if phase == "middle":
        return _propose_move_middle(record, target_spec, rng)

    return record["n_ck"].copy().astype(int)




def _middle_concentration_caps(
    n_ck: np.ndarray,
    *,
    target_ssdi: float,
    structure_bias: float,
) -> Dict[str, float]:
    """
    Middle-stage realism caps.

    目的：
    - 防止 middle 阶段把质量集中到极少数 row / col / cell
    - mild / moderate SSDI 时更严格
    - bias 偏 coverage 时可略微放宽，但仍保留上限

    返回：
    - row_cap  : 单个 row 最大占全局比例
    - col_cap  : 单个 col 最大占全局比例
    - cell_cap : 单个 cell 最大占全局比例
    - miss_cap : middle 阶段允许的最大 missing_rate（超过就重罚）
    """
    C, K = n_ck.shape
    s = float(np.clip(target_ssdi, 0.0, 1.0))
    b = float(np.clip(structure_bias, -1.0, 1.0))
    right_relax = max(0.0, b)

    # mild / moderate 阶段要更保守，避免一上来就“像高缺失极端”
    row_cap = float(np.clip(
        1.50 / max(C, 1) + 0.05 + 0.04 * s + 0.03 * right_relax,
        0.14, 0.28
    ))
    col_cap = float(np.clip(
        2.50 / max(K, 1) + 0.035 + 0.05 * s + 0.03 * right_relax,
        0.08, 0.18
    ))
    cell_cap = float(np.clip(
        3.50 / max(C * K, 1) + 0.015 + 0.03 * s + 0.02 * right_relax,
        0.02, 0.08
    ))
    miss_cap = float(np.clip(
        0.06 + 0.45 * s + 0.08 * right_relax,
        0.10, 0.35
    ))

    return {
        "row_cap": row_cap,
        "col_cap": col_cap,
        "cell_cap": cell_cap,
        "miss_cap": miss_cap,
    }


def _middle_concentration_penalty(
    n_ck: np.ndarray,
    *,
    target_ssdi: float,
    structure_bias: float,
) -> float:
    """
    对 middle 候选增加 realism penalty。

    罚这些情况：
    - 单个 row 过大
    - 单个 col 过大
    - 单个 cell 过大
    - missing_rate 在 mild / moderate SSDI 下过高
    - mild / moderate 阶段出现空 row / 空 col
    """
    arr = np.asarray(n_ck, dtype=float)
    N = float(max(1.0, arr.sum()))
    row_share = arr.sum(axis=1) / N
    col_share = arr.sum(axis=0) / N
    cell_share = arr / N

    caps = _middle_concentration_caps(
        arr.astype(int),
        target_ssdi=target_ssdi,
        structure_bias=structure_bias,
    )

    row_over = max(0.0, float(row_share.max()) - caps["row_cap"])
    col_over = max(0.0, float(col_share.max()) - caps["col_cap"])
    cell_over = max(0.0, float(cell_share.max()) - caps["cell_cap"])

    missing_rate = float(np.mean(arr <= 0))
    miss_over = max(0.0, missing_rate - caps["miss_cap"])

    empty_rows = int(np.sum(arr.sum(axis=1) <= 0))
    empty_cols = int(np.sum(arr.sum(axis=0) <= 0))

    s = float(np.clip(target_ssdi, 0.0, 1.0))
    mild_mid = (s <= 0.45)

    penalty = 0.0
    penalty += 10.0 * (row_over ** 2)
    penalty += 12.0 * (col_over ** 2)
    penalty += 22.0 * (cell_over ** 2)
    penalty += 8.0 * (miss_over ** 2)

    if mild_mid:
        penalty += 3.0 * float(empty_rows + empty_cols)

    return float(penalty)


def _middle_existing_receiver_pool(
    n_ck: np.ndarray,
    *,
    src: Tuple[int, int],
    target_ssdi: float,
    structure_bias: float,
    axis_mix: float = 0.35,
) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    """
    在 middle 阶段，为“已有非0 support”建立 receiver pool。

    规则：
    - 只允许已有非0 support 承接（middle 更像微调，不靠疯狂开新 support）
    - 超过 row/col/cell cap 的位置不允许继续承接
    - 既允许偏向较大 row/col，也必须受容量上限约束
    - 仍然偏好较弱 existing support，避免一直灌主峰
    """
    out = np.asarray(n_ck, dtype=int)
    C, K = out.shape
    N = float(max(1, int(out.sum())))
    si, sj = map(int, src)

    row_tot = out.sum(axis=1).astype(float)
    col_tot = out.sum(axis=0).astype(float)
    row_share = row_tot / N
    col_share = col_tot / N
    cell_share = out.astype(float) / N

    row_max = max(float(row_share.max()), EPS)
    col_max = max(float(col_share.max()), EPS)

    caps = _middle_concentration_caps(
        out,
        target_ssdi=target_ssdi,
        structure_bias=structure_bias,
    )

    cands: List[Tuple[int, int, int]] = []
    weights: List[float] = []

    for i in range(C):
        for j in range(K):
            if i == si and j == sj:
                continue

            v = int(out[i, j])
            if v <= 0:
                continue

            # 已经接近上限的位置，不再继续填
            if row_share[i] >= 0.985 * caps["row_cap"]:
                continue
            if col_share[j] >= 0.985 * caps["col_cap"]:
                continue
            if cell_share[i, j] >= 0.985 * caps["cell_cap"]:
                continue

            row_room = max(0.02, (caps["row_cap"] - row_share[i]) / max(caps["row_cap"], EPS))
            col_room = max(0.02, (caps["col_cap"] - col_share[j]) / max(caps["col_cap"], EPS))
            cell_room = max(0.02, (caps["cell_cap"] - cell_share[i, j]) / max(caps["cell_cap"], EPS))

            # 大 row/col 可以略微优先，但不能压倒“弱 existing support”
            large_axis = 0.5 * (
                row_share[i] / row_max +
                col_share[j] / col_max
            )
            axis_pref = (1.0 - axis_mix) * 1.0 + axis_mix * (0.45 + 0.55 * large_axis)

            weak_pref = 1.0 / np.sqrt(1.0 + float(v))

            diff_bonus = 1.0
            if i != si:
                diff_bonus += 0.35
            if j != sj:
                diff_bonus += 0.35

            w = row_room * col_room * cell_room * axis_pref * weak_pref * diff_bonus
            if w <= 0:
                continue

            cands.append((i, j, v))
            weights.append(float(w))

    if not cands:
        return [], np.zeros(0, dtype=float)

    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = weights_arr / max(float(weights_arr.sum()), EPS)
    return cands, weights_arr


def _middle_sample_diverse_receivers(
    candidates: List[Tuple[int, int, int]],
    weights: np.ndarray,
    rng: np.random.Generator,
    *,
    n_pick: int,
) -> List[Tuple[int, int, int]]:
    """
    从 receiver pool 里抽多个接收点，但鼓励分散到不同 row / col。

    做法：
    - 无放回抽样
    - 若某 row / col 已经被选中过，再次选到它的概率会被压低
    """
    if len(candidates) == 0 or n_pick <= 0:
        return []

    n_pick = int(min(n_pick, len(candidates)))
    chosen: List[Tuple[int, int, int]] = []
    used_rows: set = set()
    used_cols: set = set()
    remaining = list(range(len(candidates)))

    for _ in range(n_pick):
        local_scores = []
        for idx in remaining:
            i, j, _ = candidates[idx]
            w = float(weights[idx])

            if i in used_rows:
                w *= 0.40
            if j in used_cols:
                w *= 0.40

            local_scores.append(max(w, 1e-12))

        probs = np.asarray(local_scores, dtype=float)
        probs = probs / probs.sum()

        local_pick = int(rng.choice(np.arange(len(remaining)), p=probs))
        real_idx = remaining[local_pick]

        i, j, v = candidates[real_idx]
        chosen.append((int(i), int(j), int(v)))
        used_rows.add(int(i))
        used_cols.add(int(j))
        remaining.pop(local_pick)

        if not remaining:
            break

    return chosen




def _propose_move_middle(
    record: Dict[str, Any],
    target_spec: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Middle：修复版

    这版目标：
    1) below_lcdmax 的 middle 不再靠激进 bulk create 把 mild SSDI 推成高缺失极端
    2) operator 改成“分散 + cap + mild realism”
    3) 保留 9 方法 seed 的主导地位：middle 主要是微调，不是推翻 seed

    改动要点：
    - create donor：优先挖小 client / 稀有 label / 小 cell
    - receiver：existing support only + row/col/cell cap + 分散抽样
    - 局部评分新增 concentration penalty / support-edit penalty / drift penalty
    - low / moderate SSDI 时显著收缩 bulk 步长
    """
    n0 = record["n_ck"].copy().astype(int)
    C, K = n0.shape
    Ntot = int(np.sum(n0))

    cur_lcd = float(record["LCD"])
    cur_lds = float(record["LDS"])
    cur_ssdi = float(record["SSDI"])
    cur_theta = float(record.get("actual_theta", _compute_theta_from_lcd_lds(cur_lcd, cur_lds)))

    target_lcd = float(target_spec["target_lcd"])
    target_lds = float(target_spec["target_lds"])
    target_ssdi = float(
        target_spec.get("target_ssdi", np.sqrt(target_lcd * target_lcd + target_lds * target_lds))
    )
    target_theta = float(
        target_spec.get("target_theta", _compute_theta_from_lcd_lds(target_lcd, target_lds))
    )

    structure_bias = float(record.get("structure_bias", 0.0))
    mode_value = _resolve_mode_value_local(structure_bias)

    d_lcd = target_lcd - cur_lcd
    d_lds = target_lds - cur_lds

    # =========================================================
    # 1) 尺度量
    # =========================================================
    m = max(1, min(C, K))
    mu = abs(C - K) / float(m)
    mean_cell = Ntot / float(max(C * K, 1))
    low_radius = (target_ssdi <= 0.35)
    moderate_radius = (0.35 < target_ssdi <= 0.55)

    mu1 = int(mu > 0.5)
    mu2 = int(mu > 1.0)
    den1 = int(mean_cell < 120)
    den2 = int(mean_cell < 60)

    # =========================================================
    # 2) middle 步长：below_lcdmax / low-mid 半径显著收敛
    # =========================================================
    gap = abs(d_lcd) + abs(d_lds)
    base_main = 1 + int(gap > 0.10) + int(gap > 0.22)
    scale_bonus = mu1 + den1

    steps_main = int(np.clip(base_main + scale_bonus, 1, 5))
    steps_small = 1
    steps_extra = int(np.clip(steps_main + 1 + mu2 + den2, 2, 6))

    if low_radius:
        steps_main = min(steps_main, 2)
        steps_extra = min(steps_extra, 3)
    elif moderate_radius:
        steps_main = min(steps_main, 3)
        steps_extra = min(steps_extra, 4)

    bulk_create_steps = int(np.clip(
        1 + low_radius + mu1 + den1 + int(target_lcd > 0.08),
        1, 3 if low_radius else 4
    ))
    bulk_fill_steps = int(np.clip(
        1 + mu1 + den1 + int(target_lcd < 0.10),
        1, 3 if low_radius else 4
    ))
    bulk_peak_steps = int(np.clip(
        1 + int(target_ssdi > 0.45) + den1,
        1, 4
    ))

    # =========================================================
    # 3) 门控
    # =========================================================
    lcd_zero_like_thr = max(0.002, min(0.02, 0.35 * max(target_lcd, 1e-6)))
    need_y_axis_escape = (target_lcd > 1e-12 and cur_lcd <= lcd_zero_like_thr)

    small_lcd_target = (
        (structure_bias < -0.05 or target_theta <= np.deg2rad(20.0))
        and target_lcd <= 0.12
    )

    n0_row_share = n0.sum(axis=1).astype(float) / max(float(n0.sum()), EPS)
    n0_col_share = n0.sum(axis=0).astype(float) / max(float(n0.sum()), EPS)
    n0_support = (n0 > 0).astype(int)

    # =========================================================
    # 4) 小工具：repeat wrappers
    # =========================================================
    def _repeat_create(arr: np.ndarray, times: int) -> np.ndarray:
        out = arr.copy().astype(int)
        for _ in range(int(max(1, times))):
            out = _apply_create_high_value_zeros(
                out, rng, steps=1,
                target_ssdi=target_ssdi,
                structure_bias=structure_bias,
            )
        return out

    def _repeat_fill(arr: np.ndarray, times: int) -> np.ndarray:
        out = arr.copy().astype(int)
        for _ in range(int(max(1, times))):
            out = _apply_fill_high_value_zeros(out, rng, steps=1)
        return out

    def _repeat_peak(arr: np.ndarray, times: int) -> np.ndarray:
        out = arr.copy().astype(int)
        for _ in range(int(max(1, times))):
            out = _apply_peak_reweight(
                out, rng, steps=1,
                target_ssdi=target_ssdi,
                structure_bias=structure_bias,
            )
        return out

    def _repeat_flatten(arr: np.ndarray, times: int) -> np.ndarray:
        out = arr.copy().astype(int)
        for _ in range(int(max(1, times))):
            out = _apply_flatten_reweight(
                out, rng, steps=1,
                target_ssdi=target_ssdi,
                structure_bias=structure_bias,
            )
        return out

    # =========================================================
    # 5) 局部评分：几何 + realism + 保 seed 语义
    # =========================================================
    def _score(arr: np.ndarray) -> float:
        mtr = compute_ssdi_metrics(arr)
        lcd = float(mtr["LCD"])
        lds = float(mtr["LDS"])
        ssdi = float(mtr["SSDI"])
        theta = float(_compute_theta_from_lcd_lds(lcd, lds))

        e_lcd = lcd - target_lcd
        e_lds = lds - target_lds
        e_ssdi = ssdi - target_ssdi
        e_theta = theta - target_theta

        score = (
            1.00 * e_lcd * e_lcd
            + 1.00 * e_lds * e_lds
            + 0.18 * e_ssdi * e_ssdi
            + 0.08 * e_theta * e_theta
        )

        # mild / moderate 右移时，仍贴 y 轴要罚
        if need_y_axis_escape and lcd <= lcd_zero_like_thr:
            escape_scale = 0.20 + 0.08 * mu1 + 0.08 * den1
            if low_radius:
                escape_scale += 0.10
            score += escape_scale + 1.5 * (lcd_zero_like_thr - lcd + 1e-6)

        # 左侧小 LCD 目标，不要被硬拉太右
        if small_lcd_target:
            overshoot_band = max(0.015, 0.25 * max(target_lcd, 1e-6))
            if lcd > target_lcd + overshoot_band:
                overshoot = lcd - (target_lcd + overshoot_band)
                score += 6.0 * overshoot * overshoot + 0.25 * overshoot

        # realism penalty：行/列/cell/缺失率过高都要罚
        score += _middle_concentration_penalty(
            arr,
            target_ssdi=target_ssdi,
            structure_bias=structure_bias,
        )

        # support 编辑量过大：below_lcdmax / mild-middle 时要罚
        support_edit_frac = float(np.mean(((arr > 0).astype(int) != n0_support)))
        if target_ssdi <= 0.45:
            score += 3.0 * (support_edit_frac ** 2)
        else:
            score += 1.0 * (support_edit_frac ** 2)

        # 每一步不要把 row/col marginals 改得太猛，尽量保 seed 语义
        arr_row_share = arr.sum(axis=1).astype(float) / max(float(arr.sum()), EPS)
        arr_col_share = arr.sum(axis=0).astype(float) / max(float(arr.sum()), EPS)
        drift = (
            np.abs(arr_row_share - n0_row_share).sum()
            + np.abs(arr_col_share - n0_col_share).sum()
        )
        drift_w = 0.25 if target_ssdi <= 0.45 else 0.12
        score += drift_w * float(drift)

        # mild-middle 出现空 row / 空 col，要额外罚
        if target_ssdi <= 0.45:
            empty_rows = int(np.sum(arr.sum(axis=1) <= 0))
            empty_cols = int(np.sum(arr.sum(axis=0) <= 0))
            score += 4.0 * float(empty_rows + empty_cols)

        # mode=-1 保持原有语义
        if mode_value == -1:
            if np.any(arr <= 0):
                score += 20.0
            score += 6.0 * (lcd * lcd)

        return float(score)

    # =========================================================
    # 6) mode=-1：保留 full-support 左端逻辑
    # =========================================================
    if mode_value == -1:
        out = n0.copy()

        if np.any(out <= 0):
            out = _zipf_pareto_refill_zeros(
                out,
                rng=rng,
                alpha=1.2,
                beta=1.2,
                max_fill_each=max(2, 1 + den1 + den2),
            )
            out = _mode_minus_one_repair(out)

        cands = [out]
        cands.append(
            _near0_support_preserving_lds_move(
                out,
                rng,
                target_lds=target_lds,
                steps=max(1, steps_main),
            )
        )

        if d_lds > 0 or target_ssdi > 0.45:
            cands.append(_repeat_peak(out, bulk_peak_steps))
        else:
            cands.append(
                _near0_support_preserving_lds_move(
                    out,
                    rng,
                    target_lds=target_lds,
                    steps=max(1, steps_extra),
                )
            )

        return min(cands, key=_score)

    # =========================================================
    # 7) 普通 middle：候选集，但明显收缩 operator 主导权
    # =========================================================
    cands: List[np.ndarray] = [n0]

    # 基础单操作
    cands.append(_repeat_create(n0, steps_small))
    cands.append(_repeat_fill(n0, steps_small))
    cands.append(_repeat_peak(n0, steps_small))
    cands.append(_repeat_flatten(n0, steps_small))

    # 四象限组合，但 mild-middle 下不再给太多大步
    cands.append(_repeat_peak(_repeat_create(n0, steps_main), steps_small))
    cands.append(_repeat_flatten(_repeat_create(n0, steps_main), steps_small))
    cands.append(_repeat_peak(_repeat_fill(n0, steps_main), steps_small))
    cands.append(_repeat_flatten(_repeat_fill(n0, steps_main), steps_small))

    # 方向增强
    if d_lcd > 0:
        c1 = _repeat_create(n0, steps_extra)
        cands.append(c1)

        if d_lds >= 0:
            cands.append(_repeat_peak(c1, steps_small))
        else:
            cands.append(_repeat_flatten(c1, steps_small))

    if d_lcd < 0:
        f1 = _repeat_fill(n0, steps_extra)
        cands.append(f1)

        if d_lds >= 0:
            cands.append(_repeat_peak(f1, steps_small))
        else:
            cands.append(_repeat_flatten(f1, steps_small))

    if d_lds > 0:
        cands.append(_repeat_peak(n0, steps_extra))

    if d_lds < 0:
        cands.append(_repeat_flatten(n0, steps_extra))

    # y-axis escape：仍然保留，但强度明显缩小
    if need_y_axis_escape:
        c2 = _repeat_create(n0, bulk_create_steps)
        cands.append(c2)

        if not low_radius:
            cands.append(_repeat_peak(c2, steps_small))
            cands.append(_repeat_flatten(c2, steps_small))
        else:
            # mild-middle 里只给 very mild 起跳，不允许大幅 create+peak
            cands.append(_repeat_flatten(c2, 1))

    # 左侧小 LCD 目标：以 fill 为主，不做大 create
    if small_lcd_target:
        f1 = _repeat_fill(n0, bulk_fill_steps)
        cands.append(f1)
        cands.append(_repeat_peak(f1, steps_small))
        cands.append(_repeat_flatten(f1, steps_small))

    # 反向微修
    if cur_lcd > target_lcd and d_lds > 0:
        cands.append(_repeat_peak(_repeat_fill(n0, steps_small), 1))

    if cur_lcd < target_lcd and d_lds < 0:
        cands.append(_repeat_flatten(_repeat_create(n0, steps_small), 1))

    best = min(cands, key=_score)
    return best


def _build_seed_for_phase(
    *,
    phase: str,
    client: int,
    label: int,
    datasize: int,
    target_ssdi: float,
    lcdtype: str,
    ldstype: str,
    alpha: Optional[float],
    beta: Optional[float],
    lcd_params: Optional[Dict[str, Any]],
    lds_params: Optional[Dict[str, Any]],
    ssdi_error: float,
    seed: Optional[int],
    max_iters: int,
    structure_bias: float,
) -> Dict[str, Any]:
    """
    Seed policy:
    - exact_zero : direct iid
    - exact_one  : direct exact1
    - near_one   : unified exact1 skeleton seed
                   (凡是 > ssdi0，被路由到 near_one，就统一按 near1 标准 seed)
    - near_zero / middle :
        first use core generator as a neutral seed near target radius
    """

    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    a = float(alpha if alpha is not None else 1.0)
    b = float(beta if beta is not None else 1.0)

    mode_value = _resolve_mode_value_local(float(structure_bias))
    force_lcd_zero = (mode_value == -1)

    if phase == "exact_zero":
        rec = _construct_exact_zero_record(
            client=client,
            label=label,
            datasize=datasize,
            ssdi=target_ssdi,
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
        )
        rec["source_stage"] = "direct exact iid"
        rec["generator_variant"] = "direct_exact_zero"
        return rec

    if phase == "exact_one":
        rec = _construct_exact_one_record(
            client=client,
            label=label,
            datasize=datasize,
            ssdi=target_ssdi,
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
        )
        rec["source_stage"] = "direct exact extremal"
        rec["generator_variant"] = "direct_exact_one"
        return rec

    if phase == "near_one":
        # [MOD]
        # 统一规则：
        # 凡是已经被判进 near_one（也就是 target_ssdi > ssdi0_local），
        # 一律从 exact1 skeleton 出发。
        #
        # 不再在 near_one 内部继续拆成：
        #   1) very-high 用 exact1 seed
        #   2) moderate-high 用 neutral/core seed
        #
        # 因为这会导致 > ssdi0 的高异质区内部标准不统一。
        rec = _construct_exact_one_record(
            client=client,
            label=label,
            datasize=datasize,
            ssdi=1.0,
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
        )

        # mode = -1 时，near1 seed 也要先去零填充，避免带 LCD 的 exact1 skeleton
        # 直接进入后续搜索时与“必须全非零”的要求冲突。
        if force_lcd_zero:
            rec["n_ck"] = _zipf_pareto_refill_zeros(
                rec["n_ck"],
                rng=rng,
                alpha=a,
                beta=b,
                max_fill_each=1,
            )
            rec = _safe_record_from_matrix(
                rec["n_ck"],
                target_ssdi=1.0,
                lcdtype=lcdtype,
                ldstype=ldstype,
                alpha=alpha,
                beta=beta,
                lcd_params=lcd_params,
                lds_params=lds_params,
                source_stage="near_one exact1 refilled seed",
                generator_variant="seed_exact_one_refilled",
            )

        rec["near_one_seed_lcd"] = float(rec["LCD"])
        rec["near_one_seed_lds"] = float(rec["LDS"])
        rec["near_one_seed_theta"] = float(
            _compute_theta_from_lcd_lds(rec["LCD"], rec["LDS"])
        )
        rec["source_stage"] = rec.get("source_stage", "near_one exact1 seed")
        rec["generator_variant"] = rec.get("generator_variant", "seed_exact_one")
        rec["target_ssdi"] = float(target_ssdi)
        return rec

    # near_zero / middle
    pack = _merge_param_pack(
        target_ssdi=target_ssdi,
        lcdtype=lcdtype,
        ldstype=ldstype,
        C=int(label),
        K=int(client),
        alpha=alpha,
        beta=beta,
        lcd_params=lcd_params,
        lds_params=lds_params,
        seed=seed,
    )
    seed_rec = _run_core_with_pack(
        client=client,
        label=label,
        datasize=datasize,
        ssdi=target_ssdi,
        lcdtype=lcdtype,
        ldstype=ldstype,
        pack=pack,
        ssdi_error=ssdi_error,
        seed=seed,
        max_iters=max_iters,
        source_stage="structured neutral seed",
        source_ssdi=target_ssdi,
        preferred_variant="v2" if target_ssdi <= 0.7 else "v1",
    )

    if force_lcd_zero:
        seed_rec["n_ck"] = _zipf_pareto_refill_zeros(
            seed_rec["n_ck"],
            rng=rng,
            alpha=a,
            beta=b,
            max_fill_each=2,
        )
        seed_rec = _safe_record_from_matrix(
            seed_rec["n_ck"],
            target_ssdi=float(target_ssdi),
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
            source_stage="structured neutral seed refilled",
            generator_variant="seed_refilled",
        )

    return seed_rec

def _stage_budget_plan(phase: str, max_iters: int) -> int:
    if phase == "near_one":
        return int(max(80, 4 * max_iters))
    if phase == "middle":
        return int(max(60, 3 * max_iters))
    if phase == "near_zero":
        return int(max(40, 2 * max_iters))
    return int(max(20, max_iters))

def _near1_left_sparse_failure_trigger(
    best: Dict[str, Any],
    target_spec: Dict[str, Any],
    *,
    phase: str,
    stage_ssdi_error: float,
) -> bool:
    """
    只在 near1 左侧 branch 且出现“大片缺失 + SSDI 仍未命中”时触发 rescue。
    """
    if phase != "near_one":
        return False

    seed_theta = float(target_spec.get("near_one_seed_theta", np.inf))
    target_theta = float(target_spec.get("target_theta", 0.0))
    is_left = bool(target_theta < seed_theta - np.deg2rad(1.0))
    if not is_left:
        return False

    ssdi_gap = abs(float(best["SSDI"]) - float(target_spec["target_ssdi"]))
    if ssdi_gap <= float(stage_ssdi_error):
        return False

    cur_missing = float(best.get("missing_rate", 0.0))
    target_missing = float(
        _target_missing_rate_from_target_geometry(
            float(target_spec["target_lcd"]),
            float(target_spec["target_lds"]),
        )
    )

    # “大片缺失”触发条件：绝对大，或显著高于目标几何对应缺失
    if cur_missing < max(0.55, target_missing + 0.25):
        return False

    return True




def _near1_left_controlled_lcd_up_move(
    n_ck: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    LCD-up move for near1-left sparse rescue:

    Goal:
    - increase LCD by turning one small non-diagonal nonzero cell into zero
    - redistribute its mass into OTHER existing small nonzero cells
    - do NOT feed peaks
    - do NOT create new nonzeros
    """
    out = n_ck.copy().astype(int)
    C, K = out.shape

    rows_desc, cols_desc = _largest_rows_cols(out)
    top_rows = set(rows_desc[: max(1, int(np.ceil(0.2 * C)))])
    top_cols = set(cols_desc[: max(1, int(np.ceil(0.2 * K)))])

    # donor: non-diagonal small existing support, preferably in top-20% row/col region
    donors = []
    donor_scores = []
    for i in range(C):
        for j in range(K):
            v = int(out[i, j])
            if v <= 0:
                continue
            if i == j:
                continue
            if (i in top_rows) or (j in top_cols):
                donors.append((i, j, v))
                donor_scores.append(1.0 / max(1.0, float(v)))  # smaller value easier to zero out

    if not donors:
        return out

    donor_scores = np.asarray(donor_scores, dtype=float)
    donor_scores = donor_scores / donor_scores.sum()
    d_idx = int(rng.choice(np.arange(len(donors)), p=donor_scores))
    di, dj, dv = donors[d_idx]

    amt = int(out[di, dj])
    if amt <= 0:
        return out

    # receivers: OTHER existing small nonzero supports, also in top-20% row/col region
    recvs = []
    recv_scores = []
    for i in range(C):
        for j in range(K):
            v = int(out[i, j])
            if v <= 0:
                continue
            if (i, j) == (di, dj):
                continue
            if i == j:
                continue
            if (i in top_rows) or (j in top_cols):
                recvs.append((i, j, v))
                recv_scores.append(1.0 / max(1.0, float(v)))  # still prefer small supports

    if not recvs:
        return out

    recv_scores = np.asarray(recv_scores, dtype=float)
    recv_scores = recv_scores / recv_scores.sum()

    split_n = min(len(recvs), int(rng.integers(2, 5)))  # distribute to 2~4 cells
    chosen_idx = rng.choice(
        np.arange(len(recvs)),
        size=split_n,
        replace=False,
        p=recv_scores,
    )
    chosen = [recvs[int(t)] for t in chosen_idx]

    out[di, dj] = 0
    base_each = amt // split_n
    rem = amt % split_n

    for s, (ri, rj, rv) in enumerate(chosen):
        give = base_each + (1 if s < rem else 0)
        out[ri, rj] += int(give)

    return out




def _rescue_near1_left_sparse_failure_from_anchor(
    *,
    target_spec: Dict[str, Any],
    structure_bias: float,
    structure_mode: str,
    lcdtype: str,
    ldstype: str,
    alpha: Optional[float],
    beta: Optional[float],
    lcd_params: Optional[Dict[str, Any]],
    lds_params: Optional[Dict[str, Any]],
    stage_ssdi_error: float,
    budget: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    """
    near1 左侧 sparse-failure rescue：

    - 从 y 轴锚点重新开始
    - 同样给一轮 budget
    - 主操作：
        1) LCD 低于 target 明显时：优先 LCD↑
        2) LCD 接近后且 LDS 偏高时：优先 LDS↓
    - 反向极小步操作（只在 overshoot 时才启用）：
        3) LCD 过高：极小步 LCD↓
        4) LDS 过低：极小步 LDS↑
    """
    rng = np.random.default_rng(None if seed is None else int(seed) + 99173)

    # 从 y 轴起点重新开始
    anchor_n = np.array(target_spec["near_one_left_anchor_n_ck"], dtype=int, copy=True)

    current = _safe_record_from_matrix(
        anchor_n,
        target_ssdi=float(target_spec["target_ssdi"]),
        lcdtype=lcdtype,
        ldstype=ldstype,
        alpha=alpha,
        beta=beta,
        lcd_params=lcd_params,
        lds_params=lds_params,
        source_stage="near1 left sparse rescue from y-axis anchor",
        generator_variant="near1_left_sparse_rescue_from_anchor",
    )
    current = _decorate_record_constraints(
        current,
        structure_bias=structure_bias,
        structure_mode=structure_mode,
        target_spec=target_spec,
        phase="near_one",
        stage_ssdi_error=stage_ssdi_error,
    )
    best = dict(current)

    def _rescue_score(rec: Dict[str, Any]) -> float:
        dlcd = float(rec["LCD"]) - float(target_spec["target_lcd"])
        dlds = float(rec["LDS"]) - float(target_spec["target_lds"])
        dssdi = float(rec["SSDI"]) - float(target_spec["target_ssdi"])
        return float(dlcd * dlcd + dlds * dlds + 0.25 * dssdi * dssdi)

    best_score = _rescue_score(best)
    no_improve = 0
    C, K = anchor_n.shape

    eps_lcd_main = 0.01
    eps_lds_main = 0.01
    eps_lcd_back = 0.004
    eps_lds_back = 0.004

    for it in range(int(max(1, budget))):
        base_n = best["n_ck"].copy().astype(int)
        cur_lcd = float(best.get("LCD", 0.0))
        cur_lds = float(best.get("LDS", 0.0))
        target_lcd = float(target_spec["target_lcd"])
        target_lds = float(target_spec["target_lds"])

        # -------------------------------------------------
        # A. LCD 明显偏低：优先提升 LCD
        # -------------------------------------------------
        if cur_lcd < target_lcd - eps_lcd_main:
            cand_n = _near1_left_controlled_lcd_up_move(base_n, rng)

        # -------------------------------------------------
        # B. LCD 已接近 target，但 LDS 偏高：削主峰降 LDS
        # -------------------------------------------------
        elif cur_lds > target_lds + eps_lds_main:
            m = min(C, K)
            diag_vals = np.array([int(base_n[i, i]) for i in range(m)], dtype=float)
            if np.all(diag_vals <= 1):
                break

            donor_prob = diag_vals / diag_vals.sum()
            donor_idx = int(rng.choice(np.arange(m), p=donor_prob))
            donor_val = int(base_n[donor_idx, donor_idx])

            floor_diag = max(1, int(np.sum(base_n)) // C // K // 2)
            movable = donor_val - floor_diag
            if movable <= 0:
                no_improve += 1
                if no_improve >= 24:
                    break
                continue

            rows_desc, cols_desc = _largest_rows_cols(base_n)
            top_rows = set(rows_desc[: max(1, int(np.ceil(0.2 * C)))])
            top_cols = set(cols_desc[: max(1, int(np.ceil(0.2 * K)))])

            recv_candidates = []
            recv_scores = []
            for i in range(C):
                for j in range(K):
                    v = int(base_n[i, j])
                    if v <= 0:
                        continue
                    if i == j:
                        continue
                    if (i in top_rows) or (j in top_cols):
                        recv_candidates.append((i, j))
                        recv_scores.append(1.0 / max(1.0, float(v)))

            if not recv_candidates:
                break

            recv_scores = np.asarray(recv_scores, dtype=float)
            recv_scores = recv_scores / recv_scores.sum()

            amt_total = int(round(donor_val * 0.0015))
            amt_total = max(3, min(12, amt_total))
            amt_total = min(amt_total, movable)
            if amt_total <= 0:
                no_improve += 1
                if no_improve >= 24:
                    break
                continue

            split_n = int(rng.integers(2, 5))
            split_n = min(split_n, amt_total, len(recv_candidates))
            if split_n <= 0:
                continue

            chosen_idx = rng.choice(
                np.arange(len(recv_candidates)),
                size=split_n,
                replace=False,
                p=recv_scores,
            )
            chosen = [recv_candidates[int(t)] for t in chosen_idx]

            cand_n = base_n.copy()
            cand_n[donor_idx, donor_idx] -= amt_total

            base_each = amt_total // split_n
            rem = amt_total % split_n
            for s, (ri, rj) in enumerate(chosen):
                give = base_each + (1 if s < rem else 0)
                cand_n[ri, rj] += int(give)

        # -------------------------------------------------
        # C. LCD 过高：极小步 LCD↓
        # -------------------------------------------------
        elif cur_lcd > target_lcd + eps_lcd_back:
            cand_n = _near1_left_controlled_lcd_down_move(base_n, rng)

        # -------------------------------------------------
        # D. LDS 过低：极小步 LDS↑
        # -------------------------------------------------
        elif cur_lds < target_lds - eps_lds_back:
            cand_n = _near1_left_controlled_lds_up_move(base_n, rng)

        # -------------------------------------------------
        # E. 都已经很近：做一个更温和的 LDS↓ 微削峰
        # -------------------------------------------------
        else:
            m = min(C, K)
            diag_vals = np.array([int(base_n[i, i]) for i in range(m)], dtype=float)
            if np.all(diag_vals <= 1):
                break

            donor_prob = diag_vals / diag_vals.sum()
            donor_idx = int(rng.choice(np.arange(m), p=donor_prob))
            donor_val = int(base_n[donor_idx, donor_idx])

            floor_diag = max(1, int(np.sum(base_n)) // C // K // 2)
            movable = donor_val - floor_diag
            if movable <= 0:
                no_improve += 1
                if no_improve >= 24:
                    break
                continue

            rows_desc, cols_desc = _largest_rows_cols(base_n)
            top_rows = set(rows_desc[: max(1, int(np.ceil(0.2 * C)))])
            top_cols = set(cols_desc[: max(1, int(np.ceil(0.2 * K)))])

            recv_candidates = []
            recv_scores = []
            for i in range(C):
                for j in range(K):
                    v = int(base_n[i, j])
                    if v <= 0:
                        continue
                    if i == j:
                        continue
                    if (i in top_rows) or (j in top_cols):
                        recv_candidates.append((i, j))
                        recv_scores.append(1.0 / max(1.0, float(v)))

            if not recv_candidates:
                break

            recv_scores = np.asarray(recv_scores, dtype=float)
            recv_scores = recv_scores / recv_scores.sum()

            amt_total = max(1, min(4, int(round(donor_val * 0.0006))))
            amt_total = min(amt_total, movable)
            if amt_total <= 0:
                no_improve += 1
                if no_improve >= 24:
                    break
                continue

            split_n = min(len(recv_candidates), max(2, amt_total))
            chosen_idx = rng.choice(
                np.arange(len(recv_candidates)),
                size=split_n,
                replace=False,
                p=recv_scores,
            )
            chosen = [recv_candidates[int(t)] for t in chosen_idx]

            cand_n = base_n.copy()
            cand_n[donor_idx, donor_idx] -= amt_total

            base_each = amt_total // split_n
            rem = amt_total % split_n
            for s, (ri, rj) in enumerate(chosen):
                give = base_each + (1 if s < rem else 0)
                cand_n[ri, rj] += int(give)

        cand = _safe_record_from_matrix(
            cand_n,
            target_ssdi=float(target_spec["target_ssdi"]),
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
            source_stage="near1 left sparse rescue from y-axis anchor",
            generator_variant="near1_left_sparse_rescue_from_anchor",
        )
        cand["iter_used"] = int(it + 1)

        cand = _decorate_record_constraints(
            cand,
            structure_bias=structure_bias,
            structure_mode=structure_mode,
            target_spec=target_spec,
            phase="near_one",
            stage_ssdi_error=stage_ssdi_error,
        )

        cand_score = _rescue_score(cand)

        if cand_score + 1e-15 < best_score:
            best = cand
            best_score = cand_score
            no_improve = 0
        else:
            no_improve += 1

        if bool(best.get("structure_success", False)):
            break

        if no_improve >= 24:
            break

    best["returned_from"] = "near1_left_sparse_failure_rescue_from_anchor"
    return best


def _search_from_seed(
    seed_record: Dict[str, Any],
    target_spec: Dict[str, Any],
    phase: str,
    structure_bias: float,
    structure_mode: Union[str, float, int, None],
    lcdtype: str,
    ldstype: str,
    alpha: Optional[float],
    beta: Optional[float],
    lcd_params: Optional[Dict[str, Any]],
    lds_params: Optional[Dict[str, Any]],
    stage_ssdi_error: float,
    budget: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # near1 右探索控制标志
    # - mode>=0.90 : near1 不允许成功后早停
    # - mode>=0.95 : near1 进入强右探索模式
    # ---------------------------------------------------------
    mode_bias = float(structure_bias)
    near1_no_early_stop = (phase == "near_one" and mode_bias >= 0.90)
    near1_hard_right_explore = (phase == "near_one" and mode_bias >= 0.95)

    # [NEW] near_zero 右极端 mode=1：命中 ssdi 后不立刻停，继续在 band 内鼓励更大 LCD
    near0_no_early_stop = (phase == "near_zero" and mode_bias >= 0.90)

    # ---------------------------------------------------------
    # 本次修改：
    # 删除 middle 的 STEP-1 FIX 依赖。
    # 保留 near0 / near1 的 best-anchor，不再让 middle 大多数时候围绕 best。
    # ---------------------------------------------------------
    middle_best_base_prob = 0.00
    middle_random_accept_p = 0.08
    default_random_accept_p = 0.08

    stage_target_spec = dict(target_spec)
    stage_target_spec["target_ssdi"] = float(
        seed_record.get("target_ssdi", target_spec.get("target_ssdi", 0.0))
    )

    if phase == "near_one":
        stage_target_spec["near_one_seed_lcd"] = float(
            seed_record.get("near_one_seed_lcd", seed_record.get("LCD", 0.0))
        )
        stage_target_spec["near_one_seed_lds"] = float(
            seed_record.get("near_one_seed_lds", seed_record.get("LDS", 0.0))
        )
        stage_target_spec["near_one_seed_theta"] = float(
            seed_record.get(
                "near_one_seed_theta",
                _compute_theta_from_lcd_lds(
                    float(seed_record.get("LCD", 0.0)),
                    float(seed_record.get("LDS", 0.0)),
                ),
            )
        )
        stage_target_spec.setdefault("near_one_lcd_margin", 1e-12)

        # near1-left 的真正统一锚点：同半径 y 轴点
        left_anchor = _build_near1_left_yaxis_anchor_record(
            seed_record,
            rng,
            target_ssdi=float(stage_target_spec["target_ssdi"]),
        )
        stage_target_spec["near_one_left_anchor_n_ck"] = left_anchor["n_ck"].copy()
        stage_target_spec["near_one_left_anchor_lcd"] = float(left_anchor["LCD"])
        stage_target_spec["near_one_left_anchor_lds"] = float(left_anchor["LDS"])
        stage_target_spec["near_one_left_anchor_theta"] = float(
            _compute_theta_from_lcd_lds(left_anchor["LCD"], left_anchor["LDS"])
        )

    current = _decorate_record_constraints(
        seed_record,
        structure_bias=structure_bias,
        structure_mode=structure_mode,
        target_spec=stage_target_spec,
        phase=phase,
        stage_ssdi_error=stage_ssdi_error,
    )
    best = dict(current)

    no_improve = 0
    for it in range(int(max(1, budget))):
        if phase in {"near_zero", "near_one"}:
            base_record = best
        elif phase == "middle":
            # 本次修改：
            # middle 恢复成以 current 为主，不再大多数时候围绕 best 精修。
            base_record = current
        else:
            base_record = current

        cand_n = _propose_move(base_record, stage_target_spec, phase, rng)

        cand = _safe_record_from_matrix(
            cand_n,
            target_ssdi=float(seed_record["target_ssdi"]),
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
            source_stage="structured local search",
            generator_variant=f"structured_{phase}",
        )
        cand["iter_used"] = int(it + 1)

        cand = _decorate_record_constraints(
            cand,
            structure_bias=structure_bias,
            structure_mode=structure_mode,
            target_spec=stage_target_spec,
            phase=phase,
            stage_ssdi_error=stage_ssdi_error,
        )

        cand_score = float(cand["structure_score"])
        cur_score = float(current["structure_score"])
        best_score = float(best["structure_score"])
        random_accept_p = middle_random_accept_p if phase == "middle" else default_random_accept_p

        if cand_score <= cur_score or rng.random() < random_accept_p:
            current = cand

        if cand_score < best_score:
            best = cand
            no_improve = 0
            # 本次修改：
            # 只保留 near_zero / near_one 的 best-anchor。
            # middle 不再一刷新 best 就强行 current = best。
            if phase in {"near_zero", "near_one"}:
                current = dict(best)
        else:
            no_improve += 1

        # -----------------------------------------------------
        # diversification
        # near1 左 branch 不再做其他 left diversification；
        # 只允许直接重置回同半径 y 轴锚点。
        # -----------------------------------------------------
        if bool(best.get("structure_success", False)):
            if not near1_no_early_stop and not near0_no_early_stop:
                if phase != "near_one" or it > int(0.35 * budget):
                    break

        if no_improve >= 18:
            mode_value = _resolve_mode_value_local(float(structure_bias))

            # mode=-1 仍保留 full-support 修补
            if mode_value == -1:
                tmp_n = best["n_ck"].copy()
                tmp_n = _zipf_pareto_refill_zeros(
                    tmp_n,
                    rng=rng,
                    alpha=1.2,
                    beta=1.2,
                    max_fill_each=2,
                )
                tmp_n = _mode_minus_one_repair(tmp_n)

            # near1 左侧：只回锚点，不做别的
            elif (
                phase == "near_one"
                and float(stage_target_spec.get("target_lcd", 0.0))
                < float(stage_target_spec.get("near_one_seed_lcd", np.inf))
            ):
                tmp_n = np.array(
                    stage_target_spec["near_one_left_anchor_n_ck"],
                    dtype=int,
                    copy=True,
                )

            # 其他情况保持原有轻度 diversify
            else:
                tmp_n = best["n_ck"].copy()
                tmp_n = _apply_peak_reweight(tmp_n, rng, steps=1)
                tmp_n = _apply_flatten_reweight(tmp_n, rng, steps=1)

            current = _safe_record_from_matrix(
                tmp_n,
                target_ssdi=float(seed_record["target_ssdi"]),
                lcdtype=lcdtype,
                ldstype=ldstype,
                alpha=alpha,
                beta=beta,
                lcd_params=lcd_params,
                lds_params=lds_params,
                source_stage="structured diversification",
                generator_variant=f"structured_{phase}_diversify",
            )
            current = _decorate_record_constraints(
                current,
                structure_bias=structure_bias,
                structure_mode=structure_mode,
                target_spec=stage_target_spec,
                phase=phase,
                stage_ssdi_error=stage_ssdi_error,
            )
            no_improve = 0

    best["total_budget_used"] = int(budget)
    best["returned_from"] = phase

    # ---------------------------------------------------------
    # near1 rescue:
    # - 普通 near1: 失败时 rescue
    # - mode>=0.90: 即使已经 success，也再 rescue 一轮，
    #   让它从左锚点/补救路径再竞争一次，避免过早锁死在局部盆地
    # ---------------------------------------------------------
    need_near1_rescue = (
        phase == "near_one"
        and (
            (not bool(best.get("structure_success", False)))
            or near1_no_early_stop
        )
    )

    if need_near1_rescue:
        rescue = _rescue_near1_left_sparse_failure_from_anchor(
            target_spec=stage_target_spec,
            structure_bias=structure_bias,
            structure_mode=structure_mode,
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
            stage_ssdi_error=stage_ssdi_error,
            budget=budget,   # 再给相同 budget
            seed=seed,
        )

        rescue_success = bool(rescue.get("structure_success", False))
        best_success = bool(best.get("structure_success", False))

        # ---------------------------------------------------------
        # near1 强右探索模式：
        # mode>=0.95 时，先看谁满足 stricter SSDI<=0.02；
        # 在都满足的前提下，LCD 越大越好。
        # ---------------------------------------------------------
        if near1_hard_right_explore:
            rescue_gap = float(abs(rescue["SSDI"] - rescue["target_ssdi"]))
            best_gap = float(abs(best["SSDI"] - best["target_ssdi"]))

            rescue_in = rescue_gap <= 0.02
            best_in = best_gap <= 0.02

            if rescue_in and best_in:
                if float(rescue["LCD"]) > float(best["LCD"]):
                    best = rescue
            elif rescue_in and (not best_in):
                best = rescue
            elif (not rescue_in) and (not best_in):
                if float(rescue.get("structure_score", 1e18)) < float(best.get("structure_score", 1e18)):
                    best = rescue

        else:
            if rescue_success and (not best_success):
                best = rescue
            elif rescue_success and best_success:
                if float(rescue["structure_score"]) < float(best["structure_score"]):
                    best = rescue
            elif (not rescue_success) and (not best_success):
                if float(rescue["structure_score"]) < float(best["structure_score"]):
                    best = rescue

    return best



def _run_stage_search(
    *,
    seed_record: Dict[str, Any],
    stage: str,
    target_spec: Dict[str, Any],
    support_policy: Dict[str, Any],
    mode_policy: Dict[str, Any],
    ssdi_error: float,
    max_iters: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:

    if stage in {'exact_zero', 'exact_one'}:
        rec = dict(seed_record)
        rec['structure_success'] = True
        rec["total_budget_used"] = 0
        rec["returned_from"] = "exact_direct_return"
        return rec

    plan = _stage_budget_plan(stage, int(mode_policy['mode_value']), max_iters)

    # ===== near1 必须把 exact1 seed 参考值带入 target_spec =====
    stage_target_spec = dict(target_spec)
    if stage == 'near_one':
        stage_target_spec["near_one_seed_lcd"] = float(seed_record.get("LCD", 0.0))
        stage_target_spec["near_one_seed_lds"] = float(seed_record.get("LDS", 0.0))
        stage_target_spec["near_one_seed_theta"] = float(
            _compute_theta_from_lcd_lds(seed_record.get("LCD", 0.0), seed_record.get("LDS", 0.0))
        )
        stage_target_spec["near_one_lcd_margin"] = 1e-12

    # near_zero: 不需要 support 两轮制
    if stage == 'near_zero':
        best = dict(seed_record)
        total_budget = 0
        cap = 0
        while total_budget < plan['upper_budget']:
            extra = min(
                plan['extra_budget'] if total_budget > 0 else plan['init_budget'],
                plan['upper_budget'] - total_budget,
            )
            best = _search_from_seed(
                best,
                stage_target_spec,
                support_policy,
                mode_policy,
                ssdi_error,
                extra,
                rng,
                'round1',
                stage,
                cap,
            )
            total_budget += extra

            if _structured_success_reworked(
                best, stage_target_spec, mode_policy, support_policy, ssdi_error, stage, cap
            ):
                best["total_budget_used"] = int(total_budget)
                best["returned_from"] = "success_early_stop"
                return best

        best["total_budget_used"] = int(total_budget)
        best["returned_from"] = "budget_exhausted_best_effort"
        return best

    cap1 = 0 if mode_policy.get('protect_nonempty_clients_round1', False) else int(
        support_policy.get('round1_empty_client_cap', 0)
    )
    _decorate_record_constraints(seed_record, support_policy, 'round1', cap1)

    first_budget = int(plan['init_budget'])
    if stage == 'near_one':
        # ===== near1 第一轮不要过大，也不要过小 =====
        first_budget = min(120, int(plan['upper_budget']))

    best = _search_from_seed(
        seed_record,
        stage_target_spec,
        support_policy,
        mode_policy,
        ssdi_error,
        first_budget,
        rng,
        'round1',
        stage,
        cap1,
    )

    if _structured_success_reworked(best, stage_target_spec, mode_policy, support_policy, ssdi_error, stage, cap1):
        best["total_budget_used"] = int(first_budget)
        best["returned_from"] = "success_early_stop"
        return best

    total_budget = int(first_budget)
    schedules = list(support_policy.get('round2_empty_client_schedule', []) or [cap1])

    # ===== near1 + mode=1 右边界停滞监控 =====
    near1_mode1 = (stage == 'near_one' and int(mode_policy.get('mode_value', 0)) == 1)
    best_lcd_seen = float(best.get('LCD', -1e18))
    stale_batches = 0
    lcd_stall_eps = 1e-10
    stale_limit = 3

    for cap in schedules:
        cap = int(cap)
        if cap < cap1:
            continue

        while total_budget < plan['upper_budget']:
            extra = min(plan['extra_budget'], plan['upper_budget'] - total_budget)

            best = _search_from_seed(
                best,
                stage_target_spec,
                support_policy,
                mode_policy,
                ssdi_error,
                extra,
                rng,
                'round2',
                stage,
                cap,
            )
            total_budget += extra

            if _structured_success_reworked(best, stage_target_spec, mode_policy, support_policy, ssdi_error, stage, cap):
                best["total_budget_used"] = int(total_budget)
                best["returned_from"] = "success_early_stop"
                return best

            if near1_mode1:
                cur_lcd = float(best.get('LCD', -1e18))
                if cur_lcd > best_lcd_seen + lcd_stall_eps:
                    best_lcd_seen = cur_lcd
                    stale_batches = 0
                else:
                    stale_batches += 1

                if stale_batches >= stale_limit:
                    best["total_budget_used"] = int(total_budget)
                    best["returned_from"] = "near1_right_boundary_stalled"
                    return best

    best["total_budget_used"] = int(total_budget)
    best["returned_from"] = "budget_exhausted_best_effort"
    return best

def _pick_best_candidate_within_band(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        raise RuntimeError("No structured candidates generated.")
    feasible = [c for c in candidates if bool(c.get("structure_success", False))]
    if feasible:
        return min(feasible, key=lambda c: float(c["structure_score"]))
    return min(candidates, key=lambda c: float(c["structure_score"]))



def _build_structured_dispatch_plan(
    *,
    target_ssdi: float,
    ssdi_lcdmax: float,
    ssdi0_local: float,
) -> Dict[str, Any]:
    """
    Build the multi-stage dispatch plan you just settled:

    1) s < 0.09
       -> near_zero

    2) 0.09 <= s <= 0.2
       -> middle
       -> near_zero fallback

    3) 0.2 < s <= ssdi_lcdmax
       -> middle only

    4) ssdi_lcdmax < s <= 0.8
       -> middle
       -> relaxed_middle (allow-empty style seed)

    5) 0.8 < s <= ssdi0
       -> middle with half budget
       -> relaxed_middle
       -> near_one fallback

    6) s > ssdi0
       -> near_one مباشرة
    """
    s = float(np.clip(target_ssdi, 0.0, 1.0))
    ssdi_lcdmax = float(max(0.0, ssdi_lcdmax))
    ssdi0_local = float(np.clip(ssdi0_local, 0.0, 1.0))

    if s < 0.09:
        return {
            "dispatch_case": "near_zero_only",
            "ssdi_lcdmax": ssdi_lcdmax,
            "ssdi0_local": ssdi0_local,
            "steps": [
                {
                    "attempt_stage": "near_zero",
                    "phase": "near_zero",
                    "budget_scale": 1.0,
                    "seed_mode": "default",
                }
            ],
        }

    if s <= 0.20:
        return {
            "dispatch_case": "middle_then_near_zero_fallback",
            "ssdi_lcdmax": ssdi_lcdmax,
            "ssdi0_local": ssdi0_local,
            "steps": [
                {
                    "attempt_stage": "middle",
                    "phase": "middle",
                    "budget_scale": 1.0,
                    "seed_mode": "default",
                },
                {
                    "attempt_stage": "near_zero_fallback",
                    "phase": "near_zero",
                    "budget_scale": 1.0,
                    "seed_mode": "default",
                },
            ],
        }

    if s <= ssdi_lcdmax:
        return {
            "dispatch_case": "middle_only_below_lcdmax",
            "ssdi_lcdmax": ssdi_lcdmax,
            "ssdi0_local": ssdi0_local,
            "steps": [
                {
                    "attempt_stage": "middle",
                    "phase": "middle",
                    "budget_scale": 1.0,
                    "seed_mode": "default",
                }
            ],
        }

    if s <= 0.80:
        return {
            "dispatch_case": "middle_then_relaxed_middle",
            "ssdi_lcdmax": ssdi_lcdmax,
            "ssdi0_local": ssdi0_local,
            "steps": [
                {
                    "attempt_stage": "middle",
                    "phase": "middle",
                    "budget_scale": 1.0,
                    "seed_mode": "default",
                },
                {
                    "attempt_stage": "relaxed_middle",
                    "phase": "middle",
                    "budget_scale": 1.0,
                    "seed_mode": "relaxed_allow_empty",
                },
            ],
        }

    if s <= ssdi0_local:
        return {
            "dispatch_case": "middle_half_then_relaxed_middle_then_near_one",
            "ssdi_lcdmax": ssdi_lcdmax,
            "ssdi0_local": ssdi0_local,
            "steps": [
                {
                    "attempt_stage": "middle_half",
                    "phase": "middle",
                    "budget_scale": 0.5,
                    "seed_mode": "default",
                },
                {
                    "attempt_stage": "relaxed_middle",
                    "phase": "middle",
                    "budget_scale": 1.0,
                    "seed_mode": "relaxed_allow_empty",
                },
                {
                    "attempt_stage": "near_one_fallback",
                    "phase": "near_one",
                    "budget_scale": 1.0,
                    "seed_mode": "default",
                },
            ],
        }

    return {
        "dispatch_case": "near_one_only_above_ssdi0",
        "ssdi_lcdmax": ssdi_lcdmax,
        "ssdi0_local": ssdi0_local,
        "steps": [
            {
                "attempt_stage": "near_one",
                "phase": "near_one",
                "budget_scale": 1.0,
                "seed_mode": "default",
            }
        ],
    }

def _build_relaxed_middle_seed(
    *,
    client: int,
    label: int,
    datasize: int,
    target_ssdi: float,
    lcdtype: str,
    ldstype: str,
    alpha: Optional[float],
    beta: Optional[float],
    lcd_params: Optional[Dict[str, Any]],
    lds_params: Optional[Dict[str, Any]],
    seed: Optional[int],
) -> Dict[str, Any]:
    """
    Build a sparse / allow-empty style seed for the relaxed-middle stage.

    Design:
    - use the merged target-aware param pack to get effective alpha / beta
    - build a sparse extremal-like matrix directly
    - DO NOT repair zeros here
    - let the later middle search refine it

    This is the cleanest way to "接入允许空的机制" on the current active path,
    because the current active structured entry does not call the old support-policy
    runner; it directly calls _search_from_seed(...).
    """
    pack = _merge_param_pack(
        target_ssdi=target_ssdi,
        lcdtype=lcdtype,
        ldstype=ldstype,
        C=int(label),
        K=int(client),
        alpha=alpha,
        beta=beta,
        lcd_params=lcd_params,
        lds_params=lds_params,
        seed=seed,
    )

    alpha_eff = float(pack["alpha"])
    beta_eff = float(pack["beta"])

    n_ck = _build_sparse_extremal_seed(
        C=int(label),
        K=int(client),
        N=int(datasize),
        alpha=alpha_eff,
        beta=beta_eff,
        ldstype=ldstype,
    )

    rec = _safe_record_from_matrix(
        n_ck,
        target_ssdi=float(target_ssdi),
        lcdtype=lcdtype,
        ldstype=ldstype,
        alpha=alpha,
        beta=beta,
        lcd_params=lcd_params,
        lds_params=lds_params,
        source_stage="structured relaxed-middle seed",
        generator_variant="seed_relaxed_middle_allow_empty",
    )
    rec["target_ssdi"] = float(target_ssdi)
    return rec


def _run_structured_dispatch_plan(
    *,
    dispatch_plan: Dict[str, Any],
    client: int,
    label: int,
    datasize: int,
    target_ssdi: float,
    structure_bias: float,
    structure_mode_name: str,
    target_spec: Dict[str, Any],
    lcdtype: str,
    ldstype: str,
    alpha: Optional[float],
    beta: Optional[float],
    lcd_params: Optional[Dict[str, Any]],
    lds_params: Optional[Dict[str, Any]],
    ssdi_error: float,
    max_iters: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    """
    Execute the multi-stage dispatch plan.

    Each stage:
    - builds its own seed
    - gets its own phase-aware SSDI tolerance
    - gets its own budget scale
    - runs _search_from_seed(...)
    - validates and records trace fields

    If one stage succeeds, stop immediately.
    Otherwise, return the best candidate across all executed stages.
    """
    candidates: List[Dict[str, Any]] = []
    executed_steps: List[str] = []

    steps = list(dispatch_plan.get("steps", []))
    dispatch_case = str(dispatch_plan.get("dispatch_case", "unknown_dispatch"))

    for idx, step in enumerate(steps):
        attempt_stage = str(step["attempt_stage"])
        phase = str(step["phase"])
        seed_mode = str(step.get("seed_mode", "default"))
        budget_scale = float(step.get("budget_scale", 1.0))

        stage_seed = None if seed is None else int(seed + 10000 * (idx + 1))
        stage_ssdi_error = _resolve_stage_ssdi_error(
            target_ssdi=float(target_ssdi),
            phase=phase,
            base_error=ssdi_error,
        )

        # ---------------------------
        # stage seed
        # ---------------------------
        if seed_mode == "relaxed_allow_empty":
            seed_record = _build_relaxed_middle_seed(
                client=client,
                label=label,
                datasize=datasize,
                target_ssdi=target_ssdi,
                lcdtype=lcdtype,
                ldstype=ldstype,
                alpha=alpha,
                beta=beta,
                lcd_params=lcd_params,
                lds_params=lds_params,
                seed=stage_seed,
            )
        else:
            seed_record = _build_seed_for_phase(
                phase=phase,
                client=client,
                label=label,
                datasize=datasize,
                target_ssdi=target_ssdi,
                lcdtype=lcdtype,
                ldstype=ldstype,
                alpha=alpha,
                beta=beta,
                lcd_params=lcd_params,
                lds_params=lds_params,
                ssdi_error=stage_ssdi_error,
                seed=stage_seed,
                max_iters=max_iters,
                structure_bias=structure_bias,
            )

        # ---------------------------
        # stage budget
        # ---------------------------
        base_budget = int(_stage_budget_plan(phase, max_iters))
        budget = int(max(1, round(base_budget * budget_scale)))

        # keep your current special rule:
        # near1 + mode>=0.95 => triple search budget
        if phase == "near_one" and float(structure_bias) >= 0.95:
            budget = int(3 * budget)

        # ---------------------------
        # run stage
        # ---------------------------
        details = _search_from_seed(
            seed_record,
            phase=phase,
            structure_bias=structure_bias,
            structure_mode=structure_mode_name,
            target_spec=target_spec,
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
            stage_ssdi_error=stage_ssdi_error,
            budget=budget,
            seed=stage_seed,
        )

        executed_steps.append(attempt_stage)
        details["attempt_stage"] = attempt_stage
        details["attempt_phase"] = phase
        details["attempt_path"] = " -> ".join(executed_steps)
        details["dispatch_case"] = dispatch_case
        details["dispatch_steps"] = [s["attempt_stage"] for s in steps]
        details["dispatch_ssdi_lcdmax"] = float(dispatch_plan.get("ssdi_lcdmax", np.nan))
        details["dispatch_ssdi0"] = float(dispatch_plan.get("ssdi0_local", np.nan))


        # 关键：把 trace 字段写进去后，再重算一次 structure_success / structure_score
        details = _decorate_record_constraints(
            details,
            structure_bias=structure_bias,
            structure_mode=structure_mode_name,
            target_spec=target_spec,
            phase=phase,
            stage_ssdi_error=stage_ssdi_error,
        )

        details = _validate_constraints(
            details,
            phase=phase,
            stage_ssdi_error=stage_ssdi_error,
        )

        candidates.append(details)

        if bool(details.get("success", False)):
            return details

    best = _pick_best_candidate_within_band(candidates)
    best["dispatch_case"] = dispatch_case
    best["dispatch_steps"] = [s["attempt_stage"] for s in steps]
    best["dispatch_ssdi_lcdmax"] = float(dispatch_plan.get("ssdi_lcdmax", np.nan))
    best["dispatch_ssdi0"] = float(dispatch_plan.get("ssdi0_local", np.nan))
    return best




def _validate_constraints(
    details: Dict[str, Any],
    *,
    phase: str,
    stage_ssdi_error: float,
) -> Dict[str, Any]:
    out = dict(details)
    out["success"] = bool(out.get("structure_success", False))
    out["is_best_effort"] = not bool(out["success"])
    out["ssdi_gap"] = abs(float(out["SSDI"]) - float(out["target_ssdi"]))
    out["time_used"] = float(out.get("time_used", out.get("time_elapsed", 0.0)))
    out["time_elapsed"] = float(out.get("time_elapsed", out["time_used"]))

    # keep schema stable
    if out["success"]:
        out["failure_primary"] = None
        out["failure_structure_detail"] = None
        out["optimization_hint"] = None
        out["failure_note"] = None
    else:
        if out.get("failure_primary") is None:
            if out["ssdi_gap"] > stage_ssdi_error:
                out["failure_primary"] = "ssdi_miss"
                out["failure_structure_detail"] = "outside layered tolerance"
                out["optimization_hint"] = "repair radius"
                out["failure_note"] = "ssdi_miss: outside layered tolerance"
            else:
                out["failure_primary"] = "theta_miss"
                out["failure_structure_detail"] = "direction not matched"
                out["optimization_hint"] = "repair angle"
                out["failure_note"] = "theta_miss: direction not matched"

    if phase == "exact_zero":
        out["returned_from"] = "exact_zero"
    elif phase == "exact_one":
        out["returned_from"] = "exact_one"

    return out


def generate_ssdi_matrix_structured(
    *,
    client: int,
    label: int,
    datasize: int,
    ssdi: float,
    structure_mode: Union[str, float, int] = 0.0,
    structure_bias: Optional[float] = None,
    lcdtype: str = "client",
    ldstype: str = "client",
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    lcd_params: Optional[Dict[str, Any]] = None,
    lds_params: Optional[Dict[str, Any]] = None,
    ssdi_error: float = 0.02,
    seed: Optional[int] = 42,
    max_iters: int = 160,
    return_details: bool = False,
    save: bool = True,
    save_metrics: bool = False,
    save_csv: bool = False,
    output_dir: str = "./single_outputs",
    verbose: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Structured generator with multi-stage dispatcher.

    Exact points are preserved:
    - exact_zero -> direct iid
    - exact_one  -> direct exact extremal

    All non-exact points now follow the new dispatch structure:
    - <0.09                   : near_zero
    - 0.09~0.2               : middle -> near_zero fallback
    - 0.2~ssdi_lcdmax        : middle
    - ssdi_lcdmax~0.8        : middle -> relaxed_middle
    - 0.8~ssdi0              : middle(half budget) -> relaxed_middle -> near_one
    - >ssdi0                 : near_one
    """
    t0 = time.time()
    seed = _normalize_seed(seed)
    target_ssdi = float(np.clip(ssdi, 0.0, 1.0))
    structure_mode = _coerce_structure_mode_value(structure_mode)

    if structure_bias is None:
        structure_bias = _structure_mode_to_bias(structure_mode)
    else:
        structure_bias = float(np.clip(structure_bias, -1.0, 1.0))

    structure_mode_name = _bias_to_structure_mode(structure_bias)

    # keep exact-point detection exactly as before
    coarse_phase = _resolve_generation_phase(
        target_ssdi,
        int(label),
        int(client),
        int(datasize),
    )

    exact_zero_tol = _resolve_stage_ssdi_error(target_ssdi, "exact_zero", ssdi_error)
    exact_one_tol = _resolve_stage_ssdi_error(target_ssdi, "exact_one", ssdi_error)

    if coarse_phase == "exact_zero":
        details = _construct_exact_zero_record(
            client=client,
            label=label,
            datasize=datasize,
            ssdi=target_ssdi,
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
        )
        target_spec = {
            "target_lcd": 0.0,
            "target_lds": 0.0,
            "target_theta": 0.0,
            "theta_max": 0.0,
            "domain_stage": "exact_zero",
            "target_dsr": 0.0,
        }
        details = _decorate_record_constraints(
            details,
            structure_bias=structure_bias,
            structure_mode=structure_mode_name,
            target_spec=target_spec,
            phase="exact_zero",
            stage_ssdi_error=exact_zero_tol,
        )
        details = _validate_constraints(
            details,
            phase="exact_zero",
            stage_ssdi_error=exact_zero_tol,
        )
        details["dispatch_case"] = "exact_zero_direct"
        details["attempt_stage"] = "exact_zero"
        details["attempt_phase"] = "exact_zero"
        details["attempt_path"] = "exact_zero"

    elif coarse_phase == "exact_one":
        details = _construct_exact_one_record(
            client=client,
            label=label,
            datasize=datasize,
            ssdi=target_ssdi,
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
        )
        geometry = _compute_geometry_bundle(
            client=client,
            label=label,
            datasize=datasize,
            target_ssdi=target_ssdi,
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
        )
        target_spec = _structured_target_components(1.0, structure_bias, geometry)
        details = _decorate_record_constraints(
            details,
            structure_bias=structure_bias,
            structure_mode=structure_mode_name,
            target_spec=target_spec,
            phase="exact_one",
            stage_ssdi_error=exact_one_tol,
        )
        details = _validate_constraints(
            details,
            phase="exact_one",
            stage_ssdi_error=exact_one_tol,
        )
        details["dispatch_case"] = "exact_one_direct"
        details["attempt_stage"] = "exact_one"
        details["attempt_phase"] = "exact_one"
        details["attempt_path"] = "exact_one"

    else:
        geometry = _compute_geometry_bundle(
            client=client,
            label=label,
            datasize=datasize,
            target_ssdi=target_ssdi,
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
        )
        target_spec = _structured_target_components(
            target_ssdi,
            structure_bias,
            geometry,
        )

        ssdi_lcdmax = float(geometry["ssdi_lcdmax"])
        ssdi0_local = float(_compute_ssdi0_local(int(label), int(client), int(datasize)))

        dispatch_plan = _build_structured_dispatch_plan(
            target_ssdi=target_ssdi,
            ssdi_lcdmax=ssdi_lcdmax,
            ssdi0_local=ssdi0_local,
        )

        details = _run_structured_dispatch_plan(
            dispatch_plan=dispatch_plan,
            client=client,
            label=label,
            datasize=datasize,
            target_ssdi=target_ssdi,
            structure_bias=structure_bias,
            structure_mode_name=structure_mode_name,
            target_spec=target_spec,
            lcdtype=lcdtype,
            ldstype=ldstype,
            alpha=alpha,
            beta=beta,
            lcd_params=lcd_params,
            lds_params=lds_params,
            ssdi_error=ssdi_error,
            max_iters=max_iters,
            seed=seed,
        )

    details["target_ssdi"] = float(target_ssdi)
    details["C"] = int(label)
    details["K"] = int(client)
    details["N"] = int(datasize)
    details["lcd_type"] = lcdtype
    details["lds_type"] = ldstype
    details["structure_bias"] = float(structure_bias)
    details["structure_mode"] = structure_mode_name
    details["time_used"] = float(time.time() - t0)
    details["time_elapsed"] = details["time_used"]
    details["guaranteed_result"] = True
    details.setdefault("source_ssdi", target_ssdi)

    saved_paths = None
    if save:
        saved_paths = _save_single_matrix_outputs(
            details["n_ck"],
            details,
            client=client,
            label=label,
            datasize=datasize,
            ssdi=ssdi,
            lcdtype=lcdtype,
            ldstype=ldstype,
            output_dir=output_dir,
            save_metrics=save_metrics,
            save_csv=save_csv,
        )
        details["saved_paths"] = saved_paths
        details["output_dir"] = saved_paths["output_dir"]
        details["npy_path"] = saved_paths["npy"]
        details["metrics_csv_path"] = saved_paths["metrics_csv"]
        details["matrix_csv_path"] = saved_paths.get("matrix_csv")

    _print_single_generation_summary(details, saved_paths, verbose=verbose)

    if return_details:
        return details["matrix_df"], details
    return details["matrix_df"]



def inspect_structured_generation_plan(
    *,
    client: int,
    label: int,
    datasize: int,
    ssdi: float,
    structure_mode: Union[str, float, int] = 0.0,
    structure_bias: Optional[float] = None,
    lcdtype: str = "client",
    ldstype: str = "client",
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    lcd_params: Optional[Dict[str, Any]] = None,
    lds_params: Optional[Dict[str, Any]] = None,
    ssdi_error: float = 0.02,
    seed: Optional[int] = None,
    max_iters: int = 160,
) -> Dict[str, Any]:
    """
    Inspect the new structured generation plan without actually running the full search.

    Returns a dictionary describing:
    - phase
    - layered SSDI tolerance
    - global geometry anchors
    - target LCD/LDS/theta
    - seed summary
    - search budget
    - expected move family
    """
    target_ssdi = float(np.clip(ssdi, 0.0, 1.0))

    if structure_bias is None:
        structure_bias = _structure_mode_to_bias(structure_mode)
    else:
        structure_bias = float(np.clip(structure_bias, -1.0, 1.0))

    structure_mode_name = _bias_to_structure_mode(structure_bias)

    phase = _resolve_generation_phase(
        target_ssdi=target_ssdi,
        C=int(label),
        K=int(client),
        N=int(datasize),
    )
    stage_ssdi_error = _resolve_stage_ssdi_error(
        target_ssdi=target_ssdi,
        phase=phase,
        base_error=ssdi_error,
    )

    geometry = _compute_geometry_bundle(
        client=client,
        label=label,
        datasize=datasize,
        target_ssdi=target_ssdi,
        lcdtype=lcdtype,
        ldstype=ldstype,
        alpha=alpha,
        beta=beta,
        lcd_params=lcd_params,
        lds_params=lds_params,
    )

    target_spec = _structured_target_components(
        target_ssdi=target_ssdi,
        structure_bias=structure_bias,
        geometry=geometry,
    )

    seed_record = _build_seed_for_phase(
        phase=phase,
        client=client,
        label=label,
        datasize=datasize,
        target_ssdi=target_ssdi,
        lcdtype=lcdtype,
        ldstype=ldstype,
        alpha=alpha,
        beta=beta,
        lcd_params=lcd_params,
        lds_params=lds_params,
        ssdi_error=stage_ssdi_error,
        seed=seed,
        max_iters=max_iters,
        structure_bias=structure_bias,
    )

    seed_theta = _compute_theta_from_lcd_lds(
        float(seed_record["LCD"]),
        float(seed_record["LDS"]),
    )
    theta_tol = _theta_tolerance_by_phase(
        phase=phase,
        theta_max=float(target_spec["theta_max"]),
    )
    budget = _stage_budget_plan(phase, max_iters)

    if phase == "near_one":
        lcd_gap = float(target_spec["target_lcd"] - seed_record["LCD"])
        lds_gap = float(target_spec["target_lds"] - seed_record["LDS"])
        if lcd_gap > 0:
            planned_move = "near1_right_down: diagonal_binaryization + slight_flatten"
        elif lds_gap > 0:
            planned_move = "near1_left_up: fill_high_value_zeros + peak_reweight"
        else:
            planned_move = "near1_left_down: fill_high_value_zeros + flatten_reweight"
    elif phase in {"near_zero", "middle"}:
        lcd_gap = float(target_spec["target_lcd"] - seed_record["LCD"])
        lds_gap = float(target_spec["target_lds"] - seed_record["LDS"])
        if lcd_gap >= 0 and lds_gap >= 0:
            planned_move = "ordinary_right_up: create_high_value_zeros + peak_reweight"
        elif lcd_gap >= 0 and lds_gap < 0:
            planned_move = "ordinary_right_down: create_high_value_zeros + flatten_reweight"
        elif lcd_gap < 0 and lds_gap >= 0:
            planned_move = "ordinary_left_up: fill_high_value_zeros + peak_reweight"
        else:
            planned_move = "ordinary_left_down: fill_high_value_zeros + flatten_reweight"
    elif phase == "exact_zero":
        planned_move = "direct_exact_zero"
    else:
        planned_move = "direct_exact_one"

    return {
        "client": int(client),
        "label": int(label),
        "datasize": int(datasize),
        "target_ssdi": float(target_ssdi),
        "structure_bias": float(structure_bias),
        "structure_mode": structure_mode_name,
        "phase": phase,
        "stage_ssdi_error": float(stage_ssdi_error),
        "theta_tolerance": float(theta_tol),
        "budget": int(budget),

        "ssdi_lcdmax": float(geometry["ssdi_lcdmax"]),
        "lcdmax_point": (
            float(geometry["lcdmax_point"][0]),
            float(geometry["lcdmax_point"][1]),
        ),
        "exact1_point": (
            float(geometry["exact1_point"][0]),
            float(geometry["exact1_point"][1]),
        ),
        "theta_lcdmax": float(geometry["theta_lcdmax"]),
        "theta_exact1": float(geometry["theta_exact1"]),

        "domain_stage": str(target_spec["domain_stage"]),
        "target_lcd": float(target_spec["target_lcd"]),
        "target_lds": float(target_spec["target_lds"]),
        "target_theta": float(target_spec["target_theta"]),
        "theta_max": float(target_spec["theta_max"]),
        "target_dsr": float(target_spec["target_dsr"]),

        "seed_ssdi": float(seed_record["SSDI"]),
        "seed_lcd": float(seed_record["LCD"]),
        "seed_lds": float(seed_record["LDS"]),
        "seed_theta": float(seed_theta),
        "seed_missing_rate": float(seed_record.get("missing_rate", 0.0)),
        "seed_source_stage": seed_record.get("source_stage"),
        "seed_generator_variant": seed_record.get("generator_variant"),

        "planned_move": planned_move,
    }

import glob


def _normalize_mode_tag(structure_mode: Any = None, structure_bias: Optional[float] = None) -> str:
    """
    Convert structure mode / bias into a short filename tag.

    Important:
    - This function is ONLY for output filename tagging.
    - It does NOT change the input semantic mapping of structure_mode.
    - Output tags are always one of: {"lds", "balance", "lcd"}.
    - Tagging rule is based on the bias value split by -0.33 / +0.33.

    Rules:
    - bias <= -0.33  -> "lds"
    - bias >=  0.33  -> "lcd"
    - otherwise      -> "balance"

    If structure_bias is available, use it first.
    Otherwise, infer from structure_mode:
    - numeric mode: use directly
    - string mode : map with the ORIGINAL input semantics
        "lds"/"lds_dominant"      -> -0.75
        "balance"/"balanced"      ->  0.0
        "lcd"/"lcd_dominant"      ->  0.75
    """

    # 1) Prefer the explicit numeric bias stored in details / pipeline.
    if structure_bias is not None:
        v = float(structure_bias)
        if v <= -0.33:
            return "skew"
        if v >= 0.33:
            return "coverage"
        return "mixed"

    # 2) Fall back to the incoming mode value.
    value = _coerce_structure_mode_value(structure_mode)

    if isinstance(value, list):
        value = value[0]

    # 2a) Numeric mode: directly split by -0.33 / +0.33.
    if isinstance(value, (int, float, np.floating)):
        v = float(value)
        if v <= -0.33:
            return "skew"
        if v >= 0.33:
            return "coverage"
        return "mixed"

    # 2b) String mode: keep ORIGINAL semantic mapping for input modes,
    #     but only output 3 filename tags.
    if isinstance(value, str):
        s = value.strip().lower()

        # Keep old input semantics unchanged.
        if s in {"skew", "lds", "lds_dominant"}:
            v = -0.75
        elif s in {"mixed", "balanced", "balance"}:
            v = 0.0
        elif s in {"coverage", "lcd", "lcd_dominant"}:
            v = 0.75
        else:
            # If the string is actually a number, treat it as numeric bias.
            try:
                v = float(s)
            except Exception:
                return "balance"

        if v <= -0.33:
            return "skew"
        if v >= 0.33:
            return "coverage"
        return "mixed"

    return "mixed"


def _next_generation_index(
    npy_dir: str,
    *,
    C: int,
    K: int,
    N: int,
    target_ssdi: float,
    mode_tag: str,
) -> int:
    """
    Find next available counter for files with the same C/K/N/SSDI/mode prefix.
    """
    prefix = f"C_{C}_K_{K}_N_{N}_SSDI_{target_ssdi:.3f}_{mode_tag}_"
    pattern = os.path.join(npy_dir, prefix + "*.npy")
    files = glob.glob(pattern)

    max_idx = 0
    for fp in files:
        name = os.path.basename(fp)
        stem = os.path.splitext(name)[0]
        tail = stem.replace(prefix, "")
        if tail.isdigit():
            max_idx = max(max_idx, int(tail))
    return max_idx + 1

def save_dataset_to_npy(
    n_ck: np.ndarray,
    npy_dir: str,
    C: int,
    K: int,
    N: int,
    target_ssdi: float,
    mode_tag: str,
    counter: Optional[int] = None,
    show_result: bool = False,
    save_txt: bool = True,
    filename_stem: Optional[str] = None,
) -> str:
    """
    Save one generated matrix as .npy (and optionally .txt).

    Default filename format:
        C_{C}_K_{K}_N_{N}_SSDI_{ssdi:.3f}_{mode}_{idx:03d}.npy

    If filename_stem is provided, save exactly as:
        {filename_stem}.npy
    """
    os.makedirs(npy_dir, exist_ok=True)

    if filename_stem is not None:
        filename = f"{filename_stem}.npy"
    else:
        if counter is None:
            counter = _next_generation_index(
                npy_dir,
                C=C,
                K=K,
                N=N,
                target_ssdi=target_ssdi,
                mode_tag=mode_tag,
            )

        filename = f"C_{C}_K_{K}_N_{N}_SSDI_{target_ssdi:.3f}_{mode_tag}_{counter:03d}.npy"

    filepath = os.path.join(npy_dir, filename)
    np.save(filepath, n_ck)

    if save_txt:
        txt_path = filepath.replace(".npy", ".txt")
        np.savetxt(txt_path, n_ck, fmt="%d", delimiter="\t")
        if show_result:
            print(f"保存文本文件: {os.path.basename(txt_path)}")

    if show_result:
        print(f"保存数据集: {filename}")
        print(f"  形状: {n_ck.shape}")
        print(f"  总样本数: {int(n_ck.sum())}")

    return filepath

def _save_single_matrix_outputs(
    n_ck: np.ndarray,
    details: Dict[str, Any],
    *,
    client: int,
    label: int,
    datasize: int,
    ssdi: float,
    lcdtype: str,
    ldstype: str,
    output_dir: str,
    save_metrics: bool = False,
    save_csv: bool = False,
) -> Dict[str, Optional[str]]:
    """
    Save outputs for one structured generation.

    Storage rule:
    - Always create a dedicated timestamp subfolder for THIS generation.
    - Folder/file naming follows the preferred compact style.
    - The file mode tag is ONLY one of: lds / balance / lcd.
    - The file mode tag is determined by bias split at -0.33 / +0.33.
    - This naming rule is independent from the input structure_mode semantic mapping.
    """
    os.makedirs(output_dir, exist_ok=True)

    mode_tag = _normalize_mode_tag(
        structure_mode=details.get("structure_mode"),
        structure_bias=details.get("structure_bias"),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"c{int(label)}_k{int(client)}_n{int(datasize)}_ssdi{float(ssdi):.3f}_mode_{mode_tag}"
    run_dir = os.path.join(output_dir, f"{stem}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    npy_path = save_dataset_to_npy(
        n_ck=n_ck,
        npy_dir=run_dir,
        C=int(label),
        K=int(client),
        N=int(datasize),
        target_ssdi=float(ssdi),
        mode_tag=mode_tag,
        counter=None,
        show_result=False,
        save_txt=True,
        filename_stem=stem,
    )

    matrix_csv_path = None
    if save_csv:
        matrix_csv_path = os.path.join(run_dir, f"{stem}.csv")
        pd.DataFrame(n_ck).to_csv(matrix_csv_path, index=False)

    metrics_csv_path = None
    metrics_json_path = None
    if save_metrics:
        metrics_json_path = os.path.join(run_dir, f"{stem}_metrics.json")
        serializable = {}
        for k, v in details.items():
            if isinstance(v, np.ndarray):
                continue
            if isinstance(v, pd.DataFrame):
                continue
            try:
                json.dumps(v)
                serializable[k] = v
            except Exception:
                serializable[k] = str(v)

        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        metrics_csv_path = os.path.join(run_dir, f"{stem}_metrics.csv")
        pd.DataFrame([serializable]).to_csv(metrics_csv_path, index=False)

    return {
        "output_dir": run_dir,
        "root_output_dir": output_dir,
        "timestamp": timestamp,
        "mode_tag": mode_tag,
        "npy": npy_path,
        "txt": npy_path.replace(".npy", ".txt"),
        "matrix_csv": matrix_csv_path,
        "metrics_csv": metrics_csv_path,
        "metrics_json": metrics_json_path,
    }