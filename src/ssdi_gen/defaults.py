# =============================================
# default_params.py
# =============================================
import numpy as np

def _normalize_seed(seed, default=42):
    return int(default if seed is None else seed)

def get_default_params(target_ssdi, mechanism, C=10, K=20, seed=42):
    """
    合并版默认参数生成器（优化版）：
    - target_ssdi ≤ 0.7：使用新版（v2）参数（低中SSDI表现好）
    - target_ssdi > 0.7：以 80% 概率使用旧版（v1）参数，20% 概率使用新版（v2）参数
      这样既保留了旧版在高SSDI的优势，又通过新版的宽扰动增加了探索性，进一步提高成功率。
    """
    seed = _normalize_seed(seed)
    threshold = 0.7
    rng = np.random.default_rng(seed)

    if target_ssdi <= threshold:
        return _get_default_params_v2(target_ssdi, mechanism, C, K, seed=seed)
    else:
        # 高 SSDI 区间：随机选择版本，v1 优先
        if rng.random() < 0.7:  # 70% 概率使用 v1
            return _get_default_params_v1(target_ssdi, mechanism, C, K, seed=seed)
        else:                    # 30% 概率使用 v2
            return _get_default_params_v2(target_ssdi, mechanism, C, K, seed=seed)



def _get_default_params_v1(target_ssdi, mechanism, C=10, K=20, seed=42):
    """
    旧版本默认参数生成器（内部使用）
    """
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    s = float(target_ssdi)
    s = np.clip(s, 0.01, 0.99)

    # 基础参数
    alpha = 3.0 - 2.2 * s
    alpha = np.clip(alpha, 0.8, 3.5)
    beta = 0.5 + 2.0 * s
    beta = np.clip(beta, 0.4, 3.5)
    missing_rate = 0.05 + 0.55 * s
    missing_rate = min(missing_rate, 0.6)

    # LDS 分段增强
    if s <= 0.4:
        lds_scale = 1.0
    elif s <= 0.6:
        lds_scale = 1.5
    else:
        lds_scale = 1.5 + 4.0 * (s - 0.6)

    # 机制参数
    if mechanism == 'lcd_client':
        tau = 0.2 + 1.2 * s
        params = {'alpha': alpha, 'beta': beta, 'missing_rate': missing_rate, 'tau': tau}
    elif mechanism == 'lcd_class':
        gamma = 0.2 + 1.2 * s
        params = {'alpha': alpha, 'beta': beta, 'missing_rate': missing_rate, 'gamma': gamma}
    elif mechanism == 'lcd_joint':
        a = 0.4 + 0.8 * s
        b = 0.4 + 0.8 * s
        params = {'alpha': alpha, 'beta': beta, 'missing_rate': missing_rate, 'a': a, 'b': b}
    elif mechanism == 'lds_client':
        strength = (0.8 + 1.8 * s) * lds_scale
        params = {'alpha': alpha, 'beta': beta, 'strength': strength}
    elif mechanism == 'lds_special':
        strength = (0.8 + 1.8 * s) * lds_scale
        num_special = max(1, int((0.1 + 0.5 * s) * K))
        params = {'alpha': alpha, 'beta': beta, 'strength': strength, 'num_special': num_special}
    elif mechanism == 'lds_lowrank':
        strength = (0.8 + 1.8 * s) * lds_scale
        rank = max(1, int((0.1 + 0.6 * s) * min(C, K)))
        params = {'alpha': alpha, 'beta': beta, 'strength': strength, 'rank': rank}
    else:
        raise ValueError(f"未知机制: {mechanism}")

    params = _randomize_params_within_bounds_v1(params, rng)
    return params


def _randomize_params_within_bounds_v1(params, rng):
    new_params = params.copy()
    new_params['_randomized'] = True

    def perturb(x, scale=0.1):
        return float(x * np.exp(rng.normal(0, scale)))

    if 'alpha' in new_params:
        val = perturb(new_params['alpha'], 0.05)
        new_params['alpha'] = np.clip(val, 0.8, 3.5)
    if 'beta' in new_params:
        val = perturb(new_params['beta'], 0.05)
        new_params['beta'] = np.clip(val, 0.4, 3.5)
    if 'strength' in new_params:
        val = perturb(new_params['strength'], 0.1)
        new_params['strength'] = np.clip(val, 0.5, 6.0)
    if 'tau' in new_params:
        val = perturb(new_params['tau'], 0.08)
        new_params['tau'] = np.clip(val, 0.01, 2.0)
    if 'gamma' in new_params:
        val = perturb(new_params['gamma'], 0.08)
        new_params['gamma'] = np.clip(val, 0.01, 2.0)
    if 'missing_rate' in new_params:
        val = perturb(new_params['missing_rate'], 0.08)
        new_params['missing_rate'] = np.clip(val, 0.01, 0.6)
    for key in ['rank', 'num_special']:
        if key in new_params:
            if rng.random() < 0.25:
                new_params[key] = int(max(1, new_params[key] + rng.integers(-1, 2)))
    return new_params



def _get_default_params_v2(target_ssdi, mechanism, C=10, K=20, seed=42):
    """
    新版本默认参数生成器（内部使用）
    """
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    s = float(np.clip(target_ssdi, 0.0, 1.0))

    # 基础尾部分布参数（扩大可达范围）
    alpha = 3.5 - 3.2 * (s ** 1.3)
    alpha = np.clip(alpha, 0.3, 6.0)
    beta = 0.3 + 3.5 * (s ** 1.2)
    beta = np.clip(beta, 0.2, 6.0)
    missing_rate = 0.02 + 0.85 * (s ** 1.5)
    missing_rate = np.clip(missing_rate, 0.01, 0.9)

    # LDS 分段增强
    if s <= 0.3:
        lds_scale = 1.0
    elif s <= 0.6:
        lds_scale = 1.8
    else:
        lds_scale = 1.8 + 6.0 * (s - 0.6)

    # 机制参数
    if mechanism == 'lcd_client':
        tau = 0.1 + 3.0 * (s ** 1.4)
        params = {'alpha': alpha, 'beta': beta, 'missing_rate': missing_rate, 'tau': tau}
    elif mechanism == 'lcd_class':
        gamma = 0.1 + 3.0 * (s ** 1.4)
        params = {'alpha': alpha, 'beta': beta, 'missing_rate': missing_rate, 'gamma': gamma}
    elif mechanism == 'lcd_joint':
        a = 0.2 + 2.5 * (s ** 1.3)
        b = 0.2 + 2.5 * (s ** 1.3)
        params = {'alpha': alpha, 'beta': beta, 'missing_rate': missing_rate, 'a': a, 'b': b}
    elif mechanism == 'lds_client':
        strength = (0.5 + 3.0 * s) * lds_scale
        params = {'alpha': alpha, 'beta': beta, 'strength': strength}
    elif mechanism == 'lds_special':
        strength = (0.5 + 3.0 * s) * lds_scale
        num_special = max(1, int((0.05 + 0.8 * s) * K))
        params = {'alpha': alpha, 'beta': beta, 'strength': strength, 'num_special': num_special}
    elif mechanism == 'lds_lowrank':
        strength = (0.5 + 3.0 * s) * lds_scale
        rank = max(1, int((0.05 + 0.9 * s) * min(C, K)))
        params = {'alpha': alpha, 'beta': beta, 'strength': strength, 'rank': rank}
    else:
        raise ValueError(f"未知机制: {mechanism}")

    params = _randomize_params_wide_v2(params, rng, target_ssdi)
    return params


def _randomize_params_wide_v2(params, rng, target_ssdi):
    s = float(np.clip(target_ssdi, 0.0, 1.0))
    new_params = params.copy()
    new_params['_randomized'] = True

    sigma = 0.05 + 0.45 * (s ** 2)

    def perturb(x):
        return float(x * np.exp(rng.normal(0, sigma)))

    for key in ['alpha', 'beta']:
        if key in new_params:
            new_params[key] = np.clip(perturb(new_params[key]), 0.2, 8.0)

    if 'missing_rate' in new_params:
        new_params['missing_rate'] = np.clip(perturb(new_params['missing_rate']), 0.01, 0.95)

    if 'strength' in new_params:
        new_params['strength'] = np.clip(perturb(new_params['strength']), 0.1, 15.0)

    for key in ['tau', 'gamma', 'a', 'b']:
        if key in new_params:
            new_params[key] = np.clip(perturb(new_params[key]), 0.01, 6.0)

    for key in ['rank', 'num_special']:
        if key in new_params:
            jump = rng.integers(-3, 4)
            new_params[key] = int(max(1, new_params[key] + jump))

    return new_params