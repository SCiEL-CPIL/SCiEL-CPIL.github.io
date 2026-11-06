import numpy as np
import time as _time_module
from collections import deque

_START_TIME = _time_module.perf_counter()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

# Online learning state
_experience = deque(maxlen=4096)
_model = None  # learned weights (dict of feature_name -> weight)
_model_bias = 0.0
_last_fit_time = 0.0
_last_fit_count = 0

def _elapsed():
    return _time_module.perf_counter() - _START_TIME

def _fallback_dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    best_index = 0
    best_score = -float('inf')
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if machine_options:
            min_duration = min(machine_options.values())
            if min_duration <= 0:
                min_duration = 1e-9
        else:
            min_duration = 1.0
        remaining_ops = op.get("remaining_ops", 1)
        if remaining_ops <= 0:
            remaining_ops = 1
        remaining_work = op.get("remaining_work", 0)
        if remaining_work is None:
            remaining_work = 0.0
        # Static heuristic: urgency * speed, tie-break by lower remaining work
        score = remaining_ops / min_duration - 0.1 * remaining_work
        if score > best_score:
            best_score = score
            best_index = i
    return int(best_index)

def _extract_features(ready_operations, time, op):
    """Extract a fixed-size feature vector for one ready operation."""
    machine_options = op.get("machine_options", {})
    durations = list(machine_options.values()) if machine_options else [1.0]
    min_duration = min(durations) if durations else 1.0
    if min_duration <= 0:
        min_duration = 1e-9
    max_duration = max(durations) if durations else 1.0
    mean_duration = float(np.mean(durations)) if durations else 1.0
    remaining_ops = op.get("remaining_ops", 1)
    if remaining_ops <= 0:
        remaining_ops = 1
    remaining_work = op.get("remaining_work", 0)
    if remaining_work is None:
        remaining_work = 0.0
    job = op.get("job", 0)
    op_idx = op.get("op", 0)
    n_ready = len(ready_operations)
    # Build a normalized feature vector
    features = {
        "remaining_ops": float(remaining_ops),
        "min_duration": float(min_duration),
        "max_duration": float(max_duration),
        "mean_duration": float(mean_duration),
        "remaining_work": float(remaining_work),
        "job": float(job),
        "op_idx": float(op_idx),
        "n_ready": float(n_ready),
        "time": float(time) if time is not None else 0.0,
        "urgency": float(remaining_ops) / max(min_duration, 1e-9),
        "speed": 1.0 / max(min_duration, 1e-9),
    }
    return features

def _score_operation(features, model, bias):
    """Score a single operation using the learned model."""
    if model is None:
        # Fallback to static heuristic
        return features["urgency"] - 0.1 * features["remaining_work"]
    score = bias
    for name, weight in model.items():
        score += weight * features.get(name, 0.0)
    return score

def _fit_model():
    """Fit a simple linear model on the experience buffer."""
    global _model, _model_bias, _last_fit_time, _last_fit_count
    if len(_experience) < 16:
        return
    X = []
    y = []
    feature_names = list(next(iter(_experience))["features"].keys())
    for sample in _experience:
        feat = sample["features"]
        row = [feat.get(name, 0.0) for name in feature_names]
        X.append(row)
        y.append(sample["reward"])
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    if X.shape[0] < 16:
        return
    # Simple Ridge regression: (X^T X + lambda I)^-1 X^T y
    lambda_reg = 1e-3
    XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
    Xty = X.T @ y
    try:
        weights = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    _model = {name: float(w) for name, w in zip(feature_names, weights)}
    _model_bias = float(np.mean(y) - float(np.mean(X) @ weights))
    _last_fit_time = _time_module.perf_counter()
    _last_fit_count += 1

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0

    # Deadline guard: if close to limit, use fast fallback
    if _time_module.perf_counter() > _DEADLINE - 2.0:
        return _fallback_dispatch(ready_operations, time)

    # Periodic model fitting (only if we have enough experience and spare time)
    global _model, _model_bias, _last_fit_time, _last_fit_count
    if _time_module.perf_counter() - _last_fit_time > 0.5 and len(_experience) >= 16:
        if _time_module.perf_counter() < _DEADLINE - 5.0:  # only fit if we have >5s left
            _fit_model()

    # Score each ready operation
    best_index = 0
    best_score = -float('inf')
    best_features = None

    for i, op in enumerate(ready_operations):
        features = _extract_features(ready_operations, time, op)
        score = _score_operation(features, _model, _model_bias)
        if score > best_score:
            best_score = score
            best_index = i
            best_features = features

    # Record experience: reward is a proxy for local makespan impact.
    # We use a negative of the chosen operation's "urgency" as a simple reward signal:
    # higher urgency (more remaining ops / shorter duration) is better.
    # This is a heuristic reward; the model will learn to adjust.
    if best_features is not None:
        reward = -best_features["urgency"]  # negative urgency = lower is better
        _experience.append({
            "features": best_features,
            "reward": reward,
            "time": time,
            "n_ready": len(ready_operations),
        })

    return int(best_index)
