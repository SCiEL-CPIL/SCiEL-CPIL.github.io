"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_deadline = time.time() + 130  # Safety margin before 150s kill


def _simulate_bin_packing(items, capacity, score_fn):
    """
    Simulate online bin packing with a given scoring function.
    items: array of item sizes
    capacity: bin capacity
    score_fn: callable(item, remaining_array) -> score_array
    Returns: number of bins used
    """
    bins = []  # list of remaining capacities
    for item in items:
        if not bins:
            bins.append(capacity - item)
            continue
        
        remaining = np.array(bins)
        valid = remaining >= item
        
        if not np.any(valid):
            bins.append(capacity - item)
            continue
        
        # Compute scores for valid bins
        scores = score_fn(item, remaining)
        # Among valid bins, pick the one with highest score
        valid_indices = np.where(valid)[0]
        valid_scores = scores[valid_indices]
        best_idx = valid_indices[np.argmax(valid_scores)]
        bins[best_idx] -= item
    
    return len(bins)


def _l1_lower_bound(items, capacity):
    total = np.sum(items)
    max_item = np.max(items)
    lb1 = int(np.ceil(total / capacity))
    lb2 = int(np.ceil(max_item))
    return max(lb1, lb2)


def _generate_sequences(n_sequences=5, n_items=1000, capacity=100):
    """Generate Weibull sequences for testing."""
    sequences = []
    for seed in range(n_sequences):
        rng = np.random.RandomState(seed)
        items = rng.weibull(2, size=n_items) * 20
        items = items[items <= capacity]
        while len(items) < n_items:
            new_items = rng.weibull(2, size=n_items) * 20
            new_items = new_items[new_items <= capacity]
            items = np.concatenate([items, new_items])[:n_items]
        sequences.append(items)
    return sequences


def _fit_parameters():
    """
    Search over a family of scoring functions and return the best parameters.
    
    We test several families:
    1. score = -(remaining - item)^alpha for various alpha
    2. score = -((remaining - item) - t)^2 for various target t
    3. score = -(remaining - item) + lambda * remaining (linear combo)
    4. Hybrid: score = -a*(remaining - item) - b*(remaining - item)^2
    """
    sequences = _generate_sequences()
    
    best_score = float('inf')
    best_params = None
    
    # Family 1: score = -(remaining - item)^alpha
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    
    for alpha in alphas:
        if time.time() > _deadline - 10:
            break
        total_excess = 0.0
        for seq in sequences:
            if time.time() > _deadline - 10:
                break
            def score_fn(item, remaining, alpha=alpha):
                valid = remaining >= item
                scores = np.full(len(remaining), -np.inf)
                if np.any(valid):
                    scores[valid] = -(remaining[valid] - item) ** alpha
                return scores
            
            bins_used = _simulate_bin_packing(seq, 100, score_fn)
            lb = _l1_lower_bound(seq, 100)
            excess = (bins_used - lb) / lb * 100
            total_excess += excess
        
        mean_excess = total_excess / len(sequences)
        if mean_excess < best_score:
            best_score = mean_excess
            best_params = ('alpha', alpha)
    
    # Family 2: score = -((remaining - item) - t)^2
    targets = [0.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
    for t in targets:
        if time.time() > _deadline - 10:
            break
        total_excess = 0.0
        for seq in sequences:
            if time.time() > _deadline - 10:
                break
            def score_fn(item, remaining, t=t):
                valid = remaining >= item
                scores = np.full(len(remaining), -np.inf)
                if np.any(valid):
                    scores[valid] = -((remaining[valid] - item) - t) ** 2
                return scores
            
            bins_used = _simulate_bin_packing(seq, 100, score_fn)
            lb = _l1_lower_bound(seq, 100)
            excess = (bins_used - lb) / lb * 100
            total_excess += excess
        
        mean_excess = total_excess / len(sequences)
        if mean_excess < best_score:
            best_score = mean_excess
            best_params = ('target', t)
    
    # Family 3: score = -a*(remaining - item) - b*(remaining - item)^2
    # This is a quadratic penalty that can have an interior minimum
    for a in [0.5, 1.0, 2.0]:
        for b in [0.01, 0.05, 0.1, 0.5, 1.0]:
            if time.time() > _deadline - 10:
                break
            total_excess = 0.0
            for seq in sequences:
                if time.time() > _deadline - 10:
                    break
                def score_fn(item, remaining, a=a, b=b):
                    valid = remaining >= item
                    scores = np.full(len(remaining), -np.inf)
                    if np.any(valid):
                        diff = remaining[valid] - item
                        scores[valid] = -a * diff - b * diff ** 2
                    return scores
                
                bins_used = _simulate_bin_packing(seq, 100, score_fn)
                lb = _l1_lower_bound(seq, 100)
                excess = (bins_used - lb) / lb * 100
                total_excess += excess
            
            mean_excess = total_excess / len(sequences)
            if mean_excess < best_score:
                best_score = mean_excess
                best_params = ('quadratic', a, b)
    
    # Family 4: Weighted combination of BF and WF
    # score = w * (item - remaining) + (1-w) * (remaining - item)
    # = w*item - w*remaining + remaining - item - w*remaining + w*item
    # = 2*w*item + (1 - 2*w)*remaining - item
    # = (2*w - 1)*item + (1 - 2*w)*remaining
    # For w > 0.5: coefficient of remaining is negative -> BF-like
    # For w < 0.5: coefficient of remaining is positive -> WF-like
    # For w = 0.5: all scores equal
    # Let's parameterize by c = 1 - 2*w, so score = -c*item + c*remaining = c*(remaining - item)
    # This is just a scaled version of the difference, so it's equivalent to BF or WF
    # Not useful. Let's try a different hybrid.
    
    # Family 5: score = -remaining * (remaining - item)^alpha
    # This penalizes both large remaining and large waste
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        if time.time() > _deadline - 10:
            break
        total_excess = 0.0
        for seq in sequences:
            if time.time() > _deadline - 10:
                break
            def score_fn(item, remaining, alpha=alpha):
                valid = remaining >= item
                scores = np.full(len(remaining), -np.inf)
                if np.any(valid):
                    diff = remaining[valid] - item
                    scores[valid] = -remaining[valid] * diff ** alpha
                return scores
            
            bins_used = _simulate_bin_packing(seq, 100, score_fn)
            lb = _l1_lower_bound(seq, 100)
            excess = (bins_used - lb) / lb * 100
            total_excess += excess
        
        mean_excess = total_excess / len(sequences)
        if mean_excess < best_score:
            best_score = mean_excess
            best_params = ('weighted_diff', alpha)
    
    if best_params is None:
        best_params = ('alpha', 1.0)  # Default to standard BF
    
    return best_params, best_score


# Try fitting at module load time, but with a time guard
_start_time = time.time()

try:
    _BEST_PARAMS, _BEST_SCORE = _fit_parameters()
except Exception:
    _BEST_PARAMS = ('alpha', 1.0)
    _BEST_SCORE = float('inf')

# Build the scoring function based on best params
_param_type = _BEST_PARAMS[0]
if _param_type == 'alpha':
    _ALPHA = _BEST_PARAMS[1]
    _USE_ALPHA = True
    _USE_TARGET = False
    _USE_QUAD = False
    _USE_WEIGHTED = False
elif _param_type == 'target':
    _TARGET = _BEST_PARAMS[1]
    _USE_ALPHA = False
    _USE_TARGET = True
    _USE_QUAD = False
    _USE_WEIGHTED = False
elif _param_type == 'quadratic':
    _QUAD_A = _BEST_PARAMS[1]
    _QUAD_B = _BEST_PARAMS[2]
    _USE_ALPHA = False
    _USE_TARGET = False
    _USE_QUAD = True
    _USE_WEIGHTED = False
elif _param_type == 'weighted_diff':
    _WD_ALPHA = _BEST_PARAMS[1]
    _USE_ALPHA = False
    _USE_TARGET = False
    _USE_QUAD = False
    _USE_WEIGHTED = True
else:
    _ALPHA = 1.0
    _USE_ALPHA = True
    _USE_TARGET = False
    _USE_QUAD = False
    _USE_WEIGHTED = False


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Returns an array of scores; the item goes to the highest-scoring bin.
    """
    bins = np.asarray(bins, dtype=np.float64)
    remaining = bins
    valid_mask = remaining >= item
    
    scores = np.full(len(bins), -np.inf, dtype=np.float64)
    
    if not np.any(valid_mask):
        return scores
    
    if _USE_ALPHA:
        scores[valid_mask] = -(remaining[valid_mask] - item) ** _ALPHA
    elif _USE_TARGET:
        scores[valid_mask] = -((remaining[valid_mask] - item) - _TARGET) ** 2
    elif _USE_QUAD:
        diff = remaining[valid_mask] - item
        scores[valid_mask] = -_QUAD_A * diff - _QUAD_B * diff ** 2
    elif _USE_WEIGHTED:
        diff = remaining[valid_mask] - item
        scores[valid_mask] = -remaining[valid_mask] * diff ** _WD_ALPHA
    
    return scores
# EVOLVE-BLOCK-END
