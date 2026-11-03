import numpy as np
import time
import random

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)."""
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _ratio_from_support(support):
    """Compute ratio for equal-weight support. support is a set or list of indices."""
    k = len(support)
    if k == 0:
        return 0.0
    h = np.zeros(N)
    for idx in support:
        h[idx] = 1.0 / k
    return _compute_ratio(h)

def _autocorr_peak(support):
    """Return the maximum autocorrelation count for a given support set.
    For equal weights 1/k, sup|h*h| = max_lag_count / k^2.
    So ratio = (1/k) / (max_lag_count / k^2) = k / max_lag_count.
    We want to maximize k / max_lag_count, equivalently minimize max_lag_count for fixed k,
    or maximize k/max_lag_count overall.
    """
    # Count pairs (i,j) in support with i-j = lag for each lag
    # lag ranges from -(N-1) to (N-1)
    # For a given lag d, count = |{i in support : i+d in support}|
    # We want max over d != 0 of count, but d=0 gives count=k (the diagonal)
    # Actually sup|h*h| includes the diagonal (lag 0) which is sum h_i^2 = k*(1/k)^2 = 1/k
    # Wait: conv[0] = sum_i h_i^2 = k * (1/k)^2 = 1/k
    # conv[lag] for lag != 0: for equal weights, conv[lag] = (1/k^2) * |{i: i in S, i+lag in S}|
    # So sup = max(1/k, max_{lag!=0} count(lag)/k^2)
    # Since 1/k = k/k^2, and count(lag) <= k, we have 1/k >= count(lag)/k^2 iff k >= count(lag)
    # count(lag) <= k always, so 1/k >= count(lag)/k^2 always. So sup is always at lag 0!
    # Wait, that means sup|h*h| = 1/k for equal weights always.
    # Then ratio = (1/k) / (1 * 1/k) = 1. That can't be right.
    # Let me recheck: ||h||_2^2 = sum h_i^2 = k*(1/k)^2 = 1/k
    # ||h||_1 = 1
    # sup|h*h|: conv[0] = sum h_i^2 = 1/k. conv[lag] for lag>0 = sum_i h_i h_{i+lag}
    # For equal weights, conv[lag] = (1/k^2) * |{i in S : i+lag in S}|
    # The max of conv is max(1/k, max_{lag!=0} count(lag)/k^2)
    # 1/k vs count(lag)/k^2: 1/k = k/k^2. So 1/k > count(lag)/k^2 iff k > count(lag).
    # count(lag) can be at most k (if lag=0). For lag != 0, count(lag) <= k-1 typically.
    # So 1/k is always >= count(lag)/k^2 for lag != 0.
    # Therefore sup|h*h| = 1/k always for equal weights.
    # Then ratio = (1/k)/(1 * 1/k) = 1.
    # But the seed gets 1.14, so something is wrong.
    # Ah, I see: the heights don't have to be equal! We can choose different weights.
    # With non-equal weights, we can potentially reduce the off-diagonal peaks while keeping diagonal high.
    # But the ratio formula: ||h||_2^2 / (||h||_1 * sup|h*h|)
    # If we make one element large and others small, ||h||_2^2 is dominated by the large one,
    # and sup|h*h| is also dominated by the large one squared, so ratio ~ 1.
    # Hmm, let me just compute directly.
    pass

def _compute_ratio_direct(support, weights):
    """Compute ratio given support indices and their weights."""
    h = np.zeros(N)
    for idx, w in zip(support, weights):
        h[idx] = w
    return _compute_ratio(h)

def _optimize_weights_for_support(support, deadline, max_iters=5000):
    """Optimize weights for a fixed support to maximize the ratio.
    Use gradient-based or coordinate descent.
    """
    k = len(support)
    if k == 0:
        return np.zeros(k), 0.0
    
    # Start with equal weights
    w = np.ones(k) / k
    
    best_w = w.copy()
    best_r = _compute_ratio_direct(support, w)
    
    # Coordinate descent: adjust each weight
    lr = 0.1
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
        
        h = np.zeros(N)
        for idx, wi in zip(support, w):
            h[idx] = wi
        
        conv = np.convolve(h, h, mode='full')
        sup_conv = np.max(conv)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        
        if l1 < 1e-15 or sup_conv < 1e-15:
            break
        
        # Gradient of log(ratio) = log(l2_sq) - log(l1) - log(sup_conv)
        # d log(l2_sq)/d h_i = 2 h_i / l2_sq
        # d log(l1)/d h_i = 1/l1
        # d log(sup_conv)/d h_i: need gradient of sup conv
        # Use the lag that achieves sup
        k_lag = np.argmax(conv)
        grad_conv = np.zeros(N)
        for i in range(N):
            idx1 = k_lag - i
            idx2 = i - k_lag
            if 0 <= idx1 < N:
                grad_conv[i] += h[idx1]
            if 0 <= idx2 < N:
                grad_conv[i] += h[idx2]
        
        # Full gradient
        grad = 2.0 * h / l2_sq - 1.0 / l1 - grad_conv / conv[k_lag]
        
        # Update weights for support indices only
        improved = False
        for j, idx in enumerate(support):
            new_wj = w[j] + lr * grad[idx]
            if new_wj < 0:
                new_wj = 0.0
            # Try this update
            w_trial = w.copy()
            w_trial[j] = new_wj
            r_trial = _compute_ratio_direct(support, w_trial)
            if r_trial > best_r:
                w = w_trial
                best_r = r_trial
                best_w = w.copy()
                improved = True
        
        if not improved:
            lr *= 0.95
            if lr < 1e-6:
                break
    
    return best_w, best_r

def _simulated_annealing_support(deadline):
    """Simulated annealing over support sets.
    State: a set of indices (support).
    Moves: add a random index, remove a random index, swap an index.
    Objective: maximize ratio (with optimal weights for that support, or just equal weights for speed).
    """
    # For speed, use equal weights during annealing, then optimize weights at the end.
    
    # Initialize with a random sparse set
    k0 = random.randint(5, 20)
    support = set(random.sample(range(N), k0))
    
    # Also try some structured starts
    starts = [
        set(range(0, N, N // k0)),  # evenly spaced
        set(random.sample(range(N), k0)),
    ]
    
    best_support = None
    best_ratio = 0.0
    
    for start_support in starts:
        support = set(start_support)
        current_ratio = _ratio_from_support(support)
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_support = set(support)
        
        # Annealing
        T = 0.5
        T_min = 1e-4
        cooling = 0.999
        
        while time.time() < deadline - 10.0:
            k = len(support)
            
            # Choose a move
            move_type = random.random()
            if move_type < 0.3 and k < 45:
                # Add
                new_idx = random.randint(0, N-1)
                if new_idx not in support:
                    support.add(new_idx)
            elif move_type < 0.6 and k > 3:
                # Remove
                to_remove = random.choice(list(support))
                support.remove(to_remove)
            else:
                # Swap: remove one, add another
                if k > 0:
                    to_remove = random.choice(list(support))
                    support.remove(to_remove)
                    new_idx = random.randint(0, N-1)
                    if new_idx not in support:
                        support.add(new_idx)
            
            new_ratio = _ratio_from_support(support)
            delta = new_ratio - current_ratio
            
            if delta > 0 or random.random() < np.exp(delta / T):
                current_ratio = new_ratio
                if current_ratio > best_ratio:
                    best_ratio = current_ratio
                    best_support = set(support)
            else:
                # Revert
                if move_type < 0.3:
                    support.discard(new_idx)
                elif move_type < 0.6:
                    support.add(to_remove)
                else:
                    support.discard(new_idx)
                    support.add(to_remove)
            
            T = max(T_min, T * cooling)
            
            if T <= T_min:
                break
    
    return best_support, best_ratio

def _local_search_support(support, deadline):
    """Local search: try all single add/remove/swap moves and take the best improvement.
    Repeat until no improvement.
    """
    support = set(support)
    current_ratio = _ratio_from_support(support)
    best_ratio = current_ratio
    best_support = set(support)
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        best_move = None
        best_move_ratio = current_ratio
        
        # Try all removals
        for idx in list(support):
            if len(support) <= 3:
                continue
            new_support = support - {idx}
            r = _ratio_from_support(new_support)
            if r > best_move_ratio:
                best_move_ratio = r
                best_move = ('remove', idx)
        
        # Try all additions
        for idx in range(N):
            if idx in support:
                continue
            new_support = support | {idx}
            r = _ratio_from_support(new_support)
            if r > best_move_ratio:
                best_move_ratio = r
                best_move = ('add', idx)
        
        # Try all swaps
        for idx in list(support):
            for new_idx in range(N):
                if new_idx in support or new_idx == idx:
                    continue
                new_support = (support - {idx}) | {new_idx}
                r = _ratio_from_support(new_support)
                if r > best_move_ratio:
                    best_move_ratio = r
                    best_move = ('swap', idx, new_idx)
        
        if best_move is not None:
            if best_move[0] == 'remove':
                support.discard(best_move[1])
            elif best_move[0] == 'add':
                support.add(best_move[1])
            elif best_move[0] == 'swap':
                support.discard(best_move[1])
                support.add(best_move[2])
            
            current_ratio = best_move_ratio
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_support = set(support)
            improved = True
    
    return best_support, best_ratio

def _build_initial_supports():
    """Generate diverse initial support sets."""
    supports = []
    
    # Equal weights, various sparsities
    for k in [5, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]:
        # Evenly spaced
        step = N // k
        support = set(i * step for i in range(k) if i * step < N)
        if len(support) > 0:
            supports.append(support)
        
        # Random
        for _ in range(3):
            support = set(random.sample(range(N), min(k, N)))
            supports.append(support)
    
    # Sidon-like: choose indices with distinct differences
    # Greedy: add index that minimizes max autocorrelation
    support = set()
    for i in range(N):
        # Check if adding i increases the max autocorrelation
        new_support = support | {i}
        # Quick check: count pairs with difference
        counts = {}
        for a in new_support:
            for b in new_support:
                d = a - b
                if d != 0:
                    counts[d] = counts.get(d, 0) + 1
        max_count = max(counts.values()) if counts else 0
        # Accept if max_count is small
        if max_count <= len(support) + 1:  # loose condition
            support.add(i)
        if len(support) >= 20:
            break
    if len(support) > 0:
        supports.append(support)
    
    return supports

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    random.seed(42)
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Phase 1: Generate initial supports and evaluate
    initial_supports = _build_initial_supports()
    
    for support in initial_supports:
        r = _ratio_from_support(support)
        if r > best_ratio:
            best_ratio = r
            h = np.zeros(N)
            k = len(support)
            for idx in support:
                h[idx] = 1.0 / k
            best_h = h
    
    # Phase 2: Simulated annealing on the best support found
    if time.time() < deadline - 30.0:
        # Reconstruct best support from best_h
        best_support = set(i for i in range(N) if best_h[i] > 1e-15)
        if len(best_support) == 0:
            best_support = set(random.sample(range(N), 10))
        
        sa_support, sa_ratio = _simulated_annealing_support(best_support, deadline)
        if sa_ratio > best_ratio:
            best_ratio = sa_ratio
            h = np.zeros(N)
            k = len(sa_support)
            for idx in sa_support:
                h[idx] = 1.0 / k
            best_h = h
            best_support = sa_support
        else:
            best_support = best_support  # keep original
    
    # Phase 3: Local search on the best support
    if time.time() < deadline - 20.0:
        ls_support, ls_ratio = _local_search_support(best_support, deadline)
        if ls_ratio > best_ratio:
            best_ratio = ls_ratio
            h = np.zeros(N)
            k = len(ls_support)
            for idx in ls_support:
                h[idx] = 1.0 / k
            best_h = h
            best_support = ls_support
    
    # Phase 4: Optimize weights for the best support
    if time.time() < deadline - 10.0:
        best_support_list = list(best_support)
        if len(best_support_list) > 0:
            opt_weights, opt_ratio = _optimize_weights_for_support(best_support_list, deadline)
            if opt_ratio > best_ratio:
                best_ratio = opt_ratio
                h = np.zeros(N)
                for idx, w in zip(best_support_list, opt_weights):
                    h[idx] = w
                best_h = h
    
    # Phase 5: Final local search with optimized weights
    if time.time() < deadline - 5.0:
        # Extract support from best_h
        final_support = set(i for i in range(N) if best_h[i] > 1e-15)
        if len(final_support) > 0:
            # Try local search on support with weight optimization
            ls2_support, ls2_ratio = _local_search_support(final_support, deadline)
            if ls2_ratio > best_ratio:
                best_ratio = ls2_ratio
                h = np.zeros(N)
                k = len(ls2_support)
                for idx in ls2_support:
                    h[idx] = 1.0 / k
                best_h = h
                final_support = ls2_support
            
            # Optimize weights again
            if len(final_support) > 0:
                final_support_list = list(final_support)
                opt_weights2, opt_ratio2 = _optimize_weights_for_support(final_support_list, deadline)
                if opt_ratio2 > best_ratio:
                    best_ratio = opt_ratio2
                    h = np.zeros(N)
                    for idx, w in zip(final_support_list, opt_weights2):
                        h[idx] = w
                    best_h = h
    
    # Ensure nonnegativity and return
    best_h = np.maximum(best_h, 0.0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N) / N
    
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
