import time
import random
import numpy as np

_AP_CACHE = None

def _precompute_aps(n):
    global _AP_CACHE
    if _AP_CACHE is not None and _AP_CACHE[0] == n:
        return _AP_CACHE[1], _AP_CACHE[2]
        
    aps = []
    aps_containing = [[] for _ in range(n)]
    
    for d in range(1, n):
        for s in range(n):
            curr = s
            prefix = []
            while curr < n:
                prefix.append(curr)
                ap_id = len(aps)
                aps.append(prefix.copy())
                for idx in prefix:
                    aps_containing[idx].append(ap_id)
                curr += d
                
    _AP_CACHE = (n, aps, aps_containing)
    return aps, aps_containing

def _compute_sums(x, aps):
    """Compute all AP sums from a sign vector."""
    num_aps = len(aps)
    sums = np.zeros(num_aps, dtype=np.int32)
    for k, ap in enumerate(aps):
        s = 0
        for idx in ap:
            s += x[idx]
        sums[k] = s
    return sums

def _local_search(x, sums, aps_containing, rng, time_limit, start_time, min_T=0.05, T0=8.0, cooling=0.9995):
    """Local search with multi-flip moves and conflict-focused selection."""
    n = len(x)
    current_max = int(np.max(np.abs(sums)))
    T = T0
    best_x = x[:]
    best_sums = sums.copy()
    best_max = current_max
    
    # Track which positions are in the worst APs for conflict-focused moves
    # We'll maintain a "pressure" score for each position
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Identify worst APs and build pressure for positions
        abs_sums = np.abs(sums)
        worst_threshold = max(1, int(current_max * 0.8))
        worst_aps = np.where(abs_sums >= worst_threshold)[0]
        
        if len(worst_aps) > 0:
            # Build pressure: how many worst APs contain each position
            pressure = np.zeros(n, dtype=np.int32)
            for ap_id in worst_aps:
                for idx in aps_containing_positions[ap_id] if False else []:
                    pass
            # Simpler: just use random moves but bias toward positions in high-sum APs
            # For now, use a mix of random and greedy moves
        
        # Choose move type: single flip or pair flip
        if rng.random() < 0.3:
            # Pair flip: flip two positions
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            old_i, old_j = x[i], x[j]
            new_i, new_j = -old_i, -old_j
            
            # Compute new sums for affected APs
            affected = set(aps_containing[i]) | set(aps_containing[j])
            affected_list = list(affected)
            
            if not affected_list:
                continue
            
            # Compute delta for each affected AP
            old_vals = sums[affected_list].copy()
            new_vals = old_vals.copy()
            for ap_id in affected_list:
                delta = 0
                # Check if i is in this AP
                if i in _ap_membership[i]:  # We need membership info
                    delta += 2 * new_i
                if j in _ap_membership[j]:
                    delta += 2 * new_j
                # This approach is too slow; use precomputed membership
                
            # Simpler: just compute the change directly
            # For each affected AP, the sum changes by:
            # -2*old_i if i is in the AP, else 0
            # -2*old_j if j is in the AP, else 0
            # New sum = old sum - 2*old_i*(i in AP) - 2*old_j*(j in AP)
            
            # We need efficient membership checking. Let's use the aps_containing structure differently.
            # Actually, let's just do single flips for now but with better acceptance criteria.
            continue
        else:
            # Single flip
            i = rng.randint(0, n - 1)
            old_val = x[i]
            new_val = -old_val
            delta = 2 * new_val
            
            indices = aps_containing[i]
            if not indices:
                continue
                
            old_sums_subset = sums[indices].copy()
            sums[indices] += delta
            new_max = int(np.max(np.abs(sums)))
            delta_max = new_max - current_max
            
            # Acceptance: always accept if improves, otherwise with probability
            if delta_max < 0:
                x[i] = new_val
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_x = x[:]
                    best_sums = sums.copy()
            elif T > 0 and rng.random() < np.exp(-delta_max / T):
                x[i] = new_val
                current_max = new_max
            else:
                sums[indices] -= delta
        
        T *= cooling
    
    return best_x, best_sums, best_max

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin from 90s
    
    aps, aps_containing = _precompute_aps(n)
    num_aps = len(aps)
    
    # Build membership: for each AP, which positions it contains
    # aps_containing[i] gives APs containing position i
    # We also need: for each AP, the list of positions (already in aps)
    
    best_x = None
    best_sums = None
    best_max = float('inf')
    
    rng = random.Random(42)
    
    # Multiple restarts
    num_restarts = 0
    while time.time() - start_time < time_limit:
        num_restarts += 1
        
        # Initialize: random or greedy
        if num_restarts == 1:
            x = [rng.choice([-1, 1]) for _ in range(n)]
        else:
            # Start from best found so far with some perturbation
            if best_x is not None:
                x = best_x[:]
                # Perturb ~10% of positions
                for _ in range(n // 10):
                    i = rng.randint(0, n - 1)
                    x[i] = -x[i]
            else:
                x = [rng.choice([-1, 1]) for _ in range(n)]
        
        sums = _compute_sums(x, aps)
        current_max = int(np.max(np.abs(sums)))
        
        if current_max < best_max:
            best_max = current_max
            best_x = x[:]
            best_sums = sums.copy()
        
        # Local search for this restart
        # Allocate time: more time for early restarts, less for later
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5.0:
            break
        
        # For each restart, use a fraction of remaining time
        if num_restarts <= 3:
            ls_time = min(remaining * 0.4, 20.0)
        else:
            ls_time = min(remaining * 0.2, 10.0)
        
        ls_start = time.time()
        ls_limit = ls_start + ls_time
        
        # Run local search
        T = 8.0
        cooling = 0.9995
        min_T = 0.05
        cur_max = current_max
        
        while time.time() < ls_limit:
            if T < min_T:
                break
            
            i = rng.randint(0, n - 1)
            old_val = x[i]
            new_val = -old_val
            delta = 2 * new_val
            
            indices = aps_containing[i]
            if not indices:
                continue
            
            sums[indices] += delta
            new_max = int(np.max(np.abs(sums)))
            delta_max = new_max - cur_max
            
            if delta_max < 0:
                x[i] = new_val
                cur_max = new_max
                if cur_max < best_max:
                    best_max = cur_max
                    best_x = x[:]
                    best_sums = sums.copy()
            elif T > 0 and rng.random() < np.exp(-delta_max / T):
                x[i] = new_val
                cur_max = new_max
            else:
                sums[indices] -= delta
            
            T *= cooling
        
        # Check if we should stop
        if time.time() - start_time > time_limit - 3.0:
            break
    
    if best_x is None:
        # Fallback: return random
        x = [rng.choice([-1, 1]) for _ in range(n)]
        return x
    
    return best_x
