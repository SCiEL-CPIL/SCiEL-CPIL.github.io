"""Combined multi-start local search with bitmask-based 3-AP checking."""
import time
import random
import math

_N = 256
_KILL = 90.0
_MARGIN = 5.0

# Precompute reciprocals
_RECIP = [0.0] + [1.0 / i for i in range(1, _N + 1)]

# Precompute conflict pairs: for each x, list of (y,z) such that {x,y,z} is a 3-AP
# We store for each x, the pairs (y,z) where y < z and all three form an AP
_conflict_pairs = [[] for _ in range(_N + 1)]

def _precompute_conflicts():
    """Precompute all conflict pairs for 3-AP detection."""
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            # Case 1: a, b, c with b as middle -> c = 2b - a
            c = 2 * b - a
            if 1 <= c <= _N and c > b:
                # AP: a, b, c
                _conflict_pairs[a].append((b, c))
                _conflict_pairs[b].append((a, c))
                _conflict_pairs[c].append((a, b))
            # Case 2: a, c, b with c as middle -> c = (a+b)/2 if even
            if (a + b) % 2 == 0:
                c = (a + b) // 2
                if 1 <= c <= _N and a < c < b:
                    # AP: a, c, b
                    _conflict_pairs[a].append((c, b))
                    _conflict_pairs[c].append((a, b))
                    _conflict_pairs[b].append((a, c))
            # Case 3: c, a, b with a as middle -> c = 2a - b
            c = 2 * a - b
            if 1 <= c <= _N and c < a:
                # AP: c, a, b
                _conflict_pairs[c].append((a, b))
                _conflict_pairs[a].append((c, b))
                _conflict_pairs[b].append((c, a))
    
    # Remove duplicates and sort for consistency
    for i in range(1, _N + 1):
        _conflict_pairs[i] = list(set(_conflict_pairs[i]))
        _conflict_pairs[i].sort()

_precompute_conflicts()

def _can_add(x, S_mask):
    """Check if x can be added to set represented by S_mask without creating a 3-AP."""
    for y, z in _conflict_pairs[x]:
        if (S_mask & (1 << y)) and (S_mask & (1 << z)):
            return False
    return True

def _get_weight(S_mask):
    """Compute the reciprocal sum for a set represented by bitmask."""
    total = 0.0
    for x in range(1, _N + 1):
        if S_mask & (1 << x):
            total += _RECIP[x]
    return total

def _greedy_fill(S_mask, order=None, deadline=None):
    """Fill S by adding elements in the given order (or ascending if None)."""
    if order is None:
        order = list(range(1, _N + 1))
    
    S = S_mask
    for x in order:
        if deadline is not None and time.time() > deadline:
            break
        if S & (1 << x):
            continue
        if _can_add(x, S):
            S |= (1 << x)
    return S

def _greedy_fill_descending(S_mask, deadline=None):
    """Fill S by adding elements in descending order."""
    return _greedy_fill(S_mask, list(range(_N, 0, -1)), deadline)

def _local_search(S_mask, deadline, rng):
    """Run local search to optimize S_mask. Returns best mask found."""
    S = S_mask
    weight = _get_weight(S)
    best_S = S
    best_weight = weight
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Strategy 1: Try 1-swaps (remove x from S, add y not in S)
        # We want 1/y > 1/x, so y < x
        best_swap_gain = 0.0
        best_x = -1
        best_y = -1
        
        # Get list of elements in S
        S_list = [x for x in range(1, _N + 1) if S & (1 << x)]
        
        for x in S_list:
            if time.time() > deadline:
                break
            
            # Try to find best y < x that can replace x
            for y in range(1, x):
                if S & (1 << y):
                    continue
                gain = _RECIP[y] - _RECIP[x]
                if gain <= 1e-15:
                    continue
                
                # Check if removing x and adding y is valid
                S_new = S & ~(1 << x)
                if _can_add(y, S_new):
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_x = x
                        best_y = y
        
        if best_y != -1:
            S = S & ~(1 << best_x)
            S |= (1 << best_y)
            weight += best_swap_gain
            if weight > best_weight:
                best_weight = weight
                best_S = S
            improved = True
            continue
        
        # Strategy 2: Try 2-swaps (remove two from S, add two not in S)
        # This is more expensive but can escape local optima
        best_2swap_gain = 0.0
        best_remove = None
        best_add = None
        
        # Only try if S is large enough
        if len(S_list) >= 2:
            # Sample some pairs to keep it fast
            num_pairs = min(50, len(S_list) * (len(S_list) - 1) // 2)
            rng.shuffle(S_list)
            checked = 0
            for i in range(len(S_list)):
                if time.time() > deadline or checked >= num_pairs:
                    break
                x1 = S_list[i]
                for j in range(i + 1, len(S_list)):
                    if time.time() > deadline or checked >= num_pairs:
                        break
                    x2 = S_list[j]
                    checked += 1
                    
                    # Try to find two elements y1, y2 not in S that can replace x1, x2
                    # We want 1/y1 + 1/y2 > 1/x1 + 1/x2
                    S_new = S & ~(1 << x1) & ~(1 << x2)
                    target_gain = _RECIP[x1] + _RECIP[x2]
                    
                    # Find best pair (y1, y2)
                    best_pair_gain = 0.0
                    best_y1 = -1
                    best_y2 = -1
                    
                    for y1 in range(1, _N + 1):
                        if S_new & (1 << y1):
                            continue
                        if not _can_add(y1, S_new):
                            continue
                        for y2 in range(1, _N + 1):
                            if y2 == y1:
                                continue
                            if S_new & (1 << y2):
                                continue
                            if not _can_add(y2, S_new | (1 << y1)):
                                continue
                            pair_gain = _RECIP[y1] + _RECIP[y2] - target_gain
                            if pair_gain > best_pair_gain:
                                best_pair_gain = pair_gain
                                best_y1 = y1
                                best_y2 = y2
                    
                    if best_pair_gain > best_2swap_gain:
                        best_2swap_gain = best_pair_gain
                        best_remove = (x1, x2)
                        best_add = (best_y1, best_y2)
            
            if best_remove is not None and best_add is not None:
                x1, x2 = best_remove
                y1, y2 = best_add
                S = S & ~(1 << x1) & ~(1 << x2)
                S |= (1 << y1) | (1 << y2)
                weight += best_2swap_gain
                if weight > best_weight:
                    best_weight = weight
                    best_S = S
                improved = True
                continue
        
        # Strategy 3: Try adding any valid element (shouldn't happen if maximal, but check)
        for y in range(1, _N + 1):
            if time.time() > deadline:
                break
            if S & (1 << y):
                continue
            if _can_add(y, S):
                S |= (1 << y)
                weight += _RECIP[y]
                if weight > best_weight:
                    best_weight = weight
                    best_S = S
                improved = True
                break
        
        if not improved:
            # Try a random perturbation: remove 2-3 random elements and re-greedy
            if time.time() > deadline - 2:
                break
            S_pert = S
            drops = rng.randint(2, 3)
            S_list_pert = [x for x in range(1, _N + 1) if S_pert & (1 << x)]
            rng.shuffle(S_list_pert)
            for x in S_list_pert[:drops]:
                S_pert &= ~(1 << x)
            
            # Re-greedy fill
            S_regreedy = _greedy_fill(S_pert, deadline=deadline)
            weight_regreedy = _get_weight(S_regreedy)
            if weight_regreedy > best_weight:
                best_weight = weight_regreedy
                best_S = S_regreedy
                improved = True
    
    return best_S

def construct():
    start = time.time()
    deadline = start + _KILL - _MARGIN
    rng = random.Random(20240517)
    
    best_S = None
    best_w = -1.0
    
    # Generate diverse initial solutions
    inits = []
    
    # 1. Greedy Ascending
    S1 = 0
    for x in range(1, _N + 1):
        if _can_add(x, S1):
            S1 |= (1 << x)
    inits.append(S1)
    
    # 2. Greedy Descending
    S2 = 0
    for x in range(_N, 0, -1):
        if _can_add(x, S2):
            S2 |= (1 << x)
    inits.append(S2)
    
    # 3. Random Greedy (multiple)
    for _ in range(3):
        S_rand = 0
        candidates = list(range(1, _N + 1))
        rng.shuffle(candidates)
        for x in candidates:
            if _can_add(x, S_rand):
                S_rand |= (1 << x)
        inits.append(S_rand)
    
    # 4. Cantor-like set (base 3, digits 0,1 only)
    S_cantor = 0
    for i in range(_N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid and 1 <= i <= _N:
            S_cantor |= (1 << i)
    inits.append(S_cantor)
    
    # 5. Hybrid: Start with small numbers, then greedy fill
    S_hybrid = 0
    for x in range(1, 20):
        if _can_add(x, S_hybrid):
            S_hybrid |= (1 << x)
    S_hybrid = _greedy_fill(S_hybrid, deadline=deadline)
    inits.append(S_hybrid)
    
    # Run Local Search on each initial solution
    for S_init in inits:
        if time.time() > deadline - 15:
            break
        S_opt = _local_search(S_init, deadline, rng)
        w = _get_weight(S_opt)
        if w > best_w:
            best_w = w
            best_S = S_opt
    
    # Additional random restarts from best solution
    if best_S is not None:
        for _ in range(5):
            if time.time() > deadline - 3:
                break
            S_pert = best_S
            drops = rng.randint(3, 5)
            S_list_pert = [x for x in range(1, _N + 1) if S_pert & (1 << x)]
            rng.shuffle(S_list_pert)
            for x in S_list_pert[:drops]:
                S_pert &= ~(1 << x)
            S_opt2 = _local_search(S_pert, deadline, rng)
            w2 = _get_weight(S_opt2)
            if w2 > best_w:
                best_w = w2
                best_S = S_opt2
    
    if not best_S:
        best_S = 1  # Just {1}
    
    # Convert bitmask to sorted list
    res = [x for x in range(1, _N + 1) if best_S & (1 << x)]
    
    # Defensive Verification
    Sset = set(res)
    ok = True
    for i, a in enumerate(res):
        for b in res[i + 1:]:
            c = 2 * b - a
            if c in Sset and c > b:
                ok = False
            c2 = 2 * a - b
            if c2 in Sset and c2 < a:
                ok = False
            if (a + b) % 2 == 0:
                c3 = (a + b) // 2
                if c3 in Sset and c3 != a and c3 != b:
                    ok = False
    if not ok:
        # Fallback to simple greedy
        S = 0
        for x in range(1, _N + 1):
            if _can_add(x, S):
                S |= (1 << x)
        res = [x for x in range(1, _N + 1) if S & (1 << x)]
    
    return res
