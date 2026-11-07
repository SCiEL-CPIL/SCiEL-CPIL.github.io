"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import gcd

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))
N_ELEMENTS = NMAX - 1  # 255 elements

# Index mapping: value v -> index i = v - 2
# Bitmask bit i corresponds to value i + 2

def _precompute():
    """
    Precompute:
    1. req[x][b] = x // gcd(x, b). x | b*c iff req[x][b] | c.
    2. mask_req[x][b] = bitmask of all c in [2, 256] such that req[x][b] | c.
    """
    req = {}
    mask_req = {}
    
    # Precompute masks for all possible divisors d (1..256)
    # mask_for_d[d] = bitmask of c in [2,256] such that d | c
    mask_for_d = [0] * (NMAX + 1)
    for d in range(1, NMAX + 1):
        mask = 0
        for c in range(2, NMAX + 1):
            if c % d == 0:
                idx = c - 2
                mask |= (1 << idx)
        mask_for_d[d] = mask
        
    # Now fill req and mask_req
    for x in range(2, NMAX + 1):
        req[x] = {}
        mask_req[x] = {}
        for b in range(2, NMAX + 1):
            if x == b:
                continue
            d = x // gcd(x, b)
            req[x][b] = d
            mask_req[x][b] = mask_for_d[d]
            
    return req, mask_req

def _get_conflict_degrees(S_mask, req, mask_req):
    """
    For each x in S, compute the number of pairs (b, c) in S, b != c, b,c != x
    such that x | b*c.
    Returns a dictionary {x: degree}
    """
    degrees = {}
    S_list = [i + 2 for i in range(N_ELEMENTS) if (S_mask >> i) & 1]
    
    for x in S_list:
        deg = 0
        x_idx = x - 2
        for b in S_list:
            if b == x:
                continue
            b_idx = b - 2
            valid_c_mask = S_mask & mask_req[x][b]
            # Exclude c=b and c=x
            valid_c_mask &= ~(1 << b_idx)
            valid_c_mask &= ~(1 << x_idx)
            if valid_c_mask:
                deg += valid_c_mask.bit_count()
                
        degrees[x] = deg
        
    return degrees

def _greedy_remove_bitmask(req, mask_req, deadline, tie_break='random'):
    """
    Start with all numbers. Iteratively remove the element with the highest conflict degree.
    tie_break: 'random', 'smallest', 'largest', 'composite_first'
    """
    S_mask = (1 << N_ELEMENTS) - 1  # All bits set
    removed = set()
    rng = random.Random(12345)
    
    def is_composite(n):
        if n < 4:
            return n > 2
        if n % 2 == 0 or n % 3 == 0:
            return True
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return True
            i += 6
        return False
    
    while time.time() < deadline:
        # Get degrees
        degrees = _get_conflict_degrees(S_mask, req, mask_req)
        
        # Find max degree
        max_deg = 0
        candidates = []
        for x, deg in degrees.items():
            if deg > max_deg:
                max_deg = deg
                candidates = [x]
            elif deg == max_deg and deg > 0:
                candidates.append(x)
                
        if max_deg == 0:
            break
            
        # Tie-breaking
        if len(candidates) > 1:
            if tie_break == 'random':
                max_x = rng.choice(candidates)
            elif tie_break == 'smallest':
                max_x = min(candidates)
            elif tie_break == 'largest':
                max_x = max(candidates)
            elif tie_break == 'composite_first':
                composites = [x for x in candidates if is_composite(x)]
                if composites:
                    max_x = rng.choice(composites)
                else:
                    max_x = rng.choice(candidates)
            else:
                max_x = rng.choice(candidates)
        else:
            max_x = candidates[0]
            
        # Remove max_x
        x_idx = max_x - 2
        S_mask &= ~(1 << x_idx)
        removed.add(max_x)
        
    # Convert back to list
    S = [i + 2 for i in range(N_ELEMENTS) if (S_mask >> i) & 1]
    return S

def _valid_addition_fast(x, S_list, req):
    """Check if x can be added to set S_list without violating the constraint."""
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            # x | b*c ?
            d = req[x][b]
            if c % d == 0:
                return False
            # b | x*c ?
            d2 = req[b][x]
            if c % d2 == 0:
                return False
            # c | x*b ?
            d3 = req[c][x]
            if b % d3 == 0:
                return False
    return True

def _local_search(S, req, deadline, max_iters=2000, seed=99999):
    """Local search: try adding, removing, swapping elements."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)
    
    for _it in range(max_iters):
        if time.time() > deadline:
            break
        
        op = rng.random()
        
        if op < 0.4 and len(S) < NMAX:
            # Try adding
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                rng.shuffle(candidates)
                added = False
                for x in candidates[:30]:
                    if _valid_addition_fast(x, S, req):
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        added = True
                        break
                if added:
                    continue
        
        elif op < 0.7 and len(S) > 2:
            # Try removing one and adding one (swap)
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                rng.shuffle(candidates)
                improved = False
                for x in candidates[:40]:
                    if _valid_addition_fast(x, S_temp, req):
                        S = S_temp + [x]
                        S_set = set(S)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        improved = True
                        break
                if not improved:
                    S = S_temp + [removed]
                    S_set = set(S)
        
        else:
            # Try removing a random element to allow re-growth
            if S:
                i = rng.randrange(len(S))
                removed = S[i]
                S.pop(i)
                S_set.discard(removed)
                # Try to add back something better
                candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
                if candidates:
                    rng.shuffle(candidates)
                    for x in candidates[:30]:
                        if _valid_addition_fast(x, S, req):
                            S.append(x)
                            S_set.add(x)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
                            break
    
    return best

def _local_search_aggressive(S, req, deadline, max_iters=5000, seed=99999):
    """More aggressive local search with higher swap probability and larger candidate pools."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)
    no_improve_count = 0
    patience = 500
    
    for _it in range(max_iters):
        if time.time() > deadline:
            break
        
        op = rng.random()
        
        if op < 0.3 and len(S) < NMAX:
            # Try adding - larger candidate pool
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                rng.shuffle(candidates)
                added = False
                for x in candidates[:50]:
                    if _valid_addition_fast(x, S, req):
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                            no_improve_count = 0
                        added = True
                        break
                if added:
                    continue
        
        elif op < 0.8 and len(S) > 2:
            # Try removing one and adding one (swap) - larger candidate pool
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                rng.shuffle(candidates)
                improved = False
                for x in candidates[:60]:
                    if _valid_addition_fast(x, S_temp, req):
                        S = S_temp + [x]
                        S_set = set(S)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                            no_improve_count = 0
                        improved = True
                        break
                if not improved:
                    S = S_temp + [removed]
                    S_set = set(S)
                    no_improve_count += 1
        
        else:
            # Try removing a random element to allow re-growth
            if S:
                i = rng.randrange(len(S))
                removed = S[i]
                S.pop(i)
                S_set.discard(removed)
                # Try to add back something better
                candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
                if candidates:
                    rng.shuffle(candidates)
                    for x in candidates[:40]:
                        if _valid_addition_fast(x, S, req):
                            S.append(x)
                            S_set.add(x)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
                                no_improve_count = 0
                            break
                no_improve_count += 1
        
        # If stuck for too long, reset to best and perturb
        if no_improve_count > patience:
            S = best[:]
            S_set = set(S)
            no_improve_count = 0
            # Perturb: remove 2-3 random elements
            perturb_size = rng.randint(2, 3)
            for _ in range(perturb_size):
                if S:
                    i = rng.randrange(len(S))
                    S_set.discard(S[i])
                    S.pop(i)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    req, mask_req = _precompute()
    
    best = None
    best_size = 0
    
    def consider(c):
        nonlocal best, best_size
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # 1. Try multiple greedy removal strategies with different tie-breaking
    tie_breaks = ['random', 'smallest', 'largest', 'composite_first']
    for tb in tie_breaks:
        if time.time() > deadline - 30:
            break
        S_remove = _greedy_remove_bitmask(req, mask_req, deadline - 30, tie_break=tb)
        consider(S_remove)
    
    # 2. Local search refinement on the best result
    if best and time.time() < deadline - 20:
        for seed in [42, 123, 999, 1234, 5678, 9999]:
            if time.time() > deadline - 15:
                break
            S_ls = _local_search(best, req, deadline - 15, max_iters=800, seed=seed)
            consider(S_ls)
    
    # 3. Aggressive local search
    if best and time.time() < deadline - 10:
        S_aggr = _local_search_aggressive(best, req, deadline - 10, max_iters=3000, seed=77777)
        consider(S_aggr)
    
    # 4. Try standard greedy add (largest first) as a baseline/comparison
    if time.time() < deadline - 10:
        chosen = []
        chosen_set = set()
        for x in range(NMAX, 1, -1):
            if time.time() > deadline - 5:
                break
            if _valid_addition_fast(x, chosen, req):
                chosen.append(x)
                chosen_set.add(x)
        consider(chosen)
        
    # 5. Try standard greedy add (smallest first)
    if time.time() < deadline - 10:
        chosen = []
        chosen_set = set()
        for x in range(2, NMAX + 1):
            if time.time() > deadline - 5:
                break
            if _valid_addition_fast(x, chosen, req):
                chosen.append(x)
                chosen_set.add(x)
        consider(chosen)
        
    # 6. Final polish: try to add any remaining valid elements to the best set
    if best and time.time() < deadline - 2:
        S = list(best)
        S_set = set(S)
        for x in range(2, NMAX + 1):
            if x not in S_set:
                if _valid_addition_fast(x, S, req):
                    S.append(x)
                    S_set.add(x)
        consider(S)
        
    # 7. Another round of local search on the final best
    if best and time.time() < deadline - 1:
        S_final = _local_search(best, req, deadline - 1, max_iters=500, seed=424242)
        consider(S_final)
        
    if best is None:
        # Fallback: all primes
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        best = [x for x in range(2, NMAX + 1) if is_prime(x)]
        
    return [int(x) for x in best]
