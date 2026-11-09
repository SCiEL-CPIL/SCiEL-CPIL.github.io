"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX = 48

# Precompute: for each a in 1..192, for each x in 1..192 (x != a),
# precompute the set of pairs (b,c) with b,c != a,x, b<c, such that a | (x + b + c)
# i.e., (b+c) % a == (-x) % a
# Store as: conflict_pairs[a][x] = set of frozensets or just a bitmask of pairs

# Better: use bitmasks. For each a and x, store a bitmask over all possible (b,c) pairs.
# But C(192,2) ~ 18k pairs, so a bitmask would be ~18k bits = ~2.3KB per (a,x) pair.
# 192*192 = 36864 pairs * 2.3KB = ~85MB. Too much.

# Alternative: for each a, precompute pair_sums_mod[a] = dict mapping (b,c) -> (b+c)%a
# Then for check: given selected set S and candidate x:
#   For each a in S: need to check if there exist b,c in S\{a} with (b+c)%a == (-x)%a
#   We can precompute for each a: for each residue r, the list of pairs with that residue.

# Fastest approach: precompute for each a a list of (residue, [pairs]) and use set intersection.

# Let me use a different strategy: precompute for each a, a dict:
#   pair_res[a] = { (b,c): (b+c)%a } for all b<c, b,c != a
# Then for checking, iterate over pairs in S\{a} and check residue.

# Actually, the fastest is to precompute for each a:
#   by_res[a][r] = set of (b,c) pairs where (b+c)%a == r
# Then for check 2: for each a in S, get target = (-x)%a, then check if any pair in by_res[a][target] 
# is a subset of S\{a}.

# To make this fast, represent S as a bitmask (192 bits = 3 uint64s or use Python int as bitmask).
# For each pair (b,c), precompute a bitmask. Then check (pair_mask & S_mask) == pair_mask.

# Let's use Python integers as bitmasks (arbitrary precision).

def _precompute():
    """Precompute conflict data structures."""
    # pair_res[a][r] = list of (b, c, bitmask) where b<c, b,c != a, (b+c)%a == r
    pair_res = {}
    for a in range(1, _N + 1):
        pair_res[a] = {}
        for b in range(1, _N + 1):
            if b == a:
                continue
            for c in range(b + 1, _N + 1):
                if c == a:
                    continue
                r = (b + c) % a
                bm = (1 << (b-1)) | (1 << (c-1))
                if r not in pair_res[a]:
                    pair_res[a][r] = []
                pair_res[a][r].append((b, c, bm))
    
    # For check 1: x divides sum of 3 elements in S.
    # Precompute for each x: list of triplets (i,j,k) with i<j<k, all != x, such that x | (i+j+k)
    # C(192,3) = 1,172,960. For each x, store bitmask of each triplet.
    # 192 * (number of valid triplets for x) bitmasks. Each triplet has 3 elements.
    # Total triplets across all x: for each triplet (i,j,k), it's divisible by some x values.
    # This could be large. Let's just compute on the fly for check 1 since S is small.
    
    return pair_res

_pair_res = None

def _get_pair_res():
    global _pair_res
    if _pair_res is None:
        _pair_res = _precompute()
    return _pair_res

def _can_add(x, selected_mask, selected_list):
    """Check if x can be added to selected without violation.
    selected_mask: bitmask of selected elements
    selected_list: list of selected elements (for iteration)
    Returns True if x can be added."""
    pr = _get_pair_res()
    n = len(selected_list)
    
    # Check 1: x divides sum of any 3 distinct elements in selected
    if n >= 3:
        for i in range(n):
            si = selected_list[i]
            for j in range(i+1, n):
                sj = selected_list[j]
                s_ij = si + sj
                for k in range(j+1, n):
                    if (s_ij + selected_list[k]) % x == 0:
                        return False
    
    # Check 2: For each a in selected, a | (x + b + c) for some b,c in selected\{a}
    x_bit = 1 << (x - 1)
    for a in selected_list:
        target_res = (-x) % a
        if target_res in pr[a]:
            for (b, c, bm) in pr[a][target_res]:
                # Check if b and c are both in selected and b != a, c != a
                # Since b,c != a by construction, just check if b,c in selected
                if (bm & selected_mask) == bm:
                    return False
    
    return True

def _is_valid(selected):
    """Check if the set is valid."""
    for a in selected:
        others = [v for v in selected if v != a]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy_lookahead(time_limit=5.0):
    """Greedy with lookahead: pick element that minimizes future conflicts."""
    start = time.time()
    selected = []
    selected_set = set()
    selected_mask = 0
    remaining = set(range(1, _N + 1))
    
    while remaining and len(selected) < _MAX:
        if time.time() - start > time_limit:
            break
        
        best_x = None
        best_score = float('inf')
        
        for x in remaining:
            if not _can_add(x, selected_mask, selected):
                continue
            
            # Score: how many other remaining elements would still be addable after adding x
            # Approximate: count elements y in remaining\{x} that can be added to selected ∪ {x}
            # This is expensive, so use a cheaper heuristic: prefer larger x (less likely to divide sums)
            # Or: count conflicts with remaining elements
            score = 0
            for y in remaining:
                if y == x:
                    continue
                # Check if y would conflict with x given current selected
                # This is expensive; use simpler heuristic
                pass
            # Use size as tiebreaker: larger elements are generally safer
            score = -x  # prefer larger
            
            if score < best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            break
        
        selected.append(best_x)
        selected_set.add(best_x)
        selected_mask |= (1 << (best_x - 1))
        remaining.remove(best_x)
    
    return selected

def _greedy_random(order_seed):
    """Greedy with a specific random ordering."""
    rng = random.Random(order_seed)
    selected = []
    selected_set = set()
    selected_mask = 0
    candidates = list(range(1, _N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected_mask, selected):
            selected.append(x)
            selected_set.add(x)
            selected_mask |= (1 << (x - 1))
    
    return selected

def _local_search(selected, time_limit=10.0):
    """Local search: try adding, swapping 1-for-1, and swapping 2-for-2."""
    if not _is_valid(selected):
        return selected
    
    start = time.time()
    best = list(selected)
    best_set = set(best)
    best_mask = 0
    for v in best:
        best_mask |= (1 << (v - 1))
    
    while time.time() - start < time_limit:
        improved = False
        candidates = [x for x in range(1, _N + 1) if x not in best_set]
        if not candidates:
            break
        
        # Try adding
        for x in candidates:
            if _can_add(x, best_mask, best):
                best.append(x)
                best_set.add(x)
                best_mask |= (1 << (x - 1))
                improved = True
                break
        
        if not improved and len(best) >= 1:
            # Try 1-for-1 swap
            swap_done = False
            for i in range(len(best)):
                if time.time() - start > time_limit:
                    break
                removed = best[i]
                temp_mask = best_mask & ~(1 << (removed - 1))
                temp_list = best[:i] + best[i+1:]
                
                for x in candidates:
                    if x in best_set:
                        continue
                    if _can_add(x, temp_mask, temp_list):
                        best = temp_list + [x]
                        best_set = best_set - {removed} | {x}
                        best_mask = temp_mask | (1 << (x - 1))
                        swap_done = True
                        break
                
                if swap_done:
                    break
        
        if not improved and not swap_done:
            # Try 2-for-1 or 1-for-2 swaps (remove 2, add 1 or remove 1, add 2)
            found = False
            if len(best) >= 2:
                # Remove 2, try to add 2 different elements
                for i in range(len(best)):
                    if time.time() - start > time_limit:
                        break
                    for j in range(i+1, len(best)):
                        r1, r2 = best[i], best[j]
                        temp_mask = best_mask & ~(1 << (r1 - 1)) & ~(1 << (r2 - 1))
                        temp_list = [v for k, v in enumerate(best) if k != i and k != j]
                        
                        added_any = False
                        for x in candidates:
                            if x in best_set:
                                continue
                            if _can_add(x, temp_mask, temp_list):
                                temp_list.append(x)
                                temp_mask |= (1 << (x - 1))
                                added_any = True
                                break
                        
                        if added_any:
                            # Try to add one more
                            for y in candidates:
                                if y in best_set or y in temp_list:
                                    continue
                                if _can_add(y, temp_mask, temp_list):
                                    temp_list.append(y)
                                    temp_mask |= (1 << (y - 1))
                                    break
                            
                            if len(temp_list) > len(best):
                                best = temp_list
                                best_set = set(best)
                                best_mask = temp_mask
                                found = True
                                break
                    if found:
                        break
            
            if not found:
                break
    
    return best

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin
    
    # Precompute
    _get_pair_res()
    
    # Try multiple greedy strategies and keep the best
    result = []
    
    # Largest-first greedy
    selected = []
    selected_set = set()
    selected_mask = 0
    for x in range(_N, 0, -1):
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected_mask, selected):
            selected.append(x)
            selected_set.add(x)
            selected_mask |= (1 << (x - 1))
    result = selected
    
    # Smallest-first greedy
    selected = []
    selected_set = set()
    selected_mask = 0
    for x in range(1, _N + 1):
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected_mask, selected):
            selected.append(x)
            selected_set.add(x)
            selected_mask |= (1 << (x - 1))
    if len(selected) > len(result):
        result = selected
    
    # Random greedy with multiple seeds
    for seed in range(100):
        if time.time() > deadline - 20:
            break
        selected = _greedy_random(seed)
        if len(selected) > len(result):
            result = selected
    
    # Local search on the best result
    if time.time() < deadline - 5:
        remaining_time = min(15.0, deadline - time.time() - 2.0)
        if remaining_time > 1.0:
            result = _local_search(result, time_limit=remaining_time)
    
    # Final validation
    if not _is_valid(result):
        # Fallback: rebuild greedily
        result = []
        selected_mask = 0
        for x in range(_N, 0, -1):
            if len(result) >= _MAX:
                break
            if _can_add(x, selected_mask, result):
                result.append(x)
                selected_mask |= (1 << (x - 1))
    
    return result
# EVOLVE-BLOCK-END
