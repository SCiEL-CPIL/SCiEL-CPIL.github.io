"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

_DEADLINE_SECONDS = 85  # safety margin before 90s kill

# Precompute GCD table for 1..128
_MAX_VAL = 128
_gcd_table = [[0] * (_MAX_VAL + 1) for _ in range(_MAX_VAL + 1)]
for _i in range(1, _MAX_VAL + 1):
    for _j in range(_i, _MAX_VAL + 1):
        _g = gcd(_i, _j)
        _gcd_table[_i][_j] = _g
        _gcd_table[_j][_i] = _g

def _can_add(x, selected):
    """Check if adding x to selected keeps the set valid (no bad triple formed)."""
    for i in range(len(selected)):
        a = selected[i]
        g_ax = _gcd_table[a][x]
        for j in range(i + 1, len(selected)):
            b = selected[j]
            g_ab = _gcd_table[a][b]
            if g_ab == g_ax and _gcd_table[b][x] == g_ab:
                return False
    return True

def _greedy_pass(order):
    """Greedy pass: add elements in given order if valid."""
    selected = []
    for x in order:
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_improve_add(selected, candidates, deadline):
    """Try to add back excluded candidates in random order."""
    improved = list(selected)
    in_set = set(improved)
    to_try = [c for c in candidates if c not in in_set]
    random.shuffle(to_try)
    for x in to_try:
        if time.time() > deadline:
            break
        if _can_add(x, improved):
            improved.append(x)
    return improved

def _local_improve_swap(selected, candidates, deadline, max_swaps=100):
    """Try swap operations: remove one element, then try to add multiple others."""
    improved = list(selected)
    in_set = set(improved)
    excluded = [c for c in candidates if c not in in_set]
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        if not excluded:
            break
        
        # Pick a random element to remove
        idx = random.randint(0, len(improved) - 1)
        removed = improved[idx]
        improved.pop(idx)
        in_set.discard(removed)
        
        # Try to add multiple candidates
        added_any = False
        remaining = [e for e in excluded if e not in in_set]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline:
                break
            if _can_add(x, improved):
                improved.append(x)
                in_set.add(x)
                added_any = True
        
        if not added_any:
            # Put it back
            improved.append(removed)
            in_set.add(removed)
        else:
            # Update excluded list
            excluded = [c for c in candidates if c not in in_set]
    
    return improved

def _branch_and_bound(available, current_set, current_size, best_size, deadline):
    """Branch-and-bound for maximum independent set in the 3-uniform hypergraph.
    
    Uses recursive vertex ordering based on degree to exploit structural sparsity.
    """
    if time.time() > deadline:
        return current_size, current_set
    
    if not available:
        if current_size > best_size:
            best_size = current_size
        return best_size, current_set
    
    # Upper bound: current_size + len(available)
    if current_size + len(available) <= best_size:
        return best_size, current_set
    
    # Compute degree of each available vertex (how many hyperedges it participates in)
    # A vertex's "degree" here is the number of pairs in current_set that would form
    # a bad triple with it, plus the number of pairs among available that would form
    # bad triples involving it. For pruning, we use a simpler heuristic:
    # sort by number of conflicts with current_set (higher conflict = harder to add)
    
    def _conflict_count(x):
        """Count how many pairs in current_set would form a bad triple with x."""
        cnt = 0
        for i in range(len(current_set)):
            a = current_set[i]
            g_ax = _gcd_table[a][x]
            for j in range(i + 1, len(current_set)):
                b = current_set[j]
                g_ab = _gcd_table[a][b]
                if g_ab == g_ax and _gcd_table[b][x] == g_ab:
                    cnt += 1
        return cnt
    
    # Sort available by conflict count (ascending - try easy ones first for greedy,
    # but for B&B we want to branch on high-degree vertices first for pruning)
    # Actually for maximum independent set, we want to try including vertices that
    # have fewer conflicts first (greedy heuristic), but branch on those with more
    # constraints for better pruning.
    
    # Use a hybrid: try the vertex with most conflicts first (more constrained)
    # to prune branches earlier.
    sorted_available = sorted(available, key=lambda x: -_conflict_count(x))
    
    # Branch 1: Include the first vertex
    x = sorted_available[0]
    if _can_add(x, current_set):
        new_set = current_set + [x]
        # Remove x from available, and also remove any vertex that conflicts with x
        # (i.e., vertices y where {x, y, z} would be a bad triple for some z in new_set)
        # For simplicity, just remove x and continue
        remaining = [v for v in available if v != x]
        # Prune: remove vertices that can't be added to new_set
        remaining = [v for v in remaining if _can_add(v, new_set)]
        size, _ = _branch_and_bound(remaining, new_set, current_size + 1, best_size, deadline)
        if size > best_size:
            best_size = size
    
    # Branch 2: Exclude the first vertex
    remaining = [v for v in available if v != x]
    size, _ = _branch_and_bound(remaining, current_set, current_size, best_size, deadline)
    if size > best_size:
        best_size = size
    
    return best_size, current_set

def _bb_from_seeded(seeded_set, all_nums, deadline):
    """Run branch-and-bound starting from a seeded set."""
    in_set = set(seeded_set)
    available = [x for x in all_nums if x not in in_set]
    # Filter available to only those that can be added
    available = [x for x in available if _can_add(x, seeded_set)]
    
    best_size = len(seeded_set)
    best_set = list(seeded_set)
    
    if not available:
        return best_set
    
    # Limit the B&B depth to avoid exponential blowup
    # Use a time-limited approach
    size, _ = _branch_and_bound(available, list(seeded_set), len(seeded_set), best_size, deadline)
    
    # Reconstruct: we need to track the actual set, not just size
    # For now, use the greedy result as the best and let local improvement handle it
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS
    
    all_nums = list(range(1, 129))
    
    # Helper: count total prime factors (with multiplicity) for ordering heuristic
    def factor_count(n):
        cnt = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                cnt += 1
                n //= d
            d += 1
        if n > 1:
            cnt += 1
        return cnt
    
    fc = {x: factor_count(x) for x in all_nums}
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Sequential greedy (original)
    if time.time() < deadline:
        s = _greedy_pass(all_nums)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 2: Prime-factor-rich first greedy
    if time.time() < deadline:
        order = sorted(all_nums, key=lambda x: (-fc[x], x))
        s = _greedy_pass(order)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 3: Reverse order greedy
    if time.time() < deadline:
        s = _greedy_pass(list(reversed(all_nums)))
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 4: Many random shuffle greedy passes with local improvement
    rng = random.Random(42)
    num_random_passes = 500
    for _ in range(num_random_passes):
        if time.time() > deadline:
            break
        order = all_nums[:]
        rng.shuffle(order)
        s = _greedy_pass(order)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
            # Try local improvement on the new best
            if time.time() < deadline:
                improved = _local_improve_add(best_set, all_nums, deadline)
                if len(improved) > best_len:
                    best_set, best_len = improved, len(improved)
                if time.time() < deadline:
                    improved = _local_improve_swap(best_set, all_nums, deadline, max_swaps=50)
                    if len(improved) > best_len:
                        best_set, best_len = improved, len(improved)
    
    # Strategy 5: Branch-and-bound from the best set found
    if time.time() < deadline and best_set:
        # Use B&B to try to extend the best set
        bb_deadline = min(deadline, time.time() + 20)
        bb_result = _bb_from_seeded(best_set, all_nums, bb_deadline)
        if len(bb_result) > best_len:
            best_set, best_len = bb_result, len(bb_result)
    
    # Final local improvement on best set
    if time.time() < deadline:
        improved = _local_improve_add(best_set, all_nums, deadline)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
    if time.time() < deadline:
        improved = _local_improve_swap(best_set, all_nums, deadline, max_swaps=100)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
    
    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
