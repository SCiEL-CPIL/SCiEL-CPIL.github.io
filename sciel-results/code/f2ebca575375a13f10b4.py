import time
import numpy as np
import random
from itertools import combinations

N = 256
DEADLINE = 88.0  # Safety margin before 90s limit


def _precompute_4ap_conflicts():
    """Precompute for each position, the set of positions that would form a 4-AP
    if both were included along with 2 other specific positions.
    
    Actually, we need: for each pair (i, j), the set of positions k such that
    {i, j, k} is part of some 4-AP. But that's expensive.
    
    Better: for each position p, precompute the set of "forbidden pairs" - 
    pairs (a, b) such that adding p would complete a 4-AP containing p, a, b.
    
    Even better for backtracking: for each position p, precompute all 4-APs
    that contain p. When we add p, we must ensure none of those 4-APs have
    all other 3 elements already in the set.
    
    For forward checking: when we add p, for each other position q not yet decided,
    check if p and q together would force a conflict (i.e., there exist r, s
    already chosen such that {p, q, r, s} forms a 4-AP).
    """
    # For each position p, store list of (d, role) where role indicates
    # which position in the 4-AP p occupies.
    # 4-AP: a, a+d, a+2d, a+3d
    # p can be at offset 0, 1, 2, 3.
    
    # conflicts[p] = list of tuples (other1, other2, other3) such that
    # {p, other1, other2, other3} forms a 4-AP.
    # This is O(N^3) to build but N=256 so ~16M operations, acceptable.
    
    conflicts = [[] for _ in range(N)]
    
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            p0 = a
            p1 = a + d
            p2 = a + 2 * d
            p3 = a + 3 * d
            # All 4 positions form a 4-AP
            # For each position in the AP, the other 3 are its conflict partners
            ap = [p0, p1, p2, p3]
            for i in range(4):
                others = tuple(ap[j] for j in range(4) if j != i)
                conflicts[ap[i]].append(others)
    
    return conflicts


def _precompute_3ap_partners():
    """For each position p, precompute the positions that form a 3-AP with p.
    For counting marginal gain: if we add p, new 3-APs are those where p is
    one of the three elements and the other two are already in the set.
    
    For each p, store:
    - as_first: positions (b, c) where (p, b, c) is a 3-AP
    - as_mid: positions (a, c) where (a, p, c) is a 3-AP  
    - as_last: positions (a, b) where (a, b, p) is a 3-AP
    """
    as_first = [[] for _ in range(N)]  # (b, c) pairs
    as_mid = [[] for _ in range(N)]    # (a, c) pairs
    as_last = [[] for _ in range(N)]   # (a, b) pairs
    
    for a in range(N):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c >= N:
                break
            as_first[a].append((b, c))
            as_mid[b].append((a, c))
            as_last[c].append((a, b))
    
    return as_first, as_mid, as_last


def _count_3ap(mask):
    """Count 3-APs in the set represented by mask (boolean array)."""
    n = len(mask)
    count = 0
    for d in range(1, n // 2 + 1):
        a_part = mask[:n - 2 * d]
        b_part = mask[d:n - d]
        c_part = mask[2 * d:]
        count += np.sum(a_part & b_part & c_part)
    return int(count)


def _backtrack_bnb(conflicts, as_first, as_mid, as_last, var_order, 
                   deadline, time_budget, current_set, current_count,
                   best_set, best_count):
    """Branch and bound backtracking.
    
    var_order: list of positions in the order they should be considered.
    current_set: set of positions already included.
    current_count: number of 3-APs in current_set.
    best_set, best_count: global best.
    
    Returns updated (best_set, best_count).
    """
    n = len(var_order)
    
    # Upper bound: current_count + sum of max possible marginal gains
    # for each undecided position (optimistic: assume all can be added)
    
    def upper_bound(idx, current_count):
        """Optimistic upper bound on final 3-AP count."""
        ub = current_count
        for j in range(idx, n):
            p = var_order[j]
            if p in current_set:
                continue
            # Max marginal gain for p: count of 3-APs involving p where
            # the other 2 elements could potentially be in the set
            # Optimistic: count all 3-APs involving p
            gain = len(as_first[p]) + len(as_mid[p]) + len(as_last[p])
            # But each 3-AP is counted from both endpoints... 
            # Actually as_first[p] counts (b,c) pairs, as_mid[p] counts (a,c) pairs
            # Each 3-AP involving p is counted exactly once across these three lists
            # because p has a unique role.
            ub += gain
        return ub
    
    # Memoization for forward checking: track which positions are "blocked"
    # by the current partial assignment.
    
    def backtrack(idx, current_set, current_count):
        nonlocal best_set, best_count
        
        if time.time() > deadline - 2:
            return
        
        # Update best
        if current_count > best_count:
            best_count = current_count
            best_set = current_set.copy()
        
        if idx >= n:
            return
        
        # Upper bound pruning
        ub = upper_bound(idx, current_count)
        if ub <= best_count:
            return
        
        p = var_order[idx]
        
        # Option 1: Include p (if it doesn't create a 4-AP)
        valid = True
        for (o1, o2, o3) in conflicts[p]:
            if o1 in current_set and o2 in current_set and o3 in current_set:
                valid = False
                break
        
        if valid:
            # Compute marginal gain
            gain = 0
            for (b, c) in as_first[p]:
                if b in current_set and c in current_set:
                    gain += 1
            for (a, c) in as_mid[p]:
                if a in current_set and c in current_set:
                    gain += 1
            for (a, b) in as_last[p]:
                if a in current_set and b in current_set:
                    gain += 1
            
            new_set = current_set | {p}
            backtrack(idx + 1, new_set, current_count + gain)
        
        # Option 2: Exclude p
        backtrack(idx + 1, current_set, current_count)
    
    backtrack(0, current_set, current_count)
    return best_set, best_count


def _greedy_init(conflicts, as_first, as_mid, as_last, var_order, deadline):
    """Greedy initialization: process var_order, include if valid and positive gain."""
    current_set = set()
    current_count = 0
    
    for p in var_order:
        if time.time() > deadline - 3:
            break
        # Check 4-AP validity
        valid = True
        for (o1, o2, o3) in conflicts[p]:
            if o1 in current_set and o2 in current_set and o3 in current_set:
                valid = False
                break
        if not valid:
            continue
        # Compute marginal gain
        gain = 0
        for (b, c) in as_first[p]:
            if b in current_set and c in current_set:
                gain += 1
        for (a, c) in as_mid[p]:
            if a in current_set and c in current_set:
                gain += 1
        for (a, b) in as_last[p]:
            if a in current_set and b in current_set:
                gain += 1
        if gain > 0:
            current_set.add(p)
            current_count += gain
    
    return current_set, current_count


def _local_search(conflicts, as_first, as_mid, as_last, current_set, current_count, deadline):
    """Local search: try adding/removing/swapping elements to improve."""
    best_set = current_set.copy()
    best_count = current_count
    improved = True
    
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        
        # Try adding each element not in set
        for p in range(N):
            if time.time() > deadline - 5:
                break
            if p in best_set:
                continue
            # Check 4-AP
            valid = True
            for (o1, o2, o3) in conflicts[p]:
                if o1 in best_set and o2 in best_set and o3 in best_set:
                    valid = False
                    break
            if not valid:
                continue
            gain = 0
            for (b, c) in as_first[p]:
                if b in best_set and c in best_set:
                    gain += 1
            for (a, c) in as_mid[p]:
                if a in best_set and c in best_set:
                    gain += 1
            for (a, b) in as_last[p]:
                if a in best_set and b in best_set:
                    gain += 1
            if gain > 0:
                best_set.add(p)
                best_count += gain
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing elements that are "expensive" (low gain relative to cost)
        for p in list(best_set):
            if time.time() > deadline - 5:
                break
            # Compute how many 3-APs p contributes
            contrib = 0
            for (b, c) in as_first[p]:
                if b in best_set and c in best_set:
                    contrib += 1
            for (a, c) in as_mid[p]:
                if a in best_set and c in best_set:
                    contrib += 1
            for (a, b) in as_last[p]:
                if a in best_set and b in best_set:
                    contrib += 1
            # If removing p frees up space for better elements, try it
            # Only remove if contrib is very low (0 or 1)
            if contrib <= 1:
                best_set.remove(p)
                best_count -= contrib
                improved = True
                break
    
    return best_set, best_count


def _swap_search(conflicts, as_first, as_mid, as_last, current_set, current_count, deadline):
    """Try swapping one element out and one in."""
    best_set = current_set.copy()
    best_count = current_count
    improved = True
    
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False
        
        for p_out in list(best_set):
            if time.time() > deadline - 5:
                break
            # Try removing p_out
            set_without = best_set - {p_out}
            count_without = best_count
            # Recompute: subtract contributions of p_out
            contrib_out = 0
            for (b, c) in as_first[p_out]:
                if b in set_without and c in set_without:
                    contrib_out += 1
            for (a, c) in as_mid[p_out]:
                if a in set_without and c in set_without:
                    contrib_out += 1
            for (a, b) in as_last[p_out]:
                if a in set_without and b in set_without:
                    contrib_out += 1
            count_without -= contrib_out
            
            # Try adding each p_in not in set
            for p_in in range(N):
                if p_in in best_set:
                    continue
                # Check 4-AP in set_without + {p_in}
                valid = True
                for (o1, o2, o3) in conflicts[p_in]:
                    if o1 in set_without and o2 in set_without and o3 in set_without:
                        valid = False
                        break
                if not valid:
                    continue
                gain_in = 0
                for (b, c) in as_first[p_in]:
                    if b in set_without and c in set_without:
                        gain_in += 1
                for (a, c) in as_mid[p_in]:
                    if a in set_without and c in set_without:
                        gain_in += 1
                for (a, b) in as_last[p_in]:
                    if a in set_without and b in set_without:
                        gain_in += 1
                
                new_count = count_without + gain_in
                if new_count > best_count:
                    best_set = set_without | {p_in}
                    best_count = new_count
                    improved = True
                    break
            if improved:
                break
    
    return best_set, best_count


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    random.seed(12345)
    
    # Precompute conflict and 3-AP data
    conflicts = _precompute_4ap_conflicts()
    as_first, as_mid, as_last = _precompute_3ap_partners()
    
    # Compute conflict density for each position
    conflict_density = [len(conflicts[p]) for p in range(N)]
    
    # Global best
    global_best_set = set()
    global_best_count = 0
    
    # Multiple restarts with different variable orderings
    n_restarts = 20
    per_restart_time = (DEADLINE - 10) / n_restarts  # ~4s per restart
    
    for restart in range(n_restarts):
        if time.time() > deadline - 8:
            break
        
        # Create variable ordering: sort by conflict density (high first) + random noise
        # Higher conflict density = more constrained = decide earlier
        if restart == 0:
            # Deterministic: highest conflict density first
            var_order = sorted(range(N), key=lambda p: -conflict_density[p])
        elif restart == 1:
            # Lowest conflict density first
            var_order = sorted(range(N), key=lambda p: conflict_density[p])
        else:
            # Random perturbation
            noise = [random.gauss(0, 1.0) for _ in range(N)]
            var_order = sorted(range(N), key=lambda p: -conflict_density[p] + noise[p] * 50)
        
        # Phase 1: Greedy initialization
        current_set, current_count = _greedy_init(
            conflicts, as_first, as_mid, as_last, var_order, deadline
        )
        
        # Phase 2: Local search
        current_set, current_count = _local_search(
            conflicts, as_first, as_mid, as_last, current_set, current_count, deadline
        )
        
        # Phase 3: Swap search
        current_set, current_count = _swap_search(
            conflicts, as_first, as_mid, as_last, current_set, current_count, deadline
        )
        
        # Phase 4: Branch and bound on a subset (last K variables)
        if time.time() < deadline - 10:
            # B&B on the tail of var_order (least constrained variables)
            k = min(30, N)  # B&B on last 30 variables
            bnb_order = var_order[-k:]
            # Start from current_set, but only decide on bnb_order variables
            # Remove bnb_order variables from current_set first
            bnb_set = current_set - set(bnb_order)
            bnb_count = 0
            # Recompute count for bnb_set
            mask_tmp = np.zeros(N, dtype=bool)
            for p in bnb_set:
                mask_tmp[p] = True
            bnb_count = _count_3ap(mask_tmp)
            
            best_bnb_set, best_bnb_count = _backtrack_bnb(
                conflicts, as_first, as_mid, as_last, bnb_order,
                deadline, per_restart_time, bnb_set, bnb_count,
                bnb_set, bnb_count
            )
            
            if best_bnb_count > current_count:
                current_set = best_bnb_set
                current_count = best_bnb_count
        
        # Phase 5: Final local search
        if time.time() < deadline - 5:
            current_set, current_count = _local_search(
                conflicts, as_first, as_mid, as_last, current_set, current_count, deadline
            )
        
        # Update global best
        if current_count > global_best_count:
            global_best_count = current_count
            global_best_set = current_set.copy()
    
    # Final polish: one more local search + swap on the global best
    if time.time() < deadline - 3:
        global_best_set, global_best_count = _local_search(
            conflicts, as_first, as_mid, as_last, global_best_set, global_best_count, deadline
        )
        global_best_set, global_best_count = _swap_search(
            conflicts, as_first, as_mid, as_last, global_best_set, global_best_count, deadline
        )
    
    # Convert to 1-indexed list
    result = sorted(list(global_best_set))
    result_1indexed = [p + 1 for p in result]
    
    if len(result_1indexed) == 0:
        return [1]
    
    return result_1indexed


# Precompute at module load time to save time during construct()
_CONFLICTS = None
_AS_FIRST = None
_AS_MID = None
_AS_LAST = None

def _ensure_precomputed():
    global _CONFLICTS, _AS_FIRST, _AS_MID, _AS_LAST
    if _CONFLICTS is None:
        _CONFLICTS = _precompute_4ap_conflicts()
        _AS_FIRST, _AS_MID, _AS_LAST = _precompute_3ap_partners()

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    random.seed(12345)
    
    # Precompute conflict and 3-AP data
    _ensure_precomputed()
    conflicts = _CONFLICTS
    as_first, as_mid, as_last = _AS_FIRST, _AS_MID, _AS_LAST
    
    # Compute conflict density for each position
    conflict_density = [len(conflicts[p]) for p in range(N)]
    
    # Global best
    global_best_set = set()
    global_best_count = 0
    
    # Multiple restarts with different variable orderings
    n_restarts = 25
    per_restart_time = (DEADLINE - 12) / n_restarts  # ~3.2s per restart
    
    for restart in range(n_restarts):
        if time.time() > deadline - 8:
            break
        
        # Create variable ordering
        if restart == 0:
            var_order = sorted(range(N), key=lambda p: -conflict_density[p])
        elif restart == 1:
            var_order = sorted(range(N), key=lambda p: conflict_density[p])
        elif restart == 2:
            # Middle-out
            mid = N // 2
            left = sorted(range(mid), key=lambda p: -conflict_density[p])
            right = sorted(range(mid, N), key=lambda p: -conflict_density[p])
            var_order = left + right
        else:
            noise = [random.gauss(0, 1.0) for _ in range(N)]
            var_order = sorted(range(N), key=lambda p: -conflict_density[p] + noise[p] * 80)
        
        # Phase 1: Greedy initialization
        current_set, current_count = _greedy_init(
            conflicts, as_first, as_mid, as_last, var_order, deadline
        )
        
        # Phase 2: Local search
        current_set, current_count = _local_search(
            conflicts, as_first, as_mid, as_last, current_set, current_count, deadline
        )
        
        # Phase 3: Swap search
        current_set, current_count = _swap_search(
            conflicts, as_first, as_mid, as_last, current_set, current_count, deadline
        )
        
        # Phase 4: Branch and bound on a subset
        if time.time() < deadline - 10:
            k = min(25, N)
            bnb_order = var_order[-k:]
            bnb_set = current_set - set(bnb_order)
            mask_tmp = np.zeros(N, dtype=bool)
            for p in bnb_set:
                mask_tmp[p] = True
            bnb_count = _count_3ap(mask_tmp)
            
            best_bnb_set, best_bnb_count = _backtrack_bnb(
                conflicts, as_first, as_mid, as_last, bnb_order,
                deadline, per_restart_time, bnb_set, bnb_count,
                bnb_set, bnb_count
            )
            
            if best_bnb_count > current_count:
                current_set = best_bnb_set
                current_count = best_bnb_count
        
        # Phase 5: Final local search
        if time.time() < deadline - 5:
            current_set, current_count = _local_search(
                conflicts, as_first, as_mid, as_last, current_set, current_count, deadline
            )
        
        # Update global best
        if current_count > global_best_count:
            global_best_count = current_count
            global_best_set = current_set.copy()
    
    # Final polish
    if time.time() < deadline - 3:
        global_best_set, global_best_count = _local_search(
            conflicts, as_first, as_mid, as_last, global_best_set, global_best_count, deadline
        )
        global_best_set, global_best_count = _swap_search(
            conflicts, as_first, as_mid, as_last, global_best_set, global_best_count, deadline
        )
    
    # Convert to 1-indexed list
    result = sorted(list(global_best_set))
    result_1indexed = [p + 1 for p in result]
    
    if len(result_1indexed) == 0:
        return [1]
    
    return result_1indexed
