import time
import random
import numpy as np
from itertools import combinations

N = 16
DEADLINE = 325.0
N_TOTAL = N * N
TIME_LIMIT = 300.0  # Hard limit for the evaluator

# Precompute all points and pairwise squared distances
_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
_POINT_INDEX = {p: i for i, p in enumerate(_ALL_POINTS)}
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles masks using Python integers as bitmasks
# iso_c_mask[i][j] = bitmask of points k such that (i, j, k) is isosceles
# Note: (i, j, k) is isosceles if d(i,k)==d(j,k) OR d(i,k)==d(i,j) OR d(j,k)==d(i,j)
iso_c_mask = [[0] * N_TOTAL for _ in range(N_TOTAL)]
for i in range(N_TOTAL):
    for j in range(i + 1, N_TOTAL):
        di_j = _D2[i, j]
        mask = 0
        for k in range(N_TOTAL):
            if k == i or k == j:
                continue
            di_k = _D2[i, k]
            dj_k = _D2[j, k]
            if di_k == dj_k or di_k == di_j or dj_k == di_j:
                mask |= (1 << k)
        iso_c_mask[i][j] = mask
        iso_c_mask[j][i] = mask

# Precompute for each point p, the list of pairs (i, j) such that (i, j, p) is isosceles
# This allows fast checking: p is blocked by S iff there exists (i, j) in blocked_pairs[p] with i in S and j in S
blocked_pairs = [[] for _ in range(N_TOTAL)]
for p in range(N_TOTAL):
    bit_p = 1 << p
    for i in range(N_TOTAL):
        for j in range(i + 1, N_TOTAL):
            if i == p or j == p:
                continue
            if iso_c_mask[i][j] & bit_p:
                blocked_pairs[p].append((i, j))

def _is_no_isosceles(S_list):
    """Check if set S is no-isosceles using precomputed masks."""
    n = len(S_list)
    if n < 3:
        return True
    S_mask = 0
    for idx in S_list:
        S_mask |= (1 << idx)
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            if iso_c_mask[a][b] & S_mask:
                return False
    return True

def _can_add(p, S_list):
    """Check if point p can be added to S without creating isosceles triangle."""
    n = len(S_list)
    if n < 2:
        return True
    S_mask = 0
    for idx in S_list:
        S_mask |= (1 << idx)
    for a in S_list:
        if iso_c_mask[p][a] & S_mask:
            return False
    return True

def _is_blocked(p, S_list):
    """Check if point p is blocked by set S (forms isosceles with some pair in S)."""
    n = len(S_list)
    if n < 2:
        return False
    S_mask = 0
    for idx in S_list:
        S_mask |= (1 << idx)
    for a, b in blocked_pairs[p]:
        if (S_mask & (1 << a)) and (S_mask & (1 << b)):
            return True
    return False

def _is_maximal(S_list):
    """Check if S is maximal: no point can be added."""
    S_set = set(S_list)
    for p in range(N_TOTAL):
        if p in S_set:
            continue
        if not _is_blocked(p, S_list):
            return False
    return True

def _reduce_to_minimal(chosen_indices, deadline_margin=5):
    """Iteratively remove points if remaining set is still maximal."""
    start_time = time.time()
    S = list(chosen_indices)
    changed = True
    while changed:
        if time.time() - start_time > DEADLINE - deadline_margin:
            break
        changed = False
        for i in range(len(S)):
            if len(S) - 1 < 2:
                continue
            S_test = S[:i] + S[i + 1:]
            if _is_maximal(S_test):
                S = S_test
                changed = True
                break
    return S

def _greedy_cover(start_time, deadline_margin=15):
    """
    Greedy set-cover approach to build a maximal no-isosceles set.
    """
    chosen = []
    chosen_set = set()
    blocked = [False] * N_TOTAL
    
    while True:
        if time.time() - start_time > DEADLINE - deadline_margin:
            return None
        
        if _is_maximal(chosen):
            return chosen
        
        unblocked = []
        for q in range(N_TOTAL):
            if q in chosen_set:
                continue
            if not blocked[q]:
                unblocked.append(q)
        
        if not unblocked:
            break
        
        addable = []
        for p in range(N_TOTAL):
            if p in chosen_set:
                continue
            if _can_add(p, chosen):
                addable.append(p)
        
        if not addable:
            break
        
        best_p = None
        best_gain = -1
        
        for p in addable:
            gain = 0
            for q in unblocked:
                for a in chosen:
                    if iso_c_mask[p][a] & (1 << q):
                        gain += 1
                        break
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        chosen.append(best_p)
        chosen_set.add(best_p)
        
        for q in range(N_TOTAL):
            if q in chosen_set:
                continue
            if not blocked[q]:
                for a in chosen:
                    if iso_c_mask[best_p][a] & (1 << q):
                        blocked[q] = True
                        break
    
    return chosen

def _walk_sat_search(S_init, start_time, deadline_margin=10):
    """
    Walk-SAT style local search to minimize the size of a maximal no-isosceles set.
    State: A boolean vector of length N_TOTAL representing inclusion in S.
    Constraints:
    1. No-isosceles: For all i, j in S, no k in S such that (i,j,k) is isosceles.
       Clause: NOT (x_i AND x_j AND x_k) for all isosceles triples (i,j,k).
    2. Maximal: For all p not in S, there exists (i,j) in S such that (i,j,p) is isosceles.
       Clause: (x_i OR x_j) for all (i,j) in blocked_pairs[p] when x_p is False.
    
    Objective: Minimize sum(x_i).
    
    We use a simulated annealing / walk-SAT hybrid.
    Score = (num_violated_no_isosceles_clauses * 1000) + (num_unblocked_points * 100) + (num_points)
    """
    
    def get_score(state):
        """
        state: set of indices or list of indices.
        Returns (score, n_isos, n_unblocked, n_points)
        """
        n_isos = 0
        n_points = len(state)
        
        # Check no-isosceles violations
        state_mask = 0
        for idx in state:
            state_mask |= (1 << idx)
        
        # We only need to check pairs in state
        state_list = list(state)
        for i in range(len(state_list)):
            for j in range(i + 1, len(state_list)):
                a = state_list[i]
                b = state_list[j]
                # Check if any k in state is isosceles with a, b
                if iso_c_mask[a][b] & state_mask:
                    n_isos += 1
        
        # Check maximality (unblocked points)
        n_unblocked = 0
        for p in range(N_TOTAL):
            if p in state:
                continue
            # p is unblocked if NO pair (i,j) in state blocks it
            is_blocked = False
            for a, b in blocked_pairs[p]:
                if (state_mask & (1 << a)) and (state_mask & (1 << b)):
                    is_blocked = True
                    break
            if not is_blocked:
                n_unblocked += 1
        
        # Weighting:
        # 1. No-isosceles violation is critical.
        # 2. Maximal violation (unblocked points) is important.
        # 3. Size minimization.
        score = n_isos * 10000 + n_unblocked * 100 + n_points
        return score, n_isos, n_unblocked, n_points

    def try_flip(state, p, add):
        """
        Try flipping point p.
        Returns (new_state, score_delta) or (None, None) if flip is invalid (e.g., leads to immediate massive penalty or is not allowed).
        Actually, in Walk-SAT we just flip and evaluate. But we can be smarter.
        """
        if add:
            if p in state:
                return None, 0
            new_state = state | {p}
        else:
            if p not in state:
                return None, 0
            new_state = state - {p}
        
        old_score, _, _, _ = get_score(state)
        new_score, _, _, _ = get_score(new_state)
        return new_state, new_score - old_score

    # Initialize
    current_state = set(S_init)
    current_score, _, _, _ = get_score(current_state)
    best_state = set(current_state)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    rng = random.Random(42)
    
    while time.time() - start_time < DEADLINE - deadline_margin:
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Pick a random point to flip
        p = rng.randint(0, N_TOTAL - 1)
        
        # Decide direction
        # If p is in state, we might want to remove it to reduce size.
        # If p is not in state, we might want to add it to fix violations.
        
        if p in current_state:
            # Try removing
            new_state, delta = try_flip(current_state, p, add=False)
            if new_state is None: continue
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_state = new_state
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_state = set(current_state)
        else:
            # Try adding
            new_state, delta = try_flip(current_state, p, add=True)
            if new_state is None: continue
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_state = new_state
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_state = set(current_state)
        
        # Optional: If we find a valid maximal set, try to reduce it
        _, n_isos, n_unblocked, n_points = get_score(best_state)
        if n_isos == 0 and n_unblocked == 0:
            # Valid maximal set found. Try to minimize size via local search (remove points)
            # This is similar to _reduce_to_minimal but faster if we use masks?
            # _reduce_to_minimal is already called in construct() usually.
            pass
            
    return list(best_state)

def _local_search_improve(S, start_time, deadline_margin=10):
    """Try to improve set S by removing one point and adding another (1-swap)."""
    S_list = list(S)
    n_S = len(S_list)
    best = S_list[:]
    best_size = n_S
    
    for i in range(n_S):
        if time.time() - start_time > DEADLINE - deadline_margin:
            break
        
        S_without = S_list[:i] + S_list[i + 1:]
        S_without_set = set(S_without)
        
        best_new = None
        best_size_local = n_S
        
        for p_add in range(N_TOTAL):
            if p_add in S_without_set:
                continue
            
            if not _can_add(p_add, S_without):
                continue
            
            S_new = S_without + [p_add]
            
            if not _is_maximal(S_new):
                continue
            
            reduced = _reduce_to_minimal(S_new, deadline_margin=5)
            
            if len(reduced) < best_size_local:
                best_size_local = len(reduced)
                best_new = reduced
        
        if best_new is not None:
            best = best_new
            best_size = best_size_local
            # Update S_list for next iteration? Or just return best?
            # Returning best immediately might be too greedy. Let's continue loop but update best.
            # Actually, if we found a smaller set, we should probably restart or update base.
            # Let's just return the best found so far in this function.
            return best
    
    return best

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy cover from empty
    result = _greedy_cover(start_time, deadline_margin=20)
    if result is not None:
        reduced = _reduce_to_minimal(result, deadline_margin=10)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
    
    # Strategy 2: Walk-SAT search starting from the greedy result
    if best_set is not None:
        ws_result = _walk_sat_search(best_set, start_time, deadline_margin=30)
        if ws_result is not None:
            # Verify and reduce
            if _is_no_isosceles(ws_result) and _is_maximal(ws_result):
                reduced = _reduce_to_minimal(ws_result, deadline_margin=10)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
            else:
                # If Walk-SAT didn't find a valid one, try to repair or ignore
                pass
    
    # Strategy 3: Random cores + greedy build (if time permits)
    for core_size in [2, 3, 4]:
        if time.time() - start_time > DEADLINE - 40:
            break
        
        n_attempts = 50
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE - 20:
                break
            
            core_indices = rng.sample(range(N_TOTAL), core_size)
            
            # Check if core is no-isosceles
            if not _is_no_isosceles(core_indices):
                continue
                
            # Build from core
            chosen = list(core_indices)
            chosen_set = set(chosen)
            blocked = [False] * N_TOTAL
            # Initialize blocked
            for q in range(N_TOTAL):
                if q in chosen_set: continue
                for a, b in blocked_pairs[q]:
                    if a in chosen_set and b in chosen_set:
                        blocked[q] = True
                        break
            
            while True:
                if time.time() - start_time > DEADLINE - 15:
                    break
                
                if _is_maximal(chosen):
                    break
                
                addable = []
                for p in range(N_TOTAL):
                    if p in chosen_set:
                        continue
                    if _can_add(p, chosen):
                        addable.append(p)
                
                if not addable:
                    break
                
                # Pick best addable
                best_p = None
                best_gain = -1
                for p in addable:
                    gain = 0
                    for q in range(N_TOTAL):
                        if q in chosen_set or blocked[q]:
                            continue
                        for a in chosen:
                            if iso_c_mask[p][a] & (1 << q):
                                gain += 1
                                break
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                
                if best_p is None:
                    break
                
                chosen.append(best_p)
                chosen_set.add(best_p)
                
                for q in range(N_TOTAL):
                    if q in chosen_set:
                        continue
                    if not blocked[q]:
                        for a in chosen:
                            if iso_c_mask[best_p][a] & (1 << q):
                                blocked[q] = True
                                break
            
            if chosen and _is_maximal(chosen):
                reduced = _reduce_to_minimal(chosen, deadline_margin=5)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
    
    # Strategy 4: Local search improvement on best set
    if best_set is not None and best_size > 10:
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 10:
                break
            improved = _local_search_improve(best_set, start_time, deadline_margin=5)
            if improved is not None and len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
            else:
                break
    
    # Fallback
    if best_set is None:
        chosen = []
        for i in range(N_TOTAL):
            if _can_add(i, chosen):
                chosen.append(i)
        best_set = _reduce_to_minimal(chosen, deadline_margin=5)
    
    # Final check
    if not _is_no_isosceles(best_set):
        # This should not happen, but just in case, return a valid small set
        # Find any valid maximal set?
        # For now, return empty? No, must be maximal.
        # Let's do a quick greedy again if this fails
        chosen = []
        for i in range(N_TOTAL):
            if _can_add(i, chosen):
                chosen.append(i)
        best_set = chosen
    
    return [_ALL_POINTS[i] for i in best_set]
