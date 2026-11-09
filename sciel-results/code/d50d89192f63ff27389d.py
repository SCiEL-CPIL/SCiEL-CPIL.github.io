"""Minimize largest Sidon subset (with diagonals) of a 12-set in [0,128] via conflict-graph search."""
import time
import random
from itertools import combinations

_DEADLINE = 80.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _is_sidon(subset):
    """Check if subset has all unordered pair sums (including diagonals) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _largest_sidon_subset_size(S):
    """
    Exhaustively compute the largest Sidon subset size.
    Uses branch-and-bound with pruning for speed.
    """
    n = len(S)
    S_sorted = sorted(S)
    
    # Precompute: for each pair of indices, the sum
    # We'll use a recursive backtracking approach with pruning
    
    best_k = 0
    
    def backtrack(start, chosen, used_sums):
        nonlocal best_k
        remaining = n - start
        if len(chosen) + remaining <= best_k:
            return
        if len(chosen) > best_k:
            best_k = len(chosen)
        
        for i in range(start, n):
            val = S_sorted[i]
            # Check all sums involving val with existing chosen elements
            conflict = False
            for j in range(len(chosen)):
                s = val + chosen[j]
                if s in used_sums:
                    conflict = True
                    break
            if not conflict:
                s_diag = val + val
                if s_diag in used_sums:
                    conflict = True
            if not conflict:
                new_chosen = chosen + [val]
                new_sums = used_sums | {val + val}
                for j in range(len(chosen)):
                    new_sums.add(val + chosen[j])
                backtrack(i + 1, new_chosen, new_sums)
    
    backtrack(0, [], set())
    return best_k

def _evaluate_set(S):
    """Return the largest Sidon subset size for set S."""
    return _largest_sidon_subset_size(S)

def _count_conflict_pairs(S):
    """
    Count the number of sum-collision pairs in the set.
    A collision occurs when two different unordered pairs (i,j) and (k,l) with i<=j, k<=l
    produce the same sum. This is a proxy for how 'dense' the set is in terms of collisions.
    Higher collision count means it's harder to find large Sidon subsets.
    """
    sums = {}
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    collisions = 0
    for v in sums.values():
        if v > 1:
            collisions += v * (v - 1) // 2
    return collisions

def _greedy_conflict_set():
    """
    Build a set greedily by adding elements that maximize local sum collisions.
    Start with a random base, then iteratively add/swap elements to increase
    the number of sum collisions.
    """
    # Start with a random set
    current = list(random.sample(range(129), 12))
    current_conflicts = _count_conflict_pairs(current)
    
    best = list(current)
    best_conflicts = current_conflicts
    
    T = 5.0
    cooling = 0.99
    
    for it in range(2000):
        if _elapsed() > _DEADLINE - 2:
            break
        
        idx = random.randint(0, 11)
        old_val = current[idx]
        new_val = random.randint(0, 128)
        if new_val == old_val or new_val in current:
            continue
        
        current[idx] = new_val
        new_conflicts = _count_conflict_pairs(current)
        
        # We want to MAXIMIZE conflicts (to make Sidon extraction harder)
        delta = new_conflicts - current_conflicts
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / max(T, 1e-9)):
            current_conflicts = new_conflicts
            if new_conflicts > best_conflicts:
                best_conflicts = new_conflicts
                best = list(current)
        else:
            current[idx] = old_val
        
        T *= cooling
    
    return best

def _local_search_minimize(S, max_iters=500):
    """
    Local search to minimize the largest Sidon subset size.
    Uses simulated annealing on the actual objective.
    """
    current = list(S)
    current_score = _evaluate_set(current)
    best = list(current)
    best_score = current_score
    
    T = 3.0
    cooling = 0.995
    
    for it in range(max_iters):
        if _elapsed() > _DEADLINE - 2:
            break
        
        idx = random.randint(0, 11)
        old_val = current[idx]
        new_val = random.randint(0, 128)
        if new_val == old_val or new_val in current:
            continue
        
        current[idx] = new_val
        new_score = _evaluate_set(current)
        
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-9)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = list(current)
        else:
            current[idx] = old_val
        
        T *= cooling
    
    return best, best_score

def construct():
    """
    Return exactly 12 distinct integers in [0,128] that minimize the largest Sidon subset size.
    """
    global _start_time
    _start_time = time.time()
    
    random.seed(42)
    
    best_set = None
    best_score = 13
    
    # Phase 1: Use greedy conflict maximization to find high-collision sets
    for r in range(8):
        if _elapsed() > _DEADLINE - 10:
            break
        
        S = _greedy_conflict_set()
        score = _evaluate_set(S)
        
        if score < best_score:
            best_score = score
            best_set = list(S)
    
    # Phase 2: Local search from best and random starts
    n_restarts = 6
    for r in range(n_restarts):
        if _elapsed() > _DEADLINE - 5:
            break
        
        if r == 0 and best_set is not None:
            S = list(best_set)
        else:
            S = random.sample(range(129), 12)
        
        score = _evaluate_set(S)
        if score < best_score:
            best_score = score
            best_set = list(S)
        
        improved_set, improved_score = _local_search_minimize(S, max_iters=300)
        if improved_score < best_score:
            best_score = improved_score
            best_set = improved_set
    
    # Phase 3: Final polish
    if best_set is not None and _elapsed() < _DEADLINE - 3:
        improved_set, improved_score = _local_search_minimize(best_set, max_iters=400)
        if improved_score < best_score:
            best_score = improved_score
            best_set = improved_set
    
    if best_set is None:
        best_set = random.sample(range(129), 12)
    
    return best_set
