"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
from scipy.optimize import linprog
import random

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Uses LP relaxation and fast local search for q=3, n=6, radius=1.
    """
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: Simple greedy
        return _generic_greedy(q, n, radius)

    # Specific optimization for q=3, n=6, radius=1
    total_words = 3 ** n
    all_words = list(itertools.product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors as list of indices
    # neighbors[i] = list of indices j such that d(all_words[i], all_words[j]) <= 1
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                neighbors[i].append(word_to_idx[new_w])
        
    # 1. LP Relaxation
    # Minimize sum(x) subject to A x >= 1, 0 <= x <= 1
    # A is total_words x total_words
    # A[i][j] = 1 if i in neighbors[j] (i.e., j covers i)
    # Note: neighbors[j] contains indices covered by word j.
    # So constraint for word i: sum_{j: i in neighbors[j]} x_j >= 1
    
    # We build A_ub = -A, b_ub = -1
    # A is sparse-ish, but 729x729 is small.
    A = np.zeros((total_words, total_words))
    for j in range(total_words):
        for i in neighbors[j]:
            A[i][j] = 1.0
            
    c = np.ones(total_words)
    bounds = [(0, 1)] * total_words
    
    try:
        res = linprog(c, A_ub=-A, b_ub=-1, bounds=bounds, method='highs')
        if res.success:
            x = res.x
            # Round to get initial set
            # Take top K where K is ceil(optimal value)
            K = int(np.ceil(res.fun))
            # Get indices of top K probabilities
            top_indices = np.argsort(x)[::-1][:K]
            initial_set = set(top_indices.tolist())
        else:
            # Fallback to Tetracode if LP fails
            initial_set = set()
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    w = tuple(h + list(tail))
                    initial_set.add(word_to_idx[w])
    except Exception:
        # Fallback to Tetracode
        initial_set = set()
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = tuple(h + list(tail))
                initial_set.add(word_to_idx[w])
                
    # Ensure initial set is valid
    # Compute coverage counts
    def compute_counts(code_set):
        counts = np.zeros(total_words, dtype=int)
        for c_idx in code_set:
            for n_idx in neighbors[c_idx]:
                counts[n_idx] += 1
        return counts

    counts = compute_counts(initial_set)
    if np.any(counts == 0):
        # Add words greedily to fix coverage
        uncovered = [i for i in range(total_words) if counts[i] == 0]
        while uncovered:
            # Find word covering most uncovered
            best_idx = -1
            max_cov = -1
            for j in range(total_words):
                if j in initial_set:
                    continue
                cov = 0
                for n_idx in neighbors[j]:
                    if counts[n_idx] == 0:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_idx = j
            if best_idx == -1:
                break
            initial_set.add(best_idx)
            for n_idx in neighbors[best_idx]:
                counts[n_idx] += 1
            # Update uncovered list (simplified: just recompute or filter)
            # For speed in this small case, just filter
            uncovered = [i for i in uncovered if counts[i] == 0]
            
    # 2. Fast Local Search
    code_set = set(initial_set)
    
    # Helper to check if removing c_idx keeps coverage valid
    def can_remove(c_idx, counts):
        # Temporarily decrement
        for n_idx in neighbors[c_idx]:
            counts[n_idx] -= 1
        if np.any(counts == 0):
            # Rollback
            for n_idx in neighbors[c_idx]:
                counts[n_idx] += 1
            return False
        # Keep decrement if valid? No, we want to check first.
        # If we return True, we should keep the decrement.
        # Let's modify can_remove to return bool and NOT rollback if True.
        # This is tricky. Let's do it manually in the loop.
        return True

    # Better:
    def try_remove(c_idx, counts):
        for n_idx in neighbors[c_idx]:
            counts[n_idx] -= 1
        if np.any(counts == 0):
            for n_idx in neighbors[c_idx]:
                counts[n_idx] += 1
            return False
        return True

    def try_add(c_idx, counts):
        for n_idx in neighbors[c_idx]:
            counts[n_idx] += 1

    def try_swap(c_remove, c_add, counts):
        # Remove c_remove
        for n_idx in neighbors[c_remove]:
            counts[n_idx] -= 1
        # Check if valid so far (ignoring c_add)
        # If any count < 1, it might be fixed by c_add.
        # So we must add c_add first to check?
        # No, we need to check if (C - {c_remove} + {c_add}) is valid.
        # So:
        # 1. Decrement c_remove
        # 2. Increment c_add
        # 3. Check min(counts) >= 1
        # 4. If fail, rollback both.
        
        # We already decremented c_remove.
        for n_idx in neighbors[c_add]:
            counts[n_idx] += 1
        if np.any(counts == 0):
            # Rollback
            for n_idx in neighbors[c_add]:
                counts[n_idx] -= 1
            for n_idx in neighbors[c_remove]:
                counts[n_idx] += 1
            return False
        return True

    improved = True
    swap_attempts = 0
    max_swaps_per_stall = 50
    
    while improved:
        improved = False
        # Try removals
        for c_idx in list(code_set):
            if try_remove(c_idx, counts):
                code_set.remove(c_idx)
                improved = True
                break # Restart removal loop
        
        if not improved:
            # Try swaps to escape local minima
            # Limit swaps to avoid infinite loops
            if swap_attempts < max_swaps_per_stall:
                # Pick random c in code
                c_remove = random.choice(list(code_set))
                # Pick random c_add not in code
                # To speed up, pick from a subset or just random
                c_add = random.randint(0, total_words - 1)
                while c_add in code_set:
                    c_add = random.randint(0, total_words - 1)
                
                if try_swap(c_remove, c_add, counts):
                    code_set.remove(c_remove)
                    code_set.add(c_add)
                    swap_attempts += 1
                    # Don't set improved=True to avoid restarting removal loop immediately?
                    # Actually, we want to try removals after a successful swap.
                    # So we continue the loop.
                    # But we need to prevent infinite swap loops.
                    # The `swap_attempts` counter resets when a removal happens?
                    # Let's reset swap_attempts when a removal happens.
                    continue
            else:
                # Reset swap attempts and try again? Or break?
                # If we are stuck, we might be at a local minimum.
                # For this problem, 73 is the target.
                # If we can't reduce further, we accept.
                break
                
        # If a removal happened, reset swap attempts
        if improved:
            swap_attempts = 0

    result = [list(all_words[i]) for i in code_set]
    return result

def _generic_greedy(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = {}
    for w in all_words:
        nb = set()
        for v in all_words:
            dist = sum(1 for i in range(n) if w[i] != v[i])
            if dist <= radius:
                nb.add(v)
        neighbors[w] = nb
    
    uncovered = set(all_words)
    code = []
    
    while uncovered:
        best_word = None
        max_covered = -1
        for w in all_words:
            if w in code:
                continue
            covered_count = len(neighbors[w] & uncovered)
            if covered_count > max_covered:
                max_covered = covered_count
                best_word = w
        
        if best_word is None:
            break
            
        code.append(best_word)
        uncovered -= neighbors[best_word]
        
    return [list(c) for c in code]

# EVOLVE-BLOCK-END
