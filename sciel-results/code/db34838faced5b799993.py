import time
import math
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 145.0

def _elapsed():
    return time.time() - _START

def _score_set(U):
    """Compute the score for a set U containing 0."""
    U = np.asarray(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n == 0:
        return 0.0
    if U[0] != 0:
        U = np.concatenate(([0], U))
        n = len(U)
    M = int(U[-1])
    if M < 1:
        return 0.0
    
    # Calculate sumset and diffset sizes using numpy for speed
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    sum_size = len(np.unique(sums))
    diff_size = len(np.unique(diffs))
    
    if sum_size == 0:
        return 0.0
        
    ratio = diff_size / sum_size
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def _golomb_greedy(n, max_iter=100000):
    """Greedy construction of a Golomb Ruler with n marks."""
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    # Start with [0, 1]
    U = [0, 1]
    diffs = set()
    for i in range(len(U)):
        for j in range(i+1, len(U)):
            diffs.add(U[j] - U[i])
            
    for _ in range(n - 2):
        placed = False
        last = U[-1]
        # Try to extend from the right
        for candidate in range(last + 1, last + 2000):
            valid = True
            new_diffs = []
            for u in U:
                d = candidate - u
                if d in diffs:
                    valid = False
                    break
                new_diffs.append(d)
            if valid:
                U.append(candidate)
                diffs.update(new_diffs)
                placed = True
                break
        if not placed:
            break
            
    return U

def _local_search_compress(U, time_limit=20.0):
    """
    Local search to compress the set (reduce M) by moving elements down.
    This is the primary refinement step for Golomb rulers to beat the raw greedy score.
    """
    U = sorted(U)
    n = len(U)
    if n < 2:
        return U
    max_val = max(U)
    cur_score = _score_set(U)
    best_U = U[:]
    best_score = cur_score
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n = len(U)
        max_val = max(U)
        
        # 1. Try moving each element slightly down (compression)
        # Iterate in reverse order to avoid index shifting issues if we were deleting,
        # but here we are just changing values.
        for i in range(n - 1, 0, -1):
            if time.time() - start_time > time_limit:
                break
            original_val = U[i]
            lower_bound = U[i-1] + 1
            upper_bound = U[i+1] - 1 if i < n - 1 else original_val
            
            # Try a few candidates below original
            for delta in range(1, 15):
                new_val = original_val - delta
                if new_val < lower_bound:
                    break
                if new_val == original_val:
                    continue
                trial = U[:]
                trial[i] = new_val
                trial.sort()
                s = _score_set(trial)
                if s > cur_score + 1e-12:
                    cur_score = s
                    U = trial
                    best_U = U[:]
                    best_score = cur_score
                    improved = True
                    break # Restart loop with new U
        
        # 2. Try adding a new element if n is small (to increase |U-U|)
        if n < 15:
            if time.time() - start_time > time_limit:
                break
            for _ in range(20):
                c = rng.randint(1, max_val + 20)
                if c not in U:
                    trial = sorted(U + [c])
                    s = _score_set(trial)
                    if s > cur_score + 1e-12:
                        cur_score = s
                        U = trial
                        best_U = U[:]
                        best_score = cur_score
                        improved = True
                        break
                        
    return best_U

def construct():
    best_U = [0]
    best_score = 0.0
    
    # 1. Try known optimal Golomb Rulers
    known_golomb = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 12],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 45],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
    }
    
    for n, U in known_golomb.items():
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # 2. Generate greedy Golomb Rulers for larger n
    # We go up to n=16 or 17 depending on time
    for n in range(14, 18):
        if _elapsed() > 40:
            break
        U = _golomb_greedy(n)
        if len(U) == n:
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
        
    # 3. Local Search on the best candidate
    # We allow "soft" Golomb rulers (allowing some duplicate differences)
    # if it reduces M significantly or improves the ratio.
    # The compression search is the key improvement over raw greedy.
    
    U = list(best_U)
    
    # Run compression search with a generous time limit
    # We split the remaining time
    time_remaining = _DEADLINE - _elapsed()
    if time_remaining > 10:
        search_time = max(5, time_remaining - 5)
        U = _local_search_compress(U, time_limit=search_time)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # 4. If time permits, try a few random restarts or perturbations
    # This helps escape local minima in the compression landscape
    if _elapsed() < _DEADLINE - 15:
        rng = np.random.RandomState(123)
        for _ in range(3):
            if _elapsed() > _DEADLINE - 10:
                break
            # Perturb the best solution
            U_perturbed = list(best_U)
            if len(U_perturbed) > 2:
                # Remove a random element (not 0)
                idx = rng.randint(1, len(U_perturbed) - 1)
                U_perturbed.pop(idx)
            
            # Run a quick compression search from this perturbed state
            # Use a short time limit
            U_new = _local_search_compress(U_perturbed, time_limit=5.0)
            s = _score_set(U_new)
            if s > best_score:
                best_score = s
                best_U = U_new[:]

    return best_U

# Keep the original function names and signatures
# The task requires `construct() -> list`
# The current program has `construct()` which calls `_main_logic()`.
# I will just define `construct` directly to be clean.

# Note: The prompt says "Keep the function names and signatures of the seed program."
# The seed program (current) has `construct()`.
# My implementation above defines `construct()` directly.
# I should ensure no other top-level `construct` definition exists.

# The code above is the complete program.
# I will re-verify the structure.
# _START, _DEADLINE, _elapsed, _score_set, _golomb_greedy, _local_search_compress, construct.
# This looks correct.

# One detail: The prompt's current program had a weird double definition of construct at the end.
# I am replacing the whole file, so I just provide the clean version.

# Let's double check the score calculation logic.
# Score = 1 + log(|U-U|/|U+U|) / log(2 max U + 1)
# My _score_set does exactly this.

# Let's double check the Golomb Ruler generation.
# Greedy adds marks to the end.
# Compression moves marks down.
# This is a standard approach for this type of problem.

# Is there any other inspiration I should incorporate?
# Inspiration 1 had `_local_search` and `_local_search_swap`.
# Inspiration 2 had a more extensive list of known Golomb rulers.
# I incorporated the known rulers up to 13 (which is good).
# I incorporated the compression idea from the current program.
# I incorporated the numpy speedup from the inspirations.

# The "gap-multiplication" strategy mentioned in the directive is a different design family.
# The directive says: "DO NOT switch to a different design family."
# So I stick with Golomb Rulers + Local Search.

# Final check on time limits.
# Total 150s.
# My deadline is 145s.
# Greedy generation: ~5s.
# Compression: ~100s.
# Perturbations: ~30s.
# Total: ~135s. Safe.

# One potential issue: `_score_set` uses `np.unique` which is O(N^2 log N^2).
# For N=17, N^2 = 289. This is very fast.
# For N=20, N^2 = 400. Still fast.
# So numpy is a good choice.

# I will output the code now.
