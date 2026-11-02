import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 145.0

def _elapsed():
    return time.time() - _START

def _score_set(U):
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
    
    # Calculate sumset and diffset sizes
    # For n <= 20, O(n^2) is fast enough
    sum_set = set()
    diff_set = set()
    for i in range(n):
        for j in range(n):
            sum_set.add(U[i] + U[j])
            diff_set.add(U[i] - U[j])
    
    sum_size = len(sum_set)
    diff_size = len(diff_set)
    
    if sum_size == 0:
        return 0.0
        
    ratio = diff_size / sum_size
    base = 2 * M + 1
    if base <= 1:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(base)

def _greedy_golomb(n, max_iter=100000):
    """Greedy construction of a Golomb Ruler with n marks."""
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    U = [0]
    # Try to add marks one by one
    # Heuristic: place new mark as close as possible to the last one,
    # such that all differences remain unique.
    
    # Start with [0, 1]
    U = [0, 1]
    diffs = set()
    for i in range(len(U)):
        for j in range(i+1, len(U)):
            diffs.add(U[j] - U[i])
            
    for _ in range(n - 2):
        placed = False
        # Try to extend from the right, or insert? 
        # Standard greedy for Golomb: add to the end.
        # Try positions from U[-1] + 1 upwards
        last = U[-1]
        for candidate in range(last + 1, last + 1000):
            # Check if adding 'candidate' creates duplicate differences
            new_diffs = set()
            valid = True
            for u in U:
                d = candidate - u
                if d in diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                # Also need to check differences among existing? No, only new ones matter
                # Wait, differences are |ui - uj|. Since candidate > all u, 
                # new diffs are candidate - u.
                # We must ensure these don't collide with existing diffs.
                U.append(candidate)
                diffs.update(new_diffs)
                placed = True
                break
        if not placed:
            # If we can't find a spot nearby, break or try wider search
            # For now, just break to avoid infinite loop
            break
            
    return U

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
    }
    
    for n, U in known_golomb.items():
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # 2. Generate greedy Golomb Rulers for larger n
    for n in range(11, 18):
        if _elapsed() > 60:
            break
        U = _greedy_golomb(n)
        if len(U) == n:
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
        
    # 3. Local Search on the best candidate
    # We allow "soft" Golomb rulers (allowing some duplicate differences)
    # if it reduces M significantly or improves the ratio.
    
    U = list(best_U)
    cur_score = best_score
    rng = np.random.RandomState(42)
    
    improved = True
    while improved and _elapsed() < 140:
        improved = False
        n = len(U)
        if n < 2:
            break
        max_val = max(U)
        
        # Try moving each element slightly
        # We focus on compressing the set (reducing M)
        for i in range(1, n):
            if _elapsed() > 142:
                break
            original_val = U[i]
            # Try values slightly smaller than original
            # But must maintain sorted order: U[i-1] < U[i] < U[i+1]
            lower_bound = U[i-1] + 1
            upper_bound = U[i+1] - 1 if i < n - 1 else original_val
            
            # Try a few candidates below original
            for delta in range(1, 10):
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
        
        # Try adding a new element if n is small
        if n < 15:
            if _elapsed() > 142:
                break
            # Try adding element near the end or in gaps
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

# Keep the original function names and signatures
def construct():
    return _construct_impl()

# Alias for the actual implementation
_construct_impl = construct
# Wait, the requirement says "Keep the function names and signatures of the seed program".
# The seed program has `construct()`.
# I defined `construct` twice. I need to fix that.

# Let's restructure to be clean.
# I will define the logic in a helper and call it from construct.

def _main_logic():
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
    }
    
    for n, U in known_golomb.items():
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # 2. Generate greedy Golomb Rulers for larger n
    for n in range(11, 18):
        if _elapsed() > 60:
            break
        U = _greedy_golomb(n)
        if len(U) == n:
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
        
    # 3. Local Search on the best candidate
    U = list(best_U)
    cur_score = best_score
    rng = np.random.RandomState(42)
    
    improved = True
    while improved and _elapsed() < 140:
        improved = False
        n = len(U)
        if n < 2:
            break
        max_val = max(U)
        
        # Try moving each element slightly to compress
        for i in range(1, n):
            if _elapsed() > 142:
                break
            original_val = U[i]
            lower_bound = U[i-1] + 1
            upper_bound = U[i+1] - 1 if i < n - 1 else original_val
            
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
                    break
        
        # Try adding a new element
        if n < 15:
            if _elapsed() > 142:
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
    return _main_logic()
