"""Construct finite set U of nonnegative integers containing 0 to maximize
the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1) on C6."""
import time
import math
import random
import numpy as np

def _score_set(U):
    """Compute the score for a given set U."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset and difference set sizes
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    sum_count = len(sumset)
    diff_count = len(diffset)
    
    if sum_count == 0:
        return 0.0
    
    ratio = diff_count / sum_count
    if ratio <= 1:
        return 1.0
    
    log_term = math.log(ratio) / math.log(2 * max_u + 1)
    return 1.0 + log_term

def _score_set_numpy(U):
    """Faster scoring using numpy for larger sets."""
    U = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U)
    if n < 2:
        return 0.0
    if 0 not in U:
        return 0.0
    max_u = int(U.max())
    if max_u == 0:
        return 0.0
    
    # Use broadcasting for sumset
    sums = U[:, None] + U[None, :]
    sum_count = len(np.unique(sums))
    
    diffs = U[:, None] - U[None, :]
    diff_count = len(np.unique(diffs))
    
    if sum_count == 0:
        return 0.0
    
    ratio = diff_count / sum_count
    if ratio <= 1:
        return 1.0
    
    log_term = math.log(ratio) / math.log(2 * max_u + 1)
    return 1.0 + log_term

def _construct_via_residue_class(M, target_residue, max_size=200, deadline=None):
    """
    Construct U as a subset of {0, 1, ..., M-1} that are congruent to target_residue mod M.
    Actually, let's try a different modular structure.
    
    Better approach: U = {x in [0, K] : x mod M == r} for some M, r, K.
    This gives an arithmetic progression with step M, starting at r.
    For an AP {r, r+M, r+2M, ..., r+(k-1)M}, we have:
    |U+U| = 2k-1 (sums range from 2r to 2r+(2k-2)M, step M)
    |U-U| = 2k-1 (diffs range from -(k-1)M to (k-1)M, step M)
    So ratio = 1, which is not helpful.
    
    Let's try a different structure: U = {x in [0, K] : x mod M in S} where S is a subset of residues.
    This is more complex. Let's try the simplest non-trivial case: S = {0, 1} mod M.
    So U = {x in [0, K] : x mod M in {0, 1}}.
    """
    # Actually, let's try a completely different structural approach.
    # Use a "Golomb ruler" like structure but with a twist.
    # Or use the fact that for U = {0} union {a_i}, we want |U-U|/|U+U| large.
    # |U-U| includes all differences a_i - a_j, and |U+U| includes all sums a_i + a_j.
    
    # Let's try: U = {0} union {b * 2^i for i=0..k-1} where b is a base.
    # For U = {0, b, 2b, 4b, ..., 2^(k-1)*b}:
    # Sums: 0, b, 2b, 3b, 4b, 5b, 6b, 7b, ... up to (2^k - 1)*b
    # Actually, the sumset of {0, 1, 2, 4, ..., 2^(k-1)} * b is {0, 1, 2, ..., 2^k - 1} * b
    # So |U+U| = 2^k
    # Differences: -(2^(k-1)*b), ..., 0, ..., 2^(k-1)*b
    # The difference set is {-(2^k - 1)*b, ..., (2^k - 1)*b} intersected with multiples of b?
    # No. For U = {0, b, 2b, 4b, ..., 2^(k-1)b}, the differences are all values of form (2^i - 2^j)*b.
    # The set {2^i - 2^j : 0 <= i, j < k} has size... let's compute.
    # For k=3: {0, 1, 2, 4} -> diffs: 0, 1, 2, 3, 4, 5, 6, 7? No.
    # 0-0=0, 0-1=-1, 0-2=-2, 0-4=-4, 1-0=1, 1-1=0, 1-2=-1, 1-4=-3, 2-0=2, 2-1=1, 2-2=0, 2-4=-2, 4-0=4, 4-1=3, 4-2=2, 4-4=0
    # Distinct: {-4, -3, -2, -1, 0, 1, 2, 3, 4} -> size 9
    # Sums: 0+0=0, 0+1=1, 0+2=2, 0+4=4, 1+0=1, 1+1=2, 1+2=3, 1+4=5, 2+0=2, 2+1=3, 2+2=4, 2+4=6, 4+0=4, 4+1=5, 4+2=6, 4+4=8
    # Distinct: {0, 1, 2, 3, 4, 5, 6, 8} -> size 8
    # Ratio = 9/8 = 1.125
    # max(U) = 4b, 2*max+1 = 8b+1
    # Score = 1 + log(1.125)/log(8b+1)
    
    # For b=1: 1 + log(1.125)/log(9) = 1 + 0.1178/2.197 = 1.0536
    # That's much better than 1.0275!
    
    # Let's verify: U = [0, 1, 2, 4]
    # Sums: {0,1,2,3,4,5,6,8} -> 8 elements
    # Diffs: {-4,-3,-2,-1,0,1,2,3,4} -> 9 elements
    # Ratio = 9/8
    # 2*max+1 = 9
    # log(9/8)/log(9) = ln(1.125)/ln(9) ≈ 0.11778/2.1972 ≈ 0.0536
    # Score ≈ 1.0536
    
    # Let's try k=4: U = [0, 1, 2, 4, 8]
    # Sums: 0 to 15, all integers -> 16 elements
    # Diffs: -8 to 8, but which ones?
    # 2^i - 2^j for i,j in 0..3
    # Values: 0, ±1, ±2, ±3, ±4, ±5, ±6, ±7, ±8
    # Are all integers from -8 to 8 achievable?
    # 8 = 8-0, 7 = 8-1, 6 = 8-2, 5 = 8-3, 4 = 4-0, 3 = 4-1, 2 = 2-0, 1 = 1-0
    # Yes, all from -8 to 8. So |U-U| = 17
    # Ratio = 17/16 = 1.0625
    # 2*max+1 = 17
    # Score = 1 + log(17/16)/log(17) = 1 + ln(1.0625)/ln(17) ≈ 1 + 0.0606/2.833 ≈ 1.0214
    # Worse than k=3!
    
    # So k=3 with b=1 gives the best among this family.
    # Let's try other bases b for k=3:
    # U = [0, b, 2b, 4b]
    # |U+U| = 8, |U-U| = 9
    # 2*max+1 = 8b+1
    # Score = 1 + log(9/8)/log(8b+1)
    # This is maximized when 8b+1 is minimized, i.e., b=1.
    
    # What about other structures? Let's try U = [0, 1, 3, 7] (differences double)
    # Sums: 0,1,2,3,4,6,7,8,10,14 -> let me compute
    # 0+0=0, 0+1=1, 0+3=3, 0+7=7, 1+0=1, 1+1=2, 1+3=4, 1+7=8, 3+0=3, 3+1=4, 3+3=6, 3+7=10, 7+0=7, 7+1=8, 7+3=10, 7+7=14
    # Distinct: {0,1,2,3,4,6,7,8,10,14} -> 10 elements
    # Diffs: 0-0=0, 0-1=-1, 0-3=-3, 0-7=-7, 1-0=1, 1-1=0, 1-3=-2, 1-7=-6, 3-0=3, 3-1=2, 3-3=0, 3-7=-4, 7-0=7, 7-1=6, 7-3=4, 7-7=0
    # Distinct: {-7,-6,-4,-3,-2,-1,0,1,2,3,4,6,7} -> 13 elements
    # Ratio = 13/10 = 1.3
    # 2*max+1 = 15
    # Score = 1 + log(1.3)/log(15) = 1 + ln(1.3)/ln(15) ≈ 1 + 0.2624/2.708 ≈ 1.0969
    # Even better!
    
    # Let's try U = [0, 1, 3, 7, 15] (one less than powers of 2)
    # This is getting complex. Let's just do a local search starting from good initial sets.
    
    return None, 0.0

def _local_search(U, deadline, max_iter=10000):
    """Local search to improve the score."""
    current = sorted(set(U))
    current_score = _score_set(current)
    best_set = list(current)
    best_score = current_score
    no_improve_count = 0
    max_val = max(current) + 20
    
    while time.time() < deadline - 2 and no_improve_count < 50:
        improved = False
        
        # Try adding an element
        for new_elem in range(0, max_val + 1):
            if new_elem in current:
                continue
            candidate = sorted(current + [new_elem])
            s = _score_set(candidate)
            if s > best_score + 1e-12:
                best_score = s
                best_set = candidate
                current = candidate
                max_val = max(current) + 10
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
        
        # Try replacing an element
        for i in range(1, len(current)):
            for new_elem in range(0, max_val + 1):
                if new_elem in current:
                    continue
                candidate = list(current)
                candidate[i] = new_elem
                candidate = sorted(candidate)
                s = _score_set(candidate)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = candidate
                    current = candidate
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try removing an element (except 0)
        for i in range(1, len(current)):
            candidate = current[:i] + current[i+1:]
            s = _score_set(candidate)
            if s > best_score + 1e-12:
                best_score = s
                best_set = candidate
                current = candidate
                improved = True
                no_improve_count = 0
                break
        
        if improved:
            continue
        
        # Try swapping two elements
        for i in range(1, len(current)):
            for j in range(i+1, len(current)):
                for new_i in range(0, max_val + 1):
                    if new_i in current:
                        continue
                    for new_j in range(0, max_val + 1):
                        if new_j in current or new_j == new_i:
                            continue
                        candidate = list(current)
                        candidate[i] = new_i
                        candidate[j] = new_j
                        candidate = sorted(candidate)
                        s = _score_set(candidate)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = candidate
                            current = candidate
                            improved = True
                            no_improve_count = 0
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            # Occasionally try a random perturbation
            if no_improve_count % 10 == 0:
                if len(current) > 2:
                    i = random.randint(1, len(current)-1)
                    new_elem = random.randint(0, max_val + 5)
                    candidate = list(current)
                    candidate[i] = new_elem
                    candidate = sorted(candidate)
                    s = _score_set(candidate)
                    if s > best_score * 0.95:  # Allow slight degradation to escape local optima
                        best_set = candidate
                        current = candidate
                        best_score = s
                        no_improve_count = 0
    
    return best_set, best_score

def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Start with known good sets
    initial_sets = [
        [0, 1, 2, 4],       # Score ~1.0536
        [0, 1, 3, 7],       # Score ~1.0969
        [0, 1, 3],
        [0, 1, 2, 5],
        [0, 1, 3, 4, 6],
        [0, 1, 2, 4, 6],
        [0, 1, 3, 5, 6],
        [0, 1, 2, 3, 5],
        [0, 1, 3, 7, 15],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 6, 12],
        [0, 1, 4, 7, 12],
    ]
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    for init_set in initial_sets:
        s = _score_set(init_set)
        if s > best_score:
            best_score = s
            best_set = list(init_set)
    
    # Local search from the best initial set
    if time.time() < deadline - 10:
        set1, score1 = _local_search(best_set, deadline - 10)
        if score1 > best_score:
            best_score = score1
            best_set = set1
    
    # Try local search from other promising initial sets
    for init_set in initial_sets[:5]:
        if time.time() < deadline - 5:
            set2, score2 = _local_search(init_set, deadline - 5)
            if score2 > best_score:
                best_score = score2
                best_set = set2
    
    # Final local search from the best found
    if time.time() < deadline - 2:
        set3, score3 = _local_search(best_set, deadline - 2)
        if score3 > best_score:
            best_score = score3
            best_set = set3
    
    return best_set
