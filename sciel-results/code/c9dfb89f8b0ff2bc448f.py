import time
import itertools
from math import log
import numpy as np

def compute_score(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    return 1.0 + log(ratio) / log(2 * max_u + 1)

def fast_score(U):
    """Fast score using numpy for larger sets."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 1:
        max_u = U[0]
        return 1.0 if max_u == 0 else 1.0
    arr = np.array(U, dtype=np.int64)
    max_u = int(arr[-1])
    if max_u == 0:
        return 1.0
    # Sumset: all a+b for a,b in U
    sums = np.add.outer(arr, arr).ravel()
    sumset_size = len(np.unique(sums))
    # Diffset: all a-b for a,b in U
    diffs = np.subtract.outer(arr, arr).ravel()
    diffset_size = len(np.unique(diffs))
    ratio = diffset_size / sumset_size
    if ratio <= 1:
        return 1.0
    return 1.0 + np.log(ratio) / np.log(2 * max_u + 1)

def construct():
    start = time.time()
    deadline = start + 115  # 115s deadline, 35s safety margin
    
    best_U = [0, 1, 3]
    best_score = fast_score(best_U)
    
    # Try structured families
    # Family 1: {0, 1, 3, 7, ..., 2^k - 1}
    for k in range(2, 20):
        if time.time() > deadline:
            break
        U = [2**i - 1 for i in range(k)]
        s = fast_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Family 2: {0, 1, 3, 9, 27, ...} powers of 3
    for k in range(2, 15):
        if time.time() > deadline:
            break
        U = [3**i for i in range(k)]
        s = fast_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Family 3: Geometric progression with tunable base b
    def try_geometric(b, m):
        U = []
        val = 1.0
        for k in range(m + 1):
            U.append(int(val))
            val *= b
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        return U
    
    def score_geometric(b, m):
        U = try_geometric(b, m)
        return fast_score(U)
    
    def ternary_search_b(m, b_lo, b_hi, iters=50):
        lo = b_lo
        hi = b_hi
        for _ in range(iters):
            if time.time() > deadline:
                break
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            f1 = score_geometric(m1, m)
            f2 = score_geometric(m2, m)
            if f1 < f2:
                lo = m1
            else:
                hi = m2
        b_opt = (lo + hi) / 2.0
        return b_opt, score_geometric(b_opt, m)
    
    # Sweep over m and use ternary search for optimal b
    for m in range(3, 35):
        if time.time() > deadline:
            break
        b_lo = 1.01
        b_hi = 50.0 if m < 10 else 10.0 if m < 20 else 5.0
        
        b_opt, s_opt = ternary_search_b(m, b_lo, b_hi, iters=40)
        if s_opt > best_score:
            best_score = s_opt
            best_U = try_geometric(b_opt, m)
    
    # Fine-tune: try more granular m values and tighter b ranges
    for m in range(3, 25):
        if time.time() > deadline:
            break
        b_lo = 1.05
        b_hi = 20.0 if m < 10 else 8.0
        b_opt, s_opt = ternary_search_b(m, b_lo, b_hi, iters=60)
        if s_opt > best_score:
            best_score = s_opt
            best_U = try_geometric(b_opt, m)
    
    # Family 4: Algebraic construction - binary basis vectors
    # Select elements whose binary representations form a linearly independent set
    # This minimizes sumset collisions (carries) while maximizing difference dispersion
    if time.time() < deadline:
        # Try sets based on binary patterns with controlled density
        for bits in range(2, 12):
            if time.time() > deadline:
                break
            # Select a subset of {0, 1, ..., 2^bits - 1} with good properties
            # Use elements with exactly one bit set (powers of 2) plus 0
            U1 = [2**i for i in range(bits)]
            s1 = fast_score(U1)
            if s1 > best_score:
                best_score = s1
                best_U = U1[:]
            
            # Try elements with at most 2 bits set
            U2 = []
            for i in range(bits):
                for j in range(i + 1, bits):
                    U2.append(2**i + 2**j)
            U2 = sorted(set(U2 + [0]))
            if len(U2) > 2:
                s2 = fast_score(U2)
                if s2 > best_score:
                    best_score = s2
                    best_U = U2[:]
            
            # Try a mix: powers of 2 up to some point, then sparse additions
            for extra_bits in range(1, min(bits, 5)):
                U3 = [2**i for i in range(bits - extra_bits)]
                for i in range(bits - extra_bits, bits):
                    U3.append(2**i + 1)
                U3 = sorted(set(U3 + [0]))
                if len(U3) > 2:
                    s3 = fast_score(U3)
                    if s3 > best_score:
                        best_score = s3
                        best_U = U3[:]
    
    # Family 5: Local polish with add/remove operations
    if time.time() < deadline:
        U = best_U[:]
        current_score = best_score
        improved = True
        max_iterations = 200
        iteration = 0
        while improved and time.time() < deadline and iteration < max_iterations:
            improved = False
            iteration += 1
            max_u = max(U) if U else 0
            n = len(U)
            
            # Generate candidates for addition
            candidates = set()
            # Small integers
            candidates.update(range(1, min(2*max_u + 10, max_u + 50)))
            # Multiplicative candidates
            for mult in [1.1, 1.2, 1.5, 1.8, 2, 2.5, 3, 4, 5, 7, 10]:
                c = int(max_u * mult)
                if c > 0:
                    candidates.add(c)
            # Gaps between existing elements
            for i in range(len(U) - 1):
                gap = U[i+1] - U[i]
                if gap > 2:
                    candidates.add(U[i] + gap // 2)
            # Filter: only consider reasonable size increases
            candidates = {c for c in candidates if c > 0 and c not in U and c <= max_u * 3}
            candidates = sorted(candidates)
            
            # Try additions
            best_add = None
            best_add_score = current_score
            for c in candidates[:50]:  # Limit candidates for speed
                if time.time() > deadline:
                    break
                new_U = sorted(U + [c])
                new_score = fast_score(new_U)
                if new_score > best_add_score:
                    best_add_score = new_score
                    best_add = c
            
            if best_add is not None:
                U = sorted(U + [best_add])
                current_score = best_add_score
                improved = True
                continue
            
            # Try removals (only if set is large enough)
            if n > 3:
                best_remove = None
                best_remove_score = current_score
                for idx, elem in enumerate(U):
                    if elem == 0:
                        continue  # Never remove 0
                    if time.time() > deadline:
                        break
                    new_U = U[:idx] + U[idx+1:]
                    new_score = fast_score(new_U)
                    if new_score > best_remove_score:
                        best_remove_score = new_score
                        best_remove = elem
                
                if best_remove is not None:
                    U = [x for x in U if x != best_remove]
                    current_score = best_remove_score
                    improved = True
        
        if current_score > best_score:
            best_score = current_score
            best_U = U
    
    # Final: verify with exact compute_score
    final_score = compute_score(best_U)
    if final_score > best_score:
        best_score = final_score
    
    return best_U
