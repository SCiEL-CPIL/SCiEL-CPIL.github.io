import time
import itertools
from math import log

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

def construct():
    start = time.time()
    deadline = start + 110  # 110s deadline, 40s safety margin
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Try structured families
    # Family 1: {0, 1, 3, 7, ..., 2^k - 1}
    for k in range(2, 20):
        if time.time() > deadline:
            break
        U = [2**i - 1 for i in range(k)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Family 2: {0, 1, 3, 9, 27, ...} powers of 3
    for k in range(2, 15):
        if time.time() > deadline:
            break
        U = [3**i for i in range(k)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Family 3: Geometric progression with tunable base b
    # U = {floor(b^k) for k=0..m}, b > 1
    # Sweep over b in [1.5, 10] and m in [3, 50]
    def try_geometric(b, m):
        U = []
        val = 1
        for k in range(m + 1):
            U.append(int(val))
            val *= b
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        return U
    
    # Coarse sweep over b
    for m in range(3, 30):
        if time.time() > deadline:
            break
        for b in [1.5, 1.7, 2.0, 2.3, 2.7, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
            if time.time() > deadline:
                break
            U = try_geometric(b, m)
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Fine-tune around best b found (if any improvement was made)
    # Try more granular b values
    for m in range(3, 25):
        if time.time() > deadline:
            break
        for b in [1.8, 1.9, 2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 2.9, 3.2, 3.3, 3.4, 3.6, 3.8, 4.2, 4.5, 4.8, 5.5, 6.5, 7.5, 8.5, 9.5]:
            if time.time() > deadline:
                break
            U = try_geometric(b, m)
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Family 4: Mixed - try to grow from best
    # Local search: try adding elements
    def try_grow(U, max_val_limit=10000):
        U = sorted(set(U))
        current_score = compute_score(U)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each candidate
            candidates = list(range(1, min(max_val_limit, max(U)*2+10)))
            # Also try some larger candidates
            candidates += [max(U)*2, max(U)*3, max(U)*5, max(U)*10]
            candidates = sorted(set(candidates))
            for c in candidates:
                if c in U:
                    continue
                new_U = U + [c]
                new_score = compute_score(new_U)
                if new_score > current_score:
                    current_score = new_score
                    U = new_U
                    improved = True
                    break  # Restart candidate search with new U
        return U, current_score
    
    # Try growing from a few starting points
    starts = [[0,1,3], [0,1,3,7], [0,1,3,9], best_U[:]]
    for start_U in starts:
        if time.time() > deadline:
            break
        U, s = try_grow(start_U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    return best_U
