import time
import math

# EVOLVE-BLOCK-START
def _score(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    maxU = max(U)
    if maxU == 0:
        return 0.0
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    n = len(U)
    len_sum = len(sumset)
    len_diff = len(diffset)
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 1:
        return 1.0  # or handle appropriately
    log_ratio = math.log2(ratio)
    log_denom = math.log2(2 * maxU + 1)
    if log_denom == 0:
        return 0.0
    return 1.0 + log_ratio / log_denom

def _greedy_construct(max_size=20, max_val=200):
    """
    Greedily construct a set U of nonnegative integers containing 0
    to maximize the score.
    """
    U = [0]
    current_score = _score(U)
    for _ in range(max_size - 1):
        best_candidate = None
        best_score = current_score
        # Try adding each candidate value
        for v in range(1, max_val + 1):
            if v in U:
                continue
            new_U = U + [v]
            s = _score(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_candidate = v
        if best_candidate is None:
            break
        U.append(best_candidate)
        U.sort()
        current_score = best_score
    return U

def _local_search(U, max_iterations=100, max_val=500):
    """
    Local search: try to improve the set by swapping or replacing elements.
    """
    U = sorted(set(U))
    current_score = _score(U)
    for _ in range(max_iterations):
        improved = False
        # Try adding a new element
        for v in range(1, max_val + 1):
            if v in U:
                continue
            new_U = U + [v]
            s = _score(new_U)
            if s > current_score + 1e-12:
                U = new_U
                current_score = s
                improved = True
                break
        if improved:
            continue
        
        # Try removing an element (except 0)
        for i in range(1, len(U)):
            new_U = [x for j, x in enumerate(U) if j != i]
            s = _score(new_U)
            if s > current_score + 1e-12:
                U = new_U
                current_score = s
                improved = True
                break
        if improved:
            continue
        
        # Try replacing an element (except 0)
        for i in range(1, len(U)):
            for v in range(1, max_val + 1):
                if v in U:
                    continue
                new_U = U[:]
                new_U[i] = v
                new_U.sort()
                s = _score(new_U)
                if s > current_score + 1e-12:
                    U = new_U
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return U

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Start with a good initial set (Sidon-like)
    U = [0, 1, 4, 6]
    current_score = _score(U)
    
    # Greedy expansion
    U = _greedy_construct(max_size=15, max_val=100)
    current_score = _score(U)
    
    # Local search
    U = _local_search(U, max_iterations=50, max_val=300)
    current_score = _score(U)
    
    # Try a few different starting points if time permits
    if time.time() < deadline:
        start2 = [0, 1, 3, 7]
        U2 = _greedy_construct(max_size=15, max_val=100)
        U2 = [0, 1, 3, 7] + U2[4:] if len(U2) > 4 else U2
        U2 = _local_search(U2, max_iterations=30, max_val=300)
        s2 = _score(U2)
        if s2 > current_score:
            U = U2
            current_score = s2
    
    if time.time() < deadline:
        start3 = [0, 2, 5, 9]
        U3 = list(start3)
        U3 = _local_search(U3, max_iterations=30, max_val=300)
        s3 = _score(U3)
        if s3 > current_score:
            U = U3
            current_score = s3
    
    return sorted(set(U))
# EVOLVE-BLOCK-END
