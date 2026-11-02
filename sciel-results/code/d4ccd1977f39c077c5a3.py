import time
import itertools

def compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if len(U) < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute U+U and U-U
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    n_sum = len(sumset)
    n_diff = len(diffset)
    if n_sum == 0:
        return 0.0
    import math
    ratio = n_diff / n_sum
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 0.0
    return 1 + math.log(ratio) / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    best_score = 0.0
    best_U = [0, 1, 3]
    
    # First, try the seed and some small hand-crafted sets
    candidates = [
        [0, 1, 3],
        [0, 1, 2],
        [0, 1, 4],
        [0, 1, 5],
        [0, 1, 6],
        [0, 1, 7],
        [0, 1, 8],
        [0, 1, 9],
        [0, 1, 10],
        [0, 1, 2, 4],
        [0, 1, 2, 5],
        [0, 1, 3, 6],
        [0, 1, 3, 7],
        [0, 1, 4, 8],
        [0, 1, 4, 9],
        [0, 1, 5, 10],
        [0, 1, 5, 11],
        [0, 1, 6, 12],
        [0, 1, 6, 13],
        [0, 1, 7, 14],
        [0, 1, 7, 15],
        [0, 2, 3],
        [0, 2, 4],
        [0, 2, 5],
        [0, 2, 6],
        [0, 2, 7],
        [0, 2, 8],
        [0, 2, 9],
        [0, 2, 10],
        [0, 3, 4],
        [0, 3, 5],
        [0, 3, 6],
        [0, 3, 7],
        [0, 3, 8],
        [0, 3, 9],
        [0, 3, 10],
        [0, 3, 11],
        [0, 4, 5],
        [0, 4, 6],
        [0, 4, 7],
        [0, 4, 8],
        [0, 4, 9],
        [0, 4, 10],
        [0, 4, 11],
        [0, 4, 12],
        [0, 5, 6],
        [0, 5, 7],
        [0, 5, 8],
        [0, 5, 9],
        [0, 5, 10],
        [0, 5, 11],
        [0, 5, 12],
        [0, 5, 13],
        [0, 6, 7],
        [0, 6, 8],
        [0, 6, 9],
        [0, 6, 10],
        [0, 6, 11],
        [0, 6, 12],
        [0, 6, 13],
        [0, 6, 14],
        [0, 7, 8],
        [0, 7, 9],
        [0, 7, 10],
        [0, 7, 11],
        [0, 7, 12],
        [0, 7, 13],
        [0, 7, 14],
        [0, 7, 15],
        [0, 8, 9],
        [0, 8, 10],
        [0, 8, 11],
        [0, 8, 12],
        [0, 8, 13],
        [0, 8, 14],
        [0, 8, 15],
        [0, 8, 16],
        [0, 9, 10],
        [0, 9, 11],
        [0, 9, 12],
        [0, 9, 13],
        [0, 9, 14],
        [0, 9, 15],
        [0, 9, 16],
        [0, 9, 17],
        [0, 10, 11],
        [0, 10, 12],
        [0, 10, 13],
        [0, 10, 14],
        [0, 10, 15],
        [0, 10, 16],
        [0, 10, 17],
        [0, 10, 18],
        [0, 10, 19],
        [0, 11, 12],
        [0, 11, 13],
        [0, 11, 14],
        [0, 11, 15],
        [0, 11, 16],
        [0, 11, 17],
        [0, 11, 18],
        [0, 11, 19],
        [0, 11, 20],
        [0, 12, 13],
        [0, 12, 14],
        [0, 12, 15],
        [0, 12, 16],
        [0, 12, 17],
        [0, 12, 18],
        [0, 12, 19],
        [0, 12, 20],
        [0, 12, 21],
        [0, 13, 14],
        [0, 13, 15],
        [0, 13, 16],
        [0, 13, 17],
        [0, 13, 18],
        [0, 13, 19],
        [0, 13, 20],
        [0, 13, 21],
        [0, 13, 22],
        [0, 14, 15],
        [0, 14, 16],
        [0, 14, 17],
        [0, 14, 18],
        [0, 14, 19],
        [0, 14, 20],
        [0, 14, 21],
        [0, 14, 22],
        [0, 14, 23],
        [0, 15, 16],
        [0, 15, 17],
        [0, 15, 18],
        [0, 15, 19],
        [0, 15, 20],
        [0, 15, 21],
        [0, 15, 22],
        [0, 15, 23],
        [0, 15, 24],
    ]
    
    for U in candidates:
        if time.time() > deadline:
            break
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Now do a local search / greedy construction
    # Start from best_U and try to improve by adding or modifying elements
    
    def try_add_element(U, max_val, time_limit):
        """Try adding a new element to U and see if it improves score."""
        U = sorted(set(U))
        current_score = compute_score(U)
        best_new = None
        best_new_score = current_score
        
        # Try adding elements from 1 to max_val
        for x in range(1, max_val + 1):
            if x in U:
                continue
            new_U = U + [x]
            new_U = sorted(set(new_U))
            s = compute_score(new_U)
            if s > best_new_score:
                best_new_score = s
                best_new = new_U
            if time.time() > time_limit:
                break
        
        return best_new, best_new_score
    
    # Greedy construction: start small, keep adding elements that improve score
    def greedy_construct(start_size, max_val, time_limit):
        U = [0]
        # Try to build up from scratch
        for _ in range(start_size):
            if time.time() > time_limit:
                break
            U, s = try_add_element(U, max_val, time_limit)
            if U is None:
                break
        return U, compute_score(U)
    
    # Try different greedy starts
    for start_size in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        U, s = greedy_construct(start_size, 50, deadline)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    # Try local search on best_U: swap elements
    def local_search(U, max_val, time_limit, iterations=100):
        U = sorted(set(U))
        current_score = compute_score(U)
        for _ in range(iterations):
            if time.time() > time_limit:
                break
            improved = False
            for i, elem in enumerate(U):
                if elem == 0:
                    continue
                for new_elem in range(1, max_val + 1):
                    if new_elem == elem:
                        continue
                    if new_elem in U:
                        continue
                    new_U = U[:i] + [new_elem] + U[i+1:]
                    new_U = sorted(set(new_U))
                    if len(new_U) != len(U):
                        continue
                    s = compute_score(new_U)
                    if s > current_score:
                        current_score = s
                        U = new_U
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return U, current_score
    
    # Local search on best_U
    U, s = local_search(best_U, 100, deadline, iterations=50)
    if s > best_score:
        best_score = s
        best_U = list(U)
    
    # Try more aggressive local search with larger max_val
    U, s = local_search(best_U, 200, deadline, iterations=30)
    if s > best_score:
        best_score = s
        best_U = list(U)
    
    # Try a few random restarts with local search
    import random
    random.seed(42)
    for _ in range(5):
        if time.time() > deadline:
            break
        # Random initial set
        size = random.randint(3, 8)
        U = [0] + random.sample(range(1, 100), size - 1)
        U = sorted(set(U))
        U, s = local_search(U, 150, deadline, iterations=30)
        if s > best_score:
            best_score = s
            best_U = list(U)
    
    return sorted(set(best_U))

