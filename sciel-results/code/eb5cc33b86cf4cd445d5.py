import time
import random

N = 512

def _can_add(S, x):
    """
    Check if x can be added to the set S (represented as an integer bitset)
    without creating a 4-term arithmetic progression.
    S has bit i set if i is in the set.
    """
    # Case 1: x is the 1st term: x, x+d, x+2d, x+3d
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if (S >> (x + d)) & 1 and (S >> (x + 2 * d)) & 1 and (S >> (x + 3 * d)) & 1:
            return False
    
    # Case 2: x is the 2nd term: x-d, x, x+d, x+2d
    max_d = min(x - 1, (N - x) // 2)
    for d in range(1, max_d + 1):
        if (S >> (x - d)) & 1 and (S >> (x + d)) & 1 and (S >> (x + 2 * d)) & 1:
            return False
    
    # Case 3: x is the 3rd term: x-2d, x-d, x, x+d
    max_d = min((x - 1) // 2, N - x)
    for d in range(1, max_d + 1):
        if (S >> (x - 2 * d)) & 1 and (S >> (x - d)) & 1 and (S >> (x + d)) & 1:
            return False
    
    # Case 4: x is the 4th term: x-3d, x-2d, x-d, x
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if (S >> (x - 3 * d)) & 1 and (S >> (x - 2 * d)) & 1 and (S >> (x - d)) & 1:
            return False
            
    return True

def _greedy_construct(order):
    """
    Greedily construct a set by adding elements in the given order.
    """
    S = 0
    for x in order:
        if _can_add(S, x):
            S |= (1 << x)
    return S

def _local_search(S):
    """
    Try to add any excluded element to S. Repeat until no more can be added.
    """
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if not (S >> x) & 1:
                if _can_add(S, x):
                    S |= (1 << x)
                    changed = True
    return S

def _popcount(S):
    """
    Count the number of bits set in S.
    """
    return bin(S).count('1')

def _constraint_score(x):
    """
    Calculate the number of potential 4-APs involving x.
    Lower score means less constrained.
    """
    score = 0
    score += (N - x) // 3
    score += min(x - 1, (N - x) // 2)
    score += min((x - 1) // 2, N - x)
    score += (x - 1) // 3
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_S = 0
    best_size = 0
    
    # Pre-compute constraint scores
    scores = {x: _constraint_score(x) for x in range(1, N + 1)}
    
    # 1. Smart Greedy: Sort by constraint score (ascending)
    smart_order = sorted(range(1, N + 1), key=lambda x: scores[x])
    S_smart = _greedy_construct(smart_order)
    S_smart = _local_search(S_smart)
    best_S = S_smart
    best_size = _popcount(S_smart)
    
    # 2. Random Greedy: A few restarts
    for _ in range(10):
        if time.time() > deadline - 5.0: break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_rand = _greedy_construct(order)
        S_rand = _local_search(S_rand)
        size = _popcount(S_rand)
        if size > best_size:
            best_size = size
            best_S = S_rand

    # 3. Iterated Local Search (ILS)
    while time.time() < deadline:
        # Perturb the best solution
        # Remove k random elements
        k = random.randint(1, 8)
        S = best_S
        elements = [i for i in range(1, N + 1) if (S >> i) & 1]
        
        if len(elements) > k:
            to_remove = random.sample(elements, k)
            for x in to_remove:
                S &= ~(1 << x)
        
        # Run local search on the perturbed set
        S = _local_search(S)
        size = _popcount(S)
        
        if size > best_size:
            best_size = size
            best_S = S
        else:
            # If no improvement, occasionally try a fresh random restart
            if random.random() < 0.05:
                order = list(range(1, N + 1))
                random.shuffle(order)
                S_rand = _greedy_construct(order)
                S_rand = _local_search(S_rand)
                size = _popcount(S_rand)
                if size > best_size:
                    best_size = size
                    best_S = S_rand

    # Convert best_S to sorted list
    result = [i for i in range(1, N + 1) if (best_S >> i) & 1]
    return result
