import random
import time

N = 256

def is_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            b = S[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def is_maximal(S, N):
    """Check if S is inclusion-maximal in [1,N]."""
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        
        # Check if x can be added
        if 2 * x in sums:
            continue
            
        valid = True
        for a in S:
            if (x + a) in sums:
                valid = False
                break
        
        if valid:
            return False
    return True

def greedy_sidon(order, N):
    """Greedy Sidon set construction given an order of elements."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def try_reduce(S, N, deadline=None):
    """Try to reduce the maximal Sidon set by removing elements."""
    S = list(S)
    improved = True
    while improved:
        improved = False
        indices = list(range(len(S)))
        random.shuffle(indices)
        
        for i in indices:
            if deadline and time.time() > deadline:
                return S
            
            candidate = S[:i] + S[i+1:]
            if len(candidate) == 0:
                continue
            
            if is_maximal(candidate, N):
                S = candidate
                improved = True
                break
    return S

def construct_powers_of_two(N):
    """Construct a maximal Sidon set using powers of 2."""
    S = []
    x = 1
    while x <= N:
        S.append(x)
        x *= 2
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best = None
    best_size = N + 1
    
    # 1. Try the explicit Powers of 2 construction (known to be small and maximal)
    S_pow2 = construct_powers_of_two(N)
    if is_maximal(S_pow2, N):
        best = S_pow2
        best_size = len(S_pow2)

    # 2. Try ascending greedy as a baseline (should also find powers of 2 or similar)
    order = list(range(1, N + 1))
    S = greedy_sidon(order, N)
    S = try_reduce(S, N, deadline)
    if len(S) < best_size:
        best = S
        best_size = len(S)
    
    # 3. Try multiple random orderings
    random.seed(42)
    for trial in range(1000):
        if time.time() > deadline:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = greedy_sidon(order, N)
        
        # Try to reduce
        S = try_reduce(S, N, deadline)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
            
            # Early stop if we hit theoretical minimum ~ sqrt(N)
            if best_size <= 16:
                break
    
    if best is None:
        # Fallback to ascending greedy
        order = list(range(1, N + 1))
        best = greedy_sidon(order, N)
    
    return best
