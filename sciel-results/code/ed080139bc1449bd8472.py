import time
import random
import math
from collections import deque

N = 12

def _bfs_reset_length(a, b):
    """
    Computes the exact shortest reset word length.
    Uses precomputed subset transitions for speed.
    """
    full_mask = (1 << N) - 1
    
    # Precompute transitions for all subsets
    # trans_a[mask] = mask after applying 'a'
    # trans_b[mask] = mask after applying 'b'
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        ma = 0
        mb = 0
        # Iterate through set bits
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            ma |= (1 << a[idx])
            mb |= (1 << b[idx])
            m ^= lsb
        trans_a[mask] = ma
        trans_b[mask] = mb
        
    # BFS
    visited = [False] * (1 << N)
    visited[full_mask] = True
    queue = deque()
    queue.append((full_mask, 0))
    
    while queue:
        mask, depth = queue.popleft()
        
        # Check if singleton
        if mask & (mask - 1) == 0:
            return depth
            
        # Try 'a'
        next_a = trans_a[mask]
        if not visited[next_a]:
            visited[next_a] = True
            queue.append((next_a, depth + 1))
            
        # Try 'b'
        next_b = trans_b[mask]
        if not visited[next_b]:
            visited[next_b] = True
            queue.append((next_b, depth + 1))
            
    return -1

def _pinwheel():
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(2, N):
        b[i] = i - 1
    return [a, b]

def _local_search_sa(a, b, deadline):
    """
    Simulated Annealing local search.
    """
    current_a, current_b = a[:], b[:]
    current_score = _bfs_reset_length(current_a, current_b)
    best_a, best_b = current_a[:], current_b[:]
    best_score = current_score
    
    random.seed(42)
    
    # SA parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    while time.time() < deadline and T > T_min:
        # Generate neighbor
        new_a = current_a[:]
        new_b = current_b[:]
        
        # Mutation: change 1 or 2 entries
        num_mutations = 1 if random.random() < 0.7 else 2
        
        for _ in range(num_mutations):
            if random.random() < 0.5:
                idx = random.randint(0, N - 1)
                new_a[idx] = random.randint(0, N - 1)
            else:
                idx = random.randint(0, N - 1)
                new_b[idx] = random.randint(0, N - 1)
        
        new_score = _bfs_reset_length(new_a, new_b)
        
        if new_score == -1:
            # Not synchronizing, treat as very bad
            delta = -1000 
        else:
            delta = new_score - current_score
            
        if delta > 0:
            # Accept improvement
            current_a, current_b = new_a, new_b
            current_score = new_score
            if current_score > best_score:
                best_a, best_b = current_a[:], current_b[:]
                best_score = current_score
        else:
            # Accept worse with probability exp(delta/T)
            if random.random() < math.exp(delta / T):
                current_a, current_b = new_a, new_b
                current_score = new_score
                
        T *= cooling_rate
        
    return best_a, best_b, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial candidate
    best_a, best_b = _pinwheel()
    best_score = _bfs_reset_length(best_a, best_b)
    
    # Try a few random restarts to find a better starting point
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        ra = [random.randint(0, N-1) for _ in range(N)]
        rb = [random.randint(0, N-1) for _ in range(N)]
        score = _bfs_reset_length(ra, rb)
        if score > best_score:
            best_a, best_b = ra, rb
            best_score = score
            
    # Run Simulated Annealing
    best_a, best_b, best_score = _local_search_sa(best_a, best_b, deadline)
    
    return [best_a, best_b]
