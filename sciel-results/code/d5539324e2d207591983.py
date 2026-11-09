import random
import time
import sys
from collections import deque

N = 12
MAX_SETS = 1 << N

def compute_threshold(trans):
    """
    trans: list of two lists, each of length N.
    trans[0][i] = destination of state i on letter 0
    trans[1][i] = destination of state i on letter 1
    Returns the length of the shortest reset word, or float('inf') if not synchronizing.
    """
    # Precompute images of all subsets for both letters
    img0 = [0] * MAX_SETS
    img1 = [0] * MAX_SETS
    
    # Precompute destination masks
    dest0 = [0] * N
    dest1 = [0] * N
    for i in range(N):
        dest0[i] = 1 << trans[0][i]
        dest1[i] = 1 << trans[1][i]
        
    # Compute images for all subsets
    for S in range(MAX_SETS):
        m0 = 0
        m1 = 0
        # Iterate over set bits only for efficiency
        temp = S
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            m0 |= dest0[i]
            m1 |= dest1[i]
            temp ^= lsb
        img0[S] = m0
        img1[S] = m1
        
    # BFS
    full_set = (1 << N) - 1
    visited = [False] * MAX_SETS
    visited[full_set] = True
    
    queue = deque()
    queue.append((full_set, 0))
    
    while queue:
        current_set, dist = queue.popleft()
        
        # Check if singleton
        if current_set & (current_set - 1) == 0:
            return dist
            
        # Expand
        next0 = img0[current_set]
        if not visited[next0]:
            visited[next0] = True
            queue.append((next0, dist + 1))
            
        next1 = img1[current_set]
        if not visited[next1]:
            visited[next1] = True
            queue.append((next1, dist + 1))
            
    return float('inf')

def construct():
    start_time = time.time()
    time_limit = 55  # Safety margin for 60s limit
    
    # Initialize with a known synchronizing automaton
    # Simple cycle + reset
    trans = [[(i + 1) % N for i in range(N)], [0] * N]
    best_trans = [row[:] for row in trans]
    best_threshold = compute_threshold(best_trans)
    
    # Local search with simulated annealing
    current_trans = [row[:] for row in trans]
    current_threshold = best_threshold
    
    # Parameters
    temperature = 10.0
    cooling = 0.9999
    
    random.seed(42)
    
    # Track best
    while time.time() - start_time < time_limit:
        # Pick a random transition to change
        q = random.randint(0, N - 1)
        a = random.randint(0, 1)
        old_dest = current_trans[a][q]
        new_dest = random.randint(0, N - 1)
        
        if new_dest == old_dest:
            continue
            
        current_trans[a][q] = new_dest
        
        # Check if still synchronizing and compute new threshold
        new_threshold = compute_threshold(current_trans)
        
        if new_threshold == float('inf'):
            # Not synchronizing, revert
            current_trans[a][q] = old_dest
            continue
            
        # Acceptance criterion
        if new_threshold >= current_threshold:
            current_threshold = new_threshold
            if new_threshold > best_threshold:
                best_threshold = new_threshold
                best_trans = [row[:] for row in current_trans]
        else:
            # Calculate acceptance probability
            delta = current_threshold - new_threshold
            # Avoid division by zero or underflow
            if temperature > 1e-10:
                prob = pow(2.718281828, -delta / temperature)
                if random.random() < prob:
                    current_threshold = new_threshold
                else:
                    current_trans[a][q] = old_dest
            else:
                current_trans[a][q] = old_dest
                
        temperature *= cooling
        
        # If temperature is too low, restart or reset
        if temperature < 1.0:
            temperature = 5.0

    return best_trans
