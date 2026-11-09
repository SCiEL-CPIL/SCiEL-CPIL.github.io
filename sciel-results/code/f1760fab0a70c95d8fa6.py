import time
import sys
import math
import random
from collections import defaultdict

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_edge_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def cycle_score(cycle):
    """Compute the max edge position in the cycle."""
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        pos = get_edge_pos(u, v)
        if pos > max_pos:
            max_pos = pos
    return max_pos

def simulated_annealing(initial_cycle, n, deadline):
    """
    Use simulated annealing to minimize the max edge position.
    Returns the best cycle found.
    """
    best_cycle = initial_cycle[:]
    best_score = cycle_score(best_cycle)
    current_cycle = initial_cycle[:]
    current_score = best_score
    
    # Temperature schedule
    T_start = 100.0
    T_end = 0.1
    cooling_rate = 0.999
    
    T = T_start
    rng = random.Random(42)
    
    while time.time() < deadline and T > T_end:
        # Generate a 2-opt move
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j or i == j - 1:
            continue
        
        # Ensure i < j
        if i > j:
            i, j = j, i
        
        # 2-opt: reverse segment from i to j
        new_cycle = current_cycle[:i] + current_cycle[i:j+1][::-1] + current_cycle[j+1:]
        
        new_score = cycle_score(new_cycle)
        
        # Accept if better, or with probability exp(-delta/T)
        if new_score < current_score:
            current_cycle = new_cycle
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_cycle = current_cycle[:]
        else:
            delta = new_score - current_score
            if delta == 0:
                accept = True
            else:
                accept_prob = math.exp(-delta / T)
                accept = rng.random() < accept_prob
            
            if accept:
                current_cycle = new_cycle
                current_score = new_score
        
        T *= cooling_rate
    
    return best_cycle

def greedy_initial_cycle(n):
    """
    Create an initial cycle using greedy heuristic:
    At each step, pick the next vertex that minimizes the max edge position so far.
    """
    used = [False] * n
    used[0] = True
    cycle = [0]
    
    for _ in range(n - 1):
        last = cycle[-1]
        best_next = None
        best_max_pos = float('inf')
        
        for v in range(n):
            if not used[v]:
                pos = get_edge_pos(last, v)
                # Also consider the edge from v back to 0 if this is the last step
                if len(cycle) == n - 2:
                    pos_back = get_edge_pos(v, 0)
                    pos = max(pos, pos_back)
                
                if pos < best_max_pos:
                    best_max_pos = pos
                    best_next = v
        
        if best_next is None:
            # Fallback: pick any unused vertex
            for v in range(n):
                if not used[v]:
                    best_next = v
                    break
        
        used[best_next] = True
        cycle.append(best_next)
    
    return cycle

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    n = N
    
    # Get a good initial cycle using greedy
    initial_cycle = greedy_initial_cycle(n)
    
    # Run simulated annealing
    best_cycle = simulated_annealing(initial_cycle, n, deadline)
    
    return best_cycle

# Test locally
if __name__ == "__main__":
    c = construct()
    max_pos = 0
    for i in range(N):
        u = c[i]
        v = c[(i+1)%N]
        a, b = min(u, v), max(u, v)
        p = edge_pos[(a, b)]
        if p > max_pos:
            max_pos = p
    print(f"Cycle: {c}")
    print(f"Score: {max_pos}")
