import time
import sys
import random

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build edge position map
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    a, b = min(u, v), max(u, v)
    edge_pos[(a, b)] = i + 1

def get_pos(u, v):
    a, b = min(u, v), max(u, v)
    return edge_pos[(a, b)]

def score_cycle(cycle):
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        pos = get_pos(u, v)
        if pos > max_pos:
            max_pos = pos
    return max_pos

def greedy_init():
    """Construct a cycle using a greedy heuristic: always pick the neighbor with smallest edge position."""
    n = N
    visited = [False] * n
    cycle = [0]
    visited[0] = True
    
    for _ in range(n - 1):
        last = cycle[-1]
        best_next = -1
        best_pos = float('inf')
        for v in range(n):
            if not visited[v]:
                # If v is the last one to be added, it must connect to 0
                if len(cycle) == n - 1:
                    # Check if v connects to 0
                    if 0 not in [w for w in range(n) if not visited[w] or w==0]: # This check is redundant if we just check edge
                         pass
                pos = get_pos(last, v)
                # Heuristic: prefer smaller positions
                # Also consider closing edge if it's the last step
                if len(cycle) == n - 1:
                    close_pos = get_pos(v, 0)
                    # Combined cost: max(pos, close_pos)
                    cost = max(pos, close_pos)
                    if cost < best_pos:
                        best_pos = cost
                        best_next = v
                else:
                    if pos < best_pos:
                        best_pos = pos
                        best_next = v
        
        if best_next == -1:
            # Fallback: random unvisited
            unvisited = [v for v in range(n) if not visited[v]]
            best_next = unvisited[0]
        
        cycle.append(best_next)
        visited[best_next] = True
    
    return cycle

def two_opt_move(cycle, i, j):
    """Perform a 2-opt move: reverse the segment from i to j."""
    n = len(cycle)
    if i > j:
        i, j = j, i
    # We reverse cycle[i..j]
    # Note: The cycle is a list. Reversing a segment changes edges at (i-1, i) and (j, j+1)
    # and internal edges.
    # To be efficient, we can just create a new list or slice.
    # Given N=24, copying is cheap.
    new_cycle = cycle[:]
    new_cycle[i:j+1] = reversed(new_cycle[i:j+1])
    return new_cycle

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = N
    best_cycle = greedy_init()
    best_score = score_cycle(best_cycle)
    
    # Simulated Annealing
    temp = 100.0
    cooling = 0.99
    min_temp = 0.1
    
    current_cycle = best_cycle[:]
    current_score = best_score
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        if temp < min_temp:
            # Restart with higher temperature if stuck? Or just break?
            # Let's try a few restarts
            temp = 100.0
            # Random restart
            current_cycle = [0] + rng.sample(range(1, n), n-1)
            current_score = score_cycle(current_cycle)
            if current_score < best_score:
                best_score = current_score
                best_cycle = current_cycle[:]
        
        # Generate 2-opt move
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
            
        new_cycle = two_opt_move(current_cycle, i, j)
        new_score = score_cycle(new_cycle)
        
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / temp):
            current_cycle = new_cycle
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_cycle = current_cycle[:]
        
        temp *= cooling
        
        # If we found a score of 24 (minimum possible for a cycle of length 24 in a complete graph? 
        # No, min score is at least 24 because there are 24 edges in the cycle.
        # The edges are distinct. The minimum possible max index is 24 if the cycle uses exactly the first 24 edges.
        # It's highly unlikely, but if best_score <= 24, we can break.
        if best_score <= 24:
            break

    return best_cycle
