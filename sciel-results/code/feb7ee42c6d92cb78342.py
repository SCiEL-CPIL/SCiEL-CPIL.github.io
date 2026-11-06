"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def _build_edge_pos(edge_order):
    pos = {}
    for i, (u, v) in enumerate(edge_order):
        pos[_edge_key(u, v)] = i + 1  # one-based
    return pos

def _cycle_score(cycle, edge_pos):
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        p = edge_pos[_edge_key(u, v)]
        if p > max_pos:
            max_pos = p
    return max_pos

def _two_opt_move(cycle, i, j):
    """Perform a 2-opt move: reverse the segment between i and j (inclusive of endpoints' connections)."""
    n = len(cycle)
    if i > j:
        i, j = j, i
    # Remove edges (i-1,i) and (j,j+1), add edges (i-1,j) and (i,j+1)
    # In cycle representation: cycle[0..n-1]
    # Edge at position k connects cycle[k] to cycle[(k+1)%n]
    # 2-opt: pick two edges, say edge a (between pos a-1 and a) and edge b (between pos b-1 and b)
    # Reverse the path between a and b
    if j - i < 2:
        return None
    new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
    return new_cycle

def _random_cycle(n, rng):
    c = list(range(n))
    rng.shuffle(c)
    return c

def _local_search(cycle, edge_pos, rng, max_iter=5000):
    n = len(cycle)
    best = cycle[:]
    best_score = _cycle_score(best, edge_pos)
    current = cycle[:]
    current_score = best_score
    
    for it in range(max_iter):
        # Pick two distinct positions for 2-opt
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j or (i + 1) % n == j or (j + 1) % n == i:
            continue
        # Ensure i < j for the move
        if i > j:
            i, j = j, i
        if j - i < 2:
            continue
        
        new_cycle = _two_opt_move(current, i, j)
        if new_cycle is None:
            continue
        new_score = _cycle_score(new_cycle, edge_pos)
        
        if new_score < current_score:
            current = new_cycle
            current_score = new_score
            if current_score < best_score:
                best = current[:]
                best_score = current_score
        # No simulated annealing for simplicity, just hill climbing
    
    return best, best_score

def construct():
    n = P['n']
    edge_order = P['edge_order']
    edge_pos = _build_edge_pos(edge_order)
    
    start_time = time.time()
    time_limit = 55  # leave 5s buffer
    
    rng = random.Random(42)
    
    # Start with trivial cycle
    best_cycle = list(range(n))
    best_score = _cycle_score(best_cycle, edge_pos)
    
    # Multiple restarts with local search
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        if r == 0:
            start_cycle = list(range(n))
        else:
            start_cycle = _random_cycle(n, rng)
        
        found_cycle, found_score = _local_search(start_cycle, edge_pos, rng, max_iter=2000)
        
        if found_score < best_score:
            best_score = found_score
            best_cycle = found_cycle
    
    # Final refinement on the best
    if time.time() - start_time < time_limit:
        refined_cycle, refined_score = _local_search(best_cycle, edge_pos, rng, max_iter=50000)
        if refined_score < best_score:
            best_score = refined_score
            best_cycle = refined_cycle
    
    return best_cycle
# EVOLVE-BLOCK-END
