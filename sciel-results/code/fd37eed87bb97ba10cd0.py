"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
NUM_COLORS = 120

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def _precompute_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    e1 = _edge_idx(a, b)
                    e2 = _edge_idx(b, c)
                    e3 = _edge_idx(c, d)
                    e4 = _edge_idx(d, a)
                    cycles.append((e1, e2, e3, e4))
                    e1 = _edge_idx(a, b)
                    e2 = _edge_idx(b, d)
                    e3 = _edge_idx(d, c)
                    e4 = _edge_idx(c, a)
                    cycles.append((e1, e2, e3, e4))
                    e1 = _edge_idx(a, c)
                    e2 = _edge_idx(c, b)
                    e3 = _edge_idx(b, d)
                    e4 = _edge_idx(d, a)
                    cycles.append((e1, e2, e3, e4))
    return cycles

_CYCLES = _precompute_cycles()

def _count_violations(colors):
    v = 0
    for e1, e2, e3, e4 in _CYCLES:
        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
            v += 1
    return v

def _count_distinct(colors):
    return len(set(colors))

def _objective(colors):
    d = _count_distinct(colors)
    v = _count_violations(colors)
    return d - 1000.0 * v

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    rng = random.Random(42)
    
    # Initial: random coloring
    colors = [rng.randint(0, NUM_COLORS - 1) for _ in range(NUM_EDGES)]
    
    # Repair to validity
    for _ in range(10000):
        if time.time() > deadline:
            break
        v = _count_violations(colors)
        if v == 0:
            break
        # Find first violating cycle
        fixed = False
        for e1, e2, e3, e4 in _CYCLES:
            c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
            if len({c1, c2, c3, c4}) == 4:
                # Merge two random edges
                edges = [e1, e2, e3, e4]
                t, s = rng.sample(edges, 2)
                colors[t] = colors[s]
                fixed = True
                break
        if not fixed:
            break
    
    best_state = list(colors)
    best_obj = _objective(best_state)
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    cur = list(colors)
    cur_obj = best_obj
    
    while time.time() < deadline:
        move_type = rng.random()
        
        if move_type < 0.5:
            # Recolor one edge
            e = rng.randint(0, NUM_EDGES - 1)
            old = cur[e]
            new = rng.randint(0, NUM_COLORS - 1)
            if new == old:
                continue
            cur[e] = new
            new_obj = _objective(cur)
            delta = new_obj - cur_obj
            if delta >= 0 or rng.random() < math.exp(delta / T):
                cur_obj = new_obj
                if new_obj > best_obj:
                    best_obj = new_obj
                    best_state = list(cur)
            else:
                cur[e] = old
        else:
            # Swap two edges
            e1, e2 = rng.sample(range(NUM_EDGES), 2)
            c1, c2 = cur[e1], cur[e2]
            if c1 == c2:
                continue
            cur[e1] = c2
            cur[e2] = c1
            new_obj = _objective(cur)
            delta = new_obj - cur_obj
            if delta >= 0 or rng.random() < math.exp(delta / T):
                cur_obj = new_obj
                if new_obj > best_obj:
                    best_obj = new_obj
                    best_state = list(cur)
            else:
                cur[e1] = c1
                cur[e2] = c2
        
        T = max(T * cooling, T_min)
        
        # Restart if stuck
        if cur_obj == best_obj and T < 1.0:
            T = 5.0
    
    # Ensure validity
    if _count_violations(best_state) > 0:
        # Try to repair best_state
        for _ in range(5000):
            v = _count_violations(best_state)
            if v == 0:
                break
            for e1, e2, e3, e4 in _CYCLES:
                c1, c2, c3, c4 = best_state[e1], best_state[e2], best_state[e3], best_state[e4]
                if len({c1, c2, c3, c4}) == 4:
                    edges = [e1, e2, e3, e4]
                    t, s = rng.sample(edges, 2)
                    best_state[t] = best_state[s]
                    break
    
    if _count_violations(best_state) > 0:
        return [0] * NUM_EDGES
    
    return best_state

# EVOLVE-BLOCK-END
