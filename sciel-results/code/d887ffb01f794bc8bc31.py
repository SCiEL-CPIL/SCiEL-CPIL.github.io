import time
import numpy as np
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Precompute all 4-cycles and edge indices
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)

    # Map edge index to (u, v)
    edge_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_map[idx] = (i, j)
            idx += 1

    # Precompute cycles
    all_cycles = []
    vertices = list(range(N))

    for quad in combinations(vertices, 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1, e2, e3, e4 = get_idx(a, b), get_idx(b, c), get_idx(c, d), get_idx(d, a)
        all_cycles.append((e1, e2, e3, e4))
        # Cycle 2: a-b-d-c-a
        e1, e2, e3, e4 = get_idx(a, b), get_idx(b, d), get_idx(d, c), get_idx(c, a)
        all_cycles.append((e1, e2, e3, e4))
        # Cycle 3: a-c-b-d-a
        e1, e2, e3, e4 = get_idx(a, c), get_idx(c, b), get_idx(b, d), get_idx(d, a)
        all_cycles.append((e1, e2, e3, e4))

    num_cycles = len(all_cycles)
    cycles_arr = np.array(all_cycles, dtype=np.int32)

    # Precompute for each edge, the cycles it belongs to and its position in them
    # edge_to_cycles[e] = list of (cycle_idx, position)
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for c_idx in range(num_cycles):
        e1, e2, e3, e4 = all_cycles[c_idx]
        edge_to_cycles[e1].append((c_idx, 0))
        edge_to_cycles[e2].append((c_idx, 1))
        edge_to_cycles[e3].append((c_idx, 2))
        edge_to_cycles[e4].append((c_idx, 3))

    # 2. Evaluation Function (Vectorized)
    def evaluate(colors):
        """
        Returns (num_unique_colors, num_violations)
        """
        num_unique = len(np.unique(colors))
        
        c0 = colors[cycles_arr[:, 0]]
        c1 = colors[cycles_arr[:, 1]]
        c2 = colors[cycles_arr[:, 2]]
        c3 = colors[cycles_arr[:, 3]]
        
        distinct_mask = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
        num_violations = np.sum(distinct_mask)
        
        return num_unique, num_violations

    # 3. Base Construction: Disjoint Triangles
    # This guarantees a valid state with 16 colors (5 triangles * 3 + 0)
    current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    best_colors = current_colors.copy()
    best_score = 1
    best_violations = 0

    triangles = []
    remaining = list(range(N))
    while len(remaining) >= 3:
        t = (remaining.pop(), remaining.pop(), remaining.pop())
        triangles.append(t)
        
    color_id = 1
    for u, v, w in triangles:
        e_uv = get_idx(u, v)
        e_vw = get_idx(v, w)
        e_wu = get_idx(w, u)
        current_colors[e_uv] = color_id
        current_colors[e_vw] = color_id + 1
        current_colors[e_wu] = color_id + 2
        color_id += 3
        
    curr_score, curr_violations = evaluate(current_colors)
    if curr_violations == 0:
        best_score = curr_score
        best_colors = current_colors.copy()
        best_violations = 0
    else:
        # Fallback to all zeros if triangle construction somehow failed (shouldn't happen)
        current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        best_colors = current_colors.copy()
        best_score = 1
        best_violations = 0
        curr_score = 1
        curr_violations = 0

    # 4. Local Search (Simulated Annealing)
    # Goal: Maximize unique colors while keeping violations at 0.
    # Strategy:
    # - Try to recolor edges to unused colors to increase diversity.
    # - If a violation is created, try to fix it or accept with penalty (SA).
    # - We track the best VALID solution found so far.

    rng = random.Random(42)
    T = 10.0
    cooling = 0.999
    T_min = 0.01
    
    # Pre-compute list of edges for random selection
    edge_indices = list(range(NUM_EDGES))
    
    steps = 0
    last_improvement = time.time()
    
    while time.time() < deadline:
        # Pick a random edge
        e = rng.choice(edge_indices)
        old_c = current_colors[e]
        
        # Pick a new color
        # Bias towards unused colors to push the score up
        if rng.random() < 0.7:
            # Try an unused color
            used = set(current_colors)
            unused = [c for c in range(MAX_COLORS) if c not in used]
            if unused:
                new_c = rng.choice(unused)
            else:
                new_c = rng.randint(0, MAX_COLORS - 1)
        else:
            new_c = rng.randint(0, MAX_COLORS - 1)
            
        if new_c == old_c:
            continue
            
        current_colors[e] = new_c
        new_score, new_violations = evaluate(current_colors)
        
        # Check if this is a new best valid solution
        if new_violations == 0 and new_score > best_score:
            best_score = new_score
            best_colors = current_colors.copy()
            best_violations = 0
            last_improvement = time.time()
            
        # SA Acceptance
        # Fitness: score - heavy penalty for violations
        # We want to stay in valid states mostly, but allow exploration.
        old_fitness = curr_score - 50.0 * curr_violations
        new_fitness = new_score - 50.0 * new_violations
        
        delta = new_fitness - old_fitness
        
        if delta >= 0:
            # Accept
            pass
        else:
            if rng.random() < np.exp(delta / T):
                # Accept
                pass
            else:
                # Reject
                current_colors[e] = old_c
                T = max(T_min, T * cooling)
                steps += 1
                continue
        
        curr_score = new_score
        curr_violations = new_violations
        T = max(T_min, T * cooling)
        steps += 1

        # If we are stuck in a valid state and not improving, we might want to restart or perturb?
        # For now, standard SA cooling should handle it.
        
        # If time is running out, ensure we return the best valid one
        if time.time() > deadline - 1.0:
            break

    # Final Verification
    final_score, final_violations = evaluate(best_colors)
    if final_violations != 0:
        # Fallback to triangle base
        current_colors = np.zeros(NUM_EDGES, dtype=np.int32)
        color_id = 1
        remaining = list(range(N))
        while len(remaining) >= 3:
            t = (remaining.pop(), remaining.pop(), remaining.pop())
            u, v, w = t
            e_uv = get_idx(u, v)
            e_vw = get_idx(v, w)
            e_wu = get_idx(w, u)
            current_colors[e_uv] = color_id
            current_colors[e_vw] = color_id + 1
            current_colors[e_wu] = color_id + 2
            color_id += 3
        return current_colors.tolist()
        
    return best_colors.tolist()

if __name__ == "__main__":
    colors = construct()
    print(f"Max colors: {len(set(colors))}")
