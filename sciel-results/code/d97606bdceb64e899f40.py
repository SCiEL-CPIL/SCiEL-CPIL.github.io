import numpy as np
import itertools
import time
import random
import sys

def get_k6_cycles():
    cycles = set()
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        v = (0,) + perm
        edges = set()
        for i in range(6):
            u, w = v[i], v[(i + 1) % 6]
            if u > w: u, w = w, u
            edges.add((u, w))
        cycles.add(frozenset(edges))
    return [list(c) for c in cycles]

K6_CYCLES = get_k6_cycles()

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

    def build_cycle_data(n):
        """Build efficient cycle checking data structure."""
        num_edges = n * (n - 1) // 2
        
        def get_idx(i, j):
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        # Precompute all C6 edge index lists
        c6_edges_list = []
        for comb in itertools.combinations(range(n), 6):
            for cycle_edges in K6_CYCLES:
                global_edges = []
                for u, v in cycle_edges:
                    gu = comb[u]
                    gv = comb[v]
                    if gu > gv: gu, gv = gv, gu
                    global_edges.append(get_idx(gu, gv))
                c6_edges_list.append(global_edges)
        
        c6_edges_arr = np.array(c6_edges_list, dtype=np.int32)
        
        # For fast evaluation: precompute per-edge cycle membership
        # edge_cycle_count[e] = number of C6s containing edge e
        # For incremental updates, we need: for each edge, which cycles contain it
        # and for each cycle, what are the other 5 edges
        
        # Build edge -> list of (cycle_idx, position_in_cycle)
        edge_to_cycles = [[] for _ in range(num_edges)]
        for ci, edges in enumerate(c6_edges_list):
            for pos, e in enumerate(edges):
                edge_to_cycles[e].append((ci, pos))
        
        return c6_edges_arr, edge_to_cycles, get_idx, num_edges

    def solve_n(n, time_limit, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        num_edges = n * (n - 1) // 2
        
        def get_idx(i, j):
            return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
        # Build cycle data
        c6_edges_list = []
        for comb in itertools.combinations(range(n), 6):
            for cycle_edges in K6_CYCLES:
                global_edges = []
                for u, v in cycle_edges:
                    gu = comb[u]
                    gv = comb[v]
                    if gu > gv: gu, gv = gv, gu
                    global_edges.append(get_idx(gu, gv))
                c6_edges_list.append(global_edges)
        
        c6_edges_arr = np.array(c6_edges_list, dtype=np.int32)
        num_cycles = len(c6_edges_arr)
        
        # Initialize colors
        colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
        
        def full_evaluate(c):
            """Full evaluation: count monochromatic C6s."""
            c6_c = c[c6_edges_arr]
            # A cycle is monochromatic if all 6 edges have the same color
            # Check: all equal to first edge
            is_mono = (c6_c[:, 1] == c6_c[:, 0]) & (c6_c[:, 2] == c6_c[:, 0]) & \
                      (c6_c[:, 3] == c6_c[:, 0]) & (c6_c[:, 4] == c6_c[:, 0]) & \
                      (c6_c[:, 5] == c6_c[:, 0])
            return int(np.sum(is_mono))
        
        current_score = full_evaluate(colors)
        if current_score == 0:
            return colors.tolist()
        
        best_score = current_score
        best_colors = colors.copy()
        
        # Track per-cycle state for incremental evaluation
        # cycle_color_count[cycle_idx][color] = how many edges in this cycle have that color
        cycle_color_count = np.zeros((num_cycles, 4), dtype=np.int32)
        for ci in range(num_cycles):
            for e in c6_edges_arr[ci]:
                cycle_color_count[ci, colors[e]] += 1
        
        # A cycle is monochromatic if cycle_color_count[ci][c] == 6 for some c
        # current_score = sum over cycles of (1 if any count==6 else 0)
        
        def get_mono_count(cc):
            """Count how many color classes have count 6 in a cycle."""
            # Return 1 if any color has count 6
            return int(np.any(cc == 6))
        
        # Precompute which cycles each edge belongs to for fast updates
        edge_to_cycles = [[] for _ in range(num_edges)]
        for ci in range(num_cycles):
            for pos, e in enumerate(c6_edges_arr[ci]):
                edge_to_cycles[e].append((ci, pos))
        
        def incremental_evaluate_after_change(colors, old_color, new_color, edge_idx):
            """Update score after changing edge_idx from old_color to new_color."""
            delta = 0
            for (ci, pos) in edge_to_cycles[edge_idx]:
                old_mono = get_mono_count(cycle_color_count[ci])
                cycle_color_count[ci, old_color] -= 1
                cycle_color_count[ci, new_color] += 1
                new_mono = get_mono_count(cycle_color_count[ci])
                delta += new_mono - old_mono
            return delta
        
        # Simulated Annealing with incremental evaluation
        T = 5.0
        cooling = 0.9997
        min_T = 0.005
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < start_time + time_limit and T > min_T:
            idx = np.random.randint(num_edges)
            old_color = colors[idx]
            new_color = np.random.randint(0, 3)
            if new_color >= old_color:
                new_color += 1
            
            # Compute delta
            delta = 0
            for (ci, pos) in edge_to_cycles[idx]:
                old_mono = get_mono_count(cycle_color_count[ci])
                cycle_color_count[ci, old_color] -= 1
                cycle_color_count[ci, new_color] += 1
                new_mono = get_mono_count(cycle_color_count[ci])
                delta += new_mono - old_mono
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                colors[idx] = new_color
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                    no_improve_count = 0
                    if best_score == 0:
                        return best_colors.tolist()
                else:
                    no_improve_count += 1
            else:
                # Revert
                for (ci, pos) in edge_to_cycles[idx]:
                    cycle_color_count[ci, old_color] += 1
                    cycle_color_count[ci, new_color] -= 1
            
            T *= cooling
            if no_improve_count > max_no_improve:
                # Restart with new temperature
                T = max(T * 2, 1.0)
                no_improve_count = 0
        
        if best_score == 0:
            return best_colors.tolist()
        return None

    # Strategy: Try n=17 first with multiple restarts
    best_result = None
    
    # Phase 1: Try n=17 with several random restarts
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        num_restarts = min(10, int(remaining_time / 10))
        for r in range(num_restarts):
            if time.time() > deadline - 15:
                break
            time_per_restart = (deadline - 15 - time.time()) / (num_restarts - r)
            result = solve_n(17, time_per_restart, seed=r * 12345 + 678)
            if result is not None:
                return {'n': 17, 'colors': result}
    
    # Phase 2: Try n=16
    if time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 5:
            num_restarts = min(5, int(remaining / 5))
            for r in range(num_restarts):
                if time.time() > deadline - 5:
                    break
                time_per = (deadline - 5 - time.time()) / (num_restarts - r)
                result = solve_n(16, time_per, seed=r * 9999 + 42)
                if result is not None:
                    return {'n': 16, 'colors': result}
    
    # Phase 3: Try n=15
    if time.time() < deadline - 3:
        remaining = deadline - 3 - time.time()
        if remaining > 3:
            result = solve_n(15, remaining, seed=777)
            if result is not None:
                return {'n': 15, 'colors': result}
    
    # Phase 4: Try n=14
    if time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        if remaining > 2:
            result = solve_n(14, remaining, seed=555)
            if result is not None:
                return {'n': 14, 'colors': result}
    
    # Fallback: construct a valid small coloring
    # For n=9, use a known pattern
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            # Use a pattern that avoids C6
            color = (i + j) % 4
            colors.append(color)
    return {'n': n, 'colors': colors}
