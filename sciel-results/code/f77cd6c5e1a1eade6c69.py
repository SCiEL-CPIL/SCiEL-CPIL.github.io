import time
import random
import sys
import numpy as np

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Combines:
    1. Incremental state updates (neighbors bitmasks + common_count) for fast cost evaluation.
    2. Conflict-driven move selection (prioritize edges in C4s).
    3. Simulated Annealing with proper revert logic (using helper functions).
    4. Search strategy from best inspiration (n=28, 27, 26...).
    """
    deadline = time.time() + 110.0  # Safety margin

    def is_valid(n, colors):
        if len(colors) != n * (n - 1) // 2:
            return False
        neighbors = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                if nu == 0:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common & (common - 1):
                        return False
        return True

    def solve_sa(n, time_limit, seed=None):
        if seed is not None:
            random.seed(seed)
            
        total_edges = n * (n - 1) // 2
        edge_map = [None] * total_edges
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[idx] = (i, j)
                idx += 1
        
        # State
        neighbors = [[0] * n for _ in range(5)]
        # common_count[c][u][v]
        common_count = [[[0] * n for _ in range(n)] for _ in range(5)]
        bad_pair_count = [0] * 5
        total_cost = 0
        
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Init state
        for e_idx in range(total_edges):
            u, v = edge_map[e_idx]
            c = colors[e_idx]
            neighbors[c][u] |= (1 << v)
            neighbors[c][v] |= (1 << u)
            
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = neighbors[c][u] & neighbors[c][v]
                    cnt = bin(common).count('1')
                    common_count[c][u][v] = cnt
                    common_count[c][v][u] = cnt
                    if cnt >= 2:
                        bad_pair_count[c] += 1
                        total_cost += 1
        
        if total_cost == 0:
            return colors
            
        best_cost = total_cost
        best_colors = colors[:]
        
        # Edge index map for conflict selection
        edge_index = {}
        for e_idx, (u, v) in enumerate(edge_map):
            edge_index[(u, v)] = e_idx
            edge_index[(v, u)] = e_idx
            
        def get_conflicting_edges():
            conflicts = set()
            for c in range(5):
                if bad_pair_count[c] == 0:
                    continue
                for u in range(n):
                    for v in range(u + 1, n):
                        if common_count[c][u][v] >= 2:
                            common = neighbors[c][u] & neighbors[c][v]
                            while common:
                                lsb = common & -common
                                w = lsb.bit_length() - 1
                                common ^= lsb
                                if w != u and w != v:
                                    e_uw = edge_index.get((u, w))
                                    if e_uw is not None:
                                        conflicts.add(e_uw)
                                    e_vw = edge_index.get((v, w))
                                    if e_vw is not None:
                                        conflicts.add(e_vw)
            return list(conflicts)
        
        # Helper to update state when removing edge (u,v) from color c
        def update_remove(u, v, c):
            nonlocal bad_pair_count, total_cost
            neighbors[c][u] &= ~(1 << v)
            neighbors[c][v] &= ~(1 << u)
            
            delta = 0
            # Pair (u,v)
            old_cnt_uv = common_count[c][u][v]
            new_cnt_uv = bin(neighbors[c][u] & neighbors[c][v]).count('1')
            common_count[c][u][v] = new_cnt_uv
            common_count[c][v][u] = new_cnt_uv
            if old_cnt_uv >= 2 and new_cnt_uv < 2:
                bad_pair_count[c] -= 1
                delta -= 1
            elif old_cnt_uv < 2 and new_cnt_uv >= 2:
                bad_pair_count[c] += 1
                delta += 1
            
            # Pairs (u,w) and (v,w)
            for w in range(n):
                if w == u or w == v:
                    continue
                if (neighbors[c][v] >> w) & 1:
                    old_cnt_uw = common_count[c][u][w]
                    new_cnt_uw = old_cnt_uw - 1
                    common_count[c][u][w] = new_cnt_uw
                    common_count[c][w][u] = new_cnt_uw
                    if old_cnt_uw >= 2 and new_cnt_uw < 2:
                        bad_pair_count[c] -= 1
                        delta -= 1
                    elif old_cnt_uw < 2 and new_cnt_uw >= 2:
                        bad_pair_count[c] += 1
                        delta += 1
                if (neighbors[c][u] >> w) & 1:
                    old_cnt_vw = common_count[c][v][w]
                    new_cnt_vw = old_cnt_vw - 1
                    common_count[c][v][w] = new_cnt_vw
                    common_count[c][w][v] = new_cnt_vw
                    if old_cnt_vw >= 2 and new_cnt_vw < 2:
                        bad_pair_count[c] -= 1
                        delta -= 1
                    elif old_cnt_vw < 2 and new_cnt_vw >= 2:
                        bad_pair_count[c] += 1
                        delta += 1
            return delta

        # Helper to update state when adding edge (u,v) to color c
        def update_add(u, v, c):
            nonlocal bad_pair_count, total_cost
            neighbors[c][u] |= (1 << v)
            neighbors[c][v] |= (1 << u)
            
            delta = 0
            # Pair (u,v)
            old_cnt_uv = common_count[c][u][v]
            new_cnt_uv = bin(neighbors[c][u] & neighbors[c][v]).count('1')
            common_count[c][u][v] = new_cnt_uv
            common_count[c][v][u] = new_cnt_uv
            if old_cnt_uv >= 2 and new_cnt_uv < 2:
                bad_pair_count[c] -= 1
                delta -= 1
            elif old_cnt_uv < 2 and new_cnt_uv >= 2:
                bad_pair_count[c] += 1
                delta += 1
            
            # Pairs (u,w) and (v,w)
            for w in range(n):
                if w == u or w == v:
                    continue
                if (neighbors[c][v] >> w) & 1:
                    old_cnt_uw = common_count[c][u][w]
                    new_cnt_uw = old_cnt_uw + 1
                    common_count[c][u][w] = new_cnt_uw
                    common_count[c][w][u] = new_cnt_uw
                    if old_cnt_uw >= 2 and new_cnt_uw < 2:
                        bad_pair_count[c] -= 1
                        delta -= 1
                    elif old_cnt_uw < 2 and new_cnt_uw >= 2:
                        bad_pair_count[c] += 1
                        delta += 1
                if (neighbors[c][u] >> w) & 1:
                    old_cnt_vw = common_count[c][v][w]
                    new_cnt_vw = old_cnt_vw + 1
                    common_count[c][v][w] = new_cnt_vw
                    common_count[c][w][v] = new_cnt_vw
                    if old_cnt_vw >= 2 and new_cnt_vw < 2:
                        bad_pair_count[c] -= 1
                        delta -= 1
                    elif old_cnt_vw < 2 and new_cnt_vw >= 2:
                        bad_pair_count[c] += 1
                        delta += 1
            return delta

        T = 1.0
        cooling_rate = 0.9995
        start_time = time.time()
        last_improvement_time = start_time
        conflict_cache = []
        
        while time.time() - start_time < time_limit:
            if not conflict_cache or random.random() < 0.05:
                conflict_cache = get_conflicting_edges()
            
            if conflict_cache and random.random() < 0.8:
                e_idx = random.choice(conflict_cache)
            else:
                e_idx = random.randint(0, total_edges - 1)
                
            u, v = edge_map[e_idx]
            old_c = colors[e_idx]
            
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            # Simulate move
            # 1. Remove from old_c
            delta_remove = update_remove(u, v, old_c)
            # 2. Add to new_c
            delta_add = update_add(u, v, new_c)
            
            delta = delta_remove + delta_add
            
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / T):
                total_cost += delta
                colors[e_idx] = new_c
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_colors = colors[:]
                    last_improvement_time = time.time()
                    if best_cost == 0:
                        return best_colors
            else:
                # Revert
                # 1. Remove from new_c
                update_remove(u, v, new_c)
                # 2. Add to old_c
                update_add(u, v, old_c)
                
            T *= cooling_rate
            if T < 0.1:
                T = 1.0
                
        return best_colors if best_cost == 0 else None

    # Search Plan
    search_plan = [
        (28, 5.0),
        (27, 10.0),
        (26, 40.0),
        (25, 10.0),
    ]
    
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors
        
    best_n, best_colors = get_seed_solution()
    
    for n, t_alloc in search_plan:
        if time.time() > deadline - 2.0:
            break
        res = solve_sa(n, t_alloc)
        if res is not None and is_valid(n, res):
            best_n = n
            best_colors = res
            return {'n': best_n, 'colors': best_colors}
            
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
