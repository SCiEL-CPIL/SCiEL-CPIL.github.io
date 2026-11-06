import time
import random
import sys
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}
    
    best_result = get_fallback(5)
    
    def precompute(n):
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        m = n * (n - 1) // 2
        
        def edge_index(u, v):
            if u > v:
                u, v = v, u
            return edge_map[(u, v)]
        
        vertices = list(range(n))
        cycles = []
        edge_to_cycles = [[] for _ in range(m)]
        
        # Generate all 6-cycles
        # C(n, 6) * 60
        for combo in itertools.combinations(vertices, 6):
            v0, v1, v2, v3, v4, v5 = combo
            # Fix v0. Permute v1..v5.
            # To avoid reverse duplicates, require v1 < v5?
            # No, just generate all 120 and deduplicate?
            # Or just generate 60 by requiring v1 < v5 in the cycle representation?
            # Cycle: v0-v1-v2-v3-v4-v5-v0.
            # Reverse: v0-v5-v4-v3-v2-v1-v0.
            # If we fix v0, and iterate perms of v1..v5, we get 120.
            # We can skip if v1 > v5?
            # Let's just generate all and use a set of frozensets of edges?
            # Too slow.
            # Standard: Fix v0. The cycle is determined by the sequence v1..v5.
            # Two sequences are equivalent if one is the reverse of the other.
            # So we can just iterate all perms and divide by 2?
            # Or iterate perms where v1 < v5?
            # Let's check:
            # If v1 < v5, is the reverse v5 < v1? Yes.
            # So exactly one of the pair will satisfy v1 < v5.
            
            for perm in itertools.permutations([v1, v2, v3, v4, v5]):
                if perm[0] > perm[-1]:
                    continue
                
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    e = edge_index(u, v)
                    cycle_edges.append(e)
                
                cycle_id = len(cycles)
                cycles.append(cycle_edges)
                for e in cycle_edges:
                    edge_to_cycles[e].append(cycle_id)
        
        # Convert to NumPy for speed
        edge_to_cycles_np = [np.array(ec, dtype=np.int32) for ec in edge_to_cycles]
        cycle_edges_np = [np.array(ce, dtype=np.int32) for ce in cycles]
        
        return edge_map, cycles, edge_to_cycles_np, cycle_edges_np, m

    def search_n(n, time_budget):
        if time.time() > deadline - 5:
            return None
            
        try:
            edge_map, cycles, edge_to_cycles_np, cycle_edges_np, m = precompute(n)
        except Exception:
            return None
            
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}
        
        # Initialize counts
        # cnt[cid, color]
        cnt = np.zeros((num_cycles, 4), dtype=np.int32)
        
        best_result = None
        
        # We will try multiple restarts
        max_restarts = 100
        restart_time = time_budget / max_restarts
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Random Init
            colors = np.array([random.randint(0, 3) for _ in range(m)], dtype=np.int32)
            
            # Compute initial counts
            cnt = np.zeros((num_cycles, 4), dtype=np.int32)
            for cid, ce in enumerate(cycle_edges_np):
                # cnt[cid, colors[ce]] += 1
                # This is slow in Python loop.
                # We can use np.add.at
                np.add.at(cnt, (cid, colors[ce]), 1)
            
            bad_count = np.sum(np.max(cnt, axis=1) == 6)
            
            if bad_count == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            # Simulated Annealing
            T = 1.0
            cooling = 0.999
            best_bad = bad_count
            best_colors = colors.copy()
            
            local_start = time.time()
            
            while time.time() - local_start < restart_time and time.time() < deadline - 5:
                # Pick random edge
                e = random.randint(0, m - 1)
                c_old = colors[e]
                
                # Pick random new color
                c_new = random.randint(0, 3)
                while c_new == c_old:
                    c_new = random.randint(0, 3)
                
                # Get cycles containing e
                cids = edge_to_cycles_np[e]
                
                if len(cids) == 0:
                    # No cycles, safe to move
                    colors[e] = c_new
                    continue
                
                # Get old counts
                old_c_old = cnt[cids, c_old]
                old_c_new = cnt[cids, c_new]
                
                # Update counts
                cnt[cids, c_old] -= 1
                cnt[cids, c_new] += 1
                
                # Calculate delta
                # New bad: cnt[cids, c_new] == 6
                # Old bad: old_c_old == 6
                new_bad = np.sum(cnt[cids, c_new] == 6)
                old_bad = np.sum(old_c_old == 6)
                delta = new_bad - old_bad
                
                # Accept
                if delta < 0 or random.random() < np.exp(-delta / T):
                    colors[e] = c_new
                    bad_count += delta
                    
                    if bad_count < best_bad:
                        best_bad = bad_count
                        best_colors = colors.copy()
                        
                    if bad_count == 0:
                        return {'n': n, 'colors': colors.tolist()}
                else:
                    # Revert
                    cnt[cids, c_old] += 1
                    cnt[cids, c_new] -= 1
                
                T *= cooling
        
        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
            
        return None

    targets = [17, 16, 15, 14, 13, 12, 11]
    budgets = {
        17: 60.0,
        16: 30.0,
        15: 10.0,
        14: 5.0,
        13: 2.0,
        12: 1.0,
        11: 0.5
    }
    
    for n in targets:
        if time.time() > deadline - 10:
            break
        
        budget = budgets.get(n, 5.0)
        if time.time() + budget > deadline - 5:
             budget = deadline - 5 - time.time()
             if budget < 1.0:
                 continue
                 
        result = search_n(n, budget)
        
        if result:
            return result
        
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
