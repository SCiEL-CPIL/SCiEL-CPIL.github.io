"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120 edges
    
    # Precompute edge index mapping
    idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_map[(i, j)] = idx
            idx += 1
    
    # Precompute all K4s and their edge indices
    k4_edges = []
    edge_to_k4 = [[] for _ in range(num_edges)]
    for v in itertools.combinations(range(n), 4):
        a, b, c, d = v
        edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        e_indices = []
        for e in edges:
            i, j = e
            if i > j:
                i, j = j, i
            ei = idx_map[(i, j)]
            e_indices.append(ei)
            edge_to_k4[ei].append(len(k4_edges))
        k4_edges.append(e_indices)
    
    num_k4s = len(k4_edges)
    
    # Convert to numpy arrays for speed
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)  # shape (1820, 6)
    edge_to_k4_arr = [np.array(lst, dtype=np.int32) for lst in edge_to_k4]
    
    # Precompute pair indices for K4 edge positions
    pairs_idx = [(0,1),(0,2),(0,3),(0,4),(0,5),
                 (1,2),(1,3),(1,4),(1,5),
                 (2,3),(2,4),(2,5),
                 (3,4),(3,5),
                 (4,5)]
    pair_a = np.array([p[0] for p in pairs_idx], dtype=np.int32)
    pair_b = np.array([p[1] for p in pairs_idx], dtype=np.int32)
    
    def count_violations_fast(colors_np):
        """Fast violation count using numpy."""
        k4_colours = colors_np[k4_edges_arr]  # shape (1820, 6)
        # Count equal pairs for each K4
        num_equal_pairs = np.zeros(num_k4s, dtype=np.int32)
        for pi in range(15):
            num_equal_pairs += (k4_colours[:, pair_a[pi]] == k4_colours[:, pair_b[pi]]).astype(np.int32)
        violations = np.sum(num_equal_pairs >= 2)
        return violations
    
    def compute_slack(colors_np):
        """Compute slack for each K4: number of equal pairs (0=valid with 6 distinct, 1=valid with 5 distinct, >=2=violation)."""
        k4_colours = colors_np[k4_edges_arr]
        slack = np.zeros(num_k4s, dtype=np.int32)
        for pi in range(15):
            slack += (k4_colours[:, pair_a[pi]] == k4_colours[:, pair_b[pi]]).astype(np.int32)
        return slack
    
    def is_valid_slack(slack_arr):
        return np.all(slack_arr < 2)
    
    # Base solution: (i+j) mod 17, uses 17 colours
    base_colors_list = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = base_colors_list[:]
    best_num = len(set(best_colors))  # 17
    
    best_colors_np = np.array(best_colors, dtype=np.int32)
    assert count_violations_fast(best_colors_np) == 0
    
    def try_eliminate_colour_greedy(cur_colors_np, target_colour):
        """Greedy approach: assign edges one by one, trying colours in order."""
        cur = cur_colors_np.copy()
        target_edges = np.where(cur == target_colour)[0].tolist()
        
        if not target_edges:
            return None
        
        other_colours = [c for c in set(cur.tolist()) if c != target_colour]
        if not other_colours:
            return None
        
        # Sort target_edges by number of K4s they affect (most constrained first)
        target_edges.sort(key=lambda ei: len(edge_to_k4_arr[ei]), reverse=True)
        
        slack = compute_slack(cur)
        
        for ei in target_edges:
            orig = cur[ei]
            found = False
            for new_c in other_colours:
                cur[ei] = new_c
                # Only recompute slack for affected K4s
                affected = edge_to_k4_arr[ei]
                if len(affected) > 0:
                    k4_sub = cur[k4_edges_arr[affected]]
                    new_slack_sub = np.zeros(len(affected), dtype=np.int32)
                    for pi in range(15):
                        new_slack_sub += (k4_sub[:, pair_a[pi]] == k4_sub[:, pair_b[pi]]).astype(np.int32)
                    slack[affected] = new_slack_sub
                if np.all(slack < 2):
                    found = True
                    break
            if not found:
                cur[ei] = orig
                return None
        
        if count_violations_fast(cur) == 0:
            return cur
        return None
    
    def try_eliminate_colour_backtrack(cur_colors_np, target_colour, max_depth=30):
        """Backtracking with limited depth."""
        cur = cur_colors_np.copy()
        target_edges = np.where(cur == target_colour)[0].tolist()
        
        if not target_edges:
            return None
        
        other_colours = [c for c in set(cur.tolist()) if c != target_colour]
        if not other_colours:
            return None
        
        # Sort target_edges by number of K4s they affect (most constrained first)
        target_edges.sort(key=lambda ei: len(edge_to_k4_arr[ei]), reverse=True)
        
        slack = compute_slack(cur)
        
        def backtrack(edge_idx, depth):
            if edge_idx == len(target_edges):
                return True
            if depth >= max_depth:
                return False
            
            ei = target_edges[edge_idx]
            orig = cur[ei]
            
            for new_c in other_colours:
                cur[ei] = new_c
                affected = edge_to_k4_arr[ei]
                if len(affected) > 0:
                    k4_sub = cur[k4_edges_arr[affected]]
                    new_slack_sub = np.zeros(len(affected), dtype=np.int32)
                    for pi in range(15):
                        new_slack_sub += (k4_sub[:, pair_a[pi]] == k4_sub[:, pair_b[pi]]).astype(np.int32)
                    slack[affected] = new_slack_sub
                
                if np.all(slack < 2):
                    if backtrack(edge_idx + 1, depth + 1):
                        return True
                
                # Revert
                cur[ei] = orig
                if len(affected) > 0:
                    k4_sub = cur[k4_edges_arr[affected]]
                    new_slack_sub = np.zeros(len(affected), dtype=np.int32)
                    for pi in range(15):
                        new_slack_sub += (k4_sub[:, pair_a[pi]] == k4_sub[:, pair_b[pi]]).astype(np.int32)
                    slack[affected] = new_slack_sub
            
            return False
        
        if backtrack(0, 0):
            if count_violations_fast(cur) == 0:
                return cur
        return None
    
    # Local search with slack tracking
    def local_search(current_np, max_iters=50000):
        """Local search that tries to reduce palette size."""
        cur = current_np.copy()
        slack = compute_slack(cur)
        cur_palette = sorted(set(cur.tolist()))
        best_local = cur.copy()
        best_local_num = len(cur_palette)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            ei = random.randint(0, num_edges - 1)
            old_c = cur[ei]
            
            if len(cur_palette) <= 1:
                break
            
            # Try a random colour from palette (excluding current)
            other_palette = [c for c in cur_palette if c != old_c]
            if not other_palette:
                continue
            new_c = random.choice(other_palette)
            
            cur[ei] = new_c
            affected = edge_to_k4_arr[ei]
            if len(affected) > 0:
                k4_sub = cur[k4_edges_arr[affected]]
                new_slack_sub = np.zeros(len(affected), dtype=np.int32)
                for pi in range(15):
                    new_slack_sub += (k4_sub[:, pair_a[pi]] == k4_sub[:, pair_b[pi]]).astype(np.int32)
                slack[affected] = new_slack_sub
            
            if np.all(slack < 2):
                # Valid move - check if palette reduced
                new_palette = set(cur.tolist())
                if len(new_palette) < best_local_num:
                    best_local = cur.copy()
                    best_local_num = len(new_palette)
                    cur_palette = sorted(new_palette)
            else:
                # Revert
                cur[ei] = old_c
                if len(affected) > 0:
                    k4_sub = cur[k4_edges_arr[affected]]
                    new_slack_sub = np.zeros(len(affected), dtype=np.int32)
                    for pi in range(15):
                        new_slack_sub += (k4_sub[:, pair_a[pi]] == k4_sub[:, pair_b[pi]]).astype(np.int32)
                    slack[affected] = new_slack_sub
        
        return best_local, best_local_num
    
    # Main search loop with multiple strategies
    current_np = best_colors_np.copy()
    
    for iteration in range(300):
        if time.time() > deadline:
            break
        
        cur_cols = sorted(set(current_np.tolist()))
        if len(cur_cols) <= best_num:
            break
        
        # Strategy 1: Try to eliminate colours greedily
        eliminated = False
        for c in reversed(cur_cols):
            if time.time() > deadline:
                break
            
            result = try_eliminate_colour_greedy(current_np, c)
            if result is not None:
                current_np = result
                eliminated = True
                break
        
        if not eliminated and time.time() < deadline - 10:
            # Strategy 2: Try backtracking elimination
            for c in reversed(cur_cols):
                if time.time() > deadline:
                    break
                result = try_eliminate_colour_backtrack(current_np, c)
                if result is not None:
                    current_np = result
                    eliminated = True
                    break
        
        if not eliminated:
            # Strategy 3: Local search
            local_best, local_num = local_search(current_np, max_iters=30000)
            if local_num < best_num:
                best_colors_np = local_best.copy()
                best_num = local_num
                current_np = local_best.copy()
            elif local_num == best_num and np.any(local_best != current_np):
                # Same palette size but different structure - might enable further elimination
                current_np = local_best.copy()
        
        # Update best if current has fewer colours
        cur_num = len(set(current_np.tolist()))
        if cur_num < best_num:
            best_colors_np = current_np.copy()
            best_num = cur_num
    
    # Final check
    if count_violations_fast(best_colors_np) != 0:
        best_colors_np = np.array(base_colors_list, dtype=np.int32)
        best_num = 17
    
    best_colors = best_colors_np.tolist()
    
    # Renormalize colours to 0..best_num-1
    col_map = {}
    for i, c in enumerate(sorted(set(best_colors))):
        col_map[c] = i
    best_colors = [col_map[c] for c in best_colors]
    
    return best_colors
# EVOLVE-BLOCK-END
