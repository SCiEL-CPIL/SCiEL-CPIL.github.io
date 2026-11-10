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
    
    def count_violations_fast(colors_np):
        """Fast violation count using numpy."""
        # For each K4, check if it has < 5 distinct colours
        # k4_edges_arr[i] gives 6 edge indices for K4 i
        # colors_np[k4_edges_arr[i]] gives the 6 colours
        k4_colours = colors_np[k4_edges_arr]  # shape (1820, 6)
        # Count distinct colours per K4
        # A K4 violates if it has < 5 distinct colours
        # Check: if any colour appears in at least 2 of the 6 edges, 
        # then distinct count <= 5. But we need < 5, so at least 2 pairs match.
        # Simpler: for each K4, check if max frequency > 2 (then distinct <= 4)
        # or if exactly 2 pairs match (then distinct = 4)
        # Let's just compute distinct count properly
        # For 6 elements, distinct < 5 means at least 2 are the same
        # Actually distinct < 5 means at least 2 pairs are equal
        # Let's use a different approach: check if sorted unique has length < 5
        # This is still expensive. Let me use a smarter check.
        
        # For 6 colours, distinct < 5 iff there exist i,j,k,l with c[i]==c[j] and c[k]==c[l] (i!=j, k!=l, {i,j}!= {k,l})
        # OR three are equal (which implies the above)
        # Actually simpler: distinct < 5 iff the number of pairs that are equal is >= 2
        # Number of equal pairs = sum over all pairs (i,j) of [c[i]==c[j]]
        # For 6 elements, there are C(6,2)=15 pairs
        # If distinct = 6: 0 equal pairs
        # If distinct = 5: 1 equal pair (one pair matches)
        # If distinct = 4: could be 2 equal pairs (two separate pairs match) or 3 equal pairs (one triple)
        # If distinct = 3: could be 3 pairs (two triples) or 4 pairs (one triple + one pair) etc.
        # So distinct < 5 iff number of equal pairs >= 2
        
        # Compute number of equal pairs for each K4
        # k4_colours has shape (1820, 6)
        # For each row, count pairs that are equal
        # We can do this by checking all C(6,2)=15 pairs
        pairs_idx = [(0,1),(0,2),(0,3),(0,4),(0,5),
                     (1,2),(1,3),(1,4),(1,5),
                     (2,3),(2,4),(2,5),
                     (3,4),(3,5),
                     (4,5)]
        num_equal_pairs = np.zeros(len(k4_colours), dtype=np.int32)
        for (a, b) in pairs_idx:
            num_equal_pairs += (k4_colours[:, a] == k4_colours[:, b]).astype(np.int32)
        
        violations = np.sum(num_equal_pairs >= 2)
        return violations
    
    # Base solution: (i+j) mod 17, uses 17 colours
    base_colors_list = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best_colors = base_colors_list[:]
    best_num = len(set(best_colors))  # 17
    
    best_colors_np = np.array(best_colors, dtype=np.int32)
    assert count_violations_fast(best_colors_np) == 0
    
    def try_eliminate_colour(cur_colors_np, target_colour):
        """Try to eliminate target_colour by reassigning its edges using backtracking."""
        cur = cur_colors_np.copy()
        # Find all edges with target_colour
        target_edges = np.where(cur == target_colour)[0].tolist()
        
        if not target_edges:
            return None
        
        # Get available other colours
        other_colours = [c for c in set(cur.tolist()) if c != target_colour]
        if not other_colours:
            return None
        
        # Use backtracking with constraint propagation
        # For each edge in target_edges, try assigning one of other_colours
        # Check validity incrementally
        
        # Precompute which K4s are affected by each edge
        # edge_to_k4_arr[ei] gives the K4 indices affected by edge ei
        
        def is_valid_partial(cur_np):
            """Check if current partial assignment is valid."""
            return count_violations_fast(cur_np) == 0
        
        # Backtracking search
        def backtrack(edge_idx):
            if edge_idx == len(target_edges):
                return True
            
            ei = target_edges[edge_idx]
            orig = cur[ei]
            
            for new_c in other_colours:
                cur[ei] = new_c
                if count_violations_fast(cur) == 0:
                    if backtrack(edge_idx + 1):
                        return True
            
            cur[ei] = orig
            return False
        
        if backtrack(0):
            if count_violations_fast(cur) == 0:
                return cur
        return None
    
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
        
        for ei in target_edges:
            orig = cur[ei]
            found = False
            for new_c in other_colours:
                cur[ei] = new_c
                if count_violations_fast(cur) == 0:
                    found = True
                    break
            if not found:
                cur[ei] = orig
                return None
        
        if count_violations_fast(cur) == 0:
            return cur
        return None
    
    # Iteratively try to eliminate colours
    current_np = best_colors_np.copy()
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        cur_cols = sorted(set(current_np.tolist()))
        if len(cur_cols) <= best_num:
            break
        
        # Try to eliminate each colour (start from highest to reduce palette)
        eliminated = False
        for c in reversed(cur_cols):
            if time.time() > deadline:
                break
            # Try greedy first (faster)
            result = try_eliminate_colour_greedy(current_np, c)
            if result is None:
                # Try backtracking (slower but more thorough)
                if time.time() < deadline - 5:
                    result = try_eliminate_colour(current_np, c)
            
            if result is not None:
                current_np = result
                eliminated = True
                break
        
        if not eliminated:
            # Try random local search moves
            for _ in range(20000):
                if time.time() > deadline:
                    break
                
                # Pick random edge and random colour from current palette
                ei = random.randint(0, num_edges - 1)
                cur_palette = set(current_np.tolist())
                if len(cur_palette) <= 1:
                    break
                new_c = random.choice(list(cur_palette - {current_np[ei]}))
                
                if new_c == current_np[ei]:
                    continue
                
                orig = current_np[ei]
                current_np[ei] = new_c
                viol = count_violations_fast(current_np)
                
                if viol == 0:
                    # Check if we reduced palette
                    new_num = len(set(current_np.tolist()))
                    if new_num < best_num:
                        best_colors_np = current_np.copy()
                        best_num = new_num
                else:
                    current_np[ei] = orig
    
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
