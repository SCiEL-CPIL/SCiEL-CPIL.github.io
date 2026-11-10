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
        k4_edges.append(e_indices)
    
    num_k4s = len(k4_edges)
    
    # Convert to numpy arrays for speed
    k4_edges_arr = np.array(k4_edges, dtype=np.int32)  # shape (1820, 6)
    
    # Precompute pair indices for equal-pair counting
    pairs_idx = [(0,1),(0,2),(0,3),(0,4),(0,5),
                 (1,2),(1,3),(1,4),(1,5),
                 (2,3),(2,4),(2,5),
                 (3,4),(3,5),
                 (4,5)]
    
    def count_violations_fast(colors_np):
        """Fast violation count using numpy."""
        k4_colours = colors_np[k4_edges_arr]  # shape (1820, 6)
        num_equal_pairs = np.zeros(len(k4_colours), dtype=np.int32)
        for (a, b) in pairs_idx:
            num_equal_pairs += (k4_colours[:, a] == k4_colours[:, b]).astype(np.int32)
        violations = np.sum(num_equal_pairs >= 2)
        return violations
    
    def count_violations_incremental(colors_np, changed_edge, old_c, new_c):
        """Check violations only for K4s containing the changed edge."""
        # Find K4s that contain this edge
        # We need to check which K4s have this edge
        # This is expensive to do dynamically, so let's precompute
        
        # For now, just do full check
        return count_violations_fast(colors_np)
    
    # Try to find a valid coloring with k colors
    def try_random_search(k, max_iter=500000):
        """Random search with local improvement for k colors."""
        best_violations = float('inf')
        best_colors = None
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Random initialization with k colors
            colors = np.random.randint(0, k, size=num_edges, dtype=np.int32)
            violations = count_violations_fast(colors)
            
            if violations == 0:
                return colors
            
            if violations < best_violations:
                best_violations = violations
                best_colors = colors.copy()
            
            # Local search: try to fix violations
            # Get list of violating K4s
            k4_colours = colors[k4_edges_arr]
            num_equal_pairs = np.zeros(len(k4_colours), dtype=np.int32)
            for (a, b) in pairs_idx:
                num_equal_pairs += (k4_colours[:, a] == k4_colours[:, b]).astype(np.int32)
            violating_k4s = np.where(num_equal_pairs >= 2)[0]
            
            if len(violating_k4s) == 0:
                return colors
            
            # Pick a random violating K4 and try to fix it
            for _ in range(10):
                if time.time() > deadline:
                    break
                k4_idx = random.choice(violating_k4s.tolist())
                k4_edge_indices = k4_edges_arr[k4_idx]
                
                # Try changing one edge in this K4 to a different color
                for ei in k4_edge_indices:
                    old_c = colors[ei]
                    # Try all other colors
                    for new_c in range(k):
                        if new_c == old_c:
                            continue
                        colors[ei] = new_c
                        new_violations = count_violations_fast(colors)
                        if new_violations < violations:
                            violations = new_violations
                            if violations == 0:
                                return colors
                            if violations < best_violations:
                                best_violations = violations
                                best_colors = colors.copy()
                            break
                        else:
                            colors[ei] = old_c
                    else:
                        continue
                    break
        
        return best_colors if best_violations < float('inf') else None
    
    # Try structured initialization based on modular arithmetic
    def try_structured(k):
        """Try structured colorings with k colors."""
        # Pattern 1: (i + j) mod k
        colors = np.array([(i + j) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        if count_violations_fast(colors) == 0:
            return colors
        
        # Pattern 2: (i * 2 + j) mod k
        colors = np.array([(i * 2 + j) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        if count_violations_fast(colors) == 0:
            return colors
        
        # Pattern 3: (i * 3 + j) mod k
        colors = np.array([(i * 3 + j) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        if count_violations_fast(colors) == 0:
            return colors
        
        # Pattern 4: (i + 2*j) mod k
        colors = np.array([(i + 2 * j) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        if count_violations_fast(colors) == 0:
            return colors
        
        # Pattern 5: (i * j) mod k
        colors = np.array([(i * j) % k for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
        if count_violations_fast(colors) == 0:
            return colors
        
        # Pattern 6: (i + j) % (k-1) + 1 if k > 1
        if k > 1:
            colors = np.array([(i + j) % (k - 1) + 1 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            if count_violations_fast(colors) == 0:
                return colors
        
        return None
    
    # Start with known valid solution using 17 colors
    base_colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    assert count_violations_fast(base_colors) == 0
    best_colors = base_colors.copy()
    best_num_colors = 17
    
    # Try to find solutions with fewer colors
    for k in range(13, 17):
        if time.time() > deadline:
            break
        
        # First try structured
        result = try_structured(k)
        if result is not None:
            best_colors = result
            best_num_colors = k
            break
        
        # Then try random search with multiple restarts
        for restart in range(10):
            if time.time() > deadline:
                break
            result = try_random_search(k, max_iter=100000)
            if result is not None:
                # Verify
                if count_violations_fast(result) == 0:
                    best_colors = result
                    best_num_colors = k
                    break
    
    # If we still have 17, try to improve via local search from base
    if best_num_colors == 17:
        current = base_colors.copy()
        for _ in range(500000):
            if time.time() > deadline:
                break
            
            # Pick random edge
            ei = random.randint(0, num_edges - 1)
            old_c = current[ei]
            
            # Try to change to a color that reduces palette or keeps same
            # First try to use only colors 0..14
            if old_c >= 15:
                new_c = random.randint(0, 14)
                current[ei] = new_c
                if count_violations_fast(current) == 0:
                    new_num = len(set(current.tolist()))
                    if new_num < best_num_colors:
                        best_colors = current.copy()
                        best_num_colors = new_num
                else:
                    current[ei] = old_c
            else:
                # Try random color from 0..14
                new_c = random.randint(0, 14)
                if new_c == old_c:
                    continue
                current[ei] = new_c
                if count_violations_fast(current) == 0:
                    new_num = len(set(current.tolist()))
                    if new_num < best_num_colors:
                        best_colors = current.copy()
                        best_num_colors = new_num
                else:
                    current[ei] = old_c
    
    # Final check
    if count_violations_fast(best_colors) != 0:
        best_colors = base_colors.copy()
        best_num_colors = 17
    
    # Renormalize colors to 0..best_num_colors-1
    col_map = {}
    for i, c in enumerate(sorted(set(best_colors.tolist()))):
        col_map[c] = i
    best_colors_list = [col_map[c] for c in best_colors.tolist()]
    
    return best_colors_list
# EVOLVE-BLOCK-END
