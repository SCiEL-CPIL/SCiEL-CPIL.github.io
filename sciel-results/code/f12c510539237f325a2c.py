import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    n = 16
    # Edge list in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    num_edges = len(edges)
    
    # Build edge index lookup
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    # Precompute all K4s and their edge indices
    k4s = []
    for a in range(n-3):
        for b in range(a+1, n-2):
            for c in range(b+1, n-1):
                for d in range(c+1, n):
                    e_idx = [
                        edge_idx[(a,b)],
                        edge_idx[(a,c)],
                        edge_idx[(a,d)],
                        edge_idx[(b,c)],
                        edge_idx[(b,d)],
                        edge_idx[(c,d)]
                    ]
                    k4s.append(e_idx)
    
    num_k4s = len(k4s)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_idx in enumerate(k4s):
        for e in e_idx:
            edge_to_k4s[e].append(k4_idx)
    
    # Initialize with a valid coloring using 17 colors: (i+j) % 17
    colors = [(i + j) % 17 for i, j in edges]
    current_max = max(colors)
    
    # Helper: count number of K4s with < 5 distinct colors
    def count_bad_k4s(col):
        bad = 0
        for e_idx in k4s:
            c0, c1, c2, c3, c4, c5 = col[e_idx[0]], col[e_idx[1]], col[e_idx[2]], col[e_idx[3]], col[e_idx[4]], col[e_idx[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                bad += 1
        return bad
    
    # Optimized local search with O(1) penalty updates
    def local_search_reduce_max(col, max_color, deadline):
        """Try to reduce max color by local search."""
        col = col[:]
        current_max = max_color
        bad_count = count_bad_k4s(col)
        
        if bad_count > 0:
            return None, current_max
        
        # Track which color is used most, to target for reduction
        while time.time() < deadline:
            # Find the highest color in use
            colors_in_use = set(col)
            if not colors_in_use:
                break
            target_max = max(colors_in_use)
            
            # Try to remove target_max by reassigning edges with that color
            found_improvement = False
            for idx in range(num_edges):
                if time.time() > deadline:
                    return col, current_max
                if col[idx] != target_max:
                    continue
                
                old_color = col[idx]
                # Try to assign a different color (preferably lower)
                for new_color in range(0, target_max):
                    if new_color == old_color:
                        continue
                    
                    # Apply change
                    col[idx] = new_color
                    
                    # Update bad_count incrementally
                    # Only K4s containing edge idx are affected
                    new_bad = 0
                    for k4_idx in edge_to_k4s[idx]:
                        e_idx = k4s[k4_idx]
                        c0, c1, c2, c3, c4, c5 = col[e_idx[0]], col[e_idx[1]], col[e_idx[2]], col[e_idx[3]], col[e_idx[4]], col[e_idx[5]]
                        if len({c0, c1, c2, c3, c4, c5}) < 5:
                            new_bad += 1
                    
                    if new_bad == 0:
                        # Valid! We reduced the max color
                        current_max = target_max - 1
                        found_improvement = True
                        break
                    else:
                        col[idx] = old_color
                
                if found_improvement:
                    break
            
            if not found_improvement:
                # Can't reduce further with this approach
                break
        
        if count_bad_k4s(col) == 0:
            return col, current_max
        return None, current_max
    
    # Main search: try to find a valid coloring with fewer colors
    random.seed(42)
    best_colors = colors[:]
    best_max = current_max
    
    # Phase 1: Try to reduce from 17 to 16, 15, etc.
    for target_max in range(16, 12, -1):
        if time.time() > deadline:
            break
        
        # Try random restarts with limited color palette
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize randomly with colors 0..target_max
            col = [random.randint(0, target_max) for _ in range(num_edges)]
            
            # Local search to fix violations
            improved = True
            iterations = 0
            while improved and time.time() < deadline:
                improved = False
                iterations += 1
                
                # Find a bad K4
                bad_k4_idx = -1
                for k4_idx in range(num_k4s):
                    e_idx = k4s[k4_idx]
                    c0, c1, c2, c3, c4, c5 = col[e_idx[0]], col[e_idx[1]], col[e_idx[2]], col[e_idx[3]], col[e_idx[4]], col[e_idx[5]]
                    if len({c0, c1, c2, c3, c4, c5}) < 5:
                        bad_k4_idx = k4_idx
                        break
                
                if bad_k4_idx == -1:
                    break  # No bad K4s, we have a valid coloring!
                
                # Pick a random edge from this bad K4 and try to change its color
                e_idx = k4s[bad_k4_idx]
                random.shuffle(e_idx)
                
                for edge_idx in e_idx:
                    old_color = col[edge_idx]
                    # Try a new color
                    new_color = random.randint(0, target_max)
                    if new_color == old_color:
                        continue
                    
                    col[edge_idx] = new_color
                    
                    # Check if this fixes the bad K4 without creating new bad K4s
                    # Just check the specific K4 we're trying to fix
                    c0, c1, c2, c3, c4, c5 = col[e_idx[0]], col[e_idx[1]], col[e_idx[2]], col[e_idx[3]], col[e_idx[4]], col[e_idx[5]]
                    if len({c0, c1, c2, c3, c4, c5}) >= 5:
                        # Check if we created new bad K4s (sample a few)
                        created_bad = False
                        for check_k4 in edge_to_k4s[edge_idx][:5]:  # Sample up to 5
                            ce = k4s[check_k4]
                            cc0, cc1, cc2, cc3, cc4, cc5 = col[ce[0]], col[ce[1]], col[ce[2]], col[ce[3]], col[ce[4]], col[ce[5]]
                            if len({cc0, cc1, cc2, cc3, cc4, cc5}) < 5:
                                created_bad = True
                                break
                        
                        if not created_bad:
                            improved = True
                            break
                        else:
                            col[edge_idx] = old_color
                
                if time.time() > deadline:
                    break
            
            # Check if we found a valid coloring
            if count_bad_k4s(col) == 0:
                best_colors = col[:]
                best_max = target_max
                break  # Found valid coloring with this target_max, try lower
    
    # Phase 2: Try to further reduce using the best found
    if time.time() < deadline:
        result, new_max = local_search_reduce_max(best_colors, best_max, deadline)
        if result is not None:
            best_colors = result
            best_max = new_max
    
    return best_colors
