"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Step 1: Build the Clebsch graph as initial seed
    reps = []
    seen = set()
    for i in range(32):
        rep = min(i, 31 - i)
        if rep not in seen:
            seen.add(rep)
            reps.append(rep)
    reps.sort()
    
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            u = reps[i]
            v = reps[j]
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist == 1 or dist == 4:
                edges.append([i, j])
    
    # Build adjacency set for quick lookup
    adj = set()
    for e in edges:
        adj.add((e[0], e[1]))
        adj.add((e[1], e[0]))
    
    def is_triangle_free(adj_set):
        """Check if graph with given adjacency is triangle-free."""
        for (u, v) in adj_set:
            if u > v:
                continue
            for w in range(16):
                if w != u and w != v and (u, w) in adj_set and (v, w) in adj_set:
                    return False
        return True
    
    def is_not_3_colorable(adj_set):
        """Check if graph is NOT 3-colorable using backtracking."""
        # Try all 3-colorings via backtracking
        color = [-1] * 16
        
        def backtrack(v):
            if v == 16:
                return True
            for c in range(3):
                valid = True
                for u in range(v):
                    if (u, v) in adj_set and color[u] == c:
                        valid = False
                        break
                if valid:
                    color[v] = c
                    if backtrack(v + 1):
                        return True
                    color[v] = -1
            return False
        
        # Use ordering: try vertices with highest degree first for efficiency
        degrees = [0] * 16
        for (u, v) in adj_set:
            if u < v:
                degrees[u] += 1
                degrees[v] += 1
        
        order = sorted(range(16), key=lambda x: -degrees[x])
        pos = {v: i for i, v in enumerate(order)}
        
        def backtrack_optimized(idx):
            if idx == 16:
                return True
            v = order[idx]
            used_colors = set()
            for u in range(16):
                if u != v and (u, v) in adj_set:
                    if pos[u] < idx:
                        used_colors.add(color[u])
            for c in range(3):
                if c not in used_colors:
                    color[v] = c
                    if backtrack_optimized(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return not backtrack_optimized(0)
    
    best_edges = list(edges)
    best_count = len(edges)
    
    # Step 2: Local search - try to add edges
    all_possible = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    current_adj = set(adj)
    
    random.seed(42)
    
    while time.time() < deadline:
        # Find all edges that can be added without creating a triangle
        candidates = []
        for (i, j) in all_possible:
            if (i, j) not in current_adj:
                # Check if adding (i,j) creates a triangle
                creates_triangle = False
                for w in range(16):
                    if w != i and w != j:
                        if (i, w) in current_adj and (j, w) in current_adj:
                            creates_triangle = True
                            break
                if not creates_triangle:
                    candidates.append((i, j))
        
        if not candidates:
            break
        
        # Try adding each candidate and check 3-colorability
        improved = False
        random.shuffle(candidates)
        
        for (i, j) in candidates:
            if time.time() >= deadline:
                break
            
            # Tentatively add edge
            current_adj.add((i, j))
            
            if is_not_3_colorable(current_adj):
                # Keep the edge
                best_edges.append([i, j])
                best_count += 1
                improved = True
            else:
                # Remove the edge
                current_adj.remove((i, j))
        
        if not improved:
            # No more edges can be added; try edge swaps
            # Try removing one edge and adding two others
            found_swap = False
            for (r, s) in list(current_adj):
                if r > s:
                    continue
                if time.time() >= deadline:
                    break
                
                current_adj.discard((r, s))
                current_adj.discard((s, r))
                
                # Try to add back two edges
                swap_candidates = []
                for (i, j) in all_possible:
                    if (i, j) not in current_adj and (i, j) != (r, s):
                        creates_triangle = False
                        for w in range(16):
                            if w != i and w != j:
                                if (i, w) in current_adj and (j, w) in current_adj:
                                    creates_triangle = True
                                    break
                        if not creates_triangle:
                            swap_candidates.append((i, j))
                
                if len(swap_candidates) >= 2:
                    for idx1 in range(len(swap_candidates)):
                        if time.time() >= deadline:
                            break
                        e1 = swap_candidates[idx1]
                        for idx2 in range(idx1 + 1, len(swap_candidates)):
                            if time.time() >= deadline:
                                break
                            e2 = swap_candidates[idx2]
                            
                            current_adj.add(e1[0], e1[1])
                            current_adj.add(e1[1], e1[0])
                            current_adj.add(e2[0], e2[1])
                            current_adj.add(e2[1], e2[0])
                            
                            if is_not_3_colorable(current_adj):
                                # Keep the swap
                                best_edges = [list(e) for e in current_adj if e[0] < e[1]]
                                best_count = len(best_edges)
                                found_swap = True
                                break
                            else:
                                current_adj.discard(e1[0], e1[1])
                                current_adj.discard(e1[1], e1[0])
                                current_adj.discard(e2[0], e2[1])
                                current_adj.discard(e2[1], e2[0])
                        
                        if found_swap:
                            break
                
                if not found_swap:
                    # Restore removed edge
                    current_adj.add((r, s))
                    current_adj.add((s, r))
                
                if found_swap:
                    break
            
            if not found_swap:
                break
    
    return best_edges
# EVOLVE-BLOCK-END
