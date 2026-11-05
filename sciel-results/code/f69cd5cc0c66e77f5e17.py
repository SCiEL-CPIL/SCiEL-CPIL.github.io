import random
import numpy as np

def construct():
    n = 18
    random.seed(42)
    
    # Start with a random 3-partition of 18 vertices
    # Part sizes: 6, 6, 6
    vertices = list(range(n))
    random.shuffle(vertices)
    parts = [
        set(vertices[0:6]),
        set(vertices[6:12]),
        set(vertices[12:18])
    ]
    
    def build_adjacency(parts):
        """Build adjacency matrix for a 3-partite graph where parts are independent sets."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                # Edge exists if i and j are in different parts
                same_part = False
                for p in parts:
                    if i in p and j in p:
                        same_part = True
                        break
                if not same_part:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj
    
    def is_k4_free(adj):
        """Check if graph is K4-free."""
        for a in range(n):
            for b in range(a + 1, n):
                if adj[a, b] == 0:
                    continue
                for c in range(b + 1, n):
                    if adj[a, c] == 0 or adj[b, c] == 0:
                        continue
                    for d in range(c + 1, n):
                        if adj[a, d] == 0 or adj[b, d] == 0 or adj[c, d] == 0:
                            continue
                        # Found K4
                        return False
        return True
    
    def max_triangle_free_induced(adj):
        """Find the maximum size of a triangle-free induced subgraph."""
        # Use branch and bound
        best = [0]
        
        def is_triangle_free(subset):
            for i in range(len(subset)):
                for j in range(i + 1, len(subset)):
                    for k in range(j + 1, len(subset)):
                        if adj[subset[i], subset[j]] and adj[subset[i], subset[k]] and adj[subset[j], subset[k]]:
                            return False
            return True
        
        # Try all subsets using bit masks for efficiency
        # 2^18 = 262144 subsets, feasible
        for mask in range(1 << n):
            # Count bits
            if bin(mask).count('1') <= best[0]:
                continue
            subset = [i for i in range(n) if (mask >> i) & 1]
            if is_triangle_free(subset):
                best[0] = len(subset)
        
        return best[0]
    
    def local_search(parts):
        """Perform local search by swapping vertices between parts."""
        current_parts = [set(p) for p in parts]
        current_adj = build_adjacency(current_parts)
        
        if not is_k4_free(current_adj):
            return None, None
        
        current_score = max_triangle_free_induced(current_adj)
        best_parts = [set(p) for p in current_parts]
        best_score = current_score
        
        for iteration in range(200):
            improved = False
            # Try all possible swaps between pairs of parts
            for i in range(3):
                for j in range(i + 1, 3):
                    for u in list(current_parts[i]):
                        for v in list(current_parts[j]):
                            # Swap u and v
                            new_parts = [set(p) for p in current_parts]
                            new_parts[i].remove(u)
                            new_parts[i].add(v)
                            new_parts[j].remove(v)
                            new_parts[j].add(u)
                            
                            new_adj = build_adjacency(new_parts)
                            if not is_k4_free(new_adj):
                                continue
                            
                            new_score = max_triangle_free_induced(new_adj)
                            if new_score < best_score:
                                best_score = new_score
                                best_parts = [set(p) for p in new_parts]
                                current_parts = [set(p) for p in new_parts]
                                current_adj = new_adj
                                current_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        
        return best_parts, best_score
    
    # Perform local search
    best_parts, best_score = local_search(parts)
    
    if best_parts is None:
        # Fallback to original partition
        best_parts = parts
        best_score = max_triangle_free_induced(build_adjacency(best_parts))
    
    # Build edge list from best partition
    adj = build_adjacency(best_parts)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    
    return edges
