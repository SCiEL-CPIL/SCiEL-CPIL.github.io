import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] that share a factor with 255
    candidates = [x for x in range(1, 256) if x % 3 == 0 or x % 5 == 0 or x % 17 == 0]
    
    n = len(candidates)
    idx_255 = candidates.index(255)
    
    # Build adjacency using bitsets
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # For maximum clique containing 255:
    # We need a clique in the graph induced by {255} ∪ (neighbors of 255)
    # that contains 255. This is equivalent to finding a maximum clique in the
    # induced subgraph of neighbors of 255, then adding 255.
    
    # Get neighbors of 255 (excluding 255 itself)
    neighbors_of_255 = []
    for i in range(n):
        if i != idx_255 and (adj[idx_255] & (1 << i)):
            neighbors_of_255.append(i)
    
    m = len(neighbors_of_255)
    
    # Build adjacency for the induced subgraph of neighbors_of_255
    # Map: original index -> new index in subgraph
    idx_map = {}
    for new_i, old_i in enumerate(neighbors_of_255):
        idx_map[old_i] = new_i
    
    # Subgraph adjacency as bitsets
    sub_adj = [0] * m
    for i in range(m):
        old_i = neighbors_of_255[i]
        bits = adj[old_i]
        for j in range(i + 1, m):
            old_j = neighbors_of_255[j]
            if bits & (1 << old_j):
                sub_adj[i] |= (1 << j)
                sub_adj[j] |= (1 << i)
    
    # Tomita's Maximum Clique Algorithm with coloring upper bound
    # Uses bitsets for efficiency
    
    best_clique = []
    best_size = 0
    current_clique = []
    
    # Precompute color ordering (greedy coloring for upper bound)
    # We'll use a simpler approach: recursive branch and bound
    
    def color_bound(P, colors):
        """Assign colors to vertices in P such that adjacent vertices get different colors.
        The number of colors is an upper bound on the maximum clique in P."""
        # Greedy coloring
        uncolored = set(range(len(P)))
        color = 0
        while uncolored:
            color += 1
            independent_set = set()
            for v in list(uncolored):
                # Check if v can be added to current color class
                # v can be added if it's not adjacent to any vertex already in independent_set
                can_add = True
                for u in independent_set:
                    if sub_adj[v] & (1 << u):
                        can_add = False
                        break
                if can_add:
                    independent_set.add(v)
                    uncolored.discard(v)
        return color
    
    # More efficient: use the classic Tomita approach with a specific vertex ordering
    # Let's use a simpler recursive approach with greedy upper bounds
    
    # Convert to a form where we track the candidate set as a list
    # For the subgraph with m vertices
    
    def tomasi_max_clique(order, P_indices, P_size, current_size):
        """
        order: list of vertices in the current order
        P_indices: indices into 'order' for the current candidate set P
        P_size: number of vertices in P
        current_size: current clique size
        """
        nonlocal best_clique, best_size
        
        if P_size == 0:
            if current_size > best_size:
                best_size = current_size
                best_clique = list(current_clique)
            return
        
        if current_size + P_size <= best_size:
            return
        
        # Get the set of candidate vertices
        P = set(P_indices)
        
        # Greedy coloring of P for upper bound
        # Order vertices in P by index
        P_list = sorted(P)
        
        # Simple greedy coloring
        color_of = {}
        num_colors = 0
        color_classes = []
        for v in P_list:
            placed = False
            for c, cls in enumerate(color_classes):
                # Check if v is non-adjacent to all in cls
                can_place = True
                for u in cls:
                    if sub_adj[v] & (1 << u):
                        can_place = False
                        break
                if can_place:
                    color_of[v] = c
                    cls.append(v)
                    placed = True
                    break
            if not placed:
                num_colors += 1
                color_of[v] = num_colors - 1
                color_classes.append([v])
        
        # Upper bound: current_size + num_colors
        if current_size + num_colors <= best_size:
            return
        
        # Process vertices in order of increasing color
        # Vertices with lower color have tighter bounds
        for c in range(num_colors - 1, -1, -1):
            # Vertices with color c: these are the last in their color class
            # In the standard Tomita, we process one vertex at a time
            # and use the color of that vertex as the upper bound for its subtree
            
            pass
        
        # Simpler approach: iterate through P_list in reverse color order
        # Actually, let's just iterate and use the color as bound
        # For each vertex, the bound is current_size + color_of[v]
        
        # Process in order of decreasing color (tighter bound first for pruning)
        sorted_P = sorted(P_list, key=lambda v: color_of[v], reverse=True)
        
        for v in sorted_P:
            if current_size + color_of[v] <= best_size:
                break
            
            # Check if v is still in P
            if v not in P:
                continue
            
            current_clique.append(v)
            
            # New P: neighbors of v that are also in P and have index > v in P_list
            # Actually, to avoid duplicates, we only consider vertices in P that come after v
            # But since we're using sets, let's just take intersection with neighbors
            new_P = P & set((sub_adj[v] & ((1 << m) - 1)).__iter__().__next__() if False else [u for u in P if u != v and (sub_adj[v] & (1 << u))])
            
            # Simpler: new_P = {u in P : u != v and adjacent to v}
            new_P = set()
            for u in P:
                if u != v and (sub_adj[v] & (1 << u)):
                    new_P.add(u)
            
            # Recurse
            if len(new_P) > 0:
                tomasi_max_clique(order, list(new_P), len(new_P), current_size + 1)
            else:
                if current_size + 1 > best_size:
                    best_size = current_size + 1
                    best_clique = list(current_clique)
            
            current_clique.pop()
            P.discard(v)
    
    # Initialize with all m vertices
    if m > 0:
        tomasi_max_clique(list(range(m)), list(range(m)), m, 0)
    
    # best_clique contains indices into neighbors_of_255
    # Convert back to original indices and add 255
    result_indices = [neighbors_of_255[i] for i in best_clique]
    result_indices.append(idx_255)
    
    # If we didn't find anything better than just {255}, use the multiples of 3
    if best_size < 84:  # 84 neighbors + 255 = 85
        # Fallback: multiples of 3
        result_indices = [i for i, x in enumerate(candidates) if x % 3 == 0]
    
    result = [candidates[i] for i in result_indices]
    result.sort()
    
    # Verify
    assert 255 in result, "255 not in result"
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert gcd(result[i], result[j]) > 1, f"Failed: gcd({result[i]},{result[j]})=1"
    
    return result
