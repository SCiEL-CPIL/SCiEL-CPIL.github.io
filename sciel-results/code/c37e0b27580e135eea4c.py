import time
import random
import numpy as np

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Uses a randomized greedy approach with multiple restarts and local search.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Precompute all edges: (u, v, dim) where u < v and dim is the differing bit
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y, i))
    
    E = len(all_edges)  # 448
    
    # For each edge, precompute the 3 other edges in each of the 6 C4s it belongs to
    # Edge (x, y) with dim i: C4 is formed by flipping bit j (j != i)
    # The C4 vertices are: x, x^j, y, y^j
    # The 4 edges are: (x, y) [dim i], (x, x^j) [dim j], (y, y^j) [dim j], (x^j, y^j) [dim i]
    
    # Build edge index: map (min(u,v), max(u,v)) -> index in all_edges
    edge_index = {}
    for idx, (u, v, dim) in enumerate(all_edges):
        edge_index[(u, v)] = idx
    
    # For each edge, find all C4s it belongs to and the other 3 edges in each
    edge_c4s = [[] for _ in range(E)]  # edge_c4s[e] = list of (other1, other2, other3)
    
    for idx, (x, y, i) in enumerate(all_edges):
        for j in range(d):
            if j == i:
                continue
            xj = x ^ (1 << j)
            yj = y ^ (1 << j)
            # The 4 edges of this C4:
            e1 = (x, y)  # this edge
            e2 = (min(x, xj), max(x, xj))  # dim j
            e3 = (min(y, yj), max(y, yj))  # dim j
            e4 = (min(xj, yj), max(xj, yj))  # dim i
            
            idx2 = edge_index[e2]
            idx3 = edge_index[e3]
            idx4 = edge_index[e4]
            
            edge_c4s[idx].append((idx2, idx3, idx4))
    
    best_count = 0
    best_edges = []
    
    def greedy_randomized(seed):
        """Randomized greedy: process edges in random order, add if no C4 is completed."""
        random.seed(seed)
        order = list(range(E))
        random.shuffle(order)
        
        chosen = [False] * E
        count = 0
        
        for idx in order:
            # Check if adding this edge would complete any C4
            ok = True
            for (o1, o2, o3) in edge_c4s[idx]:
                if chosen[o1] and chosen[o2] and chosen[o3]:
                    ok = False
                    break
            if ok:
                chosen[idx] = True
                count += 1
        
        return chosen, count
    
    def local_search(chosen, count):
        """Try to improve by swapping: remove an edge, then greedily add others."""
        # Find edges that are part of C4s (i.e., could potentially be removed to allow additions)
        # Actually, let's try a simpler approach: try removing each edge and see if we can add more
        # For efficiency, only try edges that are "critical" (part of some blocked C4)
        
        # Find all blocked C4s
        # A C4 is blocked if 3 of its 4 edges are chosen
        # Let's try removing one edge from each blocked C4 and see if we can add 2+ new edges
        
        # First, find all edges that are in at least one C4 where the other 3 are chosen
        # This is expensive, so let's do a simpler local search:
        # Try removing each edge, then greedily add all possible edges in random order
        
        best_local = chosen[:]
        best_local_count = count
        
        # Try a few random removals
        for _ in range(20):
            if time.time() - start_time > time_limit - 10:
                break
            # Pick a random edge to remove
            rem_idx = random.randint(0, E - 1)
            if not chosen[rem_idx]:
                continue
            
            new_chosen = chosen[:]
            new_chosen[rem_idx] = False
            new_count = count - 1
            
            # Now try to add edges that were previously blocked
            # Get edges in random order
            order = list(range(E))
            random.shuffle(order)
            
            for idx in order:
                if new_chosen[idx]:
                    continue
                ok = True
                for (o1, o2, o3) in edge_c4s[idx]:
                    if new_chosen[o1] and new_chosen[o2] and new_chosen[o3]:
                        ok = False
                        break
                if ok:
                    new_chosen[idx] = True
                    new_count += 1
            
            if new_count > best_local_count:
                best_local = new_chosen
                best_local_count = new_count
        
        return best_local, best_local_count
    
    # Multiple restarts
    seed = 42
    while time.time() - start_time < time_limit - 5:
        chosen, count = greedy_randomized(seed)
        
        if count > best_count:
            best_count = count
            best_edges = [all_edges[i][0] for i in range(E) if chosen[i]]
            best_edges = [[all_edges[i][0], all_edges[i][1]] for i in range(E) if chosen[i]]
        
        # Local search
        if time.time() - start_time < time_limit - 15:
            improved_chosen, improved_count = local_search(chosen, count)
            if improved_count > best_count:
                best_count = improved_count
                best_edges = [[all_edges[i][0], all_edges[i][1]] for i in range(E) if improved_chosen[i]]
        
        seed += 1
    
    # Return the best result
    return best_edges
