import numpy as np
import time
import random
import itertools

def construct():
    n = 31
    target_edges = 91
    c4_free_edges = 90
    deadline = time.time() + 110
    
    all_pairs = list(itertools.combinations(range(n), 2))
    
    best_score = float('inf')
    best_edges = None
    
    # We will try multiple random seeds for the greedy construction
    # to find C4-free graphs with 90 edges.
    # Then we pick the best edge to add to minimize C4s.
    
    for seed in range(2000):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        shuffled_pairs = all_pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # Greedy construction of C4-free graph
        adj = np.zeros((n, n), dtype=np.int8)
        edges = []
        
        for u, v in shuffled_pairs:
            if len(edges) == c4_free_edges:
                break
            
            # Check if adding (u, v) creates a C4
            # A C4 is u-w-x-v-u. 
            # This exists if there is a path of length 3 from u to v.
            # Equivalently, if there are w, x such that u-w, w-x, x-v are edges.
            
            Nu = np.where(adj[u] == 1)[0]
            Nv = np.where(adj[v] == 1)[0]
            
            if len(Nu) == 0 or len(Nv) == 0:
                # Safe to add
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append((u, v))
                continue
            
            # Check if any edge connects Nu and Nv
            # This corresponds to the middle edge w-x of the path u-w-x-v
            # If such an edge exists, u-w-x-v-u is a C4.
            sub = adj[np.ix_(Nu, Nv)]
            if np.sum(sub) == 0:
                adj[u, v] = 1
                adj[v, u] = 1
                edges.append((u, v))
        
        if len(edges) < c4_free_edges:
            continue
            
        # We have a C4-free graph with 90 edges.
        # Now find the edge (u, v) not in the graph that minimizes the number of C4s created.
        # The number of C4s created by adding (u, v) is the number of paths of length 3 between u and v.
        # This is (A^3)[u, v].
        
        A = adj.copy()
        # Compute A^3
        # Using matrix multiplication
        A3 = A @ A @ A
        
        min_cost = float('inf')
        best_edge_to_add = None
        
        for u, v in all_pairs:
            if adj[u, v] == 0:
                cost = A3[u, v]
                if cost < min_cost:
                    min_cost = cost
                    best_edge_to_add = (u, v)
                    
        if best_edge_to_add is None:
            continue
            
        # The total number of C4s in the final graph is min_cost (since base was 0)
        if min_cost < best_score:
            best_score = min_cost
            final_edges = edges + [best_edge_to_add]
            best_edges = [[u, v] for u, v in final_edges]
            
            if best_score == 1:
                # Cannot do better than 1 (since ex=90, 91 edges must have >=1 C4)
                break
            if best_score == 0:
                break # Theoretical impossibility, but safe break

    if best_edges is None:
        # Fallback: return first 91 lexicographic edges
        return [[i, j] for i in range(n) for j in range(i + 1, n)][:91]
        
    return best_edges
