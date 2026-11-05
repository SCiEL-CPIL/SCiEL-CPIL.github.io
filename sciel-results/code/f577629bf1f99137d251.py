"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book
    (max over edges of number of triangles containing that edge).
    
    Approach: Spectral optimization via local search.
    1. Initialize with a quasi-random regular-like graph (balanced bipartite + one intra edge)
    2. Use local edge swaps to flatten the triangle distribution (A^2 degrees)
    3. Efficiently update book sizes for only affected edges.
    """
    n = 40
    target_edges = 401
    
    import random
    random.seed(42)
    
    # Step 1: Initialize with a balanced bipartite graph K20,20 + one intra-part edge
    edges_set = set()
    for a in range(20):
        for b in range(20, 40):
            edges_set.add((a, b))
    edges_set.add((0, 1))
    
    adj = np.zeros((n, n), dtype=np.int32)
    for u, v in edges_set:
        adj[u, v] = 1
        adj[v, u] = 1
        
    # Precompute neighbor sets
    neighbors = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if adj[u, v]:
                neighbors[u].add(v)
                neighbors[v].add(u)
                
    def get_book_sizes(adj, neighbors):
        books = {}
        for u in range(n):
            for v in neighbors[u]:
                if u < v:
                    common = len(neighbors[u].intersection(neighbors[v]))
                    books[(u, v)] = common
        return books

    books = get_book_sizes(adj, neighbors)
    current_max = max(books.values())
    
    best_adj = adj.copy()
    best_max = current_max
    
    max_iterations = 20000
    no_improve_count = 0
    max_no_improve = 1000
    
    for iteration in range(max_iterations):
        if current_max <= 7:
            break
            
        max_book_val = current_max
        max_edges = [e for e, b in books.items() if b == max_book_val]
        
        if not max_edges:
            break
            
        random.shuffle(max_edges)
        improved = False
        
        for u, v in max_edges:
            # Temporarily remove (u,v)
            adj[u, v] = 0
            adj[v, u] = 0
            neighbors[u].remove(v)
            neighbors[v].remove(u)
            
            best_new_edge = None
            best_new_max = current_max + 100
            
            # Sample 50 random non-edges
            for _ in range(50):
                w = random.randint(0, n-1)
                x = random.randint(0, n-1)
                if w == x or adj[w, x]:
                    continue
                if w > x: w, x = x, w
                
                if w in (u, v) or x in (u, v):
                    continue
                
                common = len(neighbors[w].intersection(neighbors[x]))
                
                if common < best_new_max:
                    # Verify full impact
                    adj[w, x] = 1
                    adj[x, w] = 1
                    neighbors[w].add(x)
                    neighbors[x].add(w)
                    
                    new_max = 0
                    affected = {u, v, w, x}
                    for av in affected:
                        for nb in list(neighbors[av]):
                            if av < nb:
                                c = len(neighbors[av].intersection(neighbors[nb]))
                                if c > new_max:
                                    new_max = c
                                    
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_new_edge = (w, x)
                    else:
                        # Revert
                        adj[w, x] = 0
                        adj[x, w] = 0
                        neighbors[w].remove(x)
                        neighbors[x].remove(w)
                        if best_new_edge:
                            # We found a better one, so we keep the state of best_new_edge?
                            # No, we reverted. We need to re-apply best_new_edge if we accept it later.
                            # This logic is flawed. Let's just store the best edge and its max.
                            pass
            
            # Apply best new edge if found
            if best_new_edge and best_new_max < current_max:
                w, x = best_new_edge
                # We need to ensure the state is clean.
                # We are currently in state: (u,v) removed, (w,x) NOT added (unless it was the last tried and not reverted).
                # To be safe, revert all temporary changes and re-apply the best one.
                # Actually, the loop above reverts non-best candidates.
                # If best_new_edge was the last one tried, it might be applied.
                # Let's just revert everything and apply best_new_edge cleanly.
                
                # Revert (w,x) if it is currently applied
                if adj[best_new_edge[0], best_new_edge[1]]:
                     adj[best_new_edge[0], best_new_edge[1]] = 0
                     adj[best_new_edge[1], best_new_edge[0]] = 0
                     neighbors[best_new_edge[0]].remove(best_new_edge[1])
                     neighbors[best_new_edge[1]].remove(best_new_edge[0])

                # Apply (w,x)
                adj[w, x] = 1
                adj[x, w] = 1
                neighbors[w].add(x)
                neighbors[x].add(w)
                
                # Update books
                if u < v: books.pop((u, v), None)
                else: books.pop((v, u), None)
                
                books[(w, x)] = len(neighbors[w].intersection(neighbors[x]))
                
                affected = {u, v, w, x}
                for av in affected:
                    for nb in list(neighbors[av]):
                        if av < nb:
                            books[(av, nb)] = len(neighbors[av].intersection(neighbors[nb]))
                
                current_max = best_new_max
                improved = True
                no_improve_count = 0
                if current_max < best_max:
                    best_max = current_max
                    best_adj = adj.copy()
                break
            else:
                # Revert (u,v)
                adj[u, v] = 1
                adj[v, u] = 1
                neighbors[u].add(v)
                neighbors[v].add(u)
                
                if best_new_edge:
                     # Revert the applied best_new_edge from the verification step if it's still there
                     if adj[best_new_edge[0], best_new_edge[1]]:
                         adj[best_new_edge[0], best_new_edge[1]] = 0
                         adj[best_new_edge[1], best_new_edge[0]] = 0
                         neighbors[best_new_edge[0]].remove(best_new_edge[1])
                         neighbors[best_new_edge[1]].remove(best_new_edge[0])
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break

    result = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u, v]:
                result.append([u, v])
    
    if len(result) != target_edges:
        result = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return result
# EVOLVE-BLOCK-END
