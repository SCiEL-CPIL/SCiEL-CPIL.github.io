import time
import random
import numpy as np

def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book size.
    Uses a deterministic near-bipartite construction to get a strong starting point,
    followed by vectorized batch local search for refinement.
    """
    
    deadline = time.time() + 140.0  # Safety margin for 150s limit
    
    n = 40
    m = 401
    
    # Helper: compute max book from adjacency matrix
    def _compute_max_book(adj):
        """Compute max book size using matrix multiplication."""
        A = adj.astype(np.int32)
        A2 = A @ A
        max_book = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    common = A2[i, j]
                    if common > max_book:
                        max_book = common
        return max_book

    def _compute_all_books(adj):
        """Compute book size for every edge. Returns dict {(u,v): book}."""
        A = adj.astype(np.int32)
        A2 = A @ A
        books = {}
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    books[(i, j)] = int(A2[i, j])
        return books

    # --- Phase 1: Deterministic near-bipartite construction ---
    # Start with K_{20,20}, remove k bipartite edges, add k+1 intra-edges
    # Total edges = 400 - k + (k+1) = 401
    
    A_verts = list(range(20))
    B_verts = list(range(20, 40))
    
    best_adj = None
    best_max = 999
    
    # Try different values of k
    for k in range(1, 16):
        # Start with full bipartite
        adj = np.zeros((n, n), dtype=bool)
        for a in A_verts:
            for b in B_verts:
                adj[a, b] = True
                adj[b, a] = True
        
        # Choose k+1 intra-A edges spread out to minimize shared common neighbors
        # Use a pattern that spreads endpoints
        intra_edges = []
        used_vertices = set()
        
        # Greedily pick intra-A edges to minimize max book
        # Book of intra-A edge (u,v) = |{b in B : adj[u,b] and adj[v,b]}|
        # Initially all 20, but we'll remove bipartite edges to reduce
        
        # Pick intra edges with vertices spread out
        candidates = []
        for i in range(20):
            for j in range(i+1, 20):
                candidates.append((i, j))
        
        # Sort by distance to prefer spread-out pairs
        candidates.sort(key=lambda p: -(abs(p[0] - p[1])))
        
        for u, v in candidates:
            if len(intra_edges) >= k + 1:
                break
            if u in used_vertices or v in used_vertices:
                continue
            intra_edges.append((u, v))
            used_vertices.add(u)
            used_vertices.add(v)
        
        # If we didn't get enough, add more allowing shared vertices
        if len(intra_edges) < k + 1:
            for u, v in candidates:
                if len(intra_edges) >= k + 1:
                    break
                if (u, v) in intra_edges or (v, u) in intra_edges:
                    continue
                intra_edges.append((u, v))
        
        # Add intra edges to adj
        for u, v in intra_edges:
            adj[u, v] = True
            adj[v, u] = True
        
        # Now greedily remove k bipartite edges to reduce max book
        # Compute current books
        current_books = _compute_all_books(adj)
        current_max = max(current_books.values())
        
        # Greedily remove bipartite edges that most reduce the max book
        removed = set()
        for _ in range(k):
            if time.time() > deadline:
                break
                
            # Find the edge with max book
            max_e = max(current_books, key=current_books.get)
            max_val = current_books[max_e]
            
            if max_val <= 3:
                break
            
            # Try to find a bipartite edge whose removal reduces max book most
            best_removal = None
            best_improvement = 0
            
            # Sample bipartite edges
            bipartite_candidates = []
            for a in A_verts:
                for b in B_verts:
                    if adj[a, b] and (a, b) not in removed and (b, a) not in removed:
                        bipartite_candidates.append((a, b))
            
            random.shuffle(bipartite_candidates)
            
            for a, b in bipartite_candidates[:50]:
                # Simulate removal
                adj[a, b] = False
                adj[b, a] = False
                
                # Recompute books (approximate: only affected edges)
                # Edges affected are those incident to a or b
                new_books = dict(current_books)
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u, v]:
                            if (u == a or v == a or u == b or v == b):
                                # Recompute this edge's book
                                common = 0
                                for w in range(n):
                                    if w != u and w != v:
                                        if adj[u, w] and adj[v, w]:
                                            common += 1
                                if (u, v) in new_books:
                                    new_books[(u, v)] = common
                                elif common > 0:
                                    new_books[(u, v)] = common
                        else:
                            if (u, v) in new_books:
                                del new_books[(u, v)]
                
                new_max = max(new_books.values()) if new_books else 0
                
                # Revert
                adj[a, b] = True
                adj[b, a] = True
                
                improvement = max_val - new_max
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_removal = (a, b)
            
            if best_removal:
                a, b = best_removal
                adj[a, b] = False
                adj[b, a] = False
                removed.add((a, b))
                
                # Update books
                current_books = _compute_all_books(adj)
        
        # Check final result
        final_max = _compute_max_book(adj)
        if final_max < best_max:
            best_max = final_max
            best_adj = adj.copy()
    
    # --- Phase 2: Vectorized batch local search ---
    if best_adj is None:
        # Fallback: simple random graph
        best_adj = np.zeros((n, n), dtype=bool)
        edges_list = []
        for i in range(n):
            for j in range(i+1, n):
                edges_list.append((i, j))
        random.shuffle(edges_list)
        for i, j in edges_list[:m]:
            best_adj[i, j] = True
            best_adj[j, i] = True
    
    adj = best_adj.copy()
    books = _compute_all_books(adj)
    best_book = max(books.values())
    best_adj_copy = adj.copy()
    
    # Local search with batch evaluation
    while time.time() < deadline:
        max_b = max(books.values()) if books else 0
        if max_b <= 3:
            break
        
        # Find worst edges
        worst_edges = [(u, v) for (u, v), b in books.items() if b == max_b]
        random.shuffle(worst_edges)
        
        improved = False
        
        for worst_e in worst_edges[:10]:
            if time.time() > deadline:
                break
                
            u, v = worst_e
            
            # Generate batch of candidate swaps
            candidates = []
            for _ in range(200):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y:
                    continue
                if (x, y) in books or (y, x) in books:
                    continue
                if (x == u and y == v) or (x == v and y == u):
                    continue
                candidates.append((x, y))
            
            if not candidates:
                continue
            
            # For each candidate, compute the new max book if we swap
            best_candidate = None
            best_new_max = max_b
            
            for x, y in candidates:
                # Simulate swap: remove (u,v), add (x,y)
                adj[u, v] = False
                adj[v, u] = False
                adj[x, y] = True
                adj[y, x] = True
                
                # Recompute affected books
                new_books = dict(books)
                
                # Remove old edge
                if (u, v) in new_books:
                    del new_books[(u, v)]
                if (v, u) in new_books:
                    del new_books[(v, u)]
                
                # Add new edge
                common_xy = 0
                for w in range(n):
                    if w != x and w != y:
                        if adj[x, w] and adj[y, w]:
                            common_xy += 1
                if common_xy > 0:
                    new_books[(x, y)] = common_xy
                elif (x, y) in new_books:
                    del new_books[(x, y)]
                
                # Update books for edges incident to u, v, x, y
                for a in [u, v, x, y]:
                    for w in range(n):
                        if w == a:
                            continue
                        if adj[a, w]:
                            key = (a, w) if a < w else (w, a)
                            # Recompute book for this edge
                            common = 0
                            for z in range(n):
                                if z != a and z != w:
                                    if adj[a, z] and adj[w, z]:
                                        common += 1
                            if common > 0:
                                new_books[key] = common
                            else:
                                if key in new_books:
                                    del new_books[key]
                        else:
                            key = (a, w) if a < w else (w, a)
                            if key in new_books:
                                del new_books[key]
                
                new_max = max(new_books.values()) if new_books else 0
                
                # Revert
                adj[u, v] = True
                adj[v, u] = True
                adj[x, y] = False
                adj[y, x] = False
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_candidate = (x, y)
            
            if best_candidate and best_new_max < best_book:
                # Accept the best swap
                x, y = best_candidate
                adj[u, v] = False
                adj[v, u] = False
                adj[x, y] = True
                adj[y, x] = True
                
                # Update books
                books = _compute_all_books(adj)
                best_book = max(books.values())
                best_adj_copy = adj.copy()
                improved = True
                break
        
        if not improved:
            # Try some random restarts or perturbations
            if random.random() < 0.1:
                # Perturb: swap a random edge with another random non-edge
                all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
                random.shuffle(all_edges)
                for i, j in all_edges:
                    if adj[i, j]:
                        # Find a non-edge to swap with
                        for x, y in all_edges:
                            if not adj[x, y]:
                                adj[i, j] = False
                                adj[j, i] = False
                                adj[x, y] = True
                                adj[y, x] = True
                                books = _compute_all_books(adj)
                                break
                        break
    
    # Return best result
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj_copy[i, j]:
                result.append([i, j])
    
    assert len(result) == m, f"Expected {m} edges, got {len(result)}"
    return result

if __name__ == "__main__":
    edges = construct()
    adj = np.zeros((40, 40), dtype=bool)
    for e in edges:
        adj[e[0], e[1]] = True
        adj[e[1], e[0]] = True
    max_b = _compute_max_book(adj)
    print(f"Max book size: {max_b}")
