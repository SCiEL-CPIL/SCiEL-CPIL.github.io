"""Improved construction: balanced bipartite + controlled internal edges to minimize max book size."""
import itertools
import random
import time

def _compute_max_book(edges):
    """Compute the largest book size for a given set of edges."""
    adj = {i: set() for i in range(40)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    max_book = 0
    for u, v in edges:
        common = adj[u] & adj[v]
        book_size = len(common)
        if book_size > max_book:
            max_book = book_size
    return max_book

def _build_graph(seed=0, max_book_target=11):
    """
    Build a graph with 40 vertices, 401 edges, trying to keep max book size small.
    
    Strategy:
    - Split vertices into two sets A (0-19) and B (20-39), each of size 20.
    - Start with complete bipartite K_{20,20} (400 edges).
    - Add 1 internal edge within A and 0 within B (or vice versa) to get 401 total.
    - But this gives book size 19 for cross-edges.
    
    Better strategy:
    - Don't use complete bipartite. Use a sparse bipartite graph plus more internal edges.
    
    Actually, let's think differently:
    - The seed has K_{20,20} + 1 edge = 401 edges, max book = 19.
    - We need to reduce the max book.
    
    New approach:
    - Use a regular bipartite graph between A and B with degree d.
    - Add internal edges to reach 401 total.
    
    If we use a d-regular bipartite graph: 20*d edges.
    Internal edges needed: 401 - 20*d.
    
    For d=10: 200 bipartite edges, 201 internal edges.
    Internal edges on 40 vertices: max possible = 20*19/2 * 2 = 380.
    201 internal edges is feasible.
    
    Book size for a bipartite edge (u,v): number of common neighbors = 0 (bipartite, no triangles in bipartite part... wait, triangles need one internal edge).
    
    Actually, a triangle involves either:
    - 3 vertices in A with 3 internal edges
    - 3 vertices in B with 3 internal edges  
    - 2 in A, 1 in B: needs 1 internal edge in A and 2 bipartite edges
    - 1 in A, 2 in B: needs 1 internal edge in B and 2 bipartite edges
    
    For a bipartite edge (u,v) with u in A, v in B:
    A triangle containing (u,v) needs a vertex w such that (u,w) and (w,v) are edges.
    w must be in A (connected to u internally) and connected to v (bipartite), OR w in B (connected to v internally) and connected to u (bipartite).
    
    So book size of (u,v) = |{w in A : (u,w) internal AND (w,v) bipartite}| + |{w in B : (v,w) internal AND (u,w) bipartite}|
    
    This is at most deg_A(u) + deg_B(v) where these are restricted.
    
    Let's try to construct:
    - A = {0..19}, B = {20..39}
    - Bipartite: 10-regular (200 edges)
    - Internal: 201 edges total, split between A and B
    
    For 10-regular bipartite, each vertex has degree 10 in the bipartite part.
    Internal degree: each vertex has some internal degree.
    
    Let's aim for max book size <= 7 (the theoretical lower bound).
    
    This is complex. Let's use a randomized search with local improvement.
    """
    random.seed(seed)
    n = 40
    
    # Start with seed graph
    best_edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    best_score = _compute_max_book(best_edges)
    
    deadline = time.time() + 100  # Leave margin
    
    # Try different random constructions
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Random bipartite + internal
        # Choose number of bipartite edges
        num_bipartite = random.randint(100, 300)
        
        # Generate random bipartite edges
        bipartite_edges = set()
        possible_bip = [(a, b) for a in range(20) for b in range(20, 40)]
        random.shuffle(possible_bip)
        for i in range(num_bipartite):
            bipartite_edges.add(possible_bip[i])
        
        num_internal_needed = 401 - num_bipartite
        
        # Generate internal edges
        internal_edges = set()
        possible_internal = []
        for i in range(40):
            for j in range(i+1, 40):
                if not (i < 20 and j >= 20) and not (j < 20 and i >= 20):
                    possible_internal.append((i, j))
        
        if num_internal_needed <= len(possible_internal):
            random.shuffle(possible_internal)
            for i in range(num_internal_needed):
                internal_edges.add(possible_internal[i])
        
        edges = list(bipartite_edges) + list(internal_edges)
        score = _compute_max_book(edges)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            if best_score <= 7:
                break
        
        # Local improvement: try swapping edges
        if attempt % 10 == 0:
            # Try to improve by swapping an edge
            edges_set = set(tuple(sorted(e)) for e in best_edges)
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                # Pick a random edge to remove
                to_remove = random.choice(list(edges_set))
                # Pick a random edge to add
                all_possible = set()
                for i in range(40):
                    for j in range(i+1, 40):
                        all_possible.add((i, j))
                candidates = all_possible - edges_set
                if not candidates:
                    break
                to_add = random.choice(list(candidates))
                
                new_edges_set = (edges_set - {to_remove}) | {to_add}
                new_edges = [list(e) for e in new_edges_set]
                new_score = _compute_max_book(new_edges)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edges = new_edges
                    edges_set = new_edges_set
                    improved = True
                    break
            
            if best_score <= 7:
                break
    
    return best_edges

def construct():
    """
    Construct a graph with 40 vertices and 401 edges minimizing the largest book size.
    """
    # Try multiple seeds
    best_result = None
    best_score = float('inf')
    
    for seed in range(20):
        result = _build_graph(seed=seed)
        score = _compute_max_book(result)
        if score < best_score:
            best_score = score
            best_result = result
        if best_score <= 7:
            break
    
    # Return the best found
    return best_result
