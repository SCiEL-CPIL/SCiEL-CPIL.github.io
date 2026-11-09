"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _compute_book_sizes(edges, n=40):
    """Compute the book size for each edge in the graph."""
    adj = set()
    for u, v in edges:
        adj.add((u, v))
        adj.add((v, u))
    
    # For each edge, count triangles containing it
    # A triangle containing edge (u,v) exists for each w such that (u,w) and (v,w) are both edges
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    adj_list = defaultdict(set)
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)
    
    max_book = 0
    for u, v in edges:
        # Count common neighbors of u and v
        common = adj_list[u] & adj_list[v]
        book_size = len(common)
        if book_size > max_book:
            max_book = book_size
    
    return max_book

def _try_improve(initial_edges, n=40, target_edges=401, time_limit=140.0):
    """Try to improve the initial graph by local search."""
    start_time = time.time()
    best_edges = list(initial_edges)
    best_score = _compute_book_sizes(best_edges, n)
    
    # Build adjacency structure for fast updates
    edge_set = set()
    for u, v in best_edges:
        edge_set.add((min(u,v), max(u,v)))
    
    adj = defaultdict(set)
    for u, v in best_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Count book sizes for all edges
    book_sizes = {}
    for u, v in best_edges:
        key = (min(u,v), max(u,v))
        common = adj[u] & adj[v]
        book_sizes[key] = len(common)
    
    max_book = max(book_sizes.values()) if book_sizes else 0
    
    # Local search: try swapping edges
    # We want to maintain exactly 401 edges
    # Try removing an edge with high book size and adding a non-edge that doesn't create too many triangles
    
    random.seed(42)
    
    # Precompute all possible edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        if iterations > 1000000:
            break
        
        # Pick a random edge to remove
        remove_idx = random.randint(0, len(best_edges) - 1)
        ru, rv = best_edges[remove_idx]
        rkey = (min(ru, rv), max(ru, rv))
        
        if book_sizes.get(rkey, 0) <= 2:
            # Removing this edge won't reduce max book significantly
            # Try to find a good edge to add
            candidates = []
            for _ in range(50):  # Sample up to 50 candidate edges to add
                cu, cv = random.choice(all_possible_edges)
                ckey = (min(cu, cv), max(cu, cv))
                if ckey not in edge_set:
                    # Estimate book size if we add this edge
                    common = adj[cu] & adj[cv]
                    new_book = len(common)
                    if new_book < max_book:
                        candidates.append((new_book, cu, cv, ckey))
            
            if candidates:
                # Pick the candidate with smallest book size
                candidates.sort(key=lambda x: x[0])
                nb, cu, cv, ckey = candidates[0]
                
                # Apply the swap
                # Remove edge (ru, rv)
                best_edges[remove_idx] = [cu, cv]
                edge_set.remove(rkey)
                edge_set.add(ckey)
                adj[ru].remove(rv)
                adj[rv].remove(ru)
                adj[cu].add(cv)
                adj[cv].add(cu)
                
                # Update book sizes
                del book_sizes[rkey]
                book_sizes[ckey] = nb
                
                # Recompute max book
                new_max = max(book_sizes.values()) if book_sizes else 0
                
                if new_max < max_book:
                    max_book = new_max
                    best_edges = list(best_edges)
                    best_score = max_book
                else:
                    # Revert
                    best_edges[remove_idx] = [ru, rv]
                    edge_set.remove(ckey)
                    edge_set.add(rkey)
                    adj[cu].remove(cv)
                    adj[cv].remove(cu)
                    adj[ru].add(rv)
                    adj[rv].add(ru)
                    del book_sizes[ckey]
                    book_sizes[rkey] = book_sizes.get(rkey, 0)
    
    return best_edges, best_score

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    
    # Start with K20,20 + one intra-part edge
    edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    # Verify we have exactly 401 edges
    assert len(edges) == 401, f"Expected 401 edges, got {len(edges)}"
    
    # Compute initial score
    initial_score = _compute_book_sizes(edges, n)
    
    # Try local search improvement
    if time.time() - start_time < 130.0:
        improved_edges, improved_score = _try_improve(edges, n, target_edges, time_limit=130.0)
        if improved_score < initial_score:
            return improved_edges
        else:
            return edges
    else:
        return edges
# EVOLVE-BLOCK-END
