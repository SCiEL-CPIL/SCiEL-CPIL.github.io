"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _book_size(edge_set, adj):
    """Return the maximum number of triangles containing any edge."""
    max_book = 0
    for u, v in edge_set:
        # Count common neighbors
        count = 0
        # Use sets for intersection
        nu = adj[u]
        nv = adj[v]
        if len(nu) < len(nv):
            for x in nu:
                if x in nv:
                    count += 1
        else:
            for x in nv:
                if x in nu:
                    count += 1
        if count > max_book:
            max_book = count
    return max_book

def _build_adj(edges):
    adj = [set() for _ in range(40)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _swap_edge(edges, u, v, a, b):
    """Replace edge (u,v) with edge (a,b). Return new edge list and success flag."""
    # Check that (a,b) is not already an edge
    edge_set = set()
    for e in edges:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
    if (min(a, b), max(a, b)) in edge_set:
        return None, False
    # Remove (u,v), add (a,b)
    new_edges = []
    for e in edges:
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            pass
        else:
            new_edges.append(list(e))
    new_edges.append([a, b])
    return new_edges, True

def _remove_edge(edges, u, v):
    """Remove edge (u,v) from edges, return new list."""
    new_edges = []
    for e in edges:
        if not ((e[0] == u and e[1] == v) or (e[0] == v and e[1] == u)):
            new_edges.append(list(e))
    return new_edges

def _add_edge(edges, u, v):
    """Add edge (u,v) to edges if not present."""
    for e in edges:
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            return None
    new_edges = [list(e) for e in edges]
    new_edges.append([u, v])
    return new_edges

def construct():
    # Start with K20,20 + one intra-partition edge
    edges = [[a, b] for a in range(20) for b in range(20, 40)]
    edges.append([0, 1])
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    adj = _build_adj(edges)
    current_score = _book_size(edges, adj)
    best_edges = [list(e) for e in edges]
    best_score = current_score
    
    # Simulated annealing / local search
    # We want to minimize the max book size
    # Moves: swap an edge for another non-edge (keep 401 edges)
    
    temp = 10.0
    cooling = 0.999
    
    rng = random.Random(42)
    
    # Precompute adjacency sets for fast lookup
    def edge_exists(u, v, adj_set):
        return v in adj_set[u]
    
    while time.time() < deadline:
        # Pick a random edge to potentially remove
        idx = rng.randint(0, len(edges) - 1)
        u, v = edges[idx]
        
        # Pick a random non-edge to add
        # Randomly pick two vertices and check they're not connected
        a = rng.randint(0, 39)
        b = rng.randint(0, 39)
        if a == b:
            continue
        if v in adj[u]:
            continue  # Already an edge
        
        # Also ensure (a,b) is not the same as (u,v)
        if (a == u and b == v) or (a == v and b == u):
            continue
        
        # Perform the swap: remove (u,v), add (a,b)
        # Update adjacency
        adj[u].discard(v)
        adj[v].discard(u)
        adj[a].add(b)
        adj[b].add(a)
        
        new_score = _book_size(edges, adj)
        
        # We need to actually track the edge list
        # Remove old edge, add new edge
        new_edges = _remove_edge(edges, u, v)
        if new_edges is None or len(new_edges) != 400:
            # Revert
            adj[u].add(v)
            adj[v].add(u)
            adj[a].discard(b)
            adj[b].discard(a)
            continue
        new_edges.append([a, b])
        
        if new_score <= current_score:
            # Accept
            current_score = new_score
            edges = new_edges
            if new_score < best_score:
                best_score = new_score
                best_edges = [list(e) for e in edges]
        else:
            # Accept with probability based on temperature
            delta = new_score - current_score
            if delta == 0:
                pass
            else:
                prob = math.exp(-delta / max(temp, 0.01))
                if rng.random() < prob:
                    current_score = new_score
                    edges = new_edges
                else:
                    # Revert
                    adj[u].add(v)
                    adj[v].add(u)
                    adj[a].discard(b)
                    adj[b].discard(a)
                    continue
        
        temp *= cooling
        if temp < 0.1:
            temp = 0.1
    
    # Return best found
    if best_score <= 6:
        return best_edges
    else:
        return best_edges
# EVOLVE-BLOCK-END
