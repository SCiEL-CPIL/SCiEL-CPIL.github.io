"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _edge_key(u, v):
    return (u, v) if u < v else (v, u)

def _face_edges(x, i, j):
    """Return the 4 edges of the 2-face spanned by dimensions i and j at corner x.
    The four vertices are x, x^i, x^j, x^(i|j) where ^i means flip bit i."""
    xi = x ^ (1 << i)
    xj = x ^ (1 << j)
    xij = x ^ (1 << i) ^ (1 << j)
    return [
        _edge_key(x, xi),
        _edge_key(x, xj),
        _edge_key(xi, xij),
        _edge_key(xj, xij),
    ]

def _is_c4_free(chosen_set, edge):
    """Check if adding edge to chosen_set creates a C4 (2-face with all 4 edges)."""
    u, v = edge
    # Determine which dimension the edge is in
    diff = u ^ v
    i = diff.bit_length() - 1
    # For each other dimension j, check the two faces containing this edge
    # Face 1: corner at u, dims (i, j)
    # Face 2: corner at v, dims (i, j) -- but actually the same face
    # The face containing edge (u,v) in dims (i,j) has 4 edges.
    # We need to check if the other 3 edges are already in chosen_set.
    for j in range(7):
        if j == i:
            continue
        # The face is determined by the 4 vertices:
        # Let's find the 4th vertex. The edge is between u and v where v = u ^ (1<<i).
        # The face in dims (i,j) through u has vertices: u, u^(1<<j), u^(1<<i), u^(1<<i)^(1<<j)
        # i.e., u, u_j, v, v_j where u_j = u^(1<<j), v_j = v^(1<<j)
        u_j = u ^ (1 << j)
        v_j = v ^ (1 << j)
        # The 4 edges of this face are:
        # (u, u_j), (u, v), (u_j, v_j), (v, v_j)
        # We already have (u,v) = edge. Check the other 3.
        e1 = _edge_key(u, u_j)
        e2 = _edge_key(u_j, v_j)
        e3 = _edge_key(v, v_j)
        if e1 in chosen_set and e2 in chosen_set and e3 in chosen_set:
            return False
    return True

def _greedy_random(edges, chosen_set, rng):
    """Randomly permuted greedy: add edge if it doesn't create a C4."""
    shuffled = list(edges)
    rng.shuffle(shuffled)
    added = 0
    for edge in shuffled:
        if edge in chosen_set:
            continue
        if _is_c4_free(chosen_set, edge):
            chosen_set.add(edge)
            added += 1
    return added

def _local_search(chosen_set, edges, rng, time_limit):
    """Try to improve by removing edges that are part of many C4s... 
    Actually, the constraint is NO C4, so we can't have any C4.
    Local search: try removing a random edge and then greedily re-adding others?
    Or: try swapping - remove an edge, add two others?
    
    Better local search: 
    1. Find all edges that are "critical" - removing them allows adding more edges.
    2. Or: random restarts with different orderings.
    
    Let's do: try removing a random edge from chosen_set, then greedily add back 
    all possible edges (including the removed one) in random order."""
    best_set = set(chosen_set)
    best_size = len(best_set)
    deadline = time.time() + time_limit
    edges_list = list(edges)
    
    while time.time() < deadline:
        # Pick a random edge to remove
        if not best_set:
            break
        remove_edge = rng.choice(list(best_set))
        candidate = set(best_set)
        candidate.discard(remove_edge)
        
        # Now greedily add edges in random order
        shuffled = list(edges_list)
        rng.shuffle(shuffled)
        for edge in shuffled:
            if edge in candidate:
                continue
            if _is_c4_free(candidate, edge):
                candidate.add(edge)
        
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = candidate
    
    return best_set

def construct():
    d = 7
    n = 2 ** d  # 128
    deadline = time.time() + 140  # safety margin
    
    # Generate all edges
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    edge_set = set(edges)
    rng = random.Random(42)
    
    best_set = set()
    best_size = 0
    
    # Initial greedy with a good ordering
    # Try multiple random restarts
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        chosen = set()
        _greedy_random(edges, chosen, rng)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = set(chosen)
        
        # Local search with remaining time
        remaining = deadline - time.time()
        if remaining > 5:
            time_per_ls = min(3.0, remaining / 5.0)
            improved = _local_search(best_set, edges, rng, time_per_ls)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    # Convert to list of [u, v]
    result = [[u, v] for (u, v) in best_set]
    return result

# EVOLVE-BLOCK-END
