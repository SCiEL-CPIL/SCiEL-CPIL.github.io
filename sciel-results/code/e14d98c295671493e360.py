"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _has_clique5(adj, n):
    """Check if graph with adjacency matrix adj on n vertices has a K5."""
    # For each 5-tuple, check if all pairs are connected
    # Use a faster approach: for each 4-clique, check if any vertex connects to all 4
    # But for n up to ~46, brute force with early termination is fine
    for combo in combinations(range(n), 5):
        all_connected = True
        for a in range(5):
            for b in range(a+1, 5):
                if not adj[combo[a]][combo[b]]:
                    all_connected = False
                    break
            if not all_connected:
                break
        if all_connected:
            return True
    return False

def _has_indep5(adj, n):
    """Check if graph has an independent set of size 5."""
    # Independent set = clique in complement
    for combo in combinations(range(n), 5):
        all_independent = True
        for a in range(5):
            for b in range(a+1, 5):
                if adj[combo[a]][combo[b]]:
                    all_independent = False
                    break
            if not all_independent:
                break
        if all_independent:
            return True
    return False

def _is_55_free(adj, n):
    """Check if graph is (5,5)-free: no K5 and no independent 5-set."""
    if _has_clique5(adj, n):
        return False
    if _has_indep5(adj, n):
        return False
    return True

def _paley_graph(p):
    """Construct Paley graph on p vertices where p is prime, p ≡ 1 (mod 4)."""
    residues = {(x * x) % p for x in range(1, p)}
    residues.discard(0)
    adj = [[False] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in residues:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def _try_add_vertex(adj, n, deadline):
    """Try to add a new vertex to the (5,5)-free graph.
    For each possible connection pattern to existing n vertices, check if it remains (5,5)-free."""
    # This is expensive; try a few random patterns
    for attempt in range(200):
        if time.time() > deadline:
            break
        # Random subset of neighbors
        new_neighbors = [random.random() < 0.5 for _ in range(n)]
        # Check: no K5 involving new vertex
        # New vertex + 4 existing vertices forming a K4 where all 4 connect to new vertex
        valid = True
        for quad in combinations(range(n), 4):
            if all(new_neighbors[v] for v in quad):
                # Check if quad forms a K4
                is_k4 = True
                for a in range(4):
                    for b in range(a+1, 4):
                        if not adj[quad[a]][quad[b]]:
                            is_k4 = False
                            break
                    if not is_k4:
                        break
                if is_k4:
                    valid = False
                    break
        if not valid:
            continue
        # Check: no independent 5-set involving new vertex
        # New vertex + 4 existing vertices that are independent and none connect to new vertex
        for quad in combinations(range(n), 4):
            if all(not new_neighbors[v] for v in quad):
                # Check if quad is independent
                is_indep = True
                for a in range(4):
                    for b in range(a+1, 4):
                        if adj[quad[a]][quad[b]]:
                            is_indep = False
                            break
                    if not is_indep:
                        break
                if is_indep:
                    valid = False
                    break
        if not valid:
            continue
        # Valid! Build new adjacency
        new_n = n + 1
        new_adj = [[False] * new_n for _ in range(new_n)]
        for i in range(n):
            for j in range(i + 1, n):
                new_adj[i][j] = adj[i][j]
                new_adj[j][i] = adj[i][j]
            new_adj[i][new_n - 1] = new_neighbors[i]
            new_adj[new_n - 1][i] = new_neighbors[i]
        return new_adj, new_n
    return None

def _local_search_flip(adj, n, deadline, max_flips=5000):
    """Try to flip edges to allow adding a vertex, or just try random edge flips to maintain (5,5)-free."""
    # This is a heuristic; try random edge flips that maintain (5,5)-free property
    best_adj = [row[:] for row in adj]
    best_n = n
    flips = 0
    while time.time() < deadline and flips < max_flips:
        flips += 1
        # Pick random edge
        i, j = random.sample(range(n), 2)
        if i > j:
            i, j = j, i
        # Try flipping
        old_val = best_adj[i][j]
        best_adj[i][j] = not old_val
        best_adj[j][i] = not old_val
        if _is_55_free(best_adj, n):
            # Keep the flip
            pass
        else:
            # Revert
            best_adj[i][j] = old_val
            best_adj[j][i] = old_val
    return best_adj, best_n

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    # Start with Paley graph on 41 vertices
    p = 41
    adj = _paley_graph(p)
    n = p
    
    # Verify it's (5,5)-free
    if not _is_55_free(adj, n):
        # Fall back to 37 if 41 somehow fails
        p = 37
        adj = _paley_graph(p)
        n = p
    
    best_n = n
    best_adj = [row[:] for row in adj]
    
    # Try to add vertices
    while time.time() < deadline - 30:
        result = _try_add_vertex(adj, n, deadline - 30)
        if result is not None:
            adj, n = result
            if n > best_n:
                best_n = n
                best_adj = [row[:] for row in adj]
        else:
            # Can't easily add more; try local search to see if we can find a different structure
            # Or try a different approach: start from scratch with a different seed
            break
    
    # Convert best_adj to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i][j]:
                edges.append((i, j))
    
    return best_n, edges
# EVOLVE-BLOCK-END
