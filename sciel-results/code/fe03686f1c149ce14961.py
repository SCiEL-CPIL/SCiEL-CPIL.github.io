"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools

def _has_k5(adj, n):
    """Check if graph with adjacency list has a clique of size 5."""
    # Use Bron-Kerbosch or simple enumeration for small graphs
    # For efficiency, we can use a backtracking approach
    def find_clique(current, start):
        if len(current) == 5:
            return True
        for i in range(start, n):
            if i in current:
                continue
            # Check if i is connected to all in current
            if all(i in adj[v] for v in current):
                if find_clique(current | {i}, i + 1):
                    return True
        return False
    
    return find_clique(set(), 0)

def _has_independent_5(adj, n):
    """Check if graph has an independent set of size 5 (clique in complement)."""
    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
    
    def find_clique(current, start):
        if len(current) == 5:
            return True
        for i in range(start, n):
            if i in current:
                continue
            if all(i in comp_adj[v] for v in current):
                if find_clique(current | {i}, i + 1):
                    return True
        return False
    
    return find_clique(set(), 0)

def _is_valid(adj, n):
    """Check if graph is (5,5)-free: no K5 and no independent set of size 5."""
    if _has_k5(adj, n):
        return False
    if _has_independent_5(adj, n):
        return False
    return True

def _add_vertex(adj, n, new_vertex_adj):
    """Add a new vertex with given adjacency to existing vertices."""
    new_adj = [set(s) for s in adj]
    new_adj.append(set())
    for i in range(n):
        if i in new_vertex_adj:
            new_adj[i].add(n)
            new_adj[n].add(i)
    return new_adj

def construct():
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Build adjacency list
    adj = [set() for _ in range(q)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    n = q
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to add vertices using local search
    random.seed(42)
    
    for attempt in range(500):
        # Try to add a new vertex
        # The new vertex's neighborhood must be a (4,4)-free set in the current graph
        # i.e., no K4 in the neighborhood and no independent set of size 4 in the neighborhood
        
        # Sample a random subset of vertices to be the neighborhood
        # We need the neighborhood to have no K4 and no independent set of size 4
        
        # Strategy: pick a random vertex v, and consider its neighborhood or non-neighborhood
        # or a random subset
        
        found = False
        for _ in range(200):
            # Pick a random subset of size around n/2
            subset_size = random.randint(10, min(20, n))
            neighborhood = set(random.sample(range(n), subset_size))
            
            # Check if this neighborhood is (4,4)-free
            # No K4 in neighborhood
            has_k4 = False
            for combo in itertools.combinations(neighborhood, 4):
                a, b, c, d = combo
                if (b in adj[a] and c in adj[a] and d in adj[a] and
                    c in adj[b] and d in adj[b] and d in adj[c]):
                    has_k4 = True
                    break
            if has_k4:
                continue
            
            # No independent set of size 4 in neighborhood
            has_ind_4 = False
            for combo in itertools.combinations(neighborhood, 4):
                a, b, c, d = combo
                if (b not in adj[a] and c not in adj[a] and d not in adj[a] and
                    c not in adj[b] and d not in adj[b] and d not in adj[c]):
                    has_ind_4 = True
                    break
            if has_ind_4:
                continue
            
            # This neighborhood is valid; try adding the vertex
            new_adj = _add_vertex(adj, n, neighborhood)
            new_n = n + 1
            
            if _is_valid(new_adj, new_n):
                adj = new_adj
                n = new_n
                best_n = n
                best_adj = [set(s) for s in adj]
                found = True
                break
        
        if not found:
            # Try with a different strategy: duplicate a vertex's neighborhood with modifications
            pass
    
    # Reconstruct edge list from best_adj
    final_edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                final_edges.append((i, j))
    
    return best_n, final_edges
# EVOLVE-BLOCK-END
