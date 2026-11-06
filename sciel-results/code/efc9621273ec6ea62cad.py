"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
import time

def _has_k_clique(adj, n, k):
    """Check if graph with adjacency list has a clique of size k using Bron-Kerbosch."""
    if k <= 1:
        return True
    
    # Use a simple backtracking with pruning
    def find_clique(current, candidates):
        if len(current) == k:
            return True
        if not candidates:
            return False
        # Pruning: if len(current) + len(candidates) < k, can't reach k
        if len(current) + len(candidates) < k:
            return False
        
        # Try each candidate
        for i in range(len(candidates)):
            v = candidates[i]
            # New candidates: those in candidates[i+1:] that are adjacent to v
            new_candidates = [candidates[j] for j in range(i+1, len(candidates)) if v in adj[candidates[j]]]
            if find_clique(current | {v}, new_candidates):
                return True
        return False
    
    return find_clique(set(), list(range(n)))

def _has_independent_k(adj, n, k):
    """Check if graph has an independent set of size k (clique in complement)."""
    # Build complement adjacency
    comp_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and j not in adj[i]:
                comp_adj[i].add(j)
    
    return _has_k_clique(comp_adj, n, k)

def _is_valid(adj, n):
    """Check if graph is (5,5)-free: no K5 and no independent set of size 5."""
    if _has_k_clique(adj, n, 5):
        return False
    if _has_independent_k(adj, n, 5):
        return False
    return True

def _compute_alpha(adj, vertices):
    """Compute independence number of induced subgraph on 'vertices' using max clique in complement."""
    if not vertices:
        return 0
    v_list = list(vertices)
    n = len(v_list)
    
    # Build local complement adjacency
    local_comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and v_list[j] not in adj[v_list[i]]:
                local_comp[i].add(j)
    
    # Find max clique in local complement
    best = 0
    def find_max_clique(current, candidates):
        nonlocal best
        if len(current) > best:
            best = len(current)
        if not candidates:
            return
        if len(current) + len(candidates) <= best:
            return
        for i in range(len(candidates)):
            v = candidates[i]
            new_candidates = [candidates[j] for j in range(i+1, len(candidates)) if v in local_comp[candidates[j]]]
            find_max_clique(current | {v}, new_candidates)
    
    find_max_clique(set(), list(range(n)))
    return best

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
    
    random.seed(42)
    start_time = time.time()
    timeout = 250  # Leave buffer for evaluator
    
    # Strategy: Try to add vertices by finding valid neighborhoods
    # A new vertex u can be added with neighborhood S if:
    # 1. S has no K4 (else u + K4 = K5)
    # 2. V\S has no independent set of size 4 (else u + I4 = I5)
    # 3. The overall graph remains (5,5)-free
    
    # We'll use a greedy approach with local search
    # Try adding vertices one at a time, checking validity
    
    for attempt in range(1000):
        if time.time() - start_time > timeout:
            break
        
        # Try to find a valid neighborhood for a new vertex
        found = False
        
        # Strategy 1: Random sampling of neighborhoods
        for _ in range(50):
            if time.time() - start_time > timeout:
                break
            
            # Sample a random subset
            subset_size = random.randint(n // 3, 2 * n // 3)
            neighborhood = set(random.sample(range(n), subset_size))
            
            # Quick check: does neighborhood have K4?
            has_k4 = _has_k_clique(adj, n, 4)  # This is too expensive, need local check
            
            # Better: check K4 in induced subgraph on neighborhood
            nhood_list = list(neighborhood)
            nhood_set = set(nhood_list)
            
            # Check for K4 in neighborhood
            k4_found = False
            for combo in itertools.combinations(nhood_list, 4):
                a, b, c, d = combo
                if (b in adj[a] and c in adj[a] and d in adj[a] and
                    c in adj[b] and d in adj[b] and d in adj[c]):
                    k4_found = True
                    break
            
            if k4_found:
                continue
            
            # Check for independent set of size 4 in V\neighborhood
            non_nhood = set(range(n)) - neighborhood
            non_nhood_list = list(non_nhood)
            
            ind4_found = False
            for combo in itertools.combinations(non_nhood_list, 4):
                a, b, c, d = combo
                if (b not in adj[a] and c not in adj[a] and d not in adj[a] and
                    c not in adj[b] and d not in adj[b] and d not in adj[c]):
                    ind4_found = True
                    break
            
            if ind4_found:
                continue
            
            # This neighborhood looks promising; add the vertex and verify
            new_adj = [set(s) for s in adj]
            new_adj.append(set())
            for i in range(n):
                if i in neighborhood:
                    new_adj[i].add(n)
                    new_adj[n].add(i)
            
            new_n = n + 1
            
            # Full verification
            if _is_valid(new_adj, new_n):
                adj = new_adj
                n = new_n
                best_n = n
                best_adj = [set(s) for s in adj]
                found = True
                break
        
        if not found:
            # Try a different strategy: duplicate a vertex with modifications
            # Pick a random vertex v, create a twin with slightly modified neighborhood
            v = random.randint(0, n - 1)
            v_nhood = adj[v]
            
            # Try adding a twin of v
            # The twin u will have the same neighborhood as v
            # Check if this is valid
            new_adj = [set(s) for s in adj]
            new_adj.append(set())
            for i in range(n):
                if i in v_nhood:
                    new_adj[i].add(n)
                    new_adj[n].add(i)
            
            new_n = n + 1
            
            if _is_valid(new_adj, new_n):
                adj = new_adj
                n = new_n
                best_n = n
                best_adj = [set(s) for s in adj]
        
        # If we've been stuck, try random edge flips to escape local optima
        if best_n < n + 1 and random.random() < 0.1:
            # Try flipping a random edge
            if n > 1:
                i, j = random.sample(range(n), 2)
                if i in adj[j]:
                    adj[i].remove(j)
                    adj[j].remove(i)
                else:
                    adj[i].add(j)
                    adj[j].add(i)
                
                if _is_valid(adj, n):
                    best_n = max(best_n, n)
                    best_adj = [set(s) for s in adj]
                else:
                    # Revert
                    if i in adj[j]:
                        adj[i].remove(j)
                        adj[j].remove(i)
                    else:
                        adj[i].add(j)
                        adj[j].add(i)
    
    # Reconstruct edge list from best_adj
    final_edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                final_edges.append((i, j))
    
    return best_n, final_edges
# EVOLVE-BLOCK-END
