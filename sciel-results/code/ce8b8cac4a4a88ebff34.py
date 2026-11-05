import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Phase 1: Multiple greedy constructions with different strategies
    for seed in range(3000):
        if time.time() > deadline - 40.0:
            break
        
        if seed < 1000:
            # Random greedy
            edges = _greedy_random(n, all_edges, seed)
        elif seed < 2000:
            # Degree-biased greedy (prefer low-degree endpoints)
            edges = _greedy_degree_biased(n, all_edges, seed)
        else:
            # Random greedy with different seed offset
            edges = _greedy_random(n, all_edges, seed + 10000)
        
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges
    
    # Phase 2: Local search with 2-edge swaps on the best solution
    if time.time() < deadline - 5.0 and best_edges:
        improved = _local_search_swap(n, best_edges, all_edges, deadline - 3.0)
        if len(improved) > best_score:
            best_edges = improved
            best_score = len(improved)
    
    # Phase 3: If time remains, try local search on a few other good solutions
    if time.time() < deadline - 5.0:
        # Run a few more greedy to get diverse starting points
        for seed in range(500, 800):
            if time.time() > deadline - 5.0:
                break
            edges = _greedy_degree_biased(n, all_edges, seed)
            if len(edges) >= best_score - 2:  # Close to best
                improved = _local_search_swap(n, edges, all_edges, deadline - 3.0)
                if len(improved) > best_score:
                    best_edges = improved
                    best_score = len(improved)
    
    return best_edges

def _greedy_random(n, all_edges, seed):
    """Standard random-order greedy."""
    random.seed(seed)
    adj = [0] * n
    edges = []
    
    shuffled = all_edges[:]
    random.shuffle(shuffled)
    
    for u, v in shuffled:
        if _can_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
    
    return edges

def _greedy_degree_biased(n, all_edges, seed):
    """Greedy that prefers edges between lower-degree vertices."""
    random.seed(seed)
    adj = [0] * n
    edges = []
    deg = [0] * n
    
    # Build priority list: sort edges by sum of degrees (dynamic)
    # We'll use a simple approach: at each step, pick the best valid edge
    # But that's O(m*n) per step. Instead, we'll do multiple passes.
    
    # Simpler: shuffle, but weight by current degree sum
    shuffled = all_edges[:]
    random.shuffle(shuffled)
    
    # Do a few passes, each time trying to add edges, prioritizing low-degree pairs
    for _pass in range(3):
        # Sort remaining edges by degree sum (ascending) + random tiebreak
        remaining = [e for e in shuffled if not ((1 << e[1]) & adj[e[0]])]
        remaining.sort(key=lambda e: (deg[e[0]] + deg[e[1]], random.random()))
        
        for u, v in remaining:
            if (1 << v) & adj[u]:
                continue  # Already an edge
            if _can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                edges.append([u, v])
    
    return edges

def _can_add(adj, u, v):
    """Check if adding edge (u,v) keeps graph K2,3-free. Uses bitmasks."""
    # u and v must not already be adjacent
    if (1 << v) & adj[u]:
        return False
    
    # Check pair (u, v): common neighbors
    if (adj[u] & adj[v]).bit_count() >= 3:
        return False
    
    # Check pairs (u, w) and (v, w) for all w != u, v
    for w in range(len(adj)):
        if w == u or w == v:
            continue
        
        # After adding (u,v): v becomes neighbor of u, u becomes neighbor of v
        # Common neighbors of u and w: old (adj[u] & adj[w]) + {v if v in adj[w]}
        cuw = (adj[u] & adj[w]).bit_count()
        if adj[w] & (1 << v):
            cuw += 1
        if cuw >= 3:
            return False
        
        # Common neighbors of v and w: old (adj[v] & adj[w]) + {u if u in adj[w]}
        cvw = (adj[v] & adj[w]).bit_count()
        if adj[w] & (1 << u):
            cvw += 1
        if cvw >= 3:
            return False
    
    return True

def _local_search_swap(n, edges, all_edges, deadline):
    """
    Local search using 2-edge swaps:
    - Remove one existing edge
    - Try to add two new edges
    - Net gain of +1 edge if successful
    Also try pure additions (0-1 moves) first.
    """
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((min(u, v), max(u, v)))
    
    current_edges = [list(e) for e in edges]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Phase A: Try to add any single missing edge
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if (min(u, v), max(u, v)) in edge_set:
                continue
            if _can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((min(u, v), max(u, v)))
                current_edges.append([u, v])
                improved = True
        
        if improved:
            continue  # Restart, more additions may now be possible
        
        # Phase B: Try 2-edge swaps (remove 1, add 2)
        edge_list = list(edge_set)
        random.shuffle(edge_list)
        
        for e in edge_list:
            if time.time() > deadline:
                break
            
            u1, v1 = e
            
            # Remove edge e
            adj[u1] ^= (1 << v1)
            adj[v1] ^= (1 << u1)
            edge_set.remove(e)
            
            # Try to find two edges to add
            added = []
            
            # Collect all candidate edges that can be added
            candidates = []
            for u2, v2 in all_edges:
                if (min(u2, v2), max(u2, v2)) in edge_set:
                    continue
                if _can_add(adj, u2, v2):
                    candidates.append((u2, v2))
            
            # Try to greedily pick two compatible edges
            if len(candidates) >= 2:
                # Try all pairs of candidates (limited to first few for speed)
                found = False
                max_check = min(len(candidates), 30)
                for i in range(max_check):
                    if found:
                        break
                    u2, v2 = candidates[i]
                    # Tentatively add first
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    edge_set.add((min(u2, v2), max(u2, v2)))
                    
                    for j in range(i + 1, max_check):
                        u3, v3 = candidates[j]
                        if (min(u3, v3), max(u3, v3)) in edge_set:
                            continue
                        if _can_add(adj, u3, v3):
                            adj[u3] |= (1 << v3)
                            adj[v3] |= (1 << u3)
                            edge_set.add((min(u3, v3), max(u3, v3)))
                            added = [(u2, v2), (u3, v3)]
                            found = True
                            break
                    
                    if not found:
                        # Revert first addition
                        adj[u2] ^= (1 << v2)
                        adj[v2] ^= (1 << u2)
                        edge_set.remove((min(u2, v2), max(u2, v2)))
                
                if found:
                    # Successful 2-edge swap
                    current_edges = [[u, v] for u, v in edge_set]
                    improved = True
                    break  # Restart loop as graph changed
                else:
                    # Revert removal
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    edge_set.add(e)
            else:
                # Revert removal
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                edge_set.add(e)
                
    return current_edges
