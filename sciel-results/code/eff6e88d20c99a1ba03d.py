import time
import random

def _compute_max_book(adj, n=40):
    """Compute the maximum book size (max triangles containing any edge)."""
    max_book = 0
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common_count = len(adj[u] & adj[v])
                if common_count > max_book:
                    max_book = common_count
    return max_book

def _adj_from_edges(edges, n=40):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _local_search(edges, time_limit):
    n = 40
    adj = _adj_from_edges(edges, n)
    current_max = _compute_max_book(adj, n)
    best_edges = set(edges)
    best_max = current_max
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Precompute all possible edges for faster non-edge selection?
    # Or just random sampling.
    
    while time.time() < deadline:
        # Pick a random edge to remove
        if not edges:
            break
        edge_to_remove = random.choice(list(edges))
        
        # Try a few random swaps
        found_improvement = False
        for _ in range(10):
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b:
                continue
            new_edge = (min(a, b), max(a, b))
            
            if new_edge == edge_to_remove:
                continue
            if new_edge in edges:
                continue
            
            # Tentative swap
            u, v = edge_to_remove
            x, y = new_edge
            
            edges.remove(edge_to_remove)
            edges.add(new_edge)
            
            adj[u].discard(v)
            adj[v].discard(u)
            adj[x].add(y)
            adj[y].add(x)
            
            new_max = _compute_max_book(adj, n)
            
            if new_max < current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_edges = set(edges)
                found_improvement = True
                break
            else:
                # Revert
                edges.remove(new_edge)
                edges.add(edge_to_remove)
                
                adj[x].discard(y)
                adj[y].discard(x)
                adj[u].add(v)
                adj[v].add(u)
        
        if not found_improvement:
            # If stuck, try a small random perturbation (restart-like)
            # Remove 5 random edges, add 5 random non-edges
            # This helps escape local minima
            if random.random() < 0.01:
                # Perturb
                for _ in range(5):
                    if edges:
                        e = random.choice(list(edges))
                        edges.remove(e)
                        u, v = e
                        adj[u].discard(v)
                        adj[v].discard(u)
                
                for _ in range(5):
                    while True:
                        a = random.randint(0, n-1)
                        b = random.randint(0, n-1)
                        if a == b: continue
                        ne = (min(a,b), max(a,b))
                        if ne not in edges:
                            break
                    edges.add(ne)
                    x, y = ne
                    adj[x].add(y)
                    adj[y].add(x)
                
                current_max = _compute_max_book(adj, n)
                # Note: We don't necessarily update best here, just continue search

    return best_edges, best_max

def _generate_candidate(K_intra, seed_val):
    """Generate a candidate graph with K_intra intra-edges in A, removing K_intra cross-edges."""
    n = 40
    edges = set()
    # Base K20,20
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
            
    random.seed(seed_val)
    
    # Select K_intra edges in A
    intra_edges = set()
    while len(intra_edges) < K_intra:
        u = random.randint(0, 19)
        v = random.randint(0, 19)
        if u != v:
            intra_edges.add((min(u,v), max(u,v)))
            
    edges.update(intra_edges)
    
    # Identify vertices involved in intra edges
    v_intra = set()
    for u, v in intra_edges:
        v_intra.add(u)
        v_intra.add(v)
        
    # Find cross edges incident to v_intra
    candidate_removals = []
    for a in range(20):
        for b in range(20, 40):
            if a in v_intra:
                candidate_removals.append((a, b))
                
    random.shuffle(candidate_removals)
    
    # Remove K_intra edges from candidate_removals
    removed_count = 0
    for e in candidate_removals:
        if removed_count >= K_intra:
            break
        if e in edges:
            edges.discard(e)
            removed_count += 1
            
    # If we didn't remove enough (unlikely), remove random cross edges
    if removed_count < K_intra:
        all_cross = [(a, b) for a in range(20) for b in range(20, 40)]
        random.shuffle(all_cross)
        for e in all_cross:
            if removed_count >= K_intra:
                break
            if e in edges:
                edges.discard(e)
                removed_count += 1
                
    # Ensure exactly 401 edges
    # Current count: 400 (base) + K_intra (added) - K_intra (removed) = 400
    # Wait, we need 401.
    # Base is 400.
    # We added K_intra. Count = 400 + K_intra.
    # We removed K_intra. Count = 400.
    # We need 401. So we must add 1 more edge.
    # Add a random intra-edge in B or A that is not already present.
    while len(edges) < 401:
        a = random.randint(0, 39)
        b = random.randint(0, 39)
        if a == b: continue
        ne = (min(a,b), max(a,b))
        if ne not in edges:
            edges.add(ne)
            
    # If somehow too many (shouldn't happen), remove random
    while len(edges) > 401:
        e = random.choice(list(edges))
        edges.discard(e)
        
    return edges

def construct():
    n = 40
    target_edges = 401
    total_time_limit = 115.0 # Safety margin
    start_time = time.time()
    
    random.seed(42)
    
    best_overall_edges = set()
    best_overall_score = float('inf')
    
    # Candidate 1: Baseline (1 intra edge)
    edges1 = _generate_candidate(1, 100)
    
    # Candidate 2: 10 intra edges
    edges2 = _generate_candidate(10, 200)
    
    # Candidate 3: 20 intra edges
    edges3 = _generate_candidate(20, 300)
    
    candidates = [edges1, edges2, edges3]
    
    # Run local search on each candidate
    # Allocate time based on potential
    # Simple split: 1/3 time each
    
    time_per_candidate = total_time_limit / len(candidates)
    
    for i, edges in enumerate(candidates):
        if time.time() - start_time > total_time_limit - 5:
            break
            
        # Ensure validity
        if len(edges) != target_edges:
            # Fix count
            while len(edges) < target_edges:
                a = random.randint(0, 39)
                b = random.randint(0, 39)
                if a < b and (a,b) not in edges:
                    edges.add((a,b))
            while len(edges) > target_edges:
                e = random.choice(list(edges))
                edges.discard(e)
                
        res_edges, res_score = _local_search(edges, time_per_candidate)
        
        if res_score < best_overall_score:
            best_overall_score = res_score
            best_overall_edges = res_edges
            
    return [list(e) for e in best_overall_edges]
