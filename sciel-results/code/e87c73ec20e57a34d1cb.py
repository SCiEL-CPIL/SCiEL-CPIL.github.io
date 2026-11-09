import time
import random
import math
from collections import defaultdict

def _compute_book_sizes(edges, n=40):
    """Compute the largest book size for the given edges."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    max_book = 0
    for u, v in edges:
        common = (adj[u] & adj[v]).bit_count()
        if common > max_book:
            max_book = common
    return max_book

def _build_structural_init(a_size, b_size, seed, max_deg_b=None):
    """
    Build a graph with a bipartite core A-B and a bipartite subgraph in B.
    A is independent. B is split into B1, B2.
    Edges in B are chosen to be bipartite and have bounded degree.
    """
    random.seed(seed)
    n = 40
    target_edges = 401
    
    A = list(range(a_size))
    B = list(range(a_size, n))
    
    edges = []
    edge_set = set()
    
    # 1. Add all A-B edges
    for a in A:
        for b in B:
            edges.append([a, b])
            edge_set.add((a, b))
    
    current_count = len(edges)
    needed_in_B = target_edges - current_count
    
    if needed_in_B < 0:
        return None

    # 2. Split B into B1 and B2 to make B-subgraph bipartite
    mid = len(B) // 2
    B1 = B[:mid]
    B2 = B[mid:]
    
    # Determine max degree constraint for B-subgraph
    # If max_deg_b is not provided, estimate from needed_in_B
    if max_deg_b is None:
        # Try to find a feasible max degree
        # needed_in_B <= min(len(B1), len(B2)) * max_deg_b
        # So max_deg_b >= needed_in_B / min(len(B1), len(B2))
        min_size = min(len(B1), len(B2))
        max_deg_b = math.ceil(needed_in_B / min_size)
        # Add a small buffer to ensure feasibility
        max_deg_b += 1

    # We want to construct a bipartite graph on B1, B2 with exactly needed_in_B edges
    # and max degree <= max_deg_b.
    
    # Strategy:
    # 1. Generate a "full" bipartite graph with max degree max_deg_b.
    #    This is a max_deg_b-regular bipartite graph if possible.
    # 2. Remove edges until we have needed_in_B.
    
    # Constructing a regular bipartite graph:
    # We can use a simple cyclic construction if sizes allow, or random.
    # For robustness, let's use a greedy random approach with degree tracking.
    
    deg = {v: 0 for v in B}
    potential_edges = []
    for u in B1:
        for v in B2:
            potential_edges.append((u, v))
    
    random.shuffle(potential_edges)
    
    added = 0
    # First pass: add edges respecting max_deg_b
    for u, v in potential_edges:
        if added == needed_in_B:
            break
        if deg[u] < max_deg_b and deg[v] < max_deg_b:
            edges.append([u, v])
            edge_set.add((u, v))
            deg[u] += 1
            deg[v] += 1
            added += 1
            
    if added < needed_in_B:
        # If we couldn't add enough edges with the constraint, 
        # it means the constraint was too tight or the graph structure is limiting.
        # In this case, we relax the constraint and just add remaining edges randomly.
        # This might increase the max degree, but it's better than having too few edges.
        for u, v in potential_edges:
            if added == needed_in_B:
                break
            if (u, v) not in edge_set:
                edges.append([u, v])
                edge_set.add((u, v))
                deg[u] += 1
                deg[v] += 1
                added += 1
                
    if added != needed_in_B:
        return None

    return edges

def _simulate_annealing(n=40, target_edges=401, time_limit=10.0, seed=42, initial_edges=None):
    """Simulated annealing with efficient bitmask updates."""
    start_time = time.time()
    random.seed(seed)
    
    if initial_edges is None:
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        initial_edges = [list(e) for e in all_possible[:target_edges]]
        
    current_edges = [list(e) for e in initial_edges]
    edge_set = set((min(u,v), max(u,v)) for u,v in current_edges)
    
    adj = [0] * n
    for u, v in current_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    book_sizes = {}
    current_max = 0
    for u, v in current_edges:
        key = (min(u,v), max(u,v))
        bs = (adj[u] & adj[v]).bit_count()
        book_sizes[key] = bs
        if bs > current_max:
            current_max = bs
            
    best_edges = [list(e) for e in current_edges]
    best_max = current_max
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    T0 = 5.0
    Tend = 0.05
    alpha = 0.9995
    
    T = T0
    while T > Tend and time.time() - start_time < time_limit:
        remove_idx = random.randint(0, len(current_edges) - 1)
        ru, rv = current_edges[remove_idx]
        rkey = (min(ru, rv), max(ru, rv))
        
        ckey = None
        for _ in range(10):
            cu, cv = random.choice(all_possible)
            ckey = (min(cu, cv), max(cu, cv))
            if ckey not in edge_set:
                break
        if ckey is None or ckey in edge_set:
            continue
            
        cu, cv = ckey
        
        affected = set()
        affected.add(rkey)
        affected.add(ckey)
        
        for v in range(n):
            if v == ru or v == rv or v == cu or v == cv:
                continue
            k = (min(v, ru), max(v, ru))
            if k in edge_set:
                affected.add(k)
            k = (min(v, rv), max(v, rv))
            if k in edge_set:
                affected.add(k)
            k = (min(v, cu), max(v, cu))
            if k in edge_set:
                affected.add(k)
            k = (min(v, cv), max(v, cv))
            if k in edge_set:
                affected.add(k)
                
        old_books = {k: book_sizes.get(k, 0) for k in affected}
        
        del book_sizes[rkey]
        edge_set.remove(rkey)
        adj[ru] &= ~(1 << rv)
        adj[rv] &= ~(1 << ru)
        
        edge_set.add(ckey)
        adj[cu] |= (1 << cv)
        adj[cv] |= (1 << cu)
        current_edges[remove_idx] = [cu, cv]
        
        new_max_affected = 0
        for k in affected:
            if k in edge_set:
                u, v = k
                bs = (adj[u] & adj[v]).bit_count()
                book_sizes[k] = bs
                if bs > new_max_affected:
                    new_max_affected = bs
            else:
                if k in book_sizes:
                    del book_sizes[k]
        
        was_max_edge = old_books.get(rkey, 0) == current_max
        if was_max_edge:
            new_max = max(book_sizes.values()) if book_sizes else 0
        else:
            new_max = max(current_max, new_max_affected)
            
        delta = new_max - current_max
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_edges = [list(e) for e in current_edges]
        else:
            edge_set.remove(ckey)
            adj[cu] &= ~(1 << cv)
            adj[cv] &= ~(1 << cu)
            if ckey in book_sizes:
                del book_sizes[ckey]
            
            edge_set.add(rkey)
            adj[ru] |= (1 << rv)
            adj[rv] |= (1 << ru)
            
            for k in affected:
                if k in old_books:
                    if k in edge_set:
                        book_sizes[k] = old_books[k]
                    else:
                        if k in book_sizes:
                            del book_sizes[k]
                else:
                    if k in book_sizes:
                        del book_sizes[k]
                        
            current_edges[remove_idx] = [ru, rv]
        
        T *= alpha
        
    return best_edges, best_max

def construct():
    start_time = time.time()
    n = 40
    target_edges = 401
    deadline = start_time + 145.0
    
    best_edges = None
    best_score = float('inf')
    
    # Try structural inits with different A sizes
    # A=12 is hypothesized to allow score 5
    # A=11 allows score 6
    # A=10 allows score 7
    configs = [
        (12, 28, 5), # a_size, b_size, max_deg_b
        (11, 29, 6),
        (10, 30, 7),
        (9, 31, 8)
    ]
    
    seeds = [42, 123, 456, 789]
    
    for a_size, b_size, max_deg in configs:
        if time.time() > deadline - 5.0:
            break
            
        for seed in seeds:
            if time.time() > deadline - 5.0:
                break
                
            init_edges = _build_structural_init(a_size, b_size, seed, max_deg)
            
            if init_edges is None:
                continue

            init_score = _compute_book_sizes(init_edges, n)
            
            if init_score < best_score:
                best_score = init_score
                best_edges = init_edges
                
            # Run SA with short time limit to polish
            time_limit = min(10.0, deadline - time.time() - 2.0)
            if time_limit < 1.0:
                continue
                
            edges, score = _simulate_annealing(n, target_edges, time_limit=time_limit, seed=seed, initial_edges=init_edges)
            
            if score < best_score:
                best_score = score
                best_edges = edges
                
    if best_edges is None:
        best_edges = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return best_edges
