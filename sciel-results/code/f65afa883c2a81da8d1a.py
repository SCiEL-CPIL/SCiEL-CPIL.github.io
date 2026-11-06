import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START

def check_c4(adj, n):
    """Check if graph has a C4. Returns True if C4 exists."""
    for u in range(n):
        for v in list(adj[u]):
            if v > u:
                if len(adj[u] & adj[v]) > 1:
                    return True
    return False

def can_add_edge(adj, u, v):
    """Check if adding edge (u,v) keeps graph C4-free."""
    if u == v:
        return False
    if v in adj[u]:
        return False
    # u and v must have no common neighbor
    return len(adj[u] & adj[v]) == 0

def find_independent_set(adj, n, target_size):
    """Find an independent set of size target_size using branch and bound.
    Returns the set or None if not found within time."""
    deadline = time.time() + 5.0
    
    # Greedy coloring upper bound for pruning
    # Simple approach: recursive backtracking with ordering
    
    # Order vertices by degree (ascending) for better pruning
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    
    # Map: position in order -> vertex
    # We'll work in the original vertex space but iterate in degree order
    
    found = [None]
    
    def dfs(start_idx, current, depth):
        if time.time() > deadline:
            return
        if depth == target_size:
            found[0] = current[:]
            return
        remaining = n - start_idx
        if remaining < target_size - depth:
            return
        
        for i in range(start_idx, n):
            v = order[i]
            # Check if v is independent from current
            independent = True
            for node in current:
                if v in adj[node]:
                    independent = False
                    break
            if independent:
                current.append(v)
                dfs(i + 1, current, depth + 1)
                current.pop()
                if found[0] is not None:
                    return
    
    dfs(0, [], 0)
    return found[0]

def compute_alpha_upper_bound(adj, n):
    """Compute a greedy upper bound on alpha using greedy coloring."""
    # Greedy coloring: assign colors greedily
    color = [-1] * n
    next_color = 0
    for v in range(n):
        # Find smallest color not used by neighbors
        used = set()
        for u in adj[v]:
            if color[u] >= 0:
                used.add(color[u])
        c = 0
        while c in used:
            c += 1
        color[v] = c
        next_color = max(next_color, c + 1)
    return next_color

def random_greedy_construct(n_target, seed, time_limit):
    """Randomly construct a C4-free graph on n_target vertices.
    Returns (adj, edges, alpha_ok) or None if timeout."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    random.seed(seed)
    
    adj = [set() for _ in range(n_target)]
    edges = []
    
    # Generate all possible edges
    all_pairs = [(u, v) for u in range(n_target) for v in range(u + 1, n_target)]
    random.shuffle(all_pairs)
    
    # Greedily add edges
    for u, v in all_pairs:
        if time.time() > deadline:
            return None
        if can_add_edge(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    # Verify C4-free
    if check_c4(adj, n_target):
        return None
    
    # Check alpha
    indep = find_independent_set(adj, n_target, 11)
    if indep is None:
        return (adj, edges, True)
    else:
        return (adj, edges, False)

def local_search_improve(adj, edges, n, time_limit):
    """Try to reduce alpha by adding edges. Returns (adj, edges, alpha_ok)."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Find all non-adjacent pairs
    non_adj_pairs = []
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u]:
                non_adj_pairs.append((u, v))
    
    random.shuffle(non_adj_pairs)
    
    # Try adding edges that might reduce alpha
    # Priority: edges between vertices in a found independent set
    current_adj = [set(s) for s in adj]
    current_edges = edges[:]
    
    # Find an independent set of size 11 if it exists
    indep = find_independent_set(current_adj, n, 11)
    if indep is None:
        return (current_adj, current_edges, True)
    
    # Try to break up this independent set by adding edges
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find current independent set
        indep = find_independent_set(current_adj, n, 11)
        if indep is None:
            return (current_adj, current_edges, True)
        
        # Try adding edges within the independent set
        for i in range(len(indep)):
            for j in range(i + 1, len(indep)):
                u, v = indep[i], indep[j]
                if can_add_edge(current_adj, u, v):
                    # Add this edge
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    current_edges.append([u, v])
                    improved = True
                    # Re-check
                    indep = find_independent_set(current_adj, n, 11)
                    if indep is None:
                        return (current_adj, current_edges, True)
                    break
            if improved:
                break
        
        if not improved:
            # Try random non-adjacent pairs
            for u, v in non_adj_pairs:
                if time.time() > deadline:
                    break
                if v not in current_adj[u] and can_add_edge(current_adj, u, v):
                    # Check if this helps: add and see if alpha decreases
                    old_adj_u = len(current_adj[u])
                    old_adj_v = len(current_adj[v])
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    current_edges.append([u, v])
                    improved = True
                    break
    
    # Final check
    indep = find_independent_set(current_adj, n, 11)
    if indep is None:
        return (current_adj, current_edges, True)
    else:
        return (current_adj, current_edges, False)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_n = 30
    best_edges = []
    
    # Fallback: 10 disjoint triangles
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        best_edges += [[a, b], [a, c], [b, c]]
    
    # Try random greedy construction for various n from high to low
    for n_target in [43, 42, 41, 40, 39]:
        if time.time() > deadline - 20:
            break
        
        # Try multiple random seeds
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            
            result = random_greedy_construct(n_target, seed, time_limit=2.0)
            if result is None:
                continue
            
            adj, edges, alpha_ok = result
            
            if alpha_ok:
                best_n = n_target
                best_edges = edges[:]
                best_result = {'n': n_target, 'edges': edges[:]}
                break  # Found valid graph for this n, try higher? No, we go high to low
            
            # If alpha not ok, try local search to fix
            if time.time() < deadline - 15:
                adj2, edges2, alpha_ok2 = local_search_improve(adj, edges, n_target, time_limit=3.0)
                if alpha_ok2:
                    best_n = n_target
                    best_edges = edges2[:]
                    best_result = {'n': n_target, 'edges': edges2[:]}
                    break
    
    if best_result is not None:
        return best_result
    
    # Fallback to triangle base
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
