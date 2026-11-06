import numpy as np
import random
import time

def _compute_score(adj, deg):
    """Compute the maximum degree sum over all triangles in the graph."""
    n = adj.shape[0]
    max_sum = 0
    # Enumerate triangles: for each edge (u,v), check common neighbors
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                # common neighbors w > v to avoid duplicates
                common = np.where(adj[u] & adj[v] & np.arange(n) > v)[0]
                for w in common:
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_sum:
                        max_sum = s
    return max_sum

def _build_adjacency(n, edges):
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    return adj

def _compute_degrees(adj):
    return adj.sum(axis=1)

def _count_edges(adj):
    return int(adj.sum()) // 2

def _find_triangles(adj):
    """Find all triangles as a list of (u, v, w) with u < v < w."""
    n = adj.shape[0]
    triangles = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                common = np.where(adj[u] & adj[v])[0]
                for w in common:
                    if w > v:
                        triangles.append((u, v, w))
    return triangles

def _get_edges(adj):
    n = adj.shape[0]
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                edges.append([u, v])
    return edges

def construct():
    n = 16
    target_edges = 65
    deadline = time.time() + 140  # safety margin before 150s limit
    
    best_adj = None
    best_score = float('inf')
    
    # Start with K_{8,8} + one edge
    adj = np.zeros((n, n), dtype=bool)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = True
            adj[v, u] = True
    adj[0, 1] = True
    adj[1, 0] = True
    
    deg = _compute_degrees(adj)
    score = _compute_score(adj, deg)
    
    if score < best_score:
        best_score = score
        best_adj = adj.copy()
    
    random.seed(42)
    
    # Simulated annealing with guided moves
    # State: adjacency matrix, degree array, current score
    current_adj = adj.copy()
    current_deg = deg.copy()
    current_score = score
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    # Track worst triangles for guided search
    def _get_worst_triangles(adj, deg, k=5):
        """Return the k worst triangles (highest degree sum)."""
        triangles = _find_triangles(adj)
        if not triangles:
            return []
        scored = [(deg[u] + deg[v] + deg[w], u, v, w) for (u, v, w) in triangles]
        scored.sort(reverse=True)
        return scored[:k]
    
    def _random_swap(current_adj, current_deg, n):
        """Perform a random edge swap to maintain edge count."""
        edges_list = _get_edges(current_adj)
        non_edges_list = []
        for u in range(n):
            for v in range(u+1, n):
                if not current_adj[u, v]:
                    non_edges_list.append((u, v))
        
        if not edges_list or not non_edges_list:
            return None, None, None, None
        
        # Pick a random edge to remove and a random non-edge to add
        # Must ensure they don't create a duplicate or self-loop
        removed = random.choice(edges_list)
        added = random.choice(non_edges_list)
        
        ru, rv = removed
        au, av = added
        
        # Check if adding would create a duplicate
        if current_adj[au, av]:
            return None, None, None, None
        
        # Apply swap
        new_adj = current_adj.copy()
        new_adj[ru, rv] = False
        new_adj[rv, ru] = False
        new_adj[au, av] = True
        new_adj[av, au] = True
        
        # Verify edge count is preserved
        if _count_edges(new_adj) != target_edges:
            return None, None, None, None
        
        new_deg = _compute_degrees(new_adj)
        new_score = _compute_score(new_adj, new_deg)
        
        return new_adj, new_deg, new_score, (removed, added)
    
    def _guided_swap(current_adj, current_deg, n, worst_tris):
        """Perform a guided swap targeting worst triangles."""
        if not worst_tris:
            return _random_swap(current_adj, current_deg, n)
        
        # Pick a random worst triangle
        _, u, v, w = random.choice(worst_tris)
        
        # Try to break this triangle by removing one of its edges
        # and adding a random non-edge elsewhere
        candidate_edges = [(u, v), (u, w), (v, w)]
        random.shuffle(candidate_edges)
        
        for removed in candidate_edges:
            ru, rv = removed
            if not current_adj[ru, rv]:
                continue
            
            # Find a random non-edge to add
            attempts = 0
            while attempts < 20:
                au = random.randint(0, n-1)
                av = random.randint(0, n-1)
                if au == av:
                    continue
                au, av = min(au, av), max(au, av)
                if current_adj[au, av]:
                    continue
                # Ensure we're not just re-adding the same edge
                if (au, av) == removed:
                    continue
                
                new_adj = current_adj.copy()
                new_adj[ru, rv] = False
                new_adj[rv, ru] = False
                new_adj[au, av] = True
                new_adj[av, au] = True
                
                if _count_edges(new_adj) != target_edges:
                    attempts += 1
                    continue
                
                new_deg = _compute_degrees(new_adj)
                new_score = _compute_score(new_adj, new_deg)
                
                return new_adj, new_deg, new_score, (removed, (au, av))
            
            attempts += 1
        
        return _random_swap(current_adj, current_deg, n)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Get worst triangles for guided search
        worst_tris = _get_worst_triangles(current_adj, current_deg, k=3)
        
        # Decide between guided and random moves
        if worst_tris and random.random() < 0.6:
            result = _guided_swap(current_adj, current_deg, n, worst_tris)
        else:
            result = _random_swap(current_adj, current_deg, n)
        
        if result[0] is None:
            continue
        
        new_adj, new_deg, new_score, _ = result
        
        if new_score < current_score:
            # Accept improvement
            current_adj = new_adj
            current_deg = new_deg
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj.copy()
        
        elif random.random() < np.exp(-(new_score - current_score) / T):
            # Accept worse with probability based on temperature
            current_adj = new_adj
            current_deg = new_deg
            current_score = new_score
        
        # Cool down
        T = max(T_min, T * cooling)
        
        # Occasionally reheat
        if iteration % 500 == 0 and T < 0.1:
            T = 0.5
    
    # Final polish: local search with tighter acceptance
    if time.time() < deadline - 10:
        T = 0.05
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            
            worst_tris = _get_worst_triangles(current_adj, current_deg, k=3)
            
            if worst_tris and random.random() < 0.7:
                result = _guided_swap(current_adj, current_deg, n, worst_tris)
            else:
                result = _random_swap(current_adj, current_deg, n)
            
            if result[0] is None:
                continue
            
            new_adj, new_deg, new_score, _ = result
            
            if new_score < current_score:
                current_adj = new_adj
                current_deg = new_deg
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_adj = current_adj.copy()
            elif random.random() < np.exp(-(new_score - current_score) / T):
                current_adj = new_adj
                current_deg = new_deg
                current_score = new_score
    
    # Return best found
    final_adj = best_adj if best_adj is not None else current_adj
    edges = _get_edges(final_adj)
    
    return {'n': n, 'edges': edges}
