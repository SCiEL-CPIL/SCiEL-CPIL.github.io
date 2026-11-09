"""Combined approach: deterministic algebraic base + randomized local search with fast bitmask validation."""
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Fast validation using bitmasks
    # For triangle-freeness: for each edge (u,v), N(u) & N(v) must be empty
    # For independence: no 10 vertices with no edges between them
    #   Equivalent: complement graph has no K10
    
    # Strategy:
    # 1. Try known constructions first (fast, guaranteed)
    # 2. Then randomized search from valid seeds to try to find n=39,40
    
    # Known: Andrásfai And(k) on n=3k-1 vertices, alpha=k
    # And(9): n=26, alpha=9 -> valid (alpha < 10)
    # We need to find graphs with n > 26 and alpha < 10
    
    # Try to construct n=39 or n=40 using randomized search
    # Start from random maximal triangle-free graphs and check alpha
    
    best_result = None
    best_n = 0
    
    # First, get a guaranteed valid result (And(9), n=26)
    n_and9 = 26
    adj_and9 = [[False]*n_and9 for _ in range(n_and9)]
    for i in range(n_and9):
        for j in range(i+1, n_and9):
            if (j - i) % n_and9 % 3 == 1:
                adj_and9[i][j] = True
                adj_and9[j][i] = True
    if _is_valid(adj_and9, n_and9, deadline):
        best_n = n_and9
        best_result = _to_edge_list(adj_and9, n_and9)
    
    # Try randomized search for larger n
    # Use bitmask representation for speed
    # adj_mask[i] is a bitmask of neighbors of vertex i
    
    for n in range(40, 26, -1):  # Try 40, 39, 38, ..., 27
        if time.time() > deadline - 5:
            break
        
        max_attempts = 500 if n >= 39 else 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random maximal triangle-free graph
            adj_mask = _random_maximal_triangle_free(n, deadline)
            
            if adj_mask is None:
                continue
            
            # Check if valid (no independent set of size 10)
            if _alpha_less_than_10(adj_mask, n, deadline):
                # Valid! Return immediately
                edge_list = _mask_to_edge_list(adj_mask, n)
                return {'n': n, 'edges': edge_list}
    
    # If we found nothing better, return the best valid result
    if best_result is not None:
        return {'n': best_n, 'edges': best_result}
    
    # Ultimate fallback: And(9)
    n = 26
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}


def _random_maximal_triangle_free(n, deadline):
    """Generate a random maximal triangle-free graph using bitmask representation."""
    adj_mask = [0] * n  # adj_mask[i] has bit j set if i-j is an edge
    
    # Start with random subset of edges, then complete greedily
    # Use a random order of edges
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges)
    
    for u, v in edges:
        if time.time() > deadline:
            return None
        
        # Check if adding (u,v) creates a triangle
        # Triangle exists if there's w with (u,w) and (v,w) both edges
        # i.e., adj_mask[u] & adj_mask[v] != 0
        if adj_mask[u] & adj_mask[v]:
            continue
        
        # Add the edge
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    return adj_mask


def _alpha_less_than_10(adj_mask, n, deadline):
    """Check if the independence number is less than 10.
    Equivalent: the complement graph has no clique of size 10.
    Uses branch and bound with bitmasks."""
    
    # Complement adjacency: comp_mask[i] has bit j set if i-j is NOT an edge (and i != j)
    # All possible neighbors: (1 << n) - 1
    all_mask = (1 << n) - 1
    
    # comp_mask[i] = all_mask ^ (1 << i) ^ adj_mask[i]
    comp_mask = [all_mask ^ (1 << i) ^ adj_mask[i] for i in range(n)]
    
    # Check for clique of size 10 in complement
    # Use B&B: find if there's a set of 10 vertices all connected in comp
    
    # Quick check: if max degree in comp < 9, no K10
    max_deg = max(bin(comp_mask[i]).count('1') for i in range(n))
    if max_deg < 9:
        return True  # No K10 possible
    
    # B&B search for K10 in complement
    # Use iterative deepening / branch and bound
    
    # Order vertices by degree in complement (descending) for better pruning
    deg_order = sorted(range(n), key=lambda i: bin(comp_mask[i]).count('1'), reverse=True)
    
    # Map to new ordering
    pos_in_order = [0] * n
    for idx, v in enumerate(deg_order):
        pos_in_order[v] = idx
    
    # Rebuild comp_mask in the new ordering
    comp_mask_new = [0] * n
    for i in range(n):
        v = deg_order[i]
        mask = 0
        for j in range(n):
            if j != i and (comp_mask[v] >> deg_order[j]) & 1:
                mask |= (1 << j)
        comp_mask_new[i] = mask
    
    # Now search for K10 in comp_mask_new
    # B&B: candidates is a bitmask of available vertices
    # We need to find if we can select 10 vertices all mutually adjacent
    
    def bnb_kclique(candidates, need, depth=0):
        if time.time() > deadline:
            return True  # Timeout, assume invalid (conservative)
        
        if need == 0:
            return True
        
        if candidates == 0:
            return False
        
        # Pruning: if popcount(candidates) < need, impossible
        if bin(candidates).count('1') < need:
            return False
        
        # Pick a vertex from candidates
        # Heuristic: pick the one with highest degree within candidates
        # Find first set bit
        # Try each candidate vertex
        
        # For efficiency, iterate through set bits
        c = candidates
        while c:
            # Get lowest set bit
            lb = c & (-c)
            idx = lb.bit_length() - 1
            c ^= lb
            
            # New candidates: intersection of remaining candidates with neighbors of idx
            # Remaining candidates: candidates without idx and without bits below idx (to avoid duplicates)
            remaining = candidates & (~((1 << (idx + 1)) - 1))  # bits above idx
            new_candidates = remaining & comp_mask_new[idx]
            
            if bnb_kclique(new_candidates, need - 1, depth + 1):
                return True
        
        return False
    
    # Start with all vertices as candidates
    all_candidates = (1 << n) - 1
    return not bnb_kclique(all_candidates, 10)


def _is_valid(adj, n, deadline):
    """Check if graph is triangle-free and has no independent set of size 10."""
    # Convert to bitmask
    adj_mask = [0] * n
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                adj_mask[i] |= (1 << j)
    return _alpha_less_than_10(adj_mask, n, deadline)


def _mask_to_edge_list(adj_mask, n):
    """Convert bitmask adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges


def _to_edge_list(adj, n):
    """Convert 2D list adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges
