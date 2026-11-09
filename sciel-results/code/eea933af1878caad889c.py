"""
Search for a large graph with no regular induced subgraph on 6+ vertices.
Uses repair-based local search with strict time management and limited full verifications.
"""
import time
import random
import itertools

def construct():
    start = time.time()
    deadline = start + 145  # 5s safety margin
    
    best_n = 0
    best_edges = []
    
    # Try n=20 first, then 19, etc.
    # Allocate more time to higher n
    time_budgets = {20: 40, 19: 20, 18: 10, 17: 5, 16: 3}
    
    for target_n in [20, 19, 18, 17, 16]:
        if time.time() > deadline - 5:
            break
            
        # Calculate remaining time for this n
        current_time = time.time()
        allowed_time = min(time_budgets.get(target_n, 5), deadline - 5 - current_time)
        if allowed_time < 2:
            continue
            
        target_deadline = current_time + allowed_time
        found = _local_search(target_n, target_deadline)
        if found:
            best_n = target_n
            best_edges = found
            break
            
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
        
    # Fallback: return a small valid graph (empty graph on 5 vertices is valid)
    return {'n': 5, 'edges': []}

def _local_search(n, deadline):
    """
    Perform local search to find a valid graph on n vertices.
    Returns list of edges if found, else None.
    """
    max_restarts = 5
    # Limit full verifications to avoid timeout
    max_full_checks_per_restart = 3
    
    for restart in range(max_restarts):
        if time.time() > deadline - 1:
            break
            
        rng = random.Random(restart * 1000 + 42)
        
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Local search loop
        clean_count = 0
        max_clean_samples = 2000 # Trigger verification check
        full_checks_done = 0
        
        while time.time() < deadline - 1:
            # Sample a random 6-subset
            subset = rng.sample(range(n), 6)
            
            # Check if this subset induces a regular graph
            if _is_regular(adj, subset):
                # Violation found!
                # Flip a random edge within this subset to break regularity
                u, v = rng.sample(subset, 2)
                
                # Flip edge (u,v)
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # Reset clean count
                clean_count = 0
            else:
                # No violation in this sample
                clean_count += 1
                
                if clean_count >= max_clean_samples:
                    # Perform full verification only if we have time and haven't done too many
                    if full_checks_done < max_full_checks_per_restart:
                        if _is_valid_full(adj, n, deadline):
                            # Convert to edges
                            edges = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if adj[i] & (1 << j):
                                        edges.append([i, j])
                            return edges
                        else:
                            # Verification failed, do a random kick
                            clean_count = 0
                            full_checks_done += 1
                            for _ in range(5):
                                u, v = rng.sample(range(n), 2)
                                if adj[u] & (1 << v):
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                else:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                    else:
                        # Too many full checks, just kick and continue sampling
                        clean_count = 0
                        for _ in range(5):
                            u, v = rng.sample(range(n), 2)
                            if adj[u] & (1 << v):
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                            else:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)

    return None

def _is_regular(adj, subset):
    """Check if the induced subgraph on 'subset' is regular."""
    mask = 0
    for v in subset:
        mask |= (1 << v)
        
    v0 = subset[0]
    d0 = (adj[v0] & mask).bit_count()
    
    for v in subset[1:]:
        dv = (adj[v] & mask).bit_count()
        if dv != d0:
            return False
    return True

def _is_valid_full(adj, n, deadline):
    """
    Exhaustive check for any regular induced subgraph of size >= 6.
    Returns True if valid (no such subgraph), False otherwise.
    """
    # Optimization: Check smaller sizes first as they are more likely to be regular?
    # Actually, larger sizes are rarer but harder to check? 
    # For n=20, size 6 is most common.
    # We must check all sizes >= 6.
    
    # To speed up, we can break early if time is tight.
    for size in range(6, n + 1):
        # If time is running out, assume invalid to be safe? 
        # No, we want to return valid graphs. If we don't finish check, we can't certify.
        # But we only call this if we think it's close.
        # If we run out of time during check, return False (invalid) to avoid returning unverified graph.
        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                return False
            if _is_regular(adj, subset):
                return False
    return True
