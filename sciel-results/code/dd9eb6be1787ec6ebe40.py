"""Program to construct a graph with no regular induced subgraph on 6+ vertices."""
import time
import random
import itertools
from collections import Counter

def _is_regular_induced(adj, subset, n):
    """Check if the induced subgraph on subset is regular."""
    m = len(subset)
    if m < 2:
        return True
    degrees = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            u = subset[i]
            v = subset[j]
            if adj[u] >> v & 1:
                degrees[i] += 1
                degrees[j] += 1
    # Check if all degrees are the same
    first = degrees[0]
    for d in degrees[1:]:
        if d != first:
            return False
    return True

def _find_regular_subgraph(adj, n, min_size=6):
    """Find any induced regular subgraph of size >= min_size.
    Returns the subset (as a tuple) if found, else None."""
    # Check subsets of size min_size first (6)
    # For larger sizes, if a k-regular graph exists, it might not contain a 6-regular one?
    # Actually, if a graph is regular, any subset is not necessarily regular.
    # But we must check ALL sizes >= 6.
    # However, checking all subsets of size 6 is the most expensive part if n is large.
    # C(21, 6) = 54264.
    # C(21, 7) = 116280.
    # C(21, 8) = 203490.
    # C(21, 9) = 293930.
    # C(21, 10) = 352716.
    # C(21, 11) = 366025.
    # ...
    # Total subsets of size >= 6 is roughly 2^21 / 2 - (C(21,0)+...+C(21,5))
    # 2^21 = 2097152.
    # So ~1M subsets. Checking 1M subsets per iteration is too slow for a loop.
    
    # Optimization: We only need to find ONE regular subgraph to break.
    # We can sample subsets or check in order of likelihood?
    # Smaller regular subgraphs (size 6) are more likely to appear in random graphs?
    # Let's just check size 6 first. If none, check size 7, etc.
    
    for size in range(min_size, n + 1):
        # To speed up, we can stop early if we find one
        for subset in itertools.combinations(range(n), size):
            if _is_regular_induced(adj, subset, n):
                return subset
    return None

def _check_graph_valid(adj, n):
    """Verify the graph has NO regular induced subgraph of size >= 6."""
    # This is expensive, so only call when we think we are done.
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_induced(adj, subset, n):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # We want to maximize n.
    # Known: 20 is possible. 21 is the goal.
    # Strategy: Local search.
    # 1. Try to find a valid graph on n=21.
    # 2. If fails within budget, try n=20.
    
    best_n = 0
    best_edges = []
    
    for target_n in [21, 20]:
        if time.time() > deadline:
            break
            
        n = target_n
        max_attempts = 20 if n == 21 else 50
        
        for attempt in range(max_attempts):
            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            # Limit steps per attempt
            max_steps = 500 if n == 21 else 1000
            found = False
            for step in range(max_steps):
                if time.time() > deadline:
                    found = True # break out
                    break
                    
                # Find a regular subgraph
                reg_sub = _find_regular_subgraph(adj, n)
                
                if reg_sub is None:
                    # No regular subgraph found. Verify to be sure.
                    if _check_graph_valid(adj, n):
                        # Construct edge list
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj[i] >> j & 1:
                                    edges.append([i, j])
                        
                        if n > best_n:
                            best_n = n
                            best_edges = edges
                        found = True
                        break
                    
                # Found a regular subgraph, flip an edge to break it
                # Pick a random edge inside the subgraph
                subset = list(reg_sub)
                # Pick two random vertices from the subset
                u_idx, v_idx = random.sample(range(len(subset)), 2)
                u = subset[u_idx]
                v = subset[v_idx]
                
                # Flip the edge
                if adj[u] >> v & 1:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # If time is up, stop
            if time.time() > deadline:
                break

    # If we didn't find n=21, we might have n=20 or less.
    # If we didn't find anything, return a trivial valid graph (e.g., n=5 empty)
    if best_n == 0:
        # Fallback: n=5 empty graph is always valid (no subset of size 6)
        return {'n': 5, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
