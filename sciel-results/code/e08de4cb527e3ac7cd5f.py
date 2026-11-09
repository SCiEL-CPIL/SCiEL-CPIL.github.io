import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    # Safety margin: 150s limit. Keep 10s buffer.
    deadline = start_time + 140.0
    
    best_n = 0
    best_edges = []
    
    # Try to find valid graphs for decreasing n
    # n=20 is the target. If we find it, we return it immediately as it's likely optimal or near-optimal.
    # If we fail to find 20, we try 19, etc.
    
    for n in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]:
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time for this specific n
        # Larger n is harder, so give it more time? Or just try a fixed number of attempts?
        # Let's try to find a valid graph for this n within a sub-deadline.
        sub_deadline = min(deadline - 2.0, start_time + 10.0) # Give 10s per n attempt initially?
        # Actually, if n=20 is hard, we might spend all time there.
        # Let's do a time-sliced search.
        
        # For n=20, try multiple random starts with repair.
        found = False
        attempts = 0
        max_attempts = 50 if n >= 18 else 200
        
        while time.time() < sub_deadline and attempts < max_attempts:
            attempts += 1
            adj = _random_graph(n)
            
            # Quick check: is it valid?
            if _is_valid_fast(adj, n):
                best_n = n
                best_edges = _get_edges(adj, n)
                found = True
                break
            
            # If not valid, try to repair it
            # Repair loop
            repair_deadline = min(sub_deadline, time.time() + 0.5) # 0.5s per repair attempt
            adj = _repair_graph(adj, n, repair_deadline)
            
            if _is_valid_fast(adj, n):
                best_n = n
                best_edges = _get_edges(adj, n)
                found = True
                break
        
        if found:
            return {'n': best_n, 'edges': best_edges}
        
        # If we didn't find a valid graph for this n, try a smaller n
        # But we must ensure we don't waste time if n=20 is possible.
        # The loop structure above tries n=20 first. If it fails, it goes to 19.
        # This is correct.
        
    # Fallback
    if best_n == 0:
        # Trivially valid small graph
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}

def _random_graph(n):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _get_edges(adj, n):
    edges = []
    for i in range(n):
        mask = adj[i]
        # Only consider j > i to avoid duplicates
        for j in range(i + 1, n):
            if mask & (1 << j):
                edges.append([i, j])
    return edges

def _is_valid_fast(adj, n):
    """
    Check if no subset of size >= 6 induces a regular graph.
    Optimized for early termination.
    """
    # Check size 6 first, as it's the most common violation and smallest.
    # If no size 6 regular subgraph, larger ones are less likely but possible.
    # However, checking all sizes is necessary for correctness.
    
    # To speed up, we can check sizes in increasing order.
    # For n=20, C(20,6) = 38760.
    # For each subset, we compute degrees.
    
    # Precompute bit masks for vertices?
    # adj[i] is the mask of neighbors of i.
    
    # We iterate through all combinations of size 6.
    # If we find a regular one, return False.
    
    # Optimization: 
    # A subset is regular if all vertices have the same degree within the subset.
    # degree(v, S) = (adj[v] & mask_S).bit_count()
    
    # We can generate masks for subsets incrementally? 
    # itertools.combinations is fine.
    
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset(adj, subset):
                return False
    return True

def _is_regular_subset(adj, subset):
    """
    Check if the induced subgraph on 'subset' is regular.
    """
    mask = 0
    for v in subset:
        mask |= (1 << v)
        
    # Calculate degree of first vertex
    v0 = subset[0]
    d0 = (adj[v0] & mask).bit_count()
    
    for v in subset[1:]:
        dv = (adj[v] & mask).bit_count()
        if dv != d0:
            return False
    return True

def _repair_graph(adj, n, deadline):
    """
    Attempt to repair a graph by flipping edges to break regular induced subgraphs.
    """
    adj = adj[:] # Copy
    
    # We will iterate, finding a violation and fixing it.
    # To avoid infinite loops, we limit the number of steps or time.
    steps = 0
    max_steps = 1000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Find a violating subset
        viol_subset = _find_violation(adj, n)
        
        if viol_subset is None:
            break # No violation, graph is valid
        
        # Pick an edge within the violating subset to flip
        # Flipping an edge (u, v) in the subset changes degrees of u and v.
        # This usually breaks regularity.
        
        # Randomly select two distinct vertices from the subset
        u, v = random.sample(viol_subset, 2)
        
        # Flip edge (u, v)
        # If edge exists, remove it. If not, add it.
        if adj[u] & (1 << v):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    return adj

def _find_violation(adj, n):
    """
    Find any subset of size >= 6 that induces a regular graph.
    Returns the subset or None.
    """
    # Check size 6 first
    for subset in itertools.combinations(range(n), 6):
        if _is_regular_subset(adj, subset):
            return subset
            
    # If no size 6 violation, check larger sizes?
    # It is possible to have no regular 6-subgraph but a regular 7-subgraph.
    # However, regular 7-subgraphs are rarer.
    # For efficiency, we might skip larger sizes in the repair loop and rely on final validation.
    # But if we return "None" here, _repair_graph stops, and then _is_valid_fast is called.
    # If _is_valid_fast finds a size 7 violation, we are stuck.
    # So we should check larger sizes in _find_violation too, but maybe with a limit?
    # Or just check size 6 in repair, and rely on the fact that most violations are size 6.
    # If a graph has no regular 6-subgraph but has a regular 7-subgraph, it's still invalid.
    # We must handle this.
    
    # Let's check size 7 as well, but it's more expensive.
    if n >= 7:
        for subset in itertools.combinations(range(n), 7):
            if _is_regular_subset(adj, subset):
                return subset
                
    # And so on?
    # For n=20, checking all sizes is expensive.
    # But we only do this in repair loop, which is limited by time.
    # Let's check up to size 8 or so, and assume larger are rare enough that if we fix 6s and 7s, we might be ok?
    # No, that's risky.
    
    # Better: In _repair_graph, we only fix size 6 violations.
    # If we run out of time fixing 6s, we return the graph.
    # Then _is_valid_fast checks all sizes.
    # If it's still invalid, we discard it.
    # This is safe.
    
    return None

# To ensure we don't miss easy wins, we can add a heuristic:
# If n is small, random search is very likely to find a valid graph.
# The repair is mainly for n=20.

# Let's refine the main loop to be more efficient.
# The current construct tries n=20, then 19...
# If n=20 takes 10s and fails, we try 19.
# This is good.

# One improvement: In _repair_graph, if we flip an edge, we might create a new violation elsewhere.
# The loop handles this by re-checking.

# Let's add a specific check for n=5 fallback to be safe.

# Final check on the logic:
# 1. construct() calls _random_graph and _is_valid_fast.
# 2. If invalid, calls _repair_graph.
# 3. _repair_graph flips edges to break size-6 regular subgraphs.
# 4. After repair, _is_valid_fast is called again.
# 5. If valid, return.

# This seems solid. The bottleneck is _is_valid_fast for n=20.
# C(20,6) = 38760 subsets.
# Each check is O(6) bit counts.
# 38760 * 6 * ~10ns = ~2ms per full check.
# This is very fast.
# So we can afford many checks.

# The repair loop might get stuck if it creates new violations faster than it fixes old ones.
# Random flipping is a good heuristic.

# Let's increase the max_attempts for n=20 to ensure we give it a good chance.
# And ensure the deadline is respected.

# One detail: random.seed is not set, so results are non-deterministic.
# This is fine for the task, as long as it finds a valid graph.
# To be more robust, we could try a few fixed seeds if random fails?
# No, just use system random.

# Let's wrap the construct function to ensure it always returns a valid graph.
# If the loop finishes without finding a valid graph for n=20..6, it returns the best found.
# If no graph found, returns n=5.

# The code looks correct.

# One potential issue: _find_violation only checks size 6 and 7.
# If a graph has no regular 6 or 7 subgraphs but has a regular 8 subgraph, _repair_graph will stop (thinking it's valid), but _is_valid_fast will reject it.
# This is handled by the outer loop: if _is_valid_fast returns False, we discard the result and try a new random graph.
# So it's safe.

# We can optimize _find_violation to only check size 6 to make repair faster, and rely on the final check for larger sizes.
# This is better because checking size 6 is the most common failure.

# Revised _find_violation:
def _find_violation_fast(adj, n):
    for subset in itertools.combinations(range(n), 6):
        if _is_regular_subset(adj, subset):
            return subset
    return None

# Use this in repair.

# Let's update the code to use _find_violation_fast in _repair_graph.

# Also, we should ensure that for small n (like 10), we don't waste time on complex repair.
# Random graphs on 10 vertices are very likely to be valid.
# So the first check will pass.

# Final structure:
# construct:
#   for n in [20, 19, ...]:
#      for attempt in range(max_attempts):
#         adj = random_graph(n)
#         if valid(adj): return
#         adj = repair(adj)
#         if valid(adj): return
#   return fallback

# This is efficient.

# Let's write the final code.
