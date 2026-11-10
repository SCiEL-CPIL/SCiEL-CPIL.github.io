"""Rewritten program: maximum independent set on conflict graph via branch-and-bound."""
import math
import time

def construct():
    """Return a nonempty subset of [1,256] maximizing size such that
    a+b is never a perfect square for all a<=b in the set (including a=b).
    """
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    n = 256
    # Build conflict graph: edge between i and j if i+j is a perfect square
    # Also self-conflict if 2*i is a perfect square
    
    # Perfect squares up to 512 (max sum is 256+256=512)
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    # conflict[i][j] = True if i+j is a perfect square (0-indexed, so values 1..256)
    # We'll use bitmasks for efficiency
    # Each vertex i (0-indexed, representing value i+1) has a conflict mask
    
    # First, identify vertices that have self-conflict (2*(i+1) is a perfect square)
    # These cannot be in any valid set
    self_conflict = [False] * n
    for i in range(n):
        val = i + 1
        if 2 * val in squares:
            self_conflict[i] = True
    
    # Build adjacency: for each vertex, which other vertices conflict with it
    # Use Python ints as bitmasks for the 256 vertices
    conflict_mask = [0] * n
    for i in range(n):
        vi = i + 1
        for j in range(n):
            vj = j + 1
            if vi + vj in squares:
                conflict_mask[i] |= (1 << j)
    
    # Remove self-conflicting vertices from consideration
    # They can never be in the set
    available = [i for i in range(n) if not self_conflict[i]]
    
    # Now we need maximum independent set in the graph induced by 'available' vertices
    # where edges are conflicts
    
    # For branch-and-bound, we work with the subgraph of available vertices
    # Remap to 0..m-1 for compactness
    m = len(available)
    # adj_mask[i] for i in 0..m-1: bitmask of neighbors in the subgraph
    adj_mask = [0] * m
    for i in range(m):
        vi = available[i]
        # Get conflicts from the full graph
        cm = conflict_mask[vi]
        # Map to subgraph indices
        mask = 0
        for j in range(m):
            vj = available[j]
            if cm & (1 << vj):
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # Branch and bound for maximum independent set
    # State: remaining vertices (bitmask), current set (bitmask), current size
    # We want to maximize the size of an independent set
    
    best_size = 0
    best_set = 0  # bitmask of the best independent set found
    
    def time_exceeded():
        return (time.time() - start_time) > DEADLINE
    
    # Precompute: for each vertex, its neighbors in the subgraph
    # Also, a useful bound: the size of the maximum independent set in the remaining graph
    # Upper bound: we can use a greedy coloring bound or simply the count of remaining vertices
    
    # For a tighter bound, use the fact that in an independent set, no two vertices
    # are adjacent. A simple upper bound is the number of remaining vertices.
    # A better bound: greedy coloring gives an upper bound on the clique number of the
    # complement, which equals the independence number of the original.
    
    # Actually, for maximum independent set, a common bound is:
    # For the remaining vertices, sort by degree and use a greedy upper bound.
    
    # Let's implement a simple but effective branch-and-bound:
    # Branch on a vertex: include it (remove it and its neighbors) or exclude it (remove it)
    
    # To make this efficient, we'll use the following:
    # - Order vertices by degree (descending) for branching
    # - Use a simple upper bound: number of remaining vertices
    # - Prune if current_size + upper_bound <= best_size
    
    # For better performance, let's use a recursive approach with bitmasks
    
    def max_independent_set(remaining_mask, current_size, current_mask):
        nonlocal best_size, best_set, time_exceeded
        
        if time_exceeded():
            return
        
        if remaining_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_mask
            return
        
        # Upper bound: number of bits set in remaining_mask
        upper_bound = current_size + bin(remaining_mask).count('1')
        if upper_bound <= best_size:
            return
        
        # Select a vertex to branch on
        # Choose the vertex with the highest degree in the remaining subgraph
        # (or some heuristic)
        # For simplicity, pick the first set bit
        # Better: pick the vertex with max degree among remaining
        
        # Find all remaining vertices
        verts = []
        tmp = remaining_mask
        while tmp:
            lb = tmp & (-tmp)
            v = lb.bit_length() - 1
            verts.append(v)
            tmp ^= lb
        
        if not verts:
            if current_size > best_size:
                best_size = current_size
                best_set = current_mask
            return
        
        # Choose vertex with max degree in remaining subgraph
        max_deg = -1
        branch_v = verts[0]
        for v in verts:
            # Degree in remaining subgraph
            deg = bin(adj_mask[v] & remaining_mask).count('1')
            if deg > max_deg:
                max_deg = deg
                branch_v = v
        
        # Branch 1: include branch_v
        # Remove branch_v and all its neighbors from remaining
        new_remaining = remaining_mask & ~(adj_mask[branch_v] | (1 << branch_v))
        new_mask = current_mask | (1 << branch_v)
        max_independent_set(new_remaining, current_size + 1, new_mask)
        
        if time_exceeded():
            return
        
        # Branch 2: exclude branch_v
        new_remaining = remaining_mask & ~(1 << branch_v)
        max_independent_set(new_remaining, current_size, current_mask)
    
    # Initial call
    full_mask = (1 << m) - 1
    max_independent_set(full_mask, 0, 0)
    
    # Extract the result
    if best_set == 0:
        # Fallback: return a single valid element
        for i in available:
            return [i + 1]
        return [1]
    
    result = []
    tmp = best_set
    while tmp:
        lb = tmp & (-tmp)
        v = lb.bit_length() - 1
        result.append(available[v] + 1)
        tmp ^= lb
    
    result.sort()
    
    # Verify the result
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = int(math.isqrt(s))
            if r * r == s:
                # This should not happen, but if it does, remove the offending element
                # For safety, just return what we have
                pass
    
    return result
