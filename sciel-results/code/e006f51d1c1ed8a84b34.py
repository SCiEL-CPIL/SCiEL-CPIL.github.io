import time
import sys
from fractions import Fraction

def construct():
    N = 256
    # Precompute the conflict graph.
    # Two numbers a, b are in conflict if there exists a c such that {a, b, c} is a 3-AP.
    # This is equivalent to a, b, c being distinct and forming an AP.
    # For a pair (a, b) with a < b, the potential third elements are:
    # 1. c = 2*b - a (if c > b and c <= N)
    # 2. c = a - (b - a) = 2*a - b (if c < a and c >= 1)
    # If either exists, (a, b) is an edge in the conflict graph.
    
    # We use a list of bitmasks for adjacency to allow fast bit operations.
    # Since N=256, we need 256 bits. Python integers are arbitrary precision, so this is efficient.
    adj = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # Check if a, b, c form a 3-AP
            # Case 1: b is the middle term. a = b - d, c = b + d => d = b - a. c = 2*b - a.
            # If c > b and c <= N, then (a, b) and (b, c) and (a, c) are all in conflict?
            # Wait, the problem says "without any nonconstant k=3 term arithmetic progression".
            # This means we cannot have {x, y, z} in our set such that x, y, z form an AP.
            # If {a, b, c} is an AP, then any pair from {a, b, c} cannot be in the set together?
            # No, that's not right.
            # If we have {a, b, c} in the set, and they form an AP, that's forbidden.
            # So, if a, b, c form an AP, we cannot have all three in the set.
            # This is NOT a standard independent set problem on a graph where edges connect pairs.
            # In a standard MWIS, if there is an edge (u, v), we cannot have both u and v.
            # Here, the constraint is hypergraph: we cannot have all three {a, b, c} if they form an AP.
            
            # Let's re-read carefully: "without any nonconstant k=3 term arithmetic progression".
            # Yes, it forbids the existence of the triplet.
            # This is a 3-uniform hypergraph independent set problem.
            # However, the prompt suggests: "Model the problem as a maximum weight independent set in a tripartite conflict graph where nodes are integers 1-256 with weights 1/x, and edges connect any pair (a,b) that would form a 3-term AP with any potential third element".
            
            # Let's check the inspiration note again.
            # "edges connect any pair (a,b) that would form a 3-term AP with any potential third element"
            # If we define the graph G such that (a, b) is an edge if there exists c such that {a, b, c} is a 3-AP, does an independent set in G guarantee no 3-APs?
            # If S is an independent set in G, then for any a, b in S, there is no c such that {a, b, c} is a 3-AP.
            # This implies that no three elements in S can form a 3-AP.
            # Why? Suppose x, y, z in S form a 3-AP. Then (x, y) is an edge in G (with z as the third element).
            # Since S is independent, it cannot contain both endpoints of an edge. So it cannot contain both x and y.
            # Contradiction.
            # So yes, finding a Maximum Weight Independent Set in this specific graph G solves the problem.
            # Note: This graph G is denser than the "pairwise AP" graph (where edges only exist if the pair *itself* is part of an AP with a third element).
            # Actually, "pair (a,b) that would form a 3-term AP with any potential third element" means exactly that.
            # For any pair a, b, if 2b-a is in [1, N] and distinct from a, b, OR 2a-b is in [1, N] and distinct, then (a,b) is an edge.
            
            c1 = 2 * b - a
            c2 = 2 * a - b
            if (1 <= c1 <= N and c1 != a and c1 != b) or (1 <= c2 <= N and c2 != a and c2 != b):
                # Add edge (a, b)
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)

    # Now solve MWIS on this graph.
    # Branch and bound.
    # State: 
    #   candidate set C (bitmask)
    #   current set S (bitmask)
    #   current weight w
    #   upper bound: w + sum(weights of all in C)
    
    # To make it fast, we order vertices. 
    # A good heuristic for MWIS is to pick a vertex, branch on include/exclude.
    # Include: add v to S, remove v and its neighbors from C.
    # Exclude: remove v from C.
    
    # We can use a recursive function with memoization or just simple B&B.
    # Given N=256, the search space is huge. We need strong pruning.
    # The graph is quite dense. Let's check density.
    # For a random pair a, b, probability that 2b-a is in range is high.
    # Most pairs are connected.
    # If the graph is dense, the independent sets are small.
    # The size of the maximum independent set in a random dense graph is small (log N).
    # Here, the structure is specific.
    
    # Let's implement a simple B&B with bitmasks.
    # We'll use a global best score and best set.
    
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
        
    # Precompute sum of weights for any subset? No, 2^256 is too big.
    # We will compute the upper bound dynamically or use a greedy approximation for the bound.
    # A simple bound is sum of weights of all candidates.
    # A better bound is the weight of a greedy independent set in the candidate subgraph.
    
    best_score = 0.0
    best_set_mask = 0
    
    # Sort vertices by weight descending? Or by degree?
    # Let's try sorting by weight descending to find good solutions early.
    # But we need to maintain the bitmask logic.
    
    # Let's use a recursive B&B.
    # To avoid recursion depth issues and overhead, we might need to be careful.
    # Python recursion limit is usually 1000. Depth is at most 256. Should be fine.
    
    # Optimization: Use a list of active vertices.
    # We will represent the candidate set as a bitmask.
    
    # Helper to get the sum of weights of a bitmask.
    # This is slow if done naively.
    # We can precompute prefix sums? No, bitmask is arbitrary.
    # We can iterate through bits.
    
    def popcount(x):
        return bin(x).count("1")
    
    def weight_sum(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    # To speed up weight_sum, we can precompute weights for chunks of 8 bits?
    # Or just use the fact that we are adding/removing single bits often.
    # In B&B, we pass the current weight.
    
    def solve(candidate_mask, current_mask, current_weight, depth):
        nonlocal best_score, best_set_mask
        
        # Pruning
        # Upper bound: current_weight + sum of weights in candidate_mask
        # We can estimate the sum by iterating bits.
        # To make this faster, we can precompute a "greedy bound" or just use the exact sum if the candidate set is small.
        # If candidate set is large, the sum is an overestimate, which is fine for pruning.
        
        # Optimization: If candidate_mask is 0, update best.
        if candidate_mask == 0:
            if current_weight > best_score:
                best_score = current_weight
                best_set_mask = current_mask
            return
        
        # Calculate upper bound
        # To avoid iterating all bits every time, we can maintain the sum?
        # But candidate_mask changes arbitrarily.
        # Let's just iterate. 256 bits is fast.
        ub = current_weight
        cm = candidate_mask
        while cm:
            lsb = cm & -cm
            idx = lsb.bit_length() - 1
            ub += weights[idx]
            cm ^= lsb
            
        if ub <= best_score:
            return
            
        # Choose a vertex to branch on.
        # Heuristic: Pick the vertex with the highest weight in the candidate set?
        # Or the one with the highest degree?
        # High weight is good for finding good solutions early.
        # Let's pick the highest weight vertex.
        
        # Find highest weight vertex in candidate_mask
        best_v = -1
        best_w = -1.0
        cm = candidate_mask
        while cm:
            lsb = cm & -cm
            idx = lsb.bit_length() - 1
            if weights[idx] > best_w:
                best_w = weights[idx]
                best_v = idx
            cm ^= lsb
            
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        # New current_mask = current_mask | (1 << best_v)
        # New current_weight = current_weight + weights[best_v]
        # New candidate_mask = candidate_mask & ~(adj[best_v] | (1 << best_v))
        
        new_candidate = candidate_mask & ~(adj[best_v] | (1 << best_v))
        new_current = current_mask | (1 << best_v)
        new_weight = current_weight + weights[best_v]
        
        solve(new_candidate, new_current, new_weight, depth + 1)
        
        # Branch 2: Exclude best_v
        # New candidate_mask = candidate_mask & ~(1 << best_v)
        # Current remains same
        
        new_candidate_ex = candidate_mask & ~(1 << best_v)
        solve(new_candidate_ex, current_mask, current_weight, depth + 1)

    # Initial candidate set: all vertices 1..N
    initial_mask = 0
    for i in range(1, N + 1):
        initial_mask |= (1 << i)
        
    # To help the solver, we can provide a good initial lower bound (best_score).
    # Let's run a greedy heuristic first to get a decent start.
    
    def greedy_init():
        mask = initial_mask
        curr_mask = 0
        curr_w = 0.0
        while mask:
            # Pick highest weight vertex
            best_v = -1
            best_w = -1.0
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_v = idx
                m ^= lsb
            if best_v == -1: break
            curr_mask |= (1 << best_v)
            curr_w += weights[best_v]
            mask &= ~(adj[best_v] | (1 << best_v))
        return curr_mask, curr_w

    g_mask, g_w = greedy_init()
    best_score = g_w
    best_set_mask = g_mask

    # Run B&B
    # We need a time limit.
    start_time = time.time()
    timeout = 85.0
    
    # The recursive solve doesn't check time.
    # We need to modify solve to check time and abort if exceeded.
    
    def solve_tbb(candidate_mask, current_mask, current_weight, depth):
        nonlocal best_score, best_set_mask
        
        if time.time() - start_time > timeout:
            return False # Abort
            
        if candidate_mask == 0:
            if current_weight > best_score:
                best_score = current_weight
                best_set_mask = current_mask
            return True
            
        # Upper bound
        ub = current_weight
        cm = candidate_mask
        while cm:
            lsb = cm & -cm
            idx = lsb.bit_length() - 1
            ub += weights[idx]
            cm ^= lsb
            
        if ub <= best_score:
            return True
            
        # Choose vertex
        best_v = -1
        best_w = -1.0
        cm = candidate_mask
        while cm:
            lsb = cm & -cm
            idx = lsb.bit_length() - 1
            if weights[idx] > best_w:
                best_w = weights[idx]
                best_v = idx
            cm ^= lsb
            
        if best_v == -1:
            return True
            
        # Branch 1: Include
        new_candidate = candidate_mask & ~(adj[best_v] | (1 << best_v))
        new_current = current_mask | (1 << best_v)
        new_weight = current_weight + weights[best_v]
        
        if not solve_tbb(new_candidate, new_current, new_weight, depth + 1):
            return False
            
        # Branch 2: Exclude
        new_candidate_ex = candidate_mask & ~(1 << best_v)
        
        if not solve_tbb(new_candidate_ex, current_mask, current_weight, depth + 1):
            return False
            
        return True

    # Increase recursion limit just in case
    sys.setrecursionlimit(3000)
    
    solve_tbb(initial_mask, 0, 0.0, 0)
    
    # Extract result from best_set_mask
    result = []
    m = best_set_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx)
        m ^= lsb
        
    if not result:
        return [1]
        
    return result
