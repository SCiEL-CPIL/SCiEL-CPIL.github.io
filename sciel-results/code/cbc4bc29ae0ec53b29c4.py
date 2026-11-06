import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph with no regular induced subgraph on 6 or more vertices.
    Strategy:
    1. Use Simulated Annealing / Local Search on the edge set of a graph with n=22.
    2. Objective: Minimize the number of regular induced subgraphs of size >= 6.
    3. If a valid graph (0 violations) is found, return it.
    4. If time runs out, return the best found (hopefully n=22 or n=21 if 22 fails).
    
    Why this is different:
    - Instead of generating random graphs and verifying (which might miss valid structures or be slow to verify if done naively), 
      we start with a random graph and actively repair violations.
    - We use bitmasks for fast degree calculations.
    - We target the smallest violations (size 6) first, as they are the most common and hardest to avoid.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # We aim for n=22 first. If that fails, we can drop to 21 or 20.
    # Let's try to find a valid graph on n=22.
    # If we find one, score is 22.
    # If we find one on n=21, score is 21.
    # If we find one on n=20, score is 20.
    
    # Representation: List of integers, where bit i of adj[j] is 1 if edge (j,i) exists.
    # For n=22, we need 22 bits. Python ints handle this fine.
    
    def count_violations(adj, n):
        """
        Counts the number of subsets of size >= 6 that induce a regular subgraph.
        Returns the count. 0 means valid.
        """
        violations = 0
        # We check sizes 6 to n.
        # For efficiency, we can stop if we find too many, but for annealing we need the count.
        # Actually, for local search, we just need to know IF there is a violation and WHICH one.
        # Let's modify this to return a violating subset if found, else None.
        # This is faster than counting all.
        
        # Check size 6 first (most common)
        # Then 7, etc.
        
        # Optimization: Pre-compute bitmasks for all subsets of size 6?
        # No, that's too many.
        
        # Iterate through all combinations of 6 vertices.
        # There are C(22, 6) = 74,613 subsets.
        # For each, check if regular.
        
        # To speed up, we can sample? No, we need to be sure.
        # 74k checks * 6 popcounts = ~450k ops. Very fast in C, but in Python?
        # 74k iterations in Python is ~0.05s. 
        # If we do this 1000 times, it's 50s. Too slow for pure Python loop.
        
        # We need a faster check.
        # Use NumPy? 
        # Or, we can be smarter: 
        # A subset S is regular if all vertices in S have the same degree within S.
        
        # Let's stick to Python bit ops but optimize the loop.
        
        for size in range(6, n + 1):
            for S in combinations(range(n), size):
                # Check if S is regular
                # Get degrees of each vertex in S within S
                # We can do this by:
                # mask_S = sum(1<<v for v in S)
                # degs = [bin(adj[v] & mask_S).count('1') for v in S]
                # if len(set(degs)) == 1: return S
                
                mask_S = 0
                for v in S:
                    mask_S |= (1 << v)
                
                # Check degrees
                first_deg = -1
                is_reg = True
                for v in S:
                    deg = bin(adj[v] & mask_S).count('1')
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return S # Return the first violation found
        
        return None # No violation found

    def get_violation(adj, n):
        """
        Returns a tuple (subset, size) of a regular induced subgraph, or None.
        Optimized to check smaller sizes first.
        """
        # Check size 6
        # To speed up, we can randomize the order of combinations?
        # Or just iterate.
        
        # Let's try to make this faster.
        # If we use a list of pre-computed masks for size 6?
        # There are 74,613 masks. Storing them is fine.
        # But iterating is the cost.
        
        # Alternative: 
        # Randomly sample subsets? 
        # If the graph is "close" to valid, there are few violations. 
        # If it's random, there are many. 
        # Finding one quickly is key for repair.
        
        # Let's just iterate. 74k checks is fast enough if we don't do it too often.
        # But in annealing, we might do it many times.
        
        # Let's try to limit the check to size 6 and 7. 
        # If we fix all size 6 and 7 violations, larger ones are less likely? 
        # Not necessarily, but it's a good heuristic.
        
        # Actually, if a graph has no regular induced subgraph of size 6, it might still have one of size 8.
        # But size 6 is the "base" constraint.
        
        # Let's check size 6 first.
        # If none, check size 7.
        # If none, check size 8...
        
        # To save time, if we are in the "annealing" phase, we might just check size 6.
        # If we find no size 6 violations, we assume it's valid? 
        # NO, that's risky. A size 8 regular subgraph is a violation.
        # However, if we are minimizing violations, fixing size 6 is the priority.
        
        # Let's do a full check but with a time limit inside?
        # No, let's just do size 6 and 7. 
        # If we pass size 6 and 7, we do a full check.
        
        # For now, let's just check size 6. It's the most frequent.
        # If we clear all size 6, we are in a good state.
        
        # Let's randomize the vertex order to avoid getting stuck in local minima of the iteration order.
        # Actually, the set of combinations is fixed.
        
        # Let's just iterate.
        for size in range(6, n + 1):
            # Shuffle the combinations? No, combinations is deterministic.
            # We can pick a random subset of combinations to check?
            # No, we need to find A violation.
            
            # Optimization: 
            # If we are checking size 6, there are ~75k subsets.
            # Let's just do it.
            
            for S in combinations(range(n), size):
                mask_S = 0
                for v in S:
                    mask_S |= (1 << v)
                
                first_deg = -1
                is_reg = True
                for v in S:
                    deg = bin(adj[v] & mask_S).count('1')
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return S
        
        return None

    # Main Loop: Simulated Annealing / Local Search
    n_target = 22
    n = n_target
    
    # Initialize random graph
    random.seed(42) # Deterministic start
    adj = [0] * n
    for i in range(n):
        for j in range(i):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_adj = None
    best_n = 0
    
    # We will try to find a valid graph.
    # If we find one for n=22, we return it.
    # If we don't, we try n=21, then n=20.
    
    for target_n in [22, 21, 20]:
        if time.time() > deadline - 10:
            break
            
        n = target_n
        # Re-initialize for this n
        adj = [0] * n
        for i in range(n):
            for j in range(i):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Local Search
        # We want to minimize violations.
        # Since we can't count them easily, we just try to eliminate them.
        
        # If the initial graph is valid, return immediately.
        if get_violation(adj, n) is None:
            best_adj = adj[:]
            best_n = n
            break
        
        # Otherwise, do local search.
        # We will flip edges to break violations.
        
        # Temperature / Acceptance probability?
        # Let's just do greedy: 
        # 1. Find a violation.
        # 2. Pick a random vertex in the violation.
        # 3. Pick a random neighbor (or non-neighbor) to flip edge with.
        # 4. If the new graph has fewer violations (or same, to escape minima), accept.
        
        # Since counting violations is expensive, we'll use a heuristic:
        # Just flip an edge and check if the *specific* violation is gone.
        # This is a "hill climbing" on the specific violation.
        
        current_adj = adj[:]
        
        # Iterate for a fixed number of steps or until time runs out
        # Let's allow 5 seconds per target_n.
        step_deadline = time.time() + 5.0
        
        for _ in range(10000):
            if time.time() > step_deadline:
                break
                
            viol = get_violation(current_adj, n)
            if viol is None:
                best_adj = current_adj[:]
                best_n = n
                break
            
            # Pick a random vertex from the violation
            v = random.choice(viol)
            
            # Pick a random other vertex to flip edge with
            # To be effective, we should flip an edge that connects v to a vertex in the violation, 
            # or to a vertex outside, to change the degree of v within the violation.
            
            # Flipping an edge (v, u) where u is in viol changes deg(v) and deg(u) in the subgraph.
            # If we flip (v, u) where u is NOT in viol, it only changes deg(v) in the subgraph.
            # This is often better to break regularity without creating new ones elsewhere?
            
            # Let's try flipping with a vertex in the violation first.
            candidates = [u for u in viol if u != v]
            if candidates:
                u = random.choice(candidates)
            else:
                u = random.randint(0, n - 1)
                while u == v:
                    u = random.randint(0, n - 1)
            
            # Flip edge (v, u)
            if (current_adj[v] >> u) & 1:
                current_adj[v] &= ~(1 << u)
                current_adj[u] &= ~(1 << v)
            else:
                current_adj[v] |= (1 << u)
                current_adj[u] |= (1 << v)
            
            # Check if this flip helped.
            # We don't want to check the whole graph every time (too slow).
            # We can check if the *same* violation is still there.
            # If it is, maybe revert? Or keep going?
            # Let's just keep going. The random flips will eventually break it.
            
        # After local search, if we have a valid graph, break.
        if best_adj is not None and best_n == n:
            break

    # If we didn't find a valid graph for 22, 21, or 20, 
    # we should return the best we have.
    # But if best_adj is None, we need a fallback.
    # The problem states n>=1.
    # A graph with 5 vertices is always valid (no subset of size 6).
    # But we want to maximize n.
    
    if best_adj is None:
        # Fallback: Return a valid graph on n=5 (trivial) or try a very fast search for n=10?
        # Let's just return n=5 to be safe, or n=0? 
        # "1 <= n <= 22".
        # Let's return n=5 empty graph.
        return {'n': 5, 'edges': []}
    
    # Convert adj to edges
    edges = []
    for i in range(best_n):
        for j in range(i):
            if (best_adj[i] >> j) & 1:
                edges.append([j, i])
                
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
