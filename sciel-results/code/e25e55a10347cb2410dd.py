import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 12
    m = 40
    total_nodes = 1 << n
    
    # Precompute neighbors for each node
    # neighbors[u] contains all v such that u is a strict subset of v or v is a strict subset of u
    neighbors = [[] for _ in range(total_nodes)]
    for u in range(total_nodes):
        # u is subset of v: v = u | bit
        # v is subset of u: v = u & ~bit
        # It's faster to iterate submasks and supermasks
        # Submasks of u
        sub = u
        while sub:
            neighbors[u].append(sub)
            sub = (sub - 1) & u
        # Supermasks of u
        # v such that (v | u) == v and v != u
        # v = u | extra
        # extra can be any non-zero subset of ~u
        not_u = (total_nodes - 1) ^ u
        extra = not_u
        while extra:
            neighbors[u].append(u | extra)
            extra = (extra - 1) & not_u
            
    # Note: The above adds u's submasks (excluding 0? No, sub loop excludes 0 if we stop at 0? 
    # "while sub" stops when sub becomes 0. So 0 is not included as a submask of u unless u=0?
    # If u=0, sub=0, loop doesn't run. So 0 has no submasks. Correct.
    # If u!=0, 0 is a submask. Does the loop include 0?
    # sub = u. ... sub becomes 0. Loop terminates. So 0 is NOT included.
    # We need 0 to be a neighbor if u is a supermask of 0 (i.e., u != 0).
    # So we must add 0 manually if u != 0.
    
    # Let's redo neighbor generation more carefully.
    neighbors = [[] for _ in range(total_nodes)]
    for u in range(total_nodes):
        # Strict subsets
        sub = u
        while sub:
            neighbors[u].append(sub)
            sub = (sub - 1) & u
        if u != 0:
            neighbors[u].append(0)
            
        # Strict supersets
        not_u = (total_nodes - 1) ^ u
        extra = not_u
        while extra:
            neighbors[u].append(u | extra)
            extra = (extra - 1) & not_u

    # Initial solution: 
    # Pick a random center and its neighborhood
    random.seed(42)
    center = random.randint(0, total_nodes - 1)
    initial_set = set([center])
    # Add all neighbors
    for nb in neighbors[center]:
        initial_set.add(nb)
    # If still less than m, add random nodes
    while len(initial_set) < m:
        initial_set.add(random.randint(0, total_nodes - 1))
    
    # If more than m, remove random ones (shouldn't happen if center has < 40 neighbors, 
    # but max degree is 11+11=22, so 1+22=23 < 40. So we will add 17 random nodes.)
    
    S = list(initial_set)
    in_S = np.zeros(total_nodes, dtype=bool)
    for u in S:
        in_S[u] = True
        
    # Calculate initial degrees
    degrees = np.zeros(total_nodes, dtype=np.int32)
    for u in S:
        for v in neighbors[u]:
            if in_S[v]:
                degrees[v] += 1
                # Wait, degrees[v] should count neighbors of v in S.
                # If u is in S and u is neighbor of v, then v's degree increases.
                # So for each u in S, for each v in neighbors[u], degrees[v] += 1.
                # This is correct.
                
    # Current score
    # Score is sum of degrees of nodes in S, divided by 2?
    # No. degrees[v] is the number of neighbors of v in S.
    # Sum of degrees over all v in S is the number of edges * 2.
    # So score = sum(degrees[u] for u in S) // 2
    score = sum(degrees[u] for u in S) // 2
    
    best_score = score
    
    # Local Search
    # We will perform best-improvement swaps
    # To speed up finding the best swap, we can iterate all u in S and v not in S.
    # There are 40 * 4056 = 162,240 pairs.
    
    while time.time() < deadline:
        best_delta = 0
        best_u = -1
        best_v = -1
        
        # Find best swap
        # Optimization: We can precompute a list of nodes not in S
        not_in_S = [i for i in range(total_nodes) if not in_S[i]]
        
        for u in S:
            deg_u = degrees[u]
            for v in not_in_S:
                # Delta = deg(v, S \ {u}) - deg(u, S)
                # deg(v, S \ {u}) = degrees[v] - (1 if u is neighbor of v else 0)
                # Note: u is in S, v is not.
                # If u is neighbor of v, then u is counted in degrees[v].
                # Since we are removing u, we subtract 1 if they are connected.
                
                is_conn = (u & v) == u or (u & v) == v
                # Faster: (u | v) == v or (u | v) == u
                # Or just check if v in neighbors[u]
                # Since we have neighbors list, we could use a set? 
                # No, bit check is faster.
                
                if is_conn:
                    new_deg_v = degrees[v] - 1
                else:
                    new_deg_v = degrees[v]
                    
                delta = new_deg_v - deg_u
                
                if delta > best_delta:
                    best_delta = delta
                    best_u = u
                    best_v = v
                    
        if best_delta <= 0:
            # No improvement found. Perturb.
            # Swap 2 random pairs to escape local optimum
            if len(S) >= 4:
                u1, u2 = random.sample(S, 2)
                v1, v2 = random.sample(not_in_S, 2)
                
                # Apply swap u1->v1, u2->v2
                # Update in_S
                in_S[u1] = False
                in_S[u2] = False
                in_S[v1] = True
                in_S[v2] = True
                
                # Update degrees
                for w in neighbors[u1]:
                    degrees[w] -= 1
                for w in neighbors[u2]:
                    degrees[w] -= 1
                for w in neighbors[v1]:
                    degrees[w] += 1
                for w in neighbors[v2]:
                    degrees[w] += 1
                    
                S.remove(u1)
                S.remove(u2)
                S.append(v1)
                S.append(v2)
                
                score = sum(degrees[u] for u in S) // 2
                if score > best_score:
                    best_score = score
        else:
            # Apply best swap
            u = best_u
            v = best_v
            
            in_S[u] = False
            in_S[v] = True
            
            # Update degrees
            for w in neighbors[u]:
                degrees[w] -= 1
            for w in neighbors[v]:
                degrees[w] += 1
                
            S.remove(u)
            S.append(v)
            
            score += best_delta
            if score > best_score:
                best_score = score

    # Convert to list of lists
    result = [list(sorted(s)) for s in S]
    
    # Ensure types are int
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]
        
    return result

