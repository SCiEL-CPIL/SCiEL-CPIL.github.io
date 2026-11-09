"""Maximum Weight Independent Set via Max Weight Clique on Complement Graph.

Nodes: integers 1..511, weight w(a) = 1/(512 - a).
Edges in original graph: gcd(a, b) > 1.
Complement graph edges: gcd(a, b) == 1.
We find the Maximum Weight Clique in the Complement Graph.
"""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    N = 512
    n = N - 1
    # Vertices 0..510 correspond to a = 1..511
    # Weights
    w = [1.0 / (N - (i + 1)) for i in range(n)]
    
    # Build adjacency bitsets for the Coprime Graph (Complement of Non-Coprime)
    # adj[i] has bit j set if gcd(i+1, j+1) == 1
    adj = [0] * n
    for i in range(n):
        ai = i + 1
        mask = 0
        for j in range(n):
            if i != j and math.gcd(ai, j + 1) == 1:
                mask |= (1 << j)
        adj[i] = mask

    # Helper to sum weights of vertices in a mask
    def mask_weight_sum(mask):
        s = 0.0
        while mask:
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            s += w[idx]
            mask ^= lsb
        return s

    best_val = 0.0
    best_set = []
    
    start = time.time()
    TIME_LIMIT = 115.0
    node_count = 0

    # Recursive DFS for Max Weight Clique
    # R_mask: current clique
    # P_mask: candidates that can be added to R (coprime to all in R)
    # curr_val: weight of current clique
    # p_sum: sum of weights of vertices in P_mask
    def expand(R_mask, P_mask, curr_val, p_sum):
        nonlocal best_val, best_set, node_count
        
        if time.time() - start > TIME_LIMIT:
            return False # Timeout
        
        node_count += 1
        
        if P_mask == 0:
            if curr_val > best_val:
                best_val = curr_val
                # Reconstruct set
                # R_mask contains the indices
                temp = R_mask
                s = []
                while temp:
                    lsb = temp & (-temp)
                    idx = lsb.bit_length() - 1
                    s.append(idx + 1)
                    temp ^= lsb
                best_set = s
            return True

        # Upper Bound: curr_val + p_sum
        if curr_val + p_sum <= best_val + 1e-12:
            return True

        # We process candidates in P_mask.
        # To improve pruning, we should pick vertices that lead to high values first.
        # However, the standard approach is to iterate through P.
        # We can order P by weight to find good incumbents faster.
        
        # Create a list of vertices in P_mask, sorted by weight descending
        # This is expensive if done at every node. 
        # Alternative: Just pick the max weight vertex in P_mask first.
        
        # Let's pick the vertex with the highest weight in P_mask.
        # We can scan P_mask to find max weight.
        
        # Optimization: If we always pick the max weight vertex, we are doing a greedy descent?
        # No, we still branch. But picking high weight first helps update best_val.
        
        # Let's iterate through all vertices in P_mask.
        # To make the loop efficient, we can extract vertices.
        
        # We need to be careful with the order.
        # If we pick v, we recurse with P_mask & adj[v].
        # Then we remove v from P_mask and continue.
        
        # To maximize the chance of finding a good solution early,
        # we should process higher weight vertices first.
        
        # Extract vertices from P_mask
        # Since n is small (511), we can just scan.
        
        # Find max weight vertex in P_mask
        # Actually, let's just iterate through P_mask bits.
        # But we want to prune.
        
        # Strategy:
        # 1. Find v in P_mask with max weight.
        # 2. Recurse on v.
        # 3. Remove v from P_mask.
        # 4. Check bound again.
        # 5. Repeat.
        
        # This is essentially a depth-first search where we always dive into the heaviest candidate first.
        
        while P_mask:
            # Find max weight vertex in P_mask
            max_w = -1.0
            max_v = -1
            temp = P_mask
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                if w[idx] > max_w:
                    max_w = w[idx]
                    max_v = idx
                temp ^= lsb
            
            if max_v == -1:
                break

            v = max_v
            v_bit = 1 << v
            
            # Branch: Include v
            new_P_mask = P_mask & adj[v]
            new_P_sum = mask_weight_sum(new_P_mask)
            new_curr_val = curr_val + w[v]
            
            # Check bound before recursing
            if new_curr_val + new_P_sum > best_val + 1e-12:
                if not expand(R_mask | v_bit, new_P_mask, new_curr_val, new_P_sum):
                    return False

            # Branch: Exclude v
            P_mask ^= v_bit
            p_sum -= w[v]
            
            # Check if remaining can improve
            if curr_val + p_sum <= best_val + 1e-12:
                break

        return True

    all_mask = (1 << n) - 1
    total_weight = sum(w)
    
    # Initial call
    # We can start with an empty clique and all vertices as candidates
    expand(0, all_mask, 0.0, total_weight)
    
    return best_set
# EVOLVE-BLOCK-END
