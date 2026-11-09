import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Start time for deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit to be safe under 60s/90s
    
    # Build adjacency masks
    # adj_mask[v] is a bitmask of vertices adjacent to v
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute valid subsets (cliques or independent sets)
    # valid[mask] is True if mask is a clique or independent set
    # We also want to group them by the lowest set bit for DP efficiency
    # lowest_bit_index[mask] -> index of the lowest set bit
    # valid_subsets_by_lsb[v] -> list of masks S such that v is the lowest set bit of S and S is valid
    
    # First, determine validity for all masks
    is_valid = [False] * (1 << n)
    is_valid[0] = True # Empty set is both clique and independent, but we don't use it as a part
    
    for mask in range(1, 1 << n):
        # Check if clique
        # A set is a clique if for every v in mask, adj_mask[v] & (mask ^ (1<<v)) == mask ^ (1<<v)
        # Or simpler: pick lowest bit v. Check if v is connected to all others.
        # Then check if rest is a clique.
        # Let's use the recursive property for speed
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if rest == 0:
            is_valid[mask] = True # Single vertex is both
            continue
            
        # Check Clique property for v against rest
        # v must be connected to all in rest
        if (adj_mask[v] & rest) == rest:
            # v is connected to all in rest. If rest is a clique, mask is a clique.
            if is_valid[rest] and True: # We need to know if rest is specifically a clique?
                # Actually, we just need to know if mask is a clique OR independent set.
                # Let's precompute is_clique and is_independent separately to be safe and clear.
                pass
                
    # Let's re-do validity calculation with explicit is_clique and is_independent
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Clique: v connected to all in rest AND rest is clique
        if (adj_mask[v] & rest) == rest:
            is_clique[mask] = is_clique[rest]
        else:
            is_clique[mask] = False
            
        # Independent: v not connected to any in rest AND rest is independent
        if (adj_mask[v] & rest) == 0:
            is_independent[mask] = is_independent[rest]
        else:
            is_independent[mask] = False
            
        is_valid[mask] = is_clique[mask] or is_independent[mask]
        
    # Group valid masks by their lowest set bit
    # This allows us to iterate only over valid subsets containing the specific lowest vertex of the current DP state
    valid_subsets_by_lsb = [[] for _ in range(n)]
    for mask in range(1, 1 << n):
        if is_valid[mask]:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            valid_subsets_by_lsb[v].append(mask)
            
    # DP to find minimum number of parts
    # dp[mask] = min parts to partition vertices in mask
    # choice[mask] = the subset S chosen for the lowest bit in mask
    INF = 10**9
    dp = [INF] * (1 << n)
    choice = [0] * (1 << n)
    dp[0] = 0
    
    for mask in range(1, 1 << n):
        # Time check
        if time.time() > deadline:
            # Fallback: return a valid partition, e.g., singletons
            # But we should try to return best found so far if possible?
            # Since we are in the middle of DP, we might not have a complete path for full_mask.
            # However, singletons are always a valid partition (14 parts).
            # We can break and return singletons.
            return [[i] for i in range(n)]
            
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        
        # Try all valid subsets S containing v and subset of mask
        best_val = INF
        best_S = 0
        
        for S in valid_subsets_by_lsb[v]:
            if (S & mask) == S: # S is subset of mask
                prev = mask ^ S
                if dp[prev] + 1 < best_val:
                    best_val = dp[prev] + 1
                    best_S = S
                    # Early termination? If best_val is 1, we can't do better.
                    if best_val == 1:
                        break
        
        dp[mask] = best_val
        choice[mask] = best_S
        
    # Reconstruct partition
    if dp[(1 << n) - 1] == INF:
        # Should not happen, but fallback
        return [[i] for i in range(n)]
        
    partition = []
    cur_mask = (1 << n) - 1
    while cur_mask:
        S = choice[cur_mask]
        if S == 0:
             # Error in reconstruction, fallback
             return [[i] for i in range(n)]
        # Extract vertices from S
        part = []
        m = S
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part.append(v)
            m ^= lsb
        partition.append(part)
        cur_mask ^= S
        
    return partition

if __name__ == "__main__":
    # Test run
    res = construct()
    print(res)
