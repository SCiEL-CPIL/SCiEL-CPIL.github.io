"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

# Precompute adjacency bitmask
N = P['n']
ES = P['edges']
ADJ = [0] * N
for u, v in ES:
    ADJ[u] |= (1 << v)
    ADJ[v] |= (1 << u)

# Precompute clique and independent set checks for all subsets
# A subset S (bitmask) is a clique if for every v in S, S & ADJ[v] contains S\{v}
# A subset S is an independent set if for every v in S, S & ADJ[v] == 0

def is_clique_mask(mask):
    """Check if the set of vertices indicated by mask forms a clique."""
    # For each vertex in mask, all other vertices in mask must be adjacent
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        # All other vertices in mask must be adjacent to v
        others = mask ^ lsb
        if (ADJ[v] & others) != others:
            return False
        m ^= lsb
    return True

def is_independent_mask(mask):
    """Check if the set of vertices indicated by mask forms an independent set."""
    m = mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        if ADJ[v] & (mask ^ lsb):
            return False
        m ^= lsb
    return True

# Precompute valid subsets (cliques or independent sets)
# For n=14, there are 2^14 = 16384 subsets, so this is feasible
VALID_SUBSETS = {}
for mask in range(1 << N):
    if is_clique_mask(mask) or is_independent_mask(mask):
        VALID_SUBSETS[mask] = True

# Enumerate all valid partitions using subset DP
# dp[mask] = minimum number of parts to partition the set of vertices in mask
# where each part is a valid subset (clique or independent set)
# and the parts are disjoint and cover mask

# We want to find the minimum number of parts
# This is equivalent to finding the minimum k such that we can partition all vertices into k valid subsets

# Use DP: dp[mask] = min over all valid submasks s of mask (dp[mask ^ s] + 1)
# But this is exponential in the number of valid submasks

# Alternative: iterate over k from 1 to N, and check if we can partition into k parts
# For small k, we can use backtracking

def can_partition_into_k_parts(k, deadline):
    """Check if we can partition all vertices into exactly k valid subsets."""
    full_mask = (1 << N) - 1
    best_time = time.time()
    
    # Use backtracking with memoization
    # State: (remaining_mask, parts_used)
    # We need to assign vertices to exactly k parts
    
    # Better approach: enumerate all valid subsets, then use inclusion-exclusion or DP
    # For k=2, 3, 4, we can do this efficiently
    
    if k == 1:
        return full_mask in VALID_SUBSETS
    
    if k == 2:
        # Check if there exists a valid subset s such that both s and full_mask^s are valid
        for s in range(1 << N):
            if s in VALID_SUBSETS and (full_mask ^ s) in VALID_SUBSETS:
                return True
        return False
    
    if k >= 3:
        # Use DP: dp[mask] = minimum number of valid subsets to cover mask
        # But we want exactly k, so we can check if dp[full_mask] <= k
        # Actually, we want to know if it's possible with exactly k, but since we can always
        # split a valid subset into smaller valid subsets (singletons are always valid),
        # if dp[full_mask] <= k, then we can achieve exactly k by splitting
        
        # Compute dp[mask] = min number of valid subsets to partition mask
        dp = [float('inf')] * (1 << N)
        dp[0] = 0
        
        for mask in range(1 << N):
            if dp[mask] == float('inf'):
                continue
            # Try all valid subsets of the complement
            remaining = full_mask ^ mask
            # Iterate over submasks of remaining
            sub = remaining
            while sub:
                if sub in VALID_SUBSETS:
                    new_mask = mask | sub
                    if dp[mask] + 1 < dp[new_mask]:
                        dp[new_mask] = dp[mask] + 1
                sub = (sub - 1) & remaining
                if time.time() > deadline:
                    return False
        
        return dp[full_mask] <= k
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Try k = 1, 2, 3, ... until we find a valid partition
    for k in range(1, N + 1):
        if time.time() > deadline - 5.0:
            break
        
        if can_partition_into_k_parts(k, deadline):
            # Now find an actual partition with k parts
            # Use backtracking to find the partition
            full_mask = (1 << N) - 1
            
            # dp[mask] = (min_parts, last_subset_used) for reconstruction
            # Or just do backtracking
            def backtrack(remaining_mask, parts_left, current_parts, deadline_bt):
                if parts_left == 0:
                    return remaining_mask == 0
                if time.time() > deadline_bt:
                    return None
                
                # Try all valid subsets of remaining_mask
                sub = remaining_mask
                while sub:
                    if sub in VALID_SUBSETS:
                        result = backtrack(remaining_mask ^ sub, parts_left - 1, current_parts + [sub], deadline_bt)
                        if result is not None:
                            return result
                    sub = (sub - 1) & remaining_mask
                
                return None
            
            result = backtrack(full_mask, k, [], deadline)
            if result is not None:
                # Convert bitmasks to vertex lists
                partition = []
                for mask in result:
                    vertices = [v for v in range(N) if mask & (1 << v)]
                    partition.append(vertices)
                return partition
    
    # Fallback: use the greedy algorithm from the seed
    es = {tuple(e) for e in ES}
    parts = []
    for v in range(N):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END
