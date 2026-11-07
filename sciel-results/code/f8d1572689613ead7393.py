"""Rewritten program: exact cochromatic number via branch-and-bound over homogeneous sets."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add(tuple(e))
        edge_set.add(tuple(reversed(e)))
    
    # adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute: for each subset, whether it's a clique or independent set
    # Represent subsets as bitmasks (int)
    # A subset is a clique if for all u in subset, (subset without u) & adj[u] == subset without u
    # A subset is independent if for all u in subset, (subset without u) & adj[u] == 0
    
    def is_clique(mask):
        if mask == 0:
            return True
        # For each vertex in mask, all other vertices in mask must be adjacent
        m = mask
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            if (mask ^ lsb) & ~adj[u]:  # any non-neighbor in the rest
                return False
            m ^= lsb
        return True
    
    def is_independent(mask):
        if mask == 0:
            return True
        m = mask
        while m:
            lsb = m & (-m)
            u = lsb.bit_length() - 1
            if (mask ^ lsb) & adj[u]:  # any neighbor in the rest
                return False
            m ^= lsb
        return True
    
    def is_homogeneous(mask):
        return is_clique(mask) or is_independent(mask)
    
    # We need to partition the full set (all n vertices) into k homogeneous sets, minimizing k.
    # Full mask:
    full_mask = (1 << n) - 1
    
    # Strategy: try k=1,2,... and for each k, try to find k homogeneous sets whose union is full_mask and are disjoint.
    # Use backtracking with memoization / branch and bound.
    
    # Precompute all homogeneous subsets for faster lookup
    # There are 2^14 = 16384 subsets, manageable.
    homo_cache = {}
    for mask in range(1, 1 << n):
        homo_cache[mask] = is_homogeneous(mask)
    
    def can_cover(mask, k, deadline):
        """Can 'mask' be covered by k disjoint homogeneous sets?"""
        if time.time() > deadline:
            return None  # timeout signal
        if k == 0:
            return mask == 0
        if mask == 0:
            return True  # remaining k sets can be empty (but we need nonempty parts...)
            # Actually, we need exactly k nonempty parts. Let's handle this differently.
        
        # If mask is homogeneous, we can cover it with 1 set (and k-1 more sets for the rest, but rest is empty)
        # Wait, we're covering 'mask' with exactly k nonempty homogeneous sets.
        # If k==1, mask must be homogeneous and nonempty.
        if k == 1:
            return mask != 0 and homo_cache[mask]
        
        # Branch: pick the lowest set bit in mask as the first vertex to place
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        
        # Try all homogeneous subsets of mask that contain v
        # Iterate over submasks of mask that contain v
        rest = mask ^ lsb
        # submask must contain v, so submask = lsb | (submask of rest)
        # We want homogeneous subsets containing v
        # Enumerate all submasks of rest, combine with lsb, check if homogeneous
        
        # To prune: we need to leave a remainder that can be covered by k-1 sets
        # The remainder is mask ^ chosen_set
        
        # Optimization: try larger sets first (greedy heuristic) to reduce branching
        # But for correctness, we need to try all.
        
        # Let's iterate over all submasks of rest in some order
        # Number of submasks of rest is 2^(popcount(rest))
        # For n=14, worst case rest has 13 bits, 8192 submasks. Manageable.
        
        r = rest
        # We'll collect candidate submasks (subsets of rest) to try
        # To improve pruning, try submasks that result in a remainder that is "easier" to cover
        # But for now, let's just iterate
        
        # Optimization: if the remainder (mask ^ (lsb | sub)) has more than (k-1) * max_homo_size vertices... 
        # Actually, simpler: just try all.
        
        # Sort submasks by size descending? Or just iterate.
        # Let's iterate over all submasks of rest
        sub = rest
        while True:
            candidate = lsb | sub
            if homo_cache[candidate]:
                remainder = mask ^ candidate
                if remainder == 0:
                    # We used 1 set to cover everything, but we need k sets total.
                    # This is only valid if k==1, but we're in k>1 branch.
                    # So remainder must be nonempty for k>1? No, if remainder is 0, we covered mask with 1 set.
                    # But we need k sets. If k>1 and remainder is 0, we need to split the set into more parts?
                    # No, the sets in the partition must be nonempty and disjoint, covering mask.
                    # If we use 1 set to cover all of mask, that's only 1 set. We need k sets.
                    # So this branch is invalid unless k==1 (handled above).
                    # So skip if remainder == 0 and k > 1.
                    pass
                else:
                    result = can_cover(remainder, k - 1, deadline)
                    if result is None:
                        return None
                    if result:
                        return True
            if sub == 0:
                break
            sub = (sub - 1) & rest
        
        return False
    
    # Find minimum k
    deadline = time.time() + 50  # 50 second safety margin
    
    # Try k from 1 to n
    best_partition = None
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        # Check if k homogeneous sets can cover full_mask
        if can_cover(full_mask, k, deadline):
            # Found a cover with k sets. Now we need to actually construct the partition.
            # Re-do the search to record the actual sets.
            partition = []
            mask = full_mask
            remaining_k = k
            while remaining_k > 0 and mask != 0:
                if time.time() > deadline:
                    break
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                rest = mask ^ lsb
                found = False
                sub = rest
                while True:
                    candidate = lsb | sub
                    if homo_cache[candidate]:
                        remainder = mask ^ candidate
                        if remainder == 0:
                            if remaining_k == 1:
                                partition.append(candidate)
                                found = True
                                break
                        else:
                            # Check if remainder can be covered by remaining_k - 1 sets
                            # We need a recursive check that also returns the partition
                            # For simplicity, let's just verify with can_cover
                            if can_cover(remainder, remaining_k - 1, deadline):
                                partition.append(candidate)
                                mask = remainder
                                remaining_k -= 1
                                found = True
                                break
                    if sub == 0:
                        break
                    sub = (sub - 1) & rest
                if not found:
                    # Should not happen if can_cover returned True
                    break
            if len(partition) == k:
                best_partition = partition
                break
    
    if best_partition is None:
        # Fallback: greedy first-fit (same as original)
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                hits = [(trial[i], trial[j]) in edge_set or (trial[j], trial[i]) in edge_set 
                        for i in range(len(trial)) for j in range(i+1, len(trial))]
                if all(hits) or not any(hits):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        best_partition = parts
    
    # Convert to list of lists of ints
    result = []
    for p in best_partition:
        if isinstance(p, int):  # bitmask
            vertices = [i for i in range(n) if p & (1 << i)]
            result.append(vertices)
        else:
            result.append([int(v) for v in p])
    
    return result

# EVOLVE-BLOCK-END
