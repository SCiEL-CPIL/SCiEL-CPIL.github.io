import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Fast Verifier using Bitmasks
    def verify_no_regular_subgraph(adj_masks, n):
        """
        Returns True if no subset of size >= 6 induces a regular graph.
        adj_masks: list of ints, where bit j is 1 if edge (i, j) exists.
        """
        # Check sizes from 6 to n
        for size in range(6, n + 1):
            if time.time() > deadline - 2:
                return False # Timeout, assume invalid to be safe in search context? 
                             # Actually, if we are verifying a candidate, we must be sure.
                             # But in local search, we just need to find violations.
                             # Let's return a special value or just False.
            
            for subset in combinations(range(n), size):
                mask_subset = 0
                for v in subset:
                    mask_subset |= (1 << v)
                
                # Check if regular
                degs = []
                for v in subset:
                    # Count bits in adj_masks[v] & mask_subset
                    # Python int bit_count is fast
                    deg = (adj_masks[v] & mask_subset).bit_count()
                    degs.append(deg)
                
                if len(set(degs)) == 1:
                    return False
        
        return True

    def find_violation(adj_masks, n):
        """
        Returns (subset, size) of a regular induced subgraph of size >= 6, or None.
        """
        # Check smaller sizes first as they are more likely to exist in random graphs
        for size in range(6, n + 1):
            if time.time() > deadline - 2:
                return None # Time out, assume no violation found (risky, but prevents hang)
            
            for subset in combinations(range(n), size):
                mask_subset = 0
                for v in subset:
                    mask_subset |= (1 << v)
                
                degs = []
                for v in subset:
                    deg = (adj_masks[v] & mask_subset).bit_count()
                    degs.append(deg)
                
                if len(set(degs)) == 1:
                    return subset, size
        return None

    def to_edge_list(adj_masks, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def random_graph_masks(n, p, seed):
        rng = random.Random(seed)
        masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        return masks

    def local_search(target_n, seed, time_limit):
        """
        Attempt to find a valid graph on target_n vertices.
        Returns (adj_masks, edges) or None.
        """
        start = time.time()
        rng = random.Random(seed)
        
        # Initialize
        # Try density 0.5 first
        adj_masks = random_graph_masks(target_n, 0.5, seed)
        
        # Check if already valid
        if verify_no_regular_subgraph(adj_masks, target_n):
            return adj_masks, to_edge_list(adj_masks, target_n)
        
        # Local Search
        # We want to minimize the number of violations.
        # Since we can only find one violation at a time efficiently, we use a greedy approach.
        # Pick a random violation, flip a random edge within it to break regularity.
        
        for _ in range(5000): # Max iterations
            if time.time() > time_limit:
                break
            
            viol = find_violation(adj_masks, target_n)
            if viol is None:
                # No violation found. Verify fully to be sure.
                if verify_no_regular_subgraph(adj_masks, target_n):
                    return adj_masks, to_edge_list(adj_masks, target_n)
                else:
                    # Verification failed but find_violation didn't find one? 
                    # This shouldn't happen if find_violation checks all sizes.
                    # Maybe it timed out?
                    continue
            
            subset, size = viol
            # Pick a random edge within the subset to flip
            i_idx, j_idx = rng.sample(range(size), 2)
            u, v = subset[i_idx], subset[j_idx]
            
            # Flip edge (u, v)
            bit = (1 << v)
            adj_masks[u] ^= bit
            bit = (1 << u)
            adj_masks[v] ^= bit
            
            # Occasionally flip a random edge outside to escape local minima
            if rng.random() < 0.2:
                a, b = rng.sample(range(target_n), 2)
                bit_a = (1 << b)
                adj_masks[a] ^= bit_a
                bit_b = (1 << a)
                adj_masks[b] ^= bit_b

        # Final check
        if verify_no_regular_subgraph(adj_masks, target_n):
            return adj_masks, to_edge_list(adj_masks, target_n)
        
        return None

    # 2. Search Strategy
    # Try n=21 first, then n=20, etc.
    # Use multiple seeds for robustness.
    
    best_result = None
    best_n = 0
    
    # Try n=21
    if time.time() < deadline - 20:
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            res = local_search(21, seed, deadline - 5)
            if res:
                adj_masks, edges = res
                # Double check
                if verify_no_regular_subgraph(adj_masks, 21):
                    best_result = {'n': 21, 'edges': edges}
                    best_n = 21
                    break # Found 21, return immediately as it's max possible for "improvement" over 20? 
                    # Task says 20 is known. 21 would be an improvement.
                    # If we find 21, we are done.
                    return best_result

    # Try n=20
    if best_n < 20:
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            res = local_search(20, seed, deadline - 5)
            if res:
                adj_masks, edges = res
                if verify_no_regular_subgraph(adj_masks, 20):
                    best_result = {'n': 20, 'edges': edges}
                    best_n = 20
                    break

    # Fallback: Try smaller n if 20 failed
    if best_n < 20:
        for target_n in [19, 18, 17, 16]:
            if time.time() > deadline - 5:
                break
            for seed in range(5):
                if time.time() > deadline - 2:
                    break
                res = local_search(target_n, seed, deadline - 2)
                if res:
                    adj_masks, edges = res
                    if verify_no_regular_subgraph(adj_masks, target_n):
                        best_result = {'n': target_n, 'edges': edges}
                        best_n = target_n
                        break
            if best_n >= target_n:
                break

    # Absolute Fallback: Small random graph
    if best_result is None:
        # Generate a small graph that is likely valid
        # n=10 random graph is very likely valid
        adj_masks = random_graph_masks(10, 0.5, 42)
        # Verify
        if not verify_no_regular_subgraph(adj_masks, 10):
            # Try a few more
            for s in range(1, 100):
                adj_masks = random_graph_masks(10, 0.5, s)
                if verify_no_regular_subgraph(adj_masks, 10):
                    break
        best_result = {'n': 10, 'edges': to_edge_list(adj_masks, 10)}
        best_n = 10

    return best_result

# EVOLVE-BLOCK-END
