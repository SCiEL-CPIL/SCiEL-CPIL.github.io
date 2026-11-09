import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin

    def get_all_6_subsets(n):
        return list(itertools.combinations(range(n), 6))

    def is_regular_subset(verts, adj_masks):
        """Check if induced subgraph on verts is regular."""
        mask = 0
        for v in verts:
            mask |= (1 << v)
        degs = set()
        for v in verts:
            deg = (adj_masks[v] & mask).bit_count()
            degs.add(deg)
            if len(degs) > 1:
                return False
        return True

    def check_full_validity(n, adj_masks):
        """Check for any regular induced subgraph of size >= 6."""
        # Check size 6
        # We assume score was 0, so no regular 6-subgraphs.
        # But we should verify or just check k > 6.
        # Actually, if we rely on local search score, it counts regular 6-subgraphs.
        # If score is 0, there are no regular 6-subgraphs.
        # We must check k > 6.
        
        # Optimization: If a graph has no regular 6-subgraph, it's unlikely to have a regular 7+ subgraph?
        # Not necessarily (C7 example).
        # We need to check all k >= 7.
        
        for k in range(7, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                degs = set()
                for v in comb:
                    deg = (adj_masks[v] & mask).bit_count()
                    degs.add(deg)
                    if len(degs) > 1:
                        break
                else:
                    # All degrees same
                    return False
        return True

    def local_search(n, time_budget):
        t_start = time.time()
        
        # Precompute 6-subsets
        subsets_6 = get_all_6_subsets(n)
        # Map subset to index for fast lookup? 
        # Or just iterate through subsets containing u,v.
        
        # For each 6-subset, we need to know which edges it contains?
        # No, we just check regularity.
        
        # To speed up delta calculation:
        # For a given edge (u, v), which 6-subsets contain both u and v?
        # We can precompute a list of subsets for each pair (u, v).
        # There are n*(n-1)/2 pairs.
        # Each pair is in binom(n-2, 4) subsets.
        # For n=20, 190 pairs * 3060 subsets = 580k entries. Memory is fine.
        
        subsets_by_edge = {}
        for u in range(n):
            for v in range(u + 1, n):
                subsets_by_edge[(u, v)] = [s for s in subsets_6 if u in s and v in s]
        
        # Random init
        adj_masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        
        # Initial score
        current_score = sum(1 for s in subsets_6 if is_regular_subset(s, adj_masks))
        
        best_score = current_score
        best_adj = adj_masks[:]
        
        if current_score == 0:
             if check_full_validity(n, adj_masks):
                 return {'n': n, 'edges': [[u, v] for u in range(n) for v in range(u+1, n) if (adj_masks[u] >> v) & 1]}
        
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        while time.time() < t_start + time_budget:
            if time.time() > deadline:
                break
                
            u, v = random.choice(edges)
            
            # Get subsets affected
            affected_subsets = subsets_by_edge.get((u, v), [])
            
            # Count regular before
            reg_before = 0
            for s in affected_subsets:
                if is_regular_subset(s, adj_masks):
                    reg_before += 1
            
            # Flip edge
            adj_masks[u] ^= (1 << v)
            adj_masks[v] ^= (1 << u)
            
            # Count regular after
            reg_after = 0
            for s in affected_subsets:
                if is_regular_subset(s, adj_masks):
                    reg_after += 1
            
            delta = reg_after - reg_before
            new_score = current_score + delta
            
            if new_score < current_score or (new_score == current_score and random.random() < 0.1):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj_masks[:]
                    
                if current_score == 0:
                    # Check full validity
                    if check_full_validity(n, adj_masks):
                        return {'n': n, 'edges': [[u, v] for u in range(n) for v in range(u+1, n) if (adj_masks[u] >> v) & 1]}
            else:
                # Revert
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
                
        # Return best found if it was valid? 
        # We only return if valid.
        # If best_score was 0, we should have returned inside the loop.
        # So this function returns None if no valid graph found within budget.
        return None

    # Try n=21 first? 
    # n=21 is harder. n=20 is known possible.
    # Let's try n=21 for a short time, then n=20 for longer.
    
    # Time allocation: 20s for n=21, 80s for n=20.
    
    result = local_search(21, 20)
    if result:
        return result
        
    result = local_search(20, 80)
    if result:
        return result
        
    # Fallback: n=19
    result = local_search(19, 10)
    if result:
        return result

    # Ultimate fallback: simple valid graph (e.g. empty graph on 5 vertices? No, n must be max).
    # Actually, an empty graph on 5 vertices is valid.
    # But we want to maximize n.
    # If we fail, return a small valid graph to get score > 0?
    # The task says "Maximize n". Score is value/seed.
    # If we return n=5, score is low.
    # But better than 0.
    
    n = 5
    edges = []
    return {'n': n, 'edges': edges}
