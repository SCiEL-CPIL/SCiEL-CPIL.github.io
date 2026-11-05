import heapq

def construct():
    denoms = range(1, 11)
    invs = [pow(d, -1, 101) for d in denoms]
    
    # State: (residue, mask)
    # We want to find the minimum number of denominators (cost) to reach each residue.
    # Since all edges have weight 1, BFS is sufficient.
    
    # best_cost[r] stores the minimum cost (size) to reach residue r.
    # best_mask[r] stores the mask that achieved this cost.
    best_cost = [float('inf')] * 101
    best_mask = [0] * 101
    
    # Queue: (cost, residue, mask)
    # Start at residue 0, mask 0, cost 0
    # Note: The task requires non-empty lists. 
    # Residue 0 with mask 0 is cost 0, but we need non-empty.
    # So we must find the first non-empty mask that sums to 0, or any mask.
    # Actually, the problem says "101 nonempty lists".
    # So for residue 0, we cannot use mask 0.
    # We must find a non-empty subset summing to 0.
    
    # We will run BFS from (0, 0).
    # When we visit a state (r, m) with m != 0, we can update best[r] if cost is lower.
    # We stop when all residues have been visited with some non-empty mask?
    # Or we can just run until queue is empty, but we can prune if we already have a cost for r.
    # Since we want min cost, once we pop a state for r with cost C, that's the min cost for r.
    # We can mark r as "done" and not expand it further? 
    # No, expanding it might lead to other residues. But for finding min cost for each r, 
    # the first time we reach r is the min cost.
    
    # However, we need the mask to reconstruct the list.
    
    queue = []
    heapq.heappush(queue, (0, 0, 0)) # cost, residue, mask
    
    # To avoid re-processing, we can keep a visited set of (residue, mask)
    # But there are only 101 * 1024 states.
    visited = set()
    visited.add((0, 0))
    
    found = [False] * 101
    found[0] = False # 0 needs non-empty
    
    while queue:
        cost, r, m = heapq.heappop(queue)
        
        # If this mask is non-empty and we haven't found a solution for r yet,
        # record it. Since BFS visits in increasing cost, the first non-empty mask 
        # for r is the optimal one.
        if m != 0 and not found[r]:
            best_cost[r] = cost
            best_mask[r] = m
            found[r] = True
            
            # If all found, we can break? 
            # We need to ensure we found all 101.
            if sum(found) == 101:
                break
                
        # Expand neighbors
        for i, d in enumerate(denoms):
            bit = 1 << i
            if m & bit:
                continue # Already used
            new_m = m | bit
            new_r = (r + invs[i]) % 101
            
            state = (new_r, new_m)
            if state not in visited:
                visited.add(state)
                heapq.heappush(queue, (cost + 1, new_r, new_m))
                
    # Construct result
    representations = [None] * 101
    for r in range(101):
        if found[r]:
            m = best_mask[r]
            terms = [d for i, d in enumerate(denoms) if (m >> i) & 1]
            representations[r] = terms
        else:
            # Fallback, shouldn't happen
            representations[r] = [1]
            
    return representations
