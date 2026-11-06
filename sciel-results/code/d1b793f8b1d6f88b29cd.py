import time
import sys
from math import lcm
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, 5s buffer
    
    N = 128
    # 1. Precompute LCMs
    # lcm[i][j] for 1 <= i <= j <= N
    # Actually, we only need to check triples.
    
    # Identify all bad triples
    # A triple (a,b,c) with a<b<c is bad if:
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    
    bad_triples = []
    # Optimization: 
    # If a|b, lcm(a,b)=b.
    # If b|c, lcm(b,c)=c.
    # Then we need lcm(a,c) == b == c. Impossible since b<c.
    # So if a|b|c, it's never bad.
    # Generally, the condition is very restrictive.
    
    # Let's just iterate all triples. N=128 is small enough.
    # C(128, 3) ~ 340k. 
    # Each check involves 3 LCMs.
    # 340k * 3 * log(N) is trivial.
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm(a, b)
            for c in range(b+1, N+1):
                if time.time() > deadline:
                    # Fallback to seed-like behavior if we run out of time
                    # But this loop is very fast, so unlikely to hit deadline here
                    pass
                lac = lcm(a, c)
                if lab != lac:
                    continue
                lbc = lcm(b, c)
                if lab == lbc:
                    bad_triples.append((a, b, c))
    
    # We have the list of all bad triples in the full set [1..N]
    # Now we want the largest subset with NO bad triple.
    # This is Max Independent Set in a 3-uniform hypergraph.
    
    # Greedy Removal Strategy:
    # Start with all nodes.
    # While there is a bad triple, remove one node from it.
    # Choose the node that appears in the most current bad triples.
    
    in_set = [True] * (N + 1)
    # Map node -> list of indices in bad_triples? 
    # Or just maintain a set of active bad triples.
    
    active_bad = set(bad_triples)
    
    # To make it fast, we can maintain counts.
    # But the set of active triples changes.
    
    # Simple loop:
    # While active_bad is not empty:
    #   Pick a triple.
    #   Find the node in it with highest "degree" (count in active_bad).
    #   Remove it.
    #   Remove all triples containing it from active_bad.
    
    # Since N is small, we can just recompute or use a simple structure.
    # However, the number of bad triples might be small.
    
    # Let's count how many bad triples there are.
    # If it's very few, we can solve it exactly or with very fast greedy.
    
    # Greedy:
    # We maintain a list of "current" bad triples.
    # We maintain a count of how many times each node appears in current bad triples.
    
    # Initial counts
    counts = [0] * (N + 1)
    for t in active_bad:
        for v in t:
            counts[v] += 1
            
    # List of active triples
    # Using a set is slow for iteration if we modify it.
    # Let's use a list and a "dead" flag or just filter.
    # Given the small size, filtering is fine.
    
    current_triples = list(active_bad)
    
    while current_triples:
        if time.time() > deadline - 10:
            break
            
        # Pick the first triple
        # Or pick a random one?
        # Greedy: Pick the triple with the "heaviest" node?
        # Actually, standard greedy for hitting set:
        # Pick the set (triple) that covers the most elements? No, we remove elements.
        # We want to remove an element that hits the most triples.
        
        # Find the element with max count in current_triples
        # We can just iterate through current_triples to find max count node?
        # Or just pick the first triple, and within it pick the max count node.
        
        # Let's pick the global max count node among all nodes involved in current_triples.
        # This requires scanning current_triples.
        
        max_node = 0
        max_count = -1
        # Optimization: only consider nodes in current_triples
        # But we have `counts` array.
        # We just need to find the node with max `counts[v]` that is in `current_triples`.
        # Since `counts` is updated, we can just check all nodes? 
        # N=128 is small.
        
        for v in range(1, N+1):
            if counts[v] > max_count:
                max_count = counts[v]
                max_node = v
                
        if max_node == 0 or max_count == 0:
            break # Should not happen if current_triples is not empty
            
        # Remove max_node
        in_set[max_node] = False
        
        # Update current_triples and counts
        new_triples = []
        for t in current_triples:
            if max_node in t:
                for v in t:
                    counts[v] -= 1
            else:
                new_triples.append(t)
        current_triples = new_triples
        
    # Now in_set is a valid configuration.
    # Let's try to add back removed nodes.
    # Local Search:
    # For each node not in set, check if adding it creates a bad triple.
    # If not, add it.
    
    # To check if adding x creates a bad triple:
    # We need to check if there exists a pair (a,b) in current set such that (a,b,x) is a bad triple.
    # We can precompute for each x, the list of pairs (a,b) such that (a,b,x) is bad.
    
    # Precompute: bad_pairs_for[x] = list of (a,b)
    bad_pairs_for = [[] for _ in range(N+1)]
    for (a,b,c) in bad_triples:
        bad_pairs_for[a].append((b,c))
        bad_pairs_for[b].append((a,c))
        bad_pairs_for[c].append((a,b))
        
    # Local search improvement
    # Iterate a few times
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        changed = False
        for x in range(1, N+1):
            if in_set[x]:
                continue
            # Check if x can be added
            # It conflicts if there is a pair (a,b) in set such that (a,b,x) is bad.
            # i.e., (a,b) is in bad_pairs_for[x] AND in_set[a] AND in_set[b].
            
            can_add = True
            for (a,b) in bad_pairs_for[x]:
                if in_set[a] and in_set[b]:
                    can_add = False
                    break
                    
            if can_add:
                in_set[x] = True
                changed = True
                # Optional: try to remove something else to add more?
                # For now, just add.
                
    # Convert to list
    result = [i for i in range(1, N+1) if in_set[i]]
    
    # Ensure non-empty
    if not result:
        return [1]
        
    return result
