import time
from itertools import combinations
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # 1. Identify all bad triples
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # and a, b, c are distinct.
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCMs for all pairs to speed up triple checking?
    # There are 128*127/2 = 8128 pairs. Storing them is fine.
    # But checking triples directly might be fast enough if we prune.
    # Let's just iterate combinations.
    
    bad_triples = []
    # To speed up, we can precompute lcm[i][j]
    # But 128 is small.
    
    # Let's find all bad triples
    # We can optimize by noting that if lcm(a,b) != L, we don't care.
    # But we need lcm(a,b) == lcm(a,c) == lcm(b,c).
    
    # Let's use a set of tuples to avoid duplicates?
    # Combinations(a, b, c) with a < b < c ensures uniqueness.
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            # If lab > n, it can still be the LCM of a,b,c?
            # Yes, e.g., a=6, b=10, lab=30. c could be 15. lcm(6,15)=30, lcm(10,15)=30.
            # So lab can be larger than n.
            for c in range(b + 1, n + 1):
                if time.time() > deadline:
                    break
                lac = lcm(a, c)
                if lac != lab:
                    continue
                lbc = lcm(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))
        
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return numbers # Fallback, though this might be invalid
    
    # 2. Greedy Vertex Cover
    # We want to remove the minimum number of elements to break all bad_triples.
    # active_triples = set of indices of bad_triples that are not yet broken
    # element_count = dict of element -> count of active triples it belongs to
    
    # For efficiency, let's maintain a list of active triples and a set of active elements?
    # Or simply iterate.
    
    # Let's use a list of triples and mark them as broken.
    # And a dictionary counting occurrences.
    
    # Initial counts
    counts = {x: 0 for x in numbers}
    for t in bad_triples:
        for x in t:
            counts[x] += 1
            
    active_triples = [True] * len(bad_triples)
    active_count = len(bad_triples)
    
    removed = set()
    
    while active_count > 0:
        if time.time() > deadline:
            break
            
        # Find element with max count among active triples
        # We need to update counts as we remove elements.
        # A simpler approach for small N:
        # Just scan all elements and calculate how many active triples they are in.
        # This is O(N * T) per step. T is ~350k, N=128. 128*350k = 45M ops per step.
        # Too slow if many steps.
        
        # Better: Maintain counts dynamically.
        # When we remove an element v, we decrement counts for all other elements in every active triple containing v.
        
        # Let's try the simple scan first, but optimize the inner loop.
        # Actually, the number of bad triples is likely much smaller than 350k.
        # Let's check the size of bad_triples.
        
        # If bad_triples is small, the simple scan is fine.
        # Let's assume it's manageable.
        
        best_elem = None
        best_count = -1
        
        # Only consider elements not yet removed
        for x in numbers:
            if x in removed:
                continue
            # Count active triples containing x
            # This is slow.
            # Let's use the dynamic count approach.
            
            pass
            
        # Let's restart with dynamic counts.
        
        # Re-initialize counts based on active_triples?
        # Or just maintain `counts` and update it.
        
        # Let's implement the dynamic update.
        
        # We need to find the element with the highest count.
        # We can just iterate through `numbers` and pick the max `counts[x]` where x not in removed.
        
        max_count = -1
        target = None
        for x in numbers:
            if x not in removed and counts[x] > max_count:
                max_count = counts[x]
                target = x
                
        if target is None:
            break
            
        # Remove target
        removed.add(target)
        
        # Update counts and active_triples
        # Find all active triples containing target
        for i, t in enumerate(bad_triples):
            if not active_triples[i]:
                continue
            if target in t:
                # This triple is now broken
                active_triples[i] = False
                active_count -= 1
                # Decrement counts for the other two elements
                for x in t:
                    if x != target:
                        counts[x] -= 1
                        
        # Note: `target in t` is O(1) for tuple of 3.
        
    # The result is the set of numbers not in removed
    result = [x for x in numbers if x not in removed]
    
    # Ensure non-empty
    if not result:
        return [1]
        
    return result
