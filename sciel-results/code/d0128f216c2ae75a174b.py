import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Candidate set: Upper half is a good antichain base
    # We can also try slightly lower ranges to see if more composites fit
    # But upper half [129, 256] has 128 elements.
    # Let's try [129, 256] first.
    
    candidates = list(range(129, 257))
    
    # Precompute conflicts
    # A conflict is a triple (a, b, c) of distinct indices in candidates
    # such that candidates[i] divides candidates[j] * candidates[k]
    
    # To speed up, we can just check divisibility on the fly during removal?
    # No, precomputation is better for counting.
    # 128^3 / 6 ~ 350k checks. Very fast.
    
    conflicts = []
    n = len(candidates)
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            for k in range(j + 1, n):
                c = candidates[k]
                if (b * c) % a == 0:
                    conflicts.append((i, j, k))
                # Also check a | bc, b | ac, c | ab?
                # The condition is: for EVERY three distinct members a,b,c, a does not divide bc.
                # This is not symmetric.
                # We need to ensure:
                # 1. a % (b*c) != 0 is wrong. a | b*c means (b*c) % a == 0.
                # So for a specific ordered triple (a,b,c), we check if a | bc.
                # The set S is valid if NO ordered triple (a,b,c) of distinct elements satisfies a | bc.
                # So we need to check all permutations?
                # No, "a, b, c" are just labels for the three distinct members.
                # The condition is: For any distinct x, y, z in S, x does not divide y*z.
                # So we must check x | y*z for all distinct x, y, z.
                # This is equivalent to checking all 3! = 6 permutations for each unordered set {x,y,z}.
                # Or simply: for each unordered triple {x,y,z}, check if x|yz or y|xz or z|xy.
                # If any is true, it's a conflict.
                
                if (a * c) % b == 0:
                    conflicts.append((i, j, k))
                    break # One conflict is enough to mark this triple as bad
                if (a * b) % c == 0:
                    conflicts.append((i, j, k))
                    break
    # Note: The loop above adds the same triple multiple times if multiple conditions hold.
    # We should use a set of tuples to avoid duplicates, or just count unique bad triples.
    
    # Let's restart the conflict generation more cleanly.
    bad_triples = set()
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            for k in range(j + 1, n):
                c = candidates[k]
                if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                    bad_triples.add((i, j, k))
    
    # Greedy removal
    S = set(range(n))
    # Map index to count of bad triples it is part of
    # We need to re-evaluate counts as we remove elements.
    
    # Efficient way:
    # Keep a list of bad triples.
    # When an element is removed, remove all triples containing it.
    # The "degree" of an element is the number of remaining bad triples it is in.
    
    remaining_triples = list(bad_triples)
    # We need a fast way to find which triples contain a given index.
    # index_to_triples = defaultdict(list)
    
    # Since n is small (128), we can just re-scan or use a simple structure.
    # Let's use a simple iterative greedy:
    # 1. Count frequency of each index in remaining_triples.
    # 2. Pick max.
    # 3. Remove index from S.
    # 4. Filter remaining_triples to remove any triple containing that index.
    
    # This is O(T * N) per removal. T ~ 10k, N ~ 128. 1.2M ops per removal.
    # 128 removals -> 150M ops. Might be slow in Python (2-5s). Acceptable.
    
    # Optimization: Use a counter and update it?
    # Hard to update counts when triples are removed.
    # Given 80s budget, simple approach is fine.
    
    active_triples = list(bad_triples)
    
    while active_triples:
        if time.time() > deadline:
            break
            
        # Count occurrences
        counts = defaultdict(int)
        for t in active_triples:
            counts[t[0]] += 1
            counts[t[1]] += 1
            counts[t[2]] += 1
            
        # Find max
        max_count = 0
        victim = -1
        for idx in S:
            if counts[idx] > max_count:
                max_count = counts[idx]
                victim = idx
                
        if victim == -1:
            break
            
        S.remove(victim)
        
        # Filter triples
        new_active = []
        for t in active_triples:
            if victim not in t:
                new_active.append(t)
        active_triples = new_active
        
    # Convert indices back to values
    result = [candidates[i] for i in S]
    
    # Ensure types are correct
    return [int(x) for x in result]
