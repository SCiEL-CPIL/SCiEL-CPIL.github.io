import time
import random

def construct():
    deadline = time.time() + 80  # 80s budget, 10s safety margin
    N = 128
    
    # Precompute all triples (a,b,c) with a<b<c such that a+b, a+c, b+c are all in 1..N
    # A triple violates if all of a,b,c,a+b,a+c,b+c are in the set
    # We need to find a large subset S of 1..N with no such triple fully contained
    
    # Approach: Start with all elements, remove elements to break all violating triples
    # Use greedy hitting set: repeatedly remove the element that breaks the most triples
    
    # First, enumerate all potential violating triples (a,b,c) where the sums are in range
    triples = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            for c in range(b+1, N+1):
                s1, s2, s3 = a+b, a+c, b+c
                if s1 <= N and s2 <= N and s3 <= N:
                    triples.append((a, b, c))
    
    # Now use greedy removal
    in_set = set(range(1, N+1))
    removed = set()
    
    # Count how many triples each element is in
    # For efficiency, we'll recompute as we go
    
    def has_violation(in_set):
        """Check if there's any violating triple in in_set"""
        for (a, b, c) in triples:
            if a in in_set and b in in_set and c in in_set:
                s1, s2, s3 = a+b, a+c, b+c
                if s1 in in_set and s2 in in_set and s3 in in_set:
                    return True, (a, b, c, s1, s2, s3)
        return False, None
    
    # Greedy removal: repeatedly find a violation and remove the most "central" element
    while time.time() < deadline - 5:
        violated, info = has_violation(in_set)
        if not violated:
            break
        a, b, c, s1, s2, s3 = info
        # The six elements involved are a, b, c, s1, s2, s3
        # Remove the one that appears in the most triples
        candidates = set([a, b, c, s1, s2, s3])
        # Count how many triples each candidate is in
        best_to_remove = None
        max_count = -1
        for elem in candidates:
            count = 0
            for (x, y, z) in triples:
                if elem in (x, y, z, x+y, x+z, y+z):
                    count += 1
            if count > max_count:
                max_count = count
                best_to_remove = elem
        if best_to_remove is not None:
            in_set.discard(best_to_remove)
            removed.add(best_to_remove)
    
    # Now try to add back removed elements
    removed_list = list(removed)
    random.shuffle(removed_list)
    for elem in removed_list:
        if time.time() > deadline - 2:
            break
        in_set.add(elem)
        violated, _ = has_violation(in_set)
        if violated:
            in_set.discard(elem)
    
    result = sorted(in_set)
    return list(result)
