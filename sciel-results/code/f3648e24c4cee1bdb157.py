import time
import random
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # 1. Identify all bad triples
    # A triple (a, b, c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We can compute lcm(a,b) = a*b//gcd(a,b)
    
    bad_triples = []
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = a * b // gcd(a, b)
            for c in range(b + 1, n + 1):
                lac = a * c // gcd(a, c)
                if lac != lab:
                    continue
                lbc = b * c // gcd(b, c)
                if lbc == lab:
                    bad_triples.append((a, b, c))
                    
    if time.time() > deadline:
        return numbers # Fallback
        
    # 2. Precompute conflict structures
    # For each vertex x, conflicts[x] is a set of frozensets of pairs {u,v}
    # such that (x, u, v) is a bad triple.
    # When checking if x can be added to S, we check if any pair in conflicts[x] is subset of S.
    
    conflicts = {x: set() for x in numbers}
    for a, b, c in bad_triples:
        conflicts[a].add(frozenset((b, c)))
        conflicts[b].add(frozenset((a, c)))
        conflicts[c].add(frozenset((a, b)))
    
    # Helper function to check if x can be added to current_set
    def can_add(x, current_set):
        for pair in conflicts[x]:
            if pair.issubset(current_set):
                return False
        return True

    best_set = set()
    best_size = -1
    
    # Strategy: Multi-start Greedy + Local Search
    # We use a "reverse greedy" approach: start with all vertices, remove those causing most conflicts.
    # Then try to add back vertices.
    
    num_restarts = 200 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize S with all numbers
        S = set(numbers)
        
        # Compute degree (number of active bad triples) for each vertex
        # degree[x] = number of bad triples containing x that are still "active" (all 3 vertices in S)
        degree = {x: 0 for x in numbers}
        for t in bad_triples:
            for x in t:
                degree[x] += 1
        
        # Greedy removal loop
        # While there are bad triples in S, remove the vertex with the highest degree.
        # Tie-breaking: random.
        
        # We need to efficiently find vertices with high degree.
        # Since n=128, we can just scan.
        
        while True:
            if time.time() > deadline - 2.0:
                break
                
            # Find max degree
            max_deg = 0
            candidates = []
            for x in S:
                if degree[x] > max_deg:
                    max_deg = degree[x]
                    candidates = [x]
                elif degree[x] == max_deg and max_deg > 0:
                    candidates.append(x)
            
            if not candidates or max_deg == 0:
                break
                
            # Randomly pick from candidates
            target = random.choice(candidates)
            
            # Remove target from S
            S.remove(target)
            
            # Update degrees for neighbors
            # For each bad triple containing target, if the other two are in S,
            # then that triple is now broken. We must decrement the degree of the other two.
            # Note: A triple is "active" only if all 3 vertices are in S.
            # When we remove target, any triple (target, u, v) where u,v in S becomes inactive.
            # So we decrement degree[u] and degree[v] for all such (u,v).
            
            # We need to iterate through bad_triples again or have a precomputed adjacency.
            # Let's precompute: for each x, list of (u, v) such that (x, u, v) is a bad triple.
            # This is essentially conflicts[x] but as a list of tuples for faster iteration?
            # Actually, conflicts[x] stores frozensets. Let's create a list version for speed.
            
            # Let's just iterate through all bad_triples and check if target is in it.
            # This is O(E) per removal. E ~ 1000-5000? 128 removals * 5000 = 640k ops. Fast enough.
            
            for t in bad_triples:
                if target in t:
                    # If the other two are in S, this triple was active and is now broken.
                    # We need to check if the other two are in S.
                    # The other two are t[0], t[1], t[2] excluding target.
                    others = [x for x in t if x != target]
                    if others[0] in S and others[1] in S:
                        degree[others[0]] -= 1
                        degree[others[1]] -= 1
                        # The triple is no longer active, so we don't need to do anything else for it.
                        # Wait, the degree of target itself? It doesn't matter since target is removed.
        
        # Now S is a valid set (no bad triples).
        # Local Search: Try to add vertices back.
        # We try to add vertices that can be added (can_add returns True).
        # We might want to be strategic: add vertices that allow more other vertices to be added?
        # Or just random order?
        
        # Let's try a simple hill-climbing:
        # While we can add any vertex, add one.
        # Which one? Random from those that can be added.
        
        improved = True
        max_local_iters = 100
        local_iter = 0
        
        while improved and local_iter < max_local_iters:
            improved = False
            local_iter += 1
            if time.time() > deadline - 2.0:
                break
            
            # Find all vertices not in S that can be added
            addable = []
            for x in numbers:
                if x not in S and can_add(x, S):
                    addable.append(x)
            
            if not addable:
                break
            
            # Randomly pick one to add
            # Or pick the one that minimizes future conflicts?
            # Let's stick with random for diversity.
            target = random.choice(addable)
            S.add(target)
            improved = True
            
            # Update degrees? 
            # When we add a vertex, we might create new bad triples.
            # We don't strictly need degrees for the "add" phase, only for the "remove" phase.
            # But if we want to do more sophisticated search, we'd need them.
            # For now, simple addition.
            
            # Note: Adding a vertex might make the set invalid if it forms a bad triple with two existing vertices.
            # But can_add checks exactly that. So S remains valid.
        
        # Record best
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
            # Early exit if we found a very large set
            # What is the theoretical max?
            # If we can get close to 128, we can stop.
            # But we don't know the max.
            # Let's just run until time runs out or we find a "good" set.
            
        # Optional: Try to remove 1 and add 2?
        # This is a 1-2 move.
        # If we remove y, can we add z1, z2?
        # This is more expensive. Let's skip for now and rely on the greedy + add-back.
        
    if best_size == -1:
        return [1]
        
    return list(best_set)
