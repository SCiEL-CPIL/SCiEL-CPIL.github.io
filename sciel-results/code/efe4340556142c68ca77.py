import time
from math import gcd
from itertools import combinations
import random

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    # Precompute GCDs
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g

    # Identify bad triples and conflicts
    # conflicts[i] will be a set of encoded pairs (a, b) such that {a, b, i} is a bad triple.
    # Encode pair (a, b) with a < b as a * (N + 1) + b
    def encode(a, b):
        if a > b:
            a, b = b, a
        return a * (N + 1) + b

    conflicts = [set() for _ in range(N + 1)]
    
    # Iterate through all triples
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            for k in range(j + 1, N + 1):
                if gcds[i][j] == gcds[j][k] == gcds[k][i]:
                    # This triple is bad
                    p_ik = encode(i, k)
                    p_ij = encode(i, j)
                    p_jk = encode(j, k)
                    
                    conflicts[i].add(p_jk)
                    conflicts[j].add(p_ik)
                    conflicts[k].add(p_ij)

    # Calculate degree for each number (number of conflicts)
    degrees = [len(conflicts[i]) for i in range(N + 1)]
    
    # Helper to check if x can be added to selected_pairs
    def can_add_pairs(x, selected_pairs):
        for p in conflicts[x]:
            if p in selected_pairs:
                return False
        return True

    # Initial greedy construction using degree ordering
    indices = list(range(1, N + 1))
    indices.sort(key=lambda x: degrees[x])
    
    selected = []
    selected_pairs = set()
    
    for x in indices:
        if time.time() > deadline:
            break
        if can_add_pairs(x, selected_pairs):
            selected.append(x)
            for y in selected[:-1]:
                selected_pairs.add(encode(x, y))

    best_set = set(selected)
    best_size = len(selected)
    
    # Local search function using pair-based checking for speed
    def local_search(s, rng):
        # s is a set
        # First, try to add elements
        improved = True
        while improved:
            improved = False
            remaining = [x for x in range(1, N + 1) if x not in s]
            rng.shuffle(remaining)
            
            # Rebuild pairs for current set to check validity efficiently?
            # No, we can just check against the existing set.
            # To check if x can be added to s, we need to ensure no pair (u,v) in s
            # is in conflicts[x].
            # This is O(|s|^2) per x if we iterate pairs of s.
            # Or O(|conflicts[x]|) if we maintain a set of pairs in s.
            # Let's maintain a set of pairs in s for O(1) lookups.
            
            s_pairs = set()
            s_list = list(s)
            for i in range(len(s_list)):
                for j in range(i+1, len(s_list)):
                    s_pairs.add(encode(s_list[i], s_list[j]))
            
            for x in remaining:
                if can_add_pairs(x, s_pairs):
                    s.add(x)
                    # Update s_pairs
                    for y in s_list:
                        s_pairs.add(encode(x, y))
                    s_list.append(x)
                    improved = True
                    break
        
        # Then try swaps: remove one, add multiple
        # We'll do a few rounds of this
        for _ in range(5):
            if time.time() > deadline:
                break
            
            improved = False
            s_list = list(s)
            rng.shuffle(s_list)
            
            for u in s_list:
                if time.time() > deadline - 5:
                    break
                
                # Remove u
                s_without_u = s - {u}
                new_s = set(s_without_u)
                
                # Rebuild pairs for new_s
                new_s_list = list(new_s)
                new_s_pairs = set()
                for i in range(len(new_s_list)):
                    for j in range(i+1, len(new_s_list)):
                        new_s_pairs.add(encode(new_s_list[i], new_s_list[j]))
                
                # Try to add from remaining
                remaining = [x for x in range(1, N + 1) if x not in new_s]
                rng.shuffle(remaining)
                
                added_count = 0
                for x in remaining:
                    if can_add_pairs(x, new_s_pairs):
                        new_s.add(x)
                        for y in new_s_list:
                            new_s_pairs.add(encode(x, y))
                        new_s_list.append(x)
                        added_count += 1
                
                # If we added more than we removed (1), we improved
                if added_count > 1:
                    s = new_s
                    improved = True
                    break
        
        return s

    # Random restarts
    rng = random.Random(42)
    
    # Run local search on the initial solution
    best_set = local_search(best_set, rng)
    best_size = len(best_set)
    
    # Multiple random restarts
    while time.time() < deadline - 10:
        # Generate a random order, maybe weighted by degree
        # Or just random shuffle
        random_indices = list(range(1, N + 1))
        rng.shuffle(random_indices)
        
        sel = []
        sel_pairs = set()
        
        for x in random_indices:
            if time.time() > deadline:
                break
            if can_add_pairs(x, sel_pairs):
                sel.append(x)
                for y in sel[:-1]:
                    sel_pairs.add(encode(x, y))
        
        current_set = set(sel)
        
        # Apply local search
        current_set = local_search(current_set, rng)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
            # If we reach a very high size, we might stop early?
            # Theoretical max is unknown, but 128 is the absolute max.
            # If we get close to 128, maybe stop?
            # For now, keep searching to maximize.

    return list(best_set)
