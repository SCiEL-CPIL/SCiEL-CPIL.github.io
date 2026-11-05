import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin

    # Universe
    N = 256
    vertices = list(range(2, N + 1))
    weights = {v: 1.0 / v for v in vertices}
    
    # Precompute conflicts
    # conflict[i] is a set of vertices conflicting with i
    # Or a bitmask for speed? 255 bits fits in Python int.
    # Let's use sets for clarity first, optimize if needed.
    # Actually, list of sets is fine for 255 nodes.
    
    conflicts = {v: set() for v in vertices}
    for i in range(len(vertices)):
        u = vertices[i]
        for j in range(i + 1, len(vertices)):
            v = vertices[j]
            # Check if lcm(u, v) <= 256
            # lcm(u, v) = u * v // gcd(u, v)
            if (u * v) // math.gcd(u, v) <= 256:
                conflicts[u].add(v)
                conflicts[v].add(u)
    
    def is_valid_set(S):
        S_set = set(S)
        for u in S_set:
            for v in conflicts[u]:
                if v in S_set:
                    return False
        return True

    def calc_score(S):
        return sum(weights[x] for x in S)

    def try_add(S, x):
        """Returns True if x can be added to S, and modifies S if so."""
        S_set = set(S)
        if x in S_set:
            return False
        for neighbor in conflicts[x]:
            if neighbor in S_set:
                return False
        S.append(x)
        return True

    def local_search(S, deadline):
        S = list(S)
        current_score = calc_score(S)
        S_set = set(S)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1-Swap
            # Sort S by weight ascending (remove low value first)
            # Sort candidates not in S by weight descending (add high value first)
            S_sorted = sorted(S, key=lambda x: weights[x])
            candidates = [v for v in vertices if v not in S_set]
            candidates.sort(key=lambda x: -weights[x])
            
            swap_made = False
            for u in S_sorted:
                if time.time() > deadline:
                    return S, current_score
                # Try to replace u with some v
                # We only care if weights[v] > weights[u]
                # Since S_sorted is ascending weight, u has low weight.
                # candidates are descending weight.
                for v in candidates:
                    if weights[v] <= weights[u]:
                        break # No further v will improve score
                    
                    # Check if S \ {u} U {v} is valid
                    # This requires v to be compatible with all elements in S \ {u}
                    valid = True
                    for w in S:
                        if w == u:
                            continue
                        if w in conflicts[v]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        S.remove(u)
                        S.append(v)
                        S_set.remove(u)
                        S_set.add(v)
                        current_score += weights[v] - weights[u]
                        improved = True
                        swap_made = True
                        break # Restart inner loop? Or just break inner for v?
                        # Breaking inner for v means we try next u.
                        # But S changed, so S_sorted and candidates are stale.
                        # We should restart the while loop or at least update structures.
                        # For simplicity and correctness, break out to restart the while loop.
                        break
                
                if swap_made:
                    break
            
            if not improved:
                # Try 2-Swap (limited)
                # Random sampling to avoid O(N^4)
                if time.time() < deadline - 5: # Reserve time
                    for _ in range(100): # 100 random 2-swaps
                        if time.time() > deadline:
                            break
                        # Pick 2 to remove
                        if len(S) < 2:
                            continue
                        u1, u2 = random.sample(S, 2)
                        # Pick 2 to add
                        candidates = [v for v in vertices if v not in S_set]
                        if len(candidates) < 2:
                            continue
                        v1, v2 = random.sample(candidates, 2)
                        
                        # Check potential gain
                        if weights[v1] + weights[v2] <= weights[u1] + weights[u2]:
                            continue
                        
                        # Check validity of S \ {u1, u2} U {v1, v2}
                        new_S = [x for x in S if x != u1 and x != u2]
                        
                        # Check v1, v2 against new_S and each other
                        valid = True
                        if v1 in conflicts[v2]:
                            valid = False
                        else:
                            for w in new_S:
                                if w in conflicts[v1] or w in conflicts[v2]:
                                    valid = False
                                    break
                        
                        if valid:
                            S = new_S + [v1, v2]
                            S_set = set(S)
                            current_score += (weights[v1] + weights[v2]) - (weights[u1] + weights[u2])
                            improved = True
                            break # Restart while loop
        
        return S, current_score

    best_solution = []
    best_score = 0.0
    
    # Initialization 1: Greedy by weight
    S_greedy = []
    for v in sorted(vertices, key=lambda x: -weights[x]):
        if time.time() > deadline:
            break
        if try_add(S_greedy, v):
            pass
    
    S_greedy, score_greedy = local_search(S_greedy, deadline)
    if score_greedy > best_score:
        best_score = score_greedy
        best_solution = S_greedy

    # Initialization 2: Structural (Large numbers + compatible small)
    S_struct = list(range(129, 257))
    # Add small numbers compatible with ALL current S_struct
    small_candidates = sorted([v for v in vertices if v < 129], key=lambda x: -weights[x])
    for v in small_candidates:
        if time.time() > deadline:
            break
        # Check compatibility with all in S_struct
        compatible = True
        for u in S_struct:
            if u in conflicts[v]:
                compatible = False
                break
        if compatible:
            # Also check with other added small numbers
            # Since we add sequentially, check against current S_struct
            # But S_struct is being modified? No, we are building a list.
            # Let's just use try_add
            if try_add(S_struct, v):
                pass
                
    S_struct, score_struct = local_search(S_struct, deadline)
    if score_struct > best_score:
        best_score = score_struct
        best_solution = S_struct

    # Random Restarts
    while time.time() < deadline - 10:
        S_rand = []
        # Randomized Greedy
        shuffled = vertices.copy()
        random.shuffle(shuffled)
        # Sort by weight with some noise? Or just shuffle and greedy?
        # Better: Sort by weight, but with probability skip?
        # Let's stick to simple random restart of Local Search from random valid set.
        
        # Generate a random valid set
        for v in vertices:
            if random.random() < 0.5: # 50% chance to consider adding
                if try_add(S_rand, v):
                    pass
        
        S_rand, score_rand = local_search(S_rand, deadline)
        if score_rand > best_score:
            best_score = score_rand
            best_solution = S_rand

    return best_solution
