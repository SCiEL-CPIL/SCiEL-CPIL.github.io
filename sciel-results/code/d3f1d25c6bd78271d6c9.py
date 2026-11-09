"""Construct a minimum maximal no-isosceles set on an N x N grid."""
N = 16

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _precompute(n):
    n2 = n * n
    points = [(x, y) for x in range(n) for y in range(n)]
    idx = lambda x, y: x * n + y
    
    # dist2[i][j]
    dist2 = np.zeros((n2, n2), dtype=np.int32)
    for i, (x1, y1) in enumerate(points):
        for j, (x2, y2) in enumerate(points):
            dist2[i, j] = (x1 - x2) ** 2 + (y1 - y2) ** 2
            
    # isos[i][j][k] is True if {i,j,k} is isosceles
    # We only need this for distinct i,j,k.
    isos = np.zeros((n2, n2, n2), dtype=bool)
    # conflict_pairs[i][j] = list of k such that {i,j,k} is isosceles
    conflict_pairs = [[[] for _ in range(n2)] for _ in range(n2)]
    
    for i in range(n2):
        for j in range(i + 1, n2):
            # d_ij
            dij = dist2[i, j]
            # Find k such that dij == dik or dij == djk or dik == djk
            # Note: k must be distinct from i and j.
            # We iterate k for efficiency? Or precompute distances from i and j?
            # dist2[i, :] and dist2[j, :] are available.
            dik = dist2[i, :]
            djk = dist2[j, :]
            
            # Condition: (dij == dik) OR (dij == djk) OR (dik == djk)
            # Exclude i and j themselves from k?
            # The triangle is defined by 3 distinct points.
            mask = np.ones(n2, dtype=bool)
            mask[i] = False
            mask[j] = False
            
            cond = ((dij == dik) | (dij == djk) | (dik == djk)) & mask
            
            ks = np.where(cond)[0]
            for k in ks:
                isos[i, j, k] = True
                isos[i, k, j] = True
                isos[j, i, k] = True
                isos[j, k, i] = True
                isos[k, i, j] = True
                isos[k, j, i] = True
                
                conflict_pairs[i][j].append(k)
                conflict_pairs[j][i].append(k)
                # k is also in conflict_pairs[i][k]? No, conflict_pairs is for pairs (a,b) -> c.
                # We need conflict_pairs[a][b] for any a,b.
                # So for triple (i,j,k):
                # conflict_pairs[i][j] gets k
                # conflict_pairs[i][k] gets j
                # conflict_pairs[j][k] gets i
                conflict_pairs[i][k].append(j)
                conflict_pairs[k][i].append(j)
                conflict_pairs[j][k].append(i)
                conflict_pairs[k][j].append(i)

    return points, dist2, isos, conflict_pairs

def construct():
    n = N
    n2 = n * n
    points, dist2, isos, conflict_pairs = _precompute(n)
    
    best_set = None
    best_size = n2 + 1
    
    rng = random.Random(42)
    
    for attempt in range(50):
        # 1. Greedy Construction
        order = list(range(n2))
        rng.shuffle(order)
        S = []
        S_set = set()
        
        for p in order:
            # Check if p can be added to S
            # Must not form isosceles with any pair in S
            valid = True
            for i, a in enumerate(S):
                for b in S[i+1:]:
                    if isos[p, a, b]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(p)
                S_set.add(p)
                
        # 2. Initialize blocked_count
        # blocked_count[p] = number of pairs (a,b) in S such that {p,a,b} is isosceles
        blocked_count = np.zeros(n2, dtype=np.int32)
        for i, a in enumerate(S):
            for b in S[i+1:]:
                for p in conflict_pairs[a][b]:
                    blocked_count[p] += 1
        
        # 3. Local Search: Remove points
        improved = True
        while improved:
            improved = False
            # Try to remove each point in S
            # We iterate in random order to avoid bias
            indices_to_try = list(range(len(S)))
            rng.shuffle(indices_to_try)
            
            for idx in indices_to_try:
                x = S[idx]
                # Check if removing x is valid
                # New S = S \ {x}
                # We need to ensure all p not in New S are blocked by New S
                
                # Simulate removal
                # For each y in S \ {x}, the pair (x,y) is removed.
                # So for each p in conflict_pairs[x][y], blocked_count[p] decreases by 1.
                # If any p not in S (and p != x, obviously) has blocked_count[p] become 0, invalid.
                
                valid = True
                # We need to track which p's counts changed?
                # Since N is small, we can just check.
                # But we need to be careful: if p is in S, we don't care about its blocked_count for maximality.
                # We only care about p NOT in S.
                
                # Optimization: Pre-compute list of "critical" points?
                # For now, just check.
                
                # Create a temporary copy of blocked_count?
                # Or just calculate the new count for affected p's.
                # Affected p's are those in union of conflict_pairs[x][y] for y in S\{x}.
                
                # Let's just use a set of affected points and check them.
                affected = set()
                for y in S:
                    if y == x: continue
                    for p in conflict_pairs[x][y]:
                        affected.add(p)
                        
                for p in affected:
                    if p in S_set:
                        continue
                    # Calculate new blocked count for p
                    # Old count is blocked_count[p]
                    # How many pairs (a,b) in S\{x} block p?
                    # It is blocked_count[p] - (number of pairs (a,b) in S containing x that block p)
                    # Pairs containing x are (x, y) for y in S.
                    # Does (x, y) block p? Yes, if p in conflict_pairs[x][y].
                    # So decrement = number of y in S such that p in conflict_pairs[x][y].
                    
                    # We can just check if p is in conflict_pairs[x][y] for each y.
                    # But we already iterated y to build 'affected'.
                    # Let's re-calculate decrement.
                    dec = 0
                    for y in S:
                        if y == x: continue
                        if p in conflict_pairs[x][y]:
                            dec += 1
                            
                    new_count = blocked_count[p] - dec
                    if new_count == 0:
                        valid = False
                        break
                        
                if valid:
                    # Commit removal
                    S_set.remove(x)
                    S.pop(idx)
                    # Update blocked_count
                    for y in S:
                        for p in conflict_pairs[x][y]:
                            blocked_count[p] -= 1
                    improved = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_set = list(S)
                    break # Restart loop with new S? Or continue? 
                         # Continuing might allow multiple removals in one pass, but 
                         # the indices are shifted. Restarting is safer.
                        
        # 4. If stuck, try swaps to escape local minimum
        # If no removals were made in the last 'while' pass, we are in a local minimum.
        # Try random swaps.
        if len(S) < best_size:
             # We might have improved, but if we are stuck, we should try to move.
             pass
             
        # Simple Swap Strategy:
        # Pick random x in S, y not in S.
        # Check if S' = S - {x} + {y} is valid.
        # If valid, accept.
        # Do this for some iterations.
        
        for _ in range(100):
            if len(S) == 0: break
            x_idx = rng.randint(0, len(S)-1)
            x = S[x_idx]
            # Pick y not in S
            # Create list of not in S
            not_in_S = [p for p in range(n2) if p not in S_set]
            if not not_in_S: continue
            y = rng.choice(not_in_S)
            
            # Check validity of S' = S - {x} + {y}
            # 1. Independence: y must not form isosceles with any pair in S - {x}
            # 2. Maximality: All p not in S' must be blocked by S'
            
            # Check Independence
            indep = True
            S_temp_set = S_set - {x}
            for a in S_temp_set:
                for b in S_temp_set:
                    if a >= b: continue # just check pairs
                    if isos[y, a, b]:
                        indep = False
                        break
                if not indep: break
            
            if not indep: continue
            
            # Check Maximality
            # New S' = S - {x} + {y}
            # Points that might become unblocked:
            # 1. Points that were blocked ONLY by pairs involving x.
            # 2. Points that were blocked by pairs involving x, but now x is gone, 
            #    but y might block them? Or not.
            
            # Efficient check:
            # Calculate blocked_count for S'.
            # We can start from current blocked_count.
            # Remove x: decrement for pairs (x, z) in S.
            # Add y: increment for pairs (y, z) in S - {x}.
            
            new_blocked = blocked_count.copy()
            
            # Subtract x's contributions
            for z in S:
                if z == x: continue
                for p in conflict_pairs[x][z]:
                    new_blocked[p] -= 1
            
            # Add y's contributions
            for z in S:
                if z == x: continue
                for p in conflict_pairs[y][z]:
                    new_blocked[p] += 1
            
            # Check if any p not in S' has new_blocked[p] == 0
            S_prime_set = S_temp_set | {y}
            max_ok = True
            for p in range(n2):
                if p in S_prime_set: continue
                if new_blocked[p] == 0:
                    max_ok = False
                    break
            
            if max_ok:
                # Accept swap
                S_set.remove(x)
                S.pop(x_idx)
                S.append(y)
                S_set.add(y)
                blocked_count = new_blocked
                # Size is same, but we might be able to remove now.
                # Break to try removals again?
                break

    # Convert back to (x, y) pairs
    if best_set is None:
         # Fallback: return a known valid set or empty? 
         # Must return valid.
         # Greedy result from last attempt
         result = [points[i] for i in S]
    else:
         result = [points[i] for i in best_set]
         
    return result

# EVOLVE-BLOCK-END
