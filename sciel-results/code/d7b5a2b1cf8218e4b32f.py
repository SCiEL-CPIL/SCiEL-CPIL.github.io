import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 20
    
    # A: adjacency matrix (n x n)
    # C: common neighbor counts (n x n), C[i][j] = number of common neighbors of i and j
    # Note: C[i][i] is degree of i. We don't strictly need C[i][i] for constraints, 
    # but it's part of the update logic if we use matrix multiplication, 
    # but we will update explicitly.
    
    def initialize_greedy():
        A = np.zeros((n, n), dtype=np.int8)
        C = np.zeros((n, n), dtype=np.int8)
        
        # Greedy add edges in random order
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            # Check if adding (u, v) is valid
            # 1. C[u][v] <= 2
            if C[u][v] > 2:
                continue
            # 2. For all w in N(u), C[w][v] <= 1
            # 3. For all w in N(v), C[w][u] <= 1
            
            valid = True
            # Neighbors of u
            for w in range(n):
                if A[u][w] == 1:
                    if C[w][v] > 1:
                        valid = False
                        break
            if not valid:
                continue
                
            # Neighbors of v
            for w in range(n):
                if A[v][w] == 1:
                    if C[w][u] > 1:
                        valid = False
                        break
            if not valid:
                continue
            
            # Add edge
            A[u][v] = 1
            A[v][u] = 1
            
            # Update C
            # C[u][v] unchanged
            # For w in N(u): C[w][v] += 1, C[v][w] += 1
            for w in range(n):
                if A[u][w] == 1: # Note: A[u][v] is now 1, but w=v is handled?
                    # If w == v, A[u][v] is 1. C[v][v] is degree.
                    # But we only care about C[w][x] where w != x for constraints.
                    # Let's just update all.
                    C[w][v] += 1
                    C[v][w] += 1
            # For w in N(v): C[w][u] += 1, C[u][w] += 1
            for w in range(n):
                if A[v][w] == 1:
                    C[w][u] += 1
                    C[u][w] += 1
            
            # Correction: The loops above include w=v and w=u?
            # If w=v in first loop: A[u][v] is 1. C[v][v] += 1. C[v][v] is degree. Correct.
            # If w=u in second loop: A[v][u] is 1. C[u][u] += 1. Correct.
            # But we also updated C[v][u] in first loop (w=v) and C[u][v] in second loop (w=u)?
            # First loop w=v: C[v][v] += 1, C[v][v] += 1? No.
            # Code: C[w][v] += 1, C[v][w] += 1.
            # If w=v: C[v][v] += 1, C[v][v] += 1. So C[v][v] += 2.
            # But degree of v increases by 1 (neighbor u). So C[v][v] should increase by 1.
            # Why 2?
            # Because w=v is in N(u) AND w=v is in N(v)? No, v is not in N(v).
            # Wait. N(v) contains u now.
            # The loop `for w in range(n): if A[v][w] == 1` iterates over neighbors of v.
            # v is NOT a neighbor of v. So w != v in the second loop.
            # u IS a neighbor of v now. So w=u is in the second loop.
            # In the first loop `for w in range(n): if A[u][w] == 1`:
            # v IS a neighbor of u now. So w=v is in the first loop.
            
            # Let's trace C[v][v] (degree of v):
            # Initially 0.
            # Add edge (u,v).
            # Loop 1 (w in N(u)): w=v is in N(u).
            #   C[v][v] += 1. C[v][v] += 1. -> C[v][v] += 2.
            # Loop 2 (w in N(v)): w=u is in N(v).
            #   C[u][u] += 1. C[u][u] += 1. -> C[u][u] += 2.
            #   C[u][v] += 1. C[v][u] += 1.
            
            # So C[v][v] increased by 2. But degree increased by 1.
            # The update logic `C[w][v] += 1` for w in N(u) adds 1 for each common neighbor?
            # No.
            # C[w][v] is count of common neighbors of w and v.
            # If we add edge (u,v), u becomes a common neighbor of w and v IF w is connected to u.
            # So for each w in N(u), C[w][v] increases by 1.
            # If w=v, v is not in N(v). So w=v is NOT in N(v).
            # But w=v IS in N(u) (since we just added it).
            # So the loop `if A[u][w] == 1` includes w=v.
            # So C[v][v] is updated.
            # But C[v][v] is the number of common neighbors of v and v, which is the degree of v.
            # The degree of v increases by 1 (neighbor u).
            # Why did we add 2?
            # Because `C[w][v] += 1` and `C[v][w] += 1` are the same cell `C[v][v]`.
            # We should only increment once per (w, v) pair?
            # No, the update is:
            # For each w in N(u), C[w][v] increases by 1.
            # For each w in N(v), C[w][u] increases by 1.
            # These are distinct pairs (w,v) and (w,u).
            # If w=v, pair is (v,v).
            # If w=u, pair is (u,u).
            # So C[v][v] is incremented by 1 from the first condition (w=v in N(u)).
            # And C[v][v] is incremented by 1 from the second condition?
            # Second condition: w in N(v). w=u. Pair (u,u).
            # So C[u][u] is incremented by 1 from second condition (w=u in N(v)).
            # And C[u][u] is incremented by 1 from first condition?
            # First condition: w in N(u). w=u? No, u not in N(u).
            # So C[u][u] is only incremented by the second loop (w=u in N(v))?
            # Wait.
            # Loop 1: w in N(u).
            # Loop 2: w in N(v).
            
            # Let's re-verify C[u][u] (degree of u).
            # Degree of u increases by 1 (neighbor v).
            # Loop 1: w in N(u). u is not in N(u). So w != u.
            #   Updates C[w][v] and C[v][w].
            # Loop 2: w in N(v). u is in N(v). So w=u is included.
            #   Updates C[u][u] and C[u][u]. -> C[u][u] += 2.
            # So C[u][u] increases by 2. But degree increases by 1.
            
            # The issue is that `C[w][u] += 1` and `C[u][w] += 1` are the same cell.
            # We should just do `C[w][u] += 1` since C is symmetric.
            
            # Let's fix the update logic.
            
            pass
        
        return A, C

    # Corrected Update Logic
    def update_C_add(A, C, u, v):
        # Add edge (u, v)
        A[u][v] = 1
        A[v][u] = 1
        
        # C[u][v] is unchanged (common neighbors don't change)
        
        # For w in N(u) (old neighbors + v? No, N(u) now includes v.
        # But the "new common neighbor" u for pair (w, v) exists if w was connected to u BEFORE?
        # Yes. The path w-u-v exists if w~u and u~v.
        # u~v is new. w~u must have existed.
        # So for all w such that A[u][w] was 1 BEFORE the addition:
        #   C[w][v] += 1
        # Note: w=v? A[u][v] was 0. So w=v is not in old N(u).
        # So we iterate over old neighbors of u.
        
        # Similarly for v.
        
        # To be safe, we can iterate over all w and check if A[u][w] is 1.
        # But we just set A[u][v]=1.
        # If we iterate over all w where A[u][w]==1, we include v.
        # If w=v, C[v][v] += 1.
        # Is this correct?
        # C[v][v] is degree of v.
        # Degree of v increases by 1 (neighbor u).
        # So C[v][v] should increase by 1.
        # If we do C[w][v] += 1 for w=v, we get C[v][v] += 1. Correct.
        # But we also do C[v][w] += 1 which is C[v][v] += 1.
        # So we would add 2.
        
        # So we must only add 1 to C[w][v] for each w.
        
        # Let's just iterate and add 1 to C[w][v] and C[v][w] is redundant if symmetric.
        # But C is symmetric.
        
        # Correct Update:
        # For all w:
        #   if A[u][w] == 1: C[w][v] += 1
        #   if A[v][w] == 1: C[w][u] += 1
        
        # Let's check w=v for first part: A[u][v] is 1. C[v][v] += 1.
        # Let's check w=u for second part: A[v][u] is 1. C[u][u] += 1.
        # Let's check w=v for second part: A[v][v] is 0. No change.
        # Let's check w=u for first part: A[u][u] is 0. No change.
        
        # What about C[v][u]?
        # It is updated by:
        # 1. w=v in first loop? C[v][v] += 1. No, C[w][v] with w=v is C[v][v].
        # 2. w=u in second loop? C[u][u] += 1. No, C[w][u] with w=u is C[u][u].
        # Where does C[v][u] get updated?
        # C[v][u] is the count of common neighbors of v and u.
        # It should NOT change.
        # My update logic `C[w][v] += 1` updates C[w][v].
        # If w is a neighbor of u, C[w][v] increases.
        # This does NOT update C[v][u] unless w=v?
        # If w=v, C[v][v] increases.
        # So C[v][u] is NOT updated. Correct.
        
        for w in range(n):
            if A[u][w] == 1:
                C[w][v] += 1
            if A[v][w] == 1:
                C[w][u] += 1
        
        # Symmetry check: C is symmetric.
        # C[w][v] += 1 implies C[v][w] += 1?
        # The constraint is on C[i][j] for i!=j.
        # If we only update C[w][v], we break symmetry.
        # We must update C[v][w] as well.
        # But wait.
        # If I update C[w][v], I should update C[v][w].
        # My loop `if A[u][w] == 1: C[w][v] += 1` does not touch C[v][w].
        # So I must do `C[v][w] += 1` as well.
        
        # Let's re-run the logic.
        # For w in N(u):
        #   C[w][v] += 1
        #   C[v][w] += 1
        # For w in N(v):
        #   C[w][u] += 1
        #   C[u][w] += 1
        
        # Let's check C[v][v] again.
        # w=v in N(u): C[v][v] += 1, C[v][v] += 1. Total +2.
        # But degree +1.
        # Why?
        # Because w=v is in N(u).
        # And w=v is NOT in N(v).
        # So only the first loop touches C[v][v].
        # But the first loop adds 2?
        # `C[w][v] += 1` and `C[v][w] += 1`.
        # If w=v, these are the same cell.
        # So we add 2.
        # This is the bug.
        
        # Fix: Only add 1 if w != v?
        # If w != v: C[w][v] and C[v][w] are distinct. Add 1 to each.
        # If w == v: C[v][v] is one cell. Add 1.
        
        # Same for u.
        
        for w in range(n):
            if A[u][w] == 1:
                if w == v:
                    C[v][v] += 1
                else:
                    C[w][v] += 1
                    C[v][w] += 1
            if A[v][w] == 1:
                if w == u:
                    C[u][u] += 1
                else:
                    C[w][u] += 1
                    C[u][w] += 1

    def update_C_remove(A, C, u, v):
        # Remove edge (u, v)
        # Reverse of add.
        # But we must use the state BEFORE removal?
        # No, we can just subtract.
        # The neighbors are the same (u is in N(v), v is in N(u)).
        
        for w in range(n):
            if A[u][w] == 1:
                if w == v:
                    C[v][v] -= 1
                else:
                    C[w][v] -= 1
                    C[v][w] -= 1
            if A[v][w] == 1:
                if w == u:
                    C[u][u] -= 1
                else:
                    C[w][u] -= 1
                    C[u][w] -= 1
        
        A[u][v] = 0
        A[v][u] = 0

    def is_valid_add(A, C, u, v):
        if A[u][v] == 1:
            return False
        if C[u][v] > 2:
            return False
        for w in range(n):
            if A[u][w] == 1:
                if C[w][v] > 1:
                    return False
            if A[v][w] == 1:
                if C[w][u] > 1:
                    return False
        return True

    # Initialize
    A, C = initialize_greedy()
    
    # Better initialization: Random regular-ish graph
    # Let's just run SA from the greedy start.
    
    best_A = A.copy()
    best_C = C.copy()
    best_count = np.sum(A) // 2
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.05
    
    # Precompute list of edges for swap?
    # Just random selection.
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        current_count = np.sum(A) // 2
        
        r = random.random()
        
        if r < 0.4:
            # Add edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or A[u][v] == 1:
                continue
                
            if is_valid_add(A, C, u, v):
                update_C_add(A, C, u, v)
                new_count = current_count + 1
                if new_count > best_count:
                    best_count = new_count
                    best_A = A.copy()
                    best_C = C.copy()
            else:
                # Reject
                pass
                
        elif r < 0.7:
            # Remove edge
            # Pick random edge
            u = random.randint(0, n - 1)
            neighbors = np.where(A[u] == 1)[0]
            if len(neighbors) == 0:
                continue
            v = neighbors[random.randint(0, len(neighbors) - 1)]
            
            update_C_remove(A, C, u, v)
            new_count = current_count - 1
            # Energy increase (we lose an edge)
            delta = 1.0
            if random.random() < math.exp(-delta / T):
                if new_count > best_count:
                    best_count = new_count
                    best_A = A.copy()
                    best_C = C.copy()
            else:
                # Undo
                update_C_add(A, C, u, v)
                
        else:
            # Swap: remove (u1, v1), add (u2, v2)
            u1 = random.randint(0, n - 1)
            neighbors1 = np.where(A[u1] == 1)[0]
            if len(neighbors1) == 0:
                continue
            v1 = neighbors1[random.randint(0, len(neighbors1) - 1)]
            
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2 or A[u2][v2] == 1:
                continue
            
            # Temporarily remove (u1, v1)
            update_C_remove(A, C, u1, v1)
            
            # Check if (u2, v2) can be added
            if is_valid_add(A, C, u2, v2):
                update_C_add(A, C, u2, v2)
                # Count is same
                # Accept with prob 1 (or high) to explore
                # But we should check if it's better? No, same count.
                # Just accept to move around.
                pass
            else:
                # Undo remove
                update_C_add(A, C, u1, v1)
                
        T *= cooling

    # Return best
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u][v] == 1:
                edges.append([u, v])
                
    return edges
