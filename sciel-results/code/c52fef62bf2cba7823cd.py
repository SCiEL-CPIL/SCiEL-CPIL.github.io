import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    
    # Precompute 3-APs and 4-APs
    # Element i (1-indexed) corresponds to index (i-1) in arrays
    # We use Python integers as bitmasks for fast set operations
    
    aps3 = []  # List of (a, b, c) 0-indexed
    aps4 = []  # List of (a, b, c, d) 0-indexed
    
    # For each element, precompute masks of elements that form 3-APs with it
    # mask3_start[v] : bitmask of 'a' such that (a, v, c) is a 3-AP (v is middle)
    # mask3_mid1[v]  : bitmask of 'b' such that (a, b, v) is a 3-AP (v is end)
    # mask3_mid2[v]  : bitmask of 'a' such that (v, b, c) is a 3-AP (v is start)
    # Actually, let's simplify:
    # For a given set A (bitmask), the number of 3-APs is sum over all 3-APs (a,b,c) of (A[a] & A[b] & A[c]).
    # To speed up delta calculation for adding v:
    # gain(v) = number of pairs (a,c) in A such that (a,v,c) or (v,b,c) or (a,b,v) is a 3-AP.
    # Specifically:
    # 1. v is middle: (a, v, c). Need a, c in A. a = v-d, c = v+d.
    # 2. v is end: (a, b, v). Need a, b in A. b = v-d, a = v-2d.
    # 3. v is start: (v, b, c). Need b, c in A. b = v+d, c = v+2d.
    
    # Let's precompute for each v:
    # pairs_mid[v]: list of (a, c) where (a, v, c) is 3-AP
    # pairs_end[v]: list of (a, b) where (a, b, v) is 3-AP
    # pairs_start[v]: list of (b, c) where (v, b, c) is 3-AP
    
    # Even faster: use bitmasks.
    # For v to be middle: count bits in (A & mask_mid1[v]) where mask_mid1[v] has bits at positions 'a' 
    # such that 'c' is also in A? No, that's not direct.
    # If v is middle, a = v-d, c = v+d.
    # We need to count d such that A[v-d] and A[v+d] are set.
    # This is equivalent to counting intersections of A with shifted versions of A.
    
    # Let's stick to precomputed lists of partners for each element, but use numpy/bitsets for speed if possible.
    # Given n=256, we can fit A in a Python int.
    # A is an int with bit i set if element i+1 is in A.
    
    # Precomputation for fast gain/loss:
    # For each v (0..255):
    #   mid_pairs[v] = bitmask of 'a' such that (a, v, a+2d) is valid? No.
    #   Let's define:
    #   left_mask[v]: bitmask where bit 'a' is set if (a, v, c) is a 3-AP. i.e., a = v-d, c = v+d.
    #   right_mask[v]: bitmask where bit 'c' is set if (a, v, c) is a 3-AP.
    #   To count 3-APs where v is middle: popcount( A & left_mask[v] & something? )
    #   No, if bit 'a' is in left_mask[v], then 'c' is determined by 'a' (c = 2v - a).
    #   We need A[a] AND A[c].
    #   This is not a simple bitwise AND of two masks unless we precompute the "partner" mask.
    
    # Alternative: Precompute for each v, a list of (other1, other2) pairs.
    # Since n=256, the number of pairs per element is small (~O(n)).
    # Total 3-APs is ~ O(n^2) = 32k.
    # Iterating over ~50-100 pairs per move is very fast in Python if we use simple list lookups.
    # The bottleneck in the previous code was likely the overhead of numpy arrays or complex logic.
    # Let's use pure Python lists and ints for A.
    
    elem_to_aps3 = [[] for _ in range(n)]
    # For each AP (a,b,c), we store the indices.
    # We will store for each element v, the list of (u, w) such that (v,u,w) or (u,v,w) or (u,w,v) is a 3-AP.
    # Actually, it's better to store the AP index and the role, or just the other two elements.
    
    # Let's precompute:
    # neighbors3[v] = list of tuples (u, w) such that {v, u, w} forms a 3-AP.
    # If we add v, gain = count of (u, w) in neighbors3[v] where A[u] and A[w].
    
    # Similarly for 4-APs:
    # neighbors4[v] = list of tuples (u, w, x) such that {v, u, w, x} forms a 4-AP.
    # If we add v, conflict if A[u] and A[w] and A[x].
    
    # This list-based approach with Python ints for A should be extremely fast.
    
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            b = a + d
            c = a + 2 * d
            if c < n:
                # AP: a, b, c
                elem_to_aps3[a].append((b, c))
                elem_to_aps3[b].append((a, c))
                elem_to_aps3[c].append((a, b))
                
    elem_to_aps4 = [[] for _ in range(n)]
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                # AP: a, b, c, e
                elem_to_aps4[a].append((b, c, e))
                elem_to_aps4[b].append((a, c, e))
                elem_to_aps4[c].append((a, b, e))
                elem_to_aps4[e].append((a, b, c))

    # Helper to count 3-APs in a set represented as int bitmask
    def count_3aps(A):
        count = 0
        # Iterate over all 3-APs
        # To avoid double counting, we only count if a < b < c
        # Our precomputation added all permutations. 
        # It is safer to iterate over the canonical list of 3-APs.
        # Let's keep a canonical list.
        return 0 # Placeholder, we'll calculate incrementally or use canonical list
        
    # Let's build canonical lists for final verification and initial count
    canonical_aps3 = []
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            b = a + d
            c = a + 2 * d
            if c < n:
                canonical_aps3.append((a, b, c))
                
    canonical_aps4 = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e < n:
                canonical_aps4.append((a, b, c, e))

    def count_3aps_mask(A):
        count = 0
        for a, b, c in canonical_aps3:
            if (A >> a) & 1 and (A >> b) & 1 and (A >> c) & 1:
                count += 1
        return count

    def count_4aps_mask(A):
        count = 0
        for a, b, c, d in canonical_aps4:
            if (A >> a) & 1 and (A >> b) & 1 and (A >> c) & 1 and (A >> d) & 1:
                count += 1
        return count

    def gain_add(A, v):
        gain = 0
        for u, w in elem_to_aps3[v]:
            if (A >> u) & 1 and (A >> w) & 1:
                gain += 1
        return gain

    def loss_remove(A, u):
        loss = 0
        for w, x in elem_to_aps3[u]:
            if (A >> w) & 1 and (A >> x) & 1:
                loss += 1
        return loss

    def find_conflicts(A, v):
        conflicts = []
        for u, w, x in elem_to_aps4[v]:
            if (A >> u) & 1 and (A >> w) & 1 and (A >> x) & 1:
                conflicts.append((u, w, x))
        return conflicts

    def run_sa(seed, time_limit):
        rng = random.Random(seed)
        
        # Greedy Initialization
        A = 0
        candidates = list(range(n))
        rng.shuffle(candidates)
        for x in candidates:
            # Check if adding x creates 4-AP
            valid = True
            for u, w, z in elem_to_aps4[x]:
                if (A >> u) & 1 and (A >> w) & 1 and (A >> z) & 1:
                    valid = False
                    break
            if valid:
                A |= (1 << x)
                
        current_score = count_3aps_mask(A)
        best_score = current_score
        best_A = A
        
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.999
        
        while time.time() < time_limit:
            if T < T_min:
                break
                
            # Choose move: Add, Remove, Swap
            move_type = rng.choice(['add', 'remove', 'swap'])
            
            delta = 0
            # Store changes for revert: list of (index, old_val)
            changes = []
            
            if move_type == 'add':
                # Pick random v not in A
                # Efficient way: find a 0 bit
                # Since n=256, we can just pick random int and check, or maintain list of empty.
                # For simplicity and speed, pick random int, if set, retry a few times or pick from list.
                # Let's use a list of empty indices for O(1) pick.
                # But maintaining list is O(n) to update. 
                # Given n=256, picking random int and checking is fast enough if density is moderate.
                # If A is dense, we might loop. 
                # Let's just pick random int.
                v = rng.randint(0, n - 1)
                if (A >> v) & 1:
                    continue
                
                gain = gain_add(A, v)
                conflicts = find_conflicts(A, v)
                
                if conflicts:
                    # Pick a random conflict to resolve by removing one of its elements
                    u, w, z = conflicts[rng.randint(0, len(conflicts) - 1)]
                    # Pick one of u, w, z to remove
                    # We must pick one that is in A (they are, by definition of conflict)
                    # Let's pick the one with highest "cost" (loss_remove) to minimize damage? 
                    # Or random. Random is simpler.
                    rem = rng.choice([u, w, z])
                    
                    loss = loss_remove(A, rem)
                    delta = gain - loss
                    
                    A |= (1 << v)
                    changes.append((v, 0))
                    A &= ~(1 << rem)
                    changes.append((rem, 1))
                else:
                    delta = gain
                    A |= (1 << v)
                    changes.append((v, 0))
                    
            elif move_type == 'remove':
                # Pick random u in A
                u = rng.randint(0, n - 1)
                if not ((A >> u) & 1):
                    continue
                
                loss = loss_remove(A, u)
                delta = -loss
                A &= ~(1 << u)
                changes.append((u, 1))
                
            else: # swap
                # Pick random u in A, v not in A
                u = rng.randint(0, n - 1)
                if not ((A >> u) & 1):
                    continue
                v = rng.randint(0, n - 1)
                if (A >> v) & 1:
                    continue
                    
                if u == v:
                    continue
                
                # Calculate delta
                # 1. Loss from removing u
                loss_u = loss_remove(A, u)
                
                # 2. Gain from adding v
                # Gain is number of pairs (w,x) in elem_to_aps3[v] such that w,x in A\{u}
                gain_v = 0
                for w, x in elem_to_aps3[v]:
                    # w and x must be in A, and not u
                    if ((A >> w) & 1) and ((A >> x) & 1) and w != u and x != u:
                        gain_v += 1
                        
                delta = gain_v - loss_u
                
                # Check 4-AP validity for swap
                # Remove u, Add v.
                # New set A' = A \ {u} U {v}
                # Check if v completes a 4-AP with elements in A\{u}
                valid = True
                for w, x, y in elem_to_aps4[v]:
                    # w, x, y must be in A\{u}
                    if ((A >> w) & 1) and ((A >> x) & 1) and ((A >> y) & 1):
                        # Ensure none of them is u
                        if w != u and x != u and y != u:
                            valid = False
                            break
                            
                if not valid:
                    continue
                
                A &= ~(1 << u)
                changes.append((u, 1))
                A |= (1 << v)
                changes.append((v, 0))
                
            # Acceptance
            if delta > 0:
                accept = True
            else:
                try:
                    prob = math.exp(delta / T)
                except OverflowError:
                    prob = 0.0
                accept = (rng.random() < prob)
            
            if accept:
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = A
            else:
                # Revert
                for idx, old_val in changes:
                    if old_val:
                        A |= (1 << idx)
                    else:
                        A &= ~(1 << idx)
                        
            T *= cooling_rate

        return best_A, best_score

    # Main logic
    best_A_global = 0
    best_score_global = 0
    
    # Run multiple SA instances with different seeds
    # Estimate time per run
    num_runs = 5
    time_per_run = (deadline - time.time()) / num_runs
    
    for i in range(num_runs):
        if time.time() > deadline - 5:
            break
        A, score = run_sa(i, time.time() + time_per_run - 1)
        if score > best_score_global:
            best_score_global = score
            best_A_global = A
            
    # Convert best_A_global to sorted list
    best_set = [i+1 for i in range(n) if (best_A_global >> i) & 1]
    
    # Final verification
    if count_4aps_mask(best_A_global) != 0:
        # Fallback to greedy
        A = 0
        candidates = list(range(n))
        random.shuffle(candidates)
        for x in candidates:
            valid = True
            for u, w, z in elem_to_aps4[x]:
                if (A >> u) & 1 and (A >> w) & 1 and (A >> z) & 1:
                    valid = False
                    break
            if valid:
                A |= (1 << x)
        best_set = [i+1 for i in range(n) if (A >> i) & 1]

    return sorted(best_set)

# EVOLVE-BLOCK-END
