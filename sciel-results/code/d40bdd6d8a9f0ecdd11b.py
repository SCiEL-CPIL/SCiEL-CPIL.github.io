import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 52  # Safety margin (90s kill, 52s work)
    
    # Precompute conflict masks:
    # For each pair (u, v) with u < v, conflict_mask[u][v] is a bitmask of all w
    # such that {u, v, w} forms a forbidden triple.
    # A forbidden triple {a, b, c} (distinct) exists iff a+b, a+c, b+c are all in S.
    # For a fixed pair (u, v), w forms a forbidden triple with them iff:
    #   u+w in S AND v+w in S AND (u+v in S or u+v == w)
    # But since we're checking if adding w would create a violation given current S,
    # we need: for the current set S, does adding w create any forbidden triple?
    # 
    # Actually, let's think differently. We precompute for each pair (u,v):
    #   conflict_mask[u][v] = bitmask of all w such that {u,v,w} would be a forbidden
    #   triple IF all three pair sums were in S. But the pair sums depend on S.
    #
    # Better approach: For each element w, precompute which pairs (u,v) would be "activated"
    # by w's presence. Actually, let's use a different decomposition.
    #
    # When we add x to S, a violation occurs iff there exist distinct a,b in S (a != b, a != x, b != x)
    # such that: a+b in S, a+x in S, b+x in S.
    # OR: x is one of the triple members.
    #
    # Let me restructure. For adding x to S:
    # Case 1: x is a triple member. Need a,b in S, a!=b, a!=x, b!=x, s.t. a+x in S, b+x in S, a+b in S.
    # Case 2: x is a pair sum. Need a,b,c in S, all distinct, s.t. one of {a+b, a+c, b+c} == x,
    #          and the other two are in S.
    #
    # For Case 1: Let K = {a in S : a+x in S}. Need a,b in K, a!=b, s.t. a+b in S.
    #   We can precompute for each x, and given K, check if any pair in K has sum in S.
    #   With bitmask: K_mask = bitmask of elements a in S where a+x in S.
    #   Then we need to check if there exist a,b in K_mask, a<b, such that (a+b) is in S_mask.
    #   This is still not O(1) unless we precompute more.
    #
    # Alternative: Precompute for every pair (u,v) the set of w that would make {u,v,w} forbidden,
    # but this depends on S. 
    #
    # Let's use a hybrid: precompute static structures, use bitmask for fast membership.
    
    # Represent S as a bitmask (Python int with up to 128 bits)
    # bit i (0-indexed) represents element (i+1) in S
    
    # Precompute for each x in 1..128:
    #   partner_mask[x] = bitmask of all a in 1..128 such that a+x <= 128
    #   (i.e., a+x is a valid element)
    
    partner_mask = [0] * (n + 1)
    for x in range(1, n + 1):
        m = 0
        for a in range(1, n + 1):
            if a + x <= n:
                m |= (1 << a)
        partner_mask[x] = m
    
    # For Case 1 of adding x: K = S ∩ partner_mask[x]
    #   K_mask = S_mask & partner_mask[x]
    #   Need to check: exists a,b in K, a<b, such that (a+b) in S.
    #   Precompute for each possible K? No, too many.
    #   Instead, iterate over bits in K if small, or use a precomputed table.
    
    # Precompute sum_in_S check: For a given S_mask, and a set K_mask,
    # check if any pair in K has sum in S.
    # We can precompute for each pair (a,b), a<b, the sum a+b.
    # Then check if (1 << (a+b)) & S_mask != 0.
    
    # For speed, precompute all pairs and their sums
    all_pairs = []  # list of (a, b, a+b)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            all_pairs.append((a, b, a + b))
    
    # For Case 2: x is a pair sum.
    #   For each pair (a,b) with a+b == x, a<b, a,b in S:
    #     Need c in S, c != a, c != b, s.t. a+c in S and b+c in S.
    #   Precompute for each x, the list of pairs (a,b) with a+b == x.
    
    sum_to_pairs = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            s = a + b
            if s <= n:
                sum_to_pairs[s].append((a, b))
    
    # For the inner check: given a,b,x,S_mask, check if exists c in S, c!=a, c!=b,
    # such that a+c in S and b+c in S.
    # c must be in S ∩ partner_mask[a] ∩ partner_mask[b], excluding a and b.
    # c_mask = S_mask & partner_mask[a] & partner_mask[b]
    # c_mask &= ~(1 << a) & ~(1 << b)
    # if c_mask != 0: violation
    
    def is_valid_addition(S_mask, x):
        """Check if adding x to S (represented by S_mask) keeps S valid."""
        # Case 1: x is a triple member
        # K = elements a in S such that a+x in S
        K_mask = S_mask & partner_mask[x]
        
        # Check if any pair in K has sum in S
        # Iterate over bits in K_mask
        km = K_mask
        while km:
            # Get lowest set bit
            lb = km & (-km)
            a = lb.bit_length() - 1
            km ^= lb
            # Check all b in K_mask, b > a
            kb = K_mask & ~((1 << (a + 1)) - 1)  # bits above a
            while kb:
                lb2 = kb & (-kb)
                b = lb2.bit_length() - 1
                kb ^= lb2
                ab = a + b
                if ab <= n and (S_mask >> ab) & 1:
                    return False
        
        # Case 2: x is a pair sum
        for (a, b) in sum_to_pairs[x]:
            if a > n or b > n:
                continue
            # Check if a and b are in S
            if not ((S_mask >> a) & 1) or not ((S_mask >> b) & 1):
                continue
            # Check if there exists c in S, c != a, c != b, s.t. a+c in S and b+c in S
            c_mask = S_mask & partner_mask[a] & partner_mask[b]
            c_mask &= ~(1 << a)
            c_mask &= ~(1 << b)
            if c_mask:
                return False
        
        return True
    
    def try_add(S_mask, x):
        if (S_mask >> x) & 1:
            return S_mask, False
        if is_valid_addition(S_mask, x):
            return S_mask | (1 << x), True
        return S_mask, False
    
    # Base set: {65, ..., 128}
    base_mask = 0
    for i in range(65, n + 1):
        base_mask |= (1 << i)
    
    candidates = list(range(1, 65))
    
    best_size = n - 64  # 64 elements
    best_mask = base_mask
    
    def mask_to_list(mask):
        return [i for i in range(1, n + 1) if (mask >> i) & 1]
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        c = candidates.copy()
        random.shuffle(c)
        
        current_mask = base_mask
        
        # Greedy add
        for x in c:
            if (current_mask >> x) & 1:
                continue
            current_mask, _ = try_add(current_mask, x)
        
        size = bin(current_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = current_mask
        
        # Local search: try swaps
        if time.time() < deadline - 3:
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                elements_in_set = mask_to_list(current_mask)
                random.shuffle(elements_in_set)
                
                for y in elements_in_set:
                    if time.time() > deadline - 1:
                        break
                    
                    # Remove y
                    trial_mask = current_mask & ~(1 << y)
                    
                    # Try to add candidates
                    added_any = False
                    c_local = [z for z in candidates if not (trial_mask >> z) & 1]
                    random.shuffle(c_local)
                    
                    for z in c_local:
                        new_mask, added = try_add(trial_mask, z)
                        if added:
                            trial_mask = new_mask
                            added_any = True
                            # Continue trying to add more
                        if time.time() > deadline - 1:
                            break
                    
                    if added_any and bin(trial_mask).count('1') > size:
                        current_mask = trial_mask
                        size = bin(current_mask).count('1')
                        improved = True
                        if size > best_size:
                            best_size = size
                            best_mask = current_mask
                        break  # Restart local search with new set
                    elif added_any:
                        # Same size but different configuration, might allow more later
                        current_mask = trial_mask
                        improved = True
                        break
    
    return mask_to_list(best_mask)
