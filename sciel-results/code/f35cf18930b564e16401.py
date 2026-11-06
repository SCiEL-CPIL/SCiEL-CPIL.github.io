import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute bad triples and build bitset conflict structures
    # conflict_bits[x] = bitmask of elements y such that there exists z where {x,y,z} is bad
    # For feasibility: x can be added to S iff for all y in S, no z in S with (x,y,z) bad
    # Better: for each x, precompute for each y the set of z that form bad triple with x,y
    # Then x can be added iff for all y in S: (conflict_z[x][y] & S_mask) == 0
    
    # Precompute all bad triples
    bad_pairs_for = [set() for _ in range(n + 1)]
    # conflict_z[x][y] = bitmask of z such that {x,y,z} is a bad triple
    conflict_z = [[0] * (n + 1) for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    # Triple (a, b, c) is bad
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))
                    # Set bits in conflict_z
                    conflict_z[a][b] |= (1 << c)
                    conflict_z[b][a] |= (1 << c)
                    conflict_z[a][c] |= (1 << b)
                    conflict_z[c][a] |= (1 << b)
                    conflict_z[b][c] |= (1 << a)
                    conflict_z[c][b] |= (1 << a)
    
    def can_add_bitset(x, S_mask):
        """Check if x can be added to set represented by S_mask using bitset operations."""
        # For each y in S_mask, check if conflict_z[x][y] intersects S_mask
        # We need to iterate over set bits in S_mask
        m = S_mask
        while m:
            # Get lowest set bit
            lb = m & (-m)
            y = lb.bit_length() - 1
            if conflict_z[x][y] & S_mask:
                return False
            m ^= lb
        return True
    
    def greedy(order, S_mask=0, S_set=None):
        if S_set is None:
            S_set = set()
        selected = S_set
        for x in order:
            if can_add_bitset(x, S_mask):
                selected.add(x)
                S_mask |= (1 << x)
        return selected, S_mask

    def local_search(S):
        improved = True
        S = set(S)
        S_mask = 0
        for x in S:
            S_mask |= (1 << x)
        
        while improved and time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add any element
            for x in range(1, n + 1):
                if not (S_mask & (1 << x)) and can_add_bitset(x, S_mask):
                    S.add(x)
                    S_mask |= (1 << x)
                    improved = True
                    break
            
            if not improved:
                # 2. Try 1-for-2 swap: remove x, add two new elements
                found_swap = False
                s_list = list(S)
                random.shuffle(s_list)
                for x in s_list:
                    if time.time() > deadline - 2:
                        break
                    x_bit = 1 << x
                    S_trial_mask = S_mask & ~x_bit
                    S_trial_set = S - {x}
                    added = 0
                    for y in range(1, n + 1):
                        if not (S_trial_mask & (1 << y)) and can_add_bitset(y, S_trial_mask):
                            S_trial_set.add(y)
                            S_trial_mask |= (1 << y)
                            added += 1
                            if added == 2:
                                break
                    if added == 2:
                        S = S_trial_set
                        S_mask = S_trial_mask
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    continue

            if not improved:
                # 3. Try 2-for-3 swap
                found_swap = False
                s_list = list(S)
                random.shuffle(s_list)
                for i in range(len(s_list)):
                    if time.time() > deadline - 2:
                        break
                    for j in range(i + 1, len(s_list)):
                        S_trial_mask = S_mask & ~(1 << s_list[i]) & ~(1 << s_list[j])
                        S_trial_set = S - {s_list[i], s_list[j]}
                        added = 0
                        for y in range(1, n + 1):
                            if not (S_trial_mask & (1 << y)) and can_add_bitset(y, S_trial_mask):
                                S_trial_set.add(y)
                                S_trial_mask |= (1 << y)
                                added += 1
                                if added == 3:
                                    break
                        if added == 3:
                            S = S_trial_set
                            S_mask = S_trial_mask
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    continue
            
            # 4. Try 1-for-1 swap: remove x, add y (different element, same size but may enable further additions)
            if not improved:
                found_swap = False
                s_list = list(S)
                random.shuffle(s_list)
                for x in s_list:
                    if time.time() > deadline - 2:
                        break
                    x_bit = 1 << x
                    S_trial_mask = S_mask & ~x_bit
                    S_trial_set = S - {x}
                    # Try to add any single element, then see if we can add more
                    best_y = None
                    for y in range(1, n + 1):
                        if not (S_trial_mask & (1 << y)) and can_add_bitset(y, S_trial_mask):
                            # Check if this enables further additions
                            S2_mask = S_trial_mask | (1 << y)
                            S2_set = S_trial_set | {y}
                            extra = 0
                            for z in range(1, n + 1):
                                if not (S2_mask & (1 << z)) and can_add_bitset(z, S2_mask):
                                    S2_set.add(z)
                                    S2_mask |= (1 << z)
                                    extra += 1
                                    if extra >= 1:
                                        break
                            if extra >= 1:
                                best_y = (y, S2_set, S2_mask)
                                break
                    if best_y:
                        S = best_y[1]
                        S_mask = best_y[2]
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    continue
        
        return S

    best_set = set()
    best_size = 0
    
    # Precompute degrees for ordering
    degrees = [len(bad_pairs_for[k]) for k in range(n + 1)]
    
    # Run multiple randomized greedy passes
    num_restarts = 200
    
    for i in range(num_restarts):
        if time.time() > deadline - 12:
            break
        
        # Different ordering strategies
        if i % 7 == 0:
            # Pure random
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif i % 7 == 1:
            # Sort by degree (least constrained first) with noise
            order = list(range(1, n + 1))
            order.sort(key=lambda x: degrees[x] + random.random() * 0.5)
        elif i % 7 == 2:
            # Sort by degree descending (most constrained first)
            order = list(range(1, n + 1))
            order.sort(key=lambda x: -degrees[x] + random.random() * 0.5)
        elif i % 7 == 3:
            # Random with bias toward low-degree elements
            order = list(range(1, n + 1))
            weights = [1.0 / (degrees[x] + 1) for x in order]
            for pos in range(len(order) - 1, 0, -1):
                total = sum(weights[:pos + 1])
                r = random.random() * total
                cum = 0
                for k in range(pos + 1):
                    cum += weights[k]
                    if r <= cum:
                        order[pos], order[k] = order[k], order[pos]
                        weights[pos], weights[k] = weights[k], weights[pos]
                        break
        elif i % 7 == 4:
            # Reverse sorted by value
            order = list(range(n, 0, -1))
            random.shuffle(order)
        elif i % 7 == 5:
            # Sort by value (ascending) with small noise
            order = list(range(1, n + 1))
            order.sort(key=lambda x: x + random.random() * 2.0)
        else:
            # Sort by value descending with small noise
            order = list(range(1, n + 1))
            order.sort(key=lambda x: -x + random.random() * 2.0)
        
        S, _ = greedy(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Final polish on best_set with multiple local search passes
    for _ in range(3):
        if time.time() > deadline - 3:
            break
        best_set = local_search(best_set)
    
    return list(best_set)
