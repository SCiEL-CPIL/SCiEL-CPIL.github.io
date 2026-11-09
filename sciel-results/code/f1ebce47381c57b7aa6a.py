import time
import random

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 55  # Safety margin

    # Precompute conflict structure using bit-parallel representation
    # Each element i (1..128) corresponds to bit (i-1)
    # conflict_mask[x] = bitmask of elements that conflict with x (can't be in set together with x)
    # Two elements x, y conflict if adding both to a set would create a forbidden triple

    # A forbidden triple {a, b, c} means: a+b, a+c, b+c are all in the set
    # When we add x, it conflicts with y if there exist a, b such that:
    # {x, y, a} or {x, y, b} forms a forbidden triple, OR
    # x and y are sums that create forbidden triples

    # Let's precompute for each pair (u, v) the set of w such that {u, v, w} is a forbidden triple
    # Then two elements x, y conflict if there exists w such that {x, y, w} is forbidden
    # AND x+w, y+w, x+y are all in range [1, n]

    # Actually, let me think more carefully:
    # A set S is valid if for NO three distinct a, b, c in S do we have a+b, a+c, b+c all in S
    # 
    # When building S incrementally, adding x is invalid if:
    # 1. There exist a, b in S such that {x, a, b} is a forbidden triple
    #    i.e., x+a in S, x+b in S, a+b in S (and x+a, x+b, a+b are all distinct from x,a,b)
    #    Wait, the condition is: a+b, a+c, b+c are in S where {a,b,c} = {x,a,b}
    #    So we need: x+a in S, x+b in S, a+b in S
    # 2. x is one of the sums: x = a+b for some a, b in S, and there exists c in S
    #    such that a+c in S and b+c in S

    # For bit-parallel, let's precompute:
    # For each x, conflict_mask[x] = bitmask of all y such that x and y cannot both be in S
    #   without violating the constraint (given some other elements)
    # This is actually context-dependent, so we need a different approach.

    # Better approach: precompute for each x, the set of "witness pairs" that would make x invalid
    # Then use bit-parallel operations to check validity quickly

    # Let's precompute:
    # For each x, pairs_conflict[x] = list of (a, b) such that if a, b in S and 
    #   (x+a in S and x+b in S and a+b in S) or (x = a+b and exists c with a+c, b+c in S)
    #   then x cannot be added

    # Actually, let's use a cleaner formulation:
    # Precompute for each x:
    #   as_member_conflicts[x] = list of (a, b) such that {x, a, b} is a forbidden triple
    #     meaning x+a, x+b, a+b are all in [1,n] and distinct from {x,a,b}
    #   as_sum_conflicts[x] = list of (a, b) such that x = a+b and there exists c in [1,n]
    #     with c != a, c != b, a+c <= n, b+c <= n

    # When checking if x can be added to S:
    # For each (a, b) in as_member_conflicts[x]:
    #   if a in S and b in S and (x+a) in S and (x+b) in S and (a+b) in S: invalid
    # For each (a, b) in as_sum_conflicts[x]:
    #   if a in S and b in S:
    #     check if exists c in S with c != a, c != b, (a+c) in S, (b+c) in S: invalid

    # For bit-parallel speed, let's precompute masks:
    # For each x and each (a,b) in as_member_conflicts[x], create a mask that checks
    # if a, b, x+a, x+b, a+b are all in S

    # Let's precompute all the conflict data
    as_member_conflicts = [[] for _ in range(n + 1)]
    as_sum_conflicts = [[] for _ in range(n + 1)]

    # Build as_member_conflicts
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = a + b
            if ab > n:
                continue
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                ac = a + c
                bc = b + c
                if ac > n or bc > n:
                    continue
                # {a, b, c} is a forbidden triple if ab, ac, bc are all in S
                # So for element c, (a, b) is a conflict pair
                # For element a, (b, c) is a conflict pair
                # For element b, (a, c) is a conflict pair
                as_member_conflicts[a].append((b, c))
                as_member_conflicts[b].append((a, c))
                as_member_conflicts[c].append((a, b))

    # Build as_sum_conflicts
    for x in range(2, n + 1):
        for u in range(1, x):
            v = x - u
            if v <= u:
                continue
            if v > n:
                continue
            # Check if there exists w such that u+w <= n and v+w <= n and w != u, w != v
            # w can be 1..n, w != u, w != v, u+w <= n, v+w <= n
            # This is possible if u < n and v < n (since we can pick w=1 if 1 != u,v)
            if u < n and v < n:
                as_sum_conflicts[x].append((u, v))

    # Remove duplicates
    for i in range(1, n + 1):
        as_member_conflicts[i] = list(set(as_member_conflicts[i]))
        as_sum_conflicts[i] = list(set(as_sum_conflicts[i]))

    # Precompute bit masks for fast checking
    def elem_bit(x):
        return 1 << (x - 1)

    # For each x, precompute the masks needed for as_member_conflicts checks
    # member_check_masks[x] = list of (mask_a, mask_b, mask_xa, mask_xb, mask_ab)
    # where we need all five bits set in S for a conflict
    member_check_masks = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for (a, b) in as_member_conflicts[x]:
            xa = x + a
            xb = x + b
            ab = a + b
            if xa <= n and xb <= n and ab <= n:
                # Need a, b, xa, xb, ab all in S
                mask_a = elem_bit(a)
                mask_b = elem_bit(b)
                mask_xa = elem_bit(xa)
                mask_xb = elem_bit(xb)
                mask_ab = elem_bit(ab)
                # Combined mask: all five must be set
                combined = mask_a | mask_b | mask_xa | mask_xb | mask_ab
                member_check_masks[x].append(combined)

    # For as_sum_conflicts, we need: a, b in S AND exists c in S with a+c in S and b+c in S
    # This is harder to precompute as a single mask. Let's precompute for each (a,b) pair
    # the set of valid c's as a bitmask
    sum_check_data = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for (a, b) in as_sum_conflicts[x]:
            # Precompute mask of valid c's
            valid_c_mask = 0
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                ac = a + c
                bc = b + c
                if ac <= n and bc <= n:
                    valid_c_mask |= elem_bit(c)
            if valid_c_mask:
                # We need: a in S, b in S, and (S & valid_c_mask) has some c such that
                # (a+c) in S and (b+c) in S
                # This is still complex. Let's store (a, b, valid_c_mask)
                sum_check_data[x].append((a, b, valid_c_mask))

    # Base set: {65, ..., 128} - no conflicts within this set
    base_set = set(range(65, n + 1))
    base_mask = 0
    for x in base_set:
        base_mask |= elem_bit(x)

    candidates = list(range(1, 65))
    best_size = len(base_set)
    best_set = base_set.copy()
    best_mask = base_mask

    def is_valid_addition_mask(S_mask, x):
        """Check if adding x to set represented by S_mask is valid."""
        # Check as_member_conflicts
        for combined_mask in member_check_masks[x]:
            if (S_mask & combined_mask) == combined_mask:
                return False

        # Check as_sum_conflicts
        for (a, b, valid_c_mask) in sum_check_data[x]:
            mask_a = elem_bit(a)
            mask_b = elem_bit(b)
            if (S_mask & mask_a) and (S_mask & mask_b):
                # Need to check if exists c in S ∩ valid_c_mask such that
                # (a+c) in S and (b+c) in S
                # Iterate through bits in S ∩ valid_c_mask
                candidates_c = S_mask & valid_c_mask
                while candidates_c:
                    # Extract lowest set bit
                    lsb = candidates_c & (-candidates_c)
                    c = lsb.bit_length()  # This gives position, need to adjust
                    # Actually, bit_length gives 1-indexed position for the bit
                    # If bit (c-1) is set, bit_length of that power of 2 is c
                    c_val = lsb.bit_length()
                    ac = a + c_val
                    bc = b + c_val
                    if ac <= n and bc <= n:
                        if (S_mask & elem_bit(ac)) and (S_mask & elem_bit(bc)):
                            return False
                    candidates_c &= (candidates_c - 1)

        return True

    # Greedy with random restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1

        # Randomize candidate order
        c = candidates.copy()
        random.shuffle(c)

        current_mask = base_mask
        current_set = base_set.copy()

        # Greedy add
        for x in c:
            if current_mask & elem_bit(x):
                continue
            if is_valid_addition_mask(current_mask, x):
                current_mask |= elem_bit(x)
                current_set.add(x)

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            best_mask = current_mask

        # Local search with bit-parallel checks
        if time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                # Try removing one element and adding another
                for y in list(current_set):
                    if y >= 65:
                        continue  # Don't remove base elements in first pass
                    # Remove y
                    current_mask &= ~elem_bit(y)
                    current_set.remove(y)

                    # Try to add any candidate
                    added_any = False
                    for z in c:
                        if current_mask & elem_bit(z):
                            continue
                        if is_valid_addition_mask(current_mask, z):
                            current_mask |= elem_bit(z)
                            current_set.add(z)
                            added_any = True
                            break

                    if added_any:
                        improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = current_set.copy()
                            best_mask = current_mask
                    else:
                        # Put y back
                        current_mask |= elem_bit(y)
                        current_set.add(y)

    # Convert to list of Python ints
    result = [int(x) for x in best_set]
    return result
