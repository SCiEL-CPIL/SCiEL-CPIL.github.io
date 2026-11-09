import numpy as np
import time
import random

N = 16
NPOINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 295  # Safety margin before 300s limit

    # Index points as 0..NPOINTS-1, point i corresponds to (i//N, i%N)
    # Precompute conflict: for each point p, a bitset of points q such that
    # adding p and q to any set S would create an isosceles triangle with some r in S.
    # Actually, we need: p and q conflict if there exists r such that (p,q,r) is isosceles.
    # Wait, that's not quite right. The constraint is: no three points in the set form an isosceles triangle.
    # So p and q can coexist in the set AS LONG AS there's no r in the set such that (p,q,r) is isosceles.
    # This is a 3-uniform hypergraph problem, not a simple graph.
    #
    # However, we can reformulate: 
    # A set S is "no-isosceles" iff for all distinct p,q,r in S, (p,q,r) is not isosceles.
    # 
    # For branch-and-bound, we can use the following approach:
    # - Maintain a candidate set of points that can still be added.
    # - Use bitsets for efficiency.
    #
    # Let's precompute: for each pair (p, q), the set of points r such that (p, q, r) is isosceles.
    # Then, if we have a current set S, a point p can be added iff for all q in S, there's no r in S 
    # such that (p, q, r) is isosceles. But wait, that's not right either.
    #
    # Actually: p can be added to S iff for all pairs (q, r) in S, (p, q, r) is not isosceles.
    #
    # Let me precompute: iso[p][q] = set of r such that (p, q, r) is isosceles.
    # Then p can be added to S iff for all q in S, iso[p][q] ∩ S = empty.
    #
    # With bitsets, this check is: for all q in S, (iso_bits[p][q] & S_bits) == 0.

    # Precompute iso_bits[p][q] as a numpy uint64 array (or use Python ints for bitsets)
    # Since NPOINTS = 256, we need 256-bit integers. Python handles big integers natively.
    
    iso_bits = [[0] * NPOINTS for _ in range(NPOINTS)]
    
    for p in range(NPOINTS):
        px, py = p // N, p % N
        for q in range(NPOINTS):
            if q == p:
                continue
            qx, qy = q // N, q % N
            bits = 0
            for r in range(NPOINTS):
                if r == p or r == q:
                    continue
                rx, ry = r // N, r % N
                d1 = (px - qx)**2 + (py - qy)**2
                d2 = (px - rx)**2 + (py - ry)**2
                d3 = (qx - rx)**2 + (qy - ry)**2
                if d1 == d2 or d1 == d3 or d2 == d3:
                    bits |= (1 << r)
            iso_bits[p][q] = bits

    # Now, can_add(p, S_bits) checks if p can be added to set S.
    def can_add(p, S_bits):
        if S_bits & (1 << p):
            return False
        for q in range(NPOINTS):
            if S_bits & (1 << q):
                if iso_bits[p][q] & S_bits:
                    return False
        return True

    # For faster checking, let's precompute for each p, the list of q values.
    # Actually, let's optimize can_add by only iterating over set bits in S.
    
    def can_add_fast(p, S_bits):
        if S_bits & (1 << p):
            return False
        # Iterate over set bits in S_bits
        s = S_bits
        while s:
            lowbit = s & (-s)
            q = lowbit.bit_length() - 1
            s ^= lowbit
            if iso_bits[p][q] & S_bits:
                return False
        return True

    # is_maximal(S_bits): check no point outside S can be added
    def is_maximal(S_bits):
        for p in range(NPOINTS):
            if not (S_bits & (1 << p)):
                if can_add_fast(p, S_bits):
                    return False
        return True

    # Strategy: Branch and bound to find minimum size maximal no-isosceles set.
    # We'll use a recursive search with pruning.
    #
    # State: current set S (as bitset), candidate set C (points that can still be added).
    # Goal: find a maximal subset of C ∪ S with minimum |S ∪ subset|.
    #
    # Actually, let's think differently. We want to find a set S such that:
    # 1. S is no-isosceles
    # 2. S is maximal (no point can be added)
    # 3. |S| is minimized
    #
    # Branch and bound approach:
    # - Start with S = empty, C = all points.
    # - At each step, either add a point to S or exclude it.
    # - Prune if |S| >= best_found.
    # - Prune if S is already maximal and |S| < best_found.
    # - Use lower bound: |S| + minimum additional points needed.
    #
    # For the lower bound, we can use: if S is not maximal, we need at least 1 more point.
    # A better bound: compute the minimum number of points needed to make S maximal.
    # This is itself a covering problem, which is hard.
    #
    # Let's use a simpler approach: randomized local search with bitset operations.

    best_size = NPOINTS + 1
    best_set_bits = 0

    # Phase 1: Greedy construction with randomization
    for trial in range(200):
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(42 + trial)
        
        S_bits = 0
        # Shuffle order
        order = list(range(NPOINTS))
        rng.shuffle(order)
        
        for p in order:
            if can_add_fast(p, S_bits):
                S_bits |= (1 << p)
        
        # Check maximality
        if is_maximal(S_bits):
            size = S_bits.bit_count()
            if size < best_size:
                best_size = size
                best_set_bits = S_bits

    # Phase 2: Local search - try to remove points while maintaining maximality
    if best_set_bits and time.time() < deadline - 10:
        current_bits = best_set_bits
        
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for p in range(NPOINTS):
                if current_bits & (1 << p):
                    new_bits = current_bits & ~(1 << p)
                    if is_maximal(new_bits):
                        current_bits = new_bits
                        improved = True
                        break
            
            if not improved:
                # Try a swap: remove p, add q
                removed = False
                for p in range(NPOINTS):
                    if current_bits & (1 << p):
                        temp_bits = current_bits & ~(1 << p)
                        for q in range(NPOINTS):
                            if not (temp_bits & (1 << q)):
                                if can_add_fast(q, temp_bits):
                                    new_bits = temp_bits | (1 << q)
                                    # Check if we can then remove something
                                    for r in range(NPOINTS):
                                        if new_bits & (1 << r):
                                            check_bits = new_bits & ~(1 << r)
                                            if is_maximal(check_bits):
                                                current_bits = check_bits
                                                removed = True
                                                break
                                    if removed:
                                        break
                        if removed:
                            break
                
                if not removed:
                    break
            
            size = current_bits.bit_count()
            if size < best_size:
                best_size = size
                best_set_bits = current_bits

    # Phase 3: More aggressive local search with multiple restarts
    if time.time() < deadline - 20:
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(1000 + restart)
            
            # Start from a random maximal set
            S_bits = 0
            order = list(range(NPOINTS))
            rng.shuffle(order)
            
            for p in order:
                if can_add_fast(p, S_bits):
                    S_bits |= (1 << p)
            
            if not is_maximal(S_bits):
                continue
            
            size = S_bits.bit_count()
            
            # Try to minimize
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                improved = False
                # Try removing one point
                for p in range(NPOINTS):
                    if S_bits & (1 << p):
                        new_bits = S_bits & ~(1 << p)
                        if is_maximal(new_bits):
                            S_bits = new_bits
                            improved = True
                            break
                
                if not improved:
                    # Try swap
                    for p in range(NPOINTS):
                        if S_bits & (1 << p):
                            temp = S_bits & ~(1 << p)
                            for q in range(NPOINTS):
                                if not (temp & (1 << q)) and can_add_fast(q, temp):
                                    new_bits = temp | (1 << q)
                                    # Try to remove
                                    for r in range(NPOINTS):
                                        if new_bits & (1 << r):
                                            check = new_bits & ~(1 << r)
                                            if is_maximal(check):
                                                S_bits = check
                                                improved = True
                                                break
                                    if improved:
                                        break
                        if improved:
                            break
                
                if not improved:
                    break
                
                size = S_bits.bit_count()
                if size < best_size:
                    best_size = size
                    best_set_bits = S_bits

    # Convert best_set_bits to list of (x, y) tuples
    result = []
    for p in range(NPOINTS):
        if best_set_bits & (1 << p):
            result.append((p // N, p % N))

    if not result:
        # Fallback: return a simple valid maximal set
        S_bits = 0
        for p in range(NPOINTS):
            if can_add_fast(p, S_bits):
                S_bits |= (1 << p)
        for p in range(NPOINTS):
            if S_bits & (1 << p):
                result.append((p // N, p % N))

    return result
