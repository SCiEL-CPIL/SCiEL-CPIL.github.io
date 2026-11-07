"""Rewritten program: bitmask-based feasibility + swap local search."""
import time
import random

_N = 128

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, 5s safety margin

    # Precompute for each x, the list of (a,b,c) triples where x plays a role
    # For fast membership: use a bitmask (128 bits fits in Python int)
    
    # Precompute all bad triples: distinct a<b<c with a+b, a+c, b+c all in 1..N
    # Store as (a, b, c, s1, s2, s3)
    _triples = []
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            for c in range(b + 1, _N + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 <= _N and s2 <= _N and s3 <= _N:
                    _triples.append((a, b, c, s1, s2, s3))
    
    # For each element x, precompute which triples involve x as a "member" (a,b,c)
    # and which as a "sum" (s1,s2,s3)
    # triples_as_member[x] = list of indices into _triples where x is a,b,or c
    # triples_as_sum[x] = list of indices into _triples where x is s1,s2,or s3
    triples_as_member = [[] for _ in range(_N + 1)]
    triples_as_sum = [[] for _ in range(_N + 1)]
    for i, (a, b, c, s1, s2, s3) in enumerate(_triples):
        for v in (a, b, c):
            triples_as_member[v].append(i)
        for v in (s1, s2, s3):
            triples_as_sum[v].append(i)
    
    def _can_add(mask, x):
        """Check if adding x to the set represented by mask keeps it valid."""
        # A new violation must involve x.
        # Case 1: x is a member (a,b,c) in some triple, and the other members + all sums are in set∪{x}
        for ti in triples_as_member[x]:
            a, b, c, s1, s2, s3 = _triples[ti]
            # Check all of a,b,c are in mask∪{x}
            if ((mask >> (a-1)) & 1) or a == x:
                if ((mask >> (b-1)) & 1) or b == x:
                    if ((mask >> (c-1)) & 1) or c == x:
                        # All three members present. Check sums.
                        if ((mask >> (s1-1)) & 1) or s1 == x:
                            if ((mask >> (s2-1)) & 1) or s2 == x:
                                if ((mask >> (s3-1)) & 1) or s3 == x:
                                    return False
        # Case 2: x is a sum (s1,s2,s3) in some triple, and all members + other sums are in set
        for ti in triples_as_sum[x]:
            a, b, c, s1, s2, s3 = _triples[ti]
            # x is one of s1,s2,s3. Check a,b,c in mask (x not in a,b,c since sums > max)
            if ((mask >> (a-1)) & 1) and ((mask >> (b-1)) & 1) and ((mask >> (c-1)) & 1):
                # All three members present. Check the other two sums (and x itself counts).
                if ((mask >> (s1-1)) & 1) or s1 == x:
                    if ((mask >> (s2-1)) & 1) or s2 == x:
                        if ((mask >> (s3-1)) & 1) or s3 == x:
                            return False
        return True
    
    def _is_valid(mask):
        """Check if the set represented by mask is valid."""
        for (a, b, c, s1, s2, s3) in _triples:
            if ((mask >> (a-1)) & 1) and ((mask >> (b-1)) & 1) and ((mask >> (c-1)) & 1):
                if ((mask >> (s1-1)) & 1) and ((mask >> (s2-1)) & 1) and ((mask >> (s3-1)) & 1):
                    return False
        return True
    
    def _greedy(order):
        """Greedy construction with given order."""
        mask = 0
        for x in order:
            if not ((mask >> (x-1)) & 1):
                if _can_add(mask, x):
                    mask |= (1 << (x-1))
        return mask
    
    def _local_search(mask, max_iters, dl):
        """Local search: try adds, then swaps to escape local minima."""
        for it in range(max_iters):
            if time.time() > dl:
                break
            # Try to add any element
            added = False
            for x in range(1, _N + 1):
                if not ((mask >> (x-1)) & 1):
                    if _can_add(mask, x):
                        mask |= (1 << (x-1))
                        added = True
                        break
            if added:
                continue
            # No add possible. Try swaps: remove y, add x (x != y, both not both in set)
            # We want to try swaps that might enable future adds.
            # Strategy: for each y in set, try removing it, then see if we can add any x not in set
            # If we can add x, keep the swap. Prefer swaps where x > y or where it enables more adds.
            swapped = False
            elements_in_set = [x for x in range(1, _N + 1) if (mask >> (x-1)) & 1]
            random.shuffle(elements_in_set)
            for y in elements_in_set:
                if time.time() > dl:
                    break
                mask2 = mask & ~(1 << (y-1))
                # Try to add some x not in mask2
                for x in range(1, _N + 1):
                    if (mask2 >> (x-1)) & 1:
                        continue
                    if _can_add(mask2, x):
                        # Swap: remove y, add x
                        mask = mask2 | (1 << (x-1))
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                break
        return mask
    
    best_mask = 0
    
    # Strategy 1: Greedy from 128 down
    m1 = _greedy(range(_N, 0, -1))
    if m1.bit_count() > best_mask.bit_count():
        best_mask = m1
    
    # Strategy 2: Greedy from 1 to 128
    m2 = _greedy(range(1, _N + 1))
    if m2.bit_count() > best_mask.bit_count():
        best_mask = m2
    
    # Strategy 3: Random greedy orders
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, _N + 1))
        random.shuffle(order)
        m3 = _greedy(order)
        if m3.bit_count() > best_mask.bit_count():
            best_mask = m3
    
    # Local search on best
    best_mask = _local_search(best_mask, 100000, deadline)
    
    # Multiple restarts with perturbation
    for _ in range(20):
        if time.time() > deadline:
            break
        # Perturb: remove 1-3 random elements from best
        perturbed = best_mask
        elems = [x for x in range(1, _N + 1) if (perturbed >> (x-1)) & 1]
        for _ in range(random.randint(1, 3)):
            if elems:
                y = random.choice(elems)
                perturbed &= ~(1 << (y-1))
                elems.remove(y)
        improved = _local_search(perturbed, 50000, deadline)
        if improved.bit_count() > best_mask.bit_count():
            best_mask = improved
    
    # Final validation
    if not _is_valid(best_mask):
        # Fallback: upper half
        best_mask = 0
        for x in range(_N // 2 + 1, _N + 1):
            best_mask |= (1 << (x-1))
    
    result = [x for x in range(1, _N + 1) if (best_mask >> (x-1)) & 1]
    return result
