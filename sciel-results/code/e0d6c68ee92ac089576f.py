"""Two Sidon sets with disjoint difference sets – Bitset + multi-start local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # safety margin before 60s kill

def _can_add_to_sidon(elem, set_elems, sum_bitset, diff_bitset, other_diff_bitset):
    """Check if elem can be added to set_elems maintaining Sidon + disjoint diffs."""
    # Check pair sums: elem + a for all a in set_elems, and elem+elem
    # Sum bitset tracks which sums are already used (1-indexed sums from 2 to 512)
    # Diff bitset tracks nonzero differences (1 to 255)
    
    # elem + elem
    s2 = elem + elem
    if s2 <= 512 and (sum_bitset >> s2) & 1:
        return False
    
    # elem + a for each a in set_elems
    for a in set_elems:
        s = elem + a
        if (sum_bitset >> s) & 1:
            return False
        # Check difference: |elem - a|
        d = abs(elem - a)
        if d > 0 and ((diff_bitset >> d) & 1 or (other_diff_bitset >> d) & 1):
            return False
    return True

def _add_to_sidon(elem, set_elems, sum_bitset, diff_bitset):
    """Update bitsets after adding elem to set."""
    s2 = elem + elem
    sum_bitset |= (1 << s2)
    for a in set_elems:
        s = elem + a
        sum_bitset |= (1 << s)
        d = abs(elem - a)
        if d > 0:
            diff_bitset |= (1 << d)
    return sum_bitset, diff_bitset

def _rebuild_bitsets(A, B):
    """Rebuild sum and diff bitsets from sets A and B."""
    diffA = 0
    diffB = 0
    sumA = 0
    sumB = 0
    for i, a in enumerate(A):
        sumA |= (1 << (a + a))
        for b in A[i+1:]:
            sumA |= (1 << (a + b))
        for b in A:
            if b != a:
                d = abs(a - b)
                diffA |= (1 << d)
    for i, b in enumerate(B):
        sumB |= (1 << (b + b))
        for c in B[i+1:]:
            sumB |= (1 << (b + c))
        for c in B:
            if c != b:
                d = abs(b - c)
                diffB |= (1 << d)
    return sumA, sumB, diffA, diffB

def _greedy_build(candidates, order, seed_diff_bitset=0):
    """Greedy build using given candidate list and order."""
    A = []
    B = []
    sumA = 0
    sumB = 0
    diffA = 0
    diffB = 0
    
    for x in order:
        # Try adding to A
        if len(A) < MAX_EACH:
            ok = True
            s2 = x + x
            if (sumA >> s2) & 1:
                ok = False
            else:
                for a in A:
                    s = x + a
                    if (sumA >> s) & 1:
                        ok = False
                        break
                    d = abs(x - a)
                    if d > 0 and ((diffA >> d) & 1 or (diffB >> d) & 1):
                        ok = False
                        break
            if ok:
                s2 = x + x
                sumA |= (1 << s2)
                for a in A:
                    s = x + a
                    sumA |= (1 << s)
                    d = abs(x - a)
                    if d > 0:
                        diffA |= (1 << d)
                A.append(x)
                continue
        
        # Try adding to B
        if len(B) < MAX_EACH:
            ok = True
            s2 = x + x
            if (sumB >> s2) & 1:
                ok = False
            else:
                for b in B:
                    s = x + b
                    if (sumB >> s) & 1:
                        ok = False
                        break
                    d = abs(x - b)
                    if d > 0 and ((diffB >> d) & 1 or (diffA >> d) & 1):
                        ok = False
                        break
            if ok:
                s2 = x + x
                sumB |= (1 << s2)
                for b in B:
                    s = x + b
                    sumB |= (1 << s)
                    d = abs(x - b)
                    if d > 0:
                        diffB |= (1 << d)
                B.append(x)
    
    return A, B

def _local_search(A, B, diffA, diffB, sumA, sumB, time_limit):
    """Local search: try adding elements and 1-for-1 swaps to increase total size."""
    A = list(A)
    B = list(B)
    start = time.time()
    
    # Convert to sets for faster membership checks
    setA = set(A)
    setB = set(B)
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        
        # Try adding new elements to A
        if len(A) < MAX_EACH:
            for x in range(1, N + 1):
                if x in setA:
                    continue
                ok = True
                s2 = x + x
                if (sumA >> s2) & 1:
                    ok = False
                else:
                    for a in A:
                        s = x + a
                        if (sumA >> s) & 1:
                            ok = False
                            break
                        d = abs(x - a)
                        if d > 0 and ((diffA >> d) & 1 or (diffB >> d) & 1):
                            ok = False
                            break
                if ok:
                    sumA |= (1 << s2)
                    for a in A:
                        sumA |= (1 << (x + a))
                        d = abs(x - a)
                        if d > 0:
                            diffA |= (1 << d)
                    A.append(x)
                    setA.add(x)
                    improved = True
                    break
        
        # Try adding new elements to B
        if len(B) < MAX_EACH:
            for x in range(1, N + 1):
                if x in setB:
                    continue
                ok = True
                s2 = x + x
                if (sumB >> s2) & 1:
                    ok = False
                else:
                    for b in B:
                        s = x + b
                        if (sumB >> s) & 1:
                            ok = False
                            break
                        d = abs(x - b)
                        if d > 0 and ((diffB >> d) & 1 or (diffA >> d) & 1):
                            ok = False
                            break
                if ok:
                    sumB |= (1 << s2)
                    for b in B:
                        sumB |= (1 << (x + b))
                        d = abs(x - b)
                        if d > 0:
                            diffB |= (1 << d)
                    B.append(x)
                    setB.add(x)
                    improved = True
                    break
        
        # Try 1-for-1 swaps: remove one from A, add one to A (or vice versa)
        # This can help when direct addition fails but a swap enables growth
        if len(A) < MAX_EACH:
            for idx_a in range(len(A)):
                a_remove = A[idx_a]
                # Try removing a_remove and adding a new x
                # First, temporarily remove a_remove
                A_temp = A[:idx_a] + A[idx_a+1:]
                setA_temp = setA - {a_remove}
                
                # Rebuild bitsets without a_remove
                sumA_temp, sumB_temp, diffA_temp, diffB_temp = _rebuild_bitsets(A_temp, B)
                
                # Try adding each x to A_temp
                for x in range(1, N + 1):
                    if x in setA_temp:
                        continue
                    ok = True
                    s2 = x + x
                    if (sumA_temp >> s2) & 1:
                        ok = False
                    else:
                        for a in A_temp:
                            s = x + a
                            if (sumA_temp >> s) & 1:
                                ok = False
                                break
                            d = abs(x - a)
                            if d > 0 and ((diffA_temp >> d) & 1 or (diffB_temp >> d) & 1):
                                ok = False
                                break
                    if ok:
                        # Accept the swap
                        A = A_temp + [x]
                        setA = setA_temp | {x}
                        sumA, sumB, diffA, diffB = _rebuild_bitsets(A, B)
                        improved = True
                        break
                if improved:
                    break
        
        if len(B) < MAX_EACH:
            for idx_b in range(len(B)):
                b_remove = B[idx_b]
                B_temp = B[:idx_b] + B[idx_b+1:]
                setB_temp = setB - {b_remove}
                
                sumA_temp, sumB_temp, diffA_temp, diffB_temp = _rebuild_bitsets(A, B_temp)
                
                for x in range(1, N + 1):
                    if x in setB_temp:
                        continue
                    ok = True
                    s2 = x + x
                    if (sumB_temp >> s2) & 1:
                        ok = False
                    else:
                        for b in B_temp:
                            s = x + b
                            if (sumB_temp >> s) & 1:
                                ok = False
                                break
                            d = abs(x - b)
                            if d > 0 and ((diffB_temp >> d) & 1 or (diffA_temp >> d) & 1):
                                ok = False
                                break
                    if ok:
                        B = B_temp + [x]
                        setB = setB_temp | {x}
                        sumA, sumB, diffA, diffB = _rebuild_bitsets(A, B)
                        improved = True
                        break
                if improved:
                    break
        
        if time.time() - start > time_limit:
            break
    
    return A, B

def construct():
    t0 = time.time()
    best_total = 0
    best_result = ([1], [2])
    
    # Multi-start with different random orderings
    random.seed(42)
    
    # First try a deterministic good ordering
    order = list(range(1, N + 1))
    A, B = _greedy_build(None, order)
    sumA, sumB, diffA, diffB = _rebuild_bitsets(A, B)
    
    total = len(A) + len(B)
    if total > best_total:
        best_total = total
        best_result = (A[:], B[:])
    
    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed
    if remaining > 5:
        A, B = _local_search(A, B, diffA, diffB, sumA, sumB, remaining - 5)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_result = (A[:], B[:])
    
    # Try several random orderings
    for trial in range(30):
        if time.time() - t0 > DEADLINE - 5:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        A, B = _greedy_build(None, order)
        sumA, sumB, diffA, diffB = _rebuild_bitsets(A, B)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_result = (A[:], B[:])
        
        # Local search on best
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 3
        if remaining > 2 and total >= best_total:
            A2, B2 = _local_search(A, B, diffA, diffB, sumA, sumB, remaining)
            total2 = len(A2) + len(B2)
            if total2 > best_total:
                best_total = total2
                best_result = (A2[:], B2[:])
    
    return list(best_result[0]), list(best_result[1])
# EVOLVE-BLOCK-END
