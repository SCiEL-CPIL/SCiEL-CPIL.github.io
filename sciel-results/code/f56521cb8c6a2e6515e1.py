"""
Rewritten solver for the dense increasing sequence with unique consecutive block products.

Strategy:
1. Greedy construction (scan 2..128 in order, include x if all block products remain unique).
2. Local improvement: try removing one element, swapping one element, or removing two elements,
   then re-filling greedily to find a longer sequence.
3. Deadline guard to return best result within 60s.
"""

import time
import sys
from itertools import combinations

def _all_block_products(seq):
    """Return a set of all nonempty consecutive block products for the given sorted sequence."""
    n = len(seq)
    prods = set()
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _is_valid(seq):
    """Check if all consecutive block products are distinct. Early-exit on first duplicate."""
    n = len(seq)
    prods = set()
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            if p in prods:
                return False
            prods.add(p)
    return True

def _greedy_build(start_seq, deadline):
    """
    Starting from a sorted list start_seq, greedily try to add each number from 2 to 128
    (inserting in sorted order) if the resulting sequence has all distinct block products.
    Returns the resulting sequence.
    """
    seq = list(start_seq)
    seq_set = set(seq)
    n = len(seq)
    
    for x in range(2, 129):
        if time.time() > deadline:
            break
        if x in seq_set:
            continue
        # Find insertion position via binary search
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        pos = lo
        # Build new sequence and validate with early exit
        new_seq = seq[:pos] + [x] + seq[pos:]
        if _is_valid(new_seq):
            seq = new_seq
            seq_set.add(x)
            n += 1
    return seq

def _local_improve(seq, deadline):
    """
    Try removing one element, swapping one element, or removing two elements,
    then re-filling greedily. Return the best sequence found (longest).
    """
    best = list(seq)
    best_len = len(best)
    n = len(seq)
    seq_set = set(seq)
    
    # Phase A: Single removals
    for i in range(n):
        if time.time() > deadline:
            return best
        reduced = seq[:i] + seq[i+1:]
        candidate = _greedy_build(reduced, deadline)
        if len(candidate) > best_len:
            best = candidate
            best_len = len(candidate)
    
    # Phase B: Single swaps (remove seq[i], try adding x not in seq)
    if time.time() > deadline - 5:
        return best
    for i in range(n):
        if time.time() > deadline:
            break
        for x in range(2, 129):
            if x in seq_set:
                continue
            # Build sequence: remove seq[i], insert x in sorted position
            reduced = seq[:i] + seq[i+1:]
            # Find insertion position for x in reduced
            lo, hi = 0, len(reduced)
            while lo < hi:
                mid = (lo + hi) // 2
                if reduced[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            swapped = reduced[:lo] + [x] + reduced[lo:]
            candidate = _greedy_build(swapped, deadline)
            if len(candidate) > best_len:
                best = candidate
                best_len = len(candidate)
                seq_set = set(best)
                # Update for subsequent iterations
                n = len(seq)  # original seq length stays the same for swap base
    
    # Phase C: Two-element removals (if time permits)
    if time.time() > deadline - 10:
        return best
    for i, j in combinations(range(n), 2):
        if time.time() > deadline:
            break
        if i == j:
            continue
        reduced = seq[:i] + seq[i+1:j] + seq[j+1:]
        candidate = _greedy_build(reduced, deadline)
        if len(candidate) > best_len:
            best = candidate
            best_len = len(candidate)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    # Phase 1: Initial greedy construction
    seq = _greedy_build([], deadline)
    
    # Phase 2: Local improvement (removals, swaps, two-removals)
    improved = _local_improve(seq, deadline)
    if len(improved) > len(seq):
        seq = improved
    
    # Phase 3: Another round of local improvement if time permits
    if time.time() < deadline - 10:
        improved2 = _local_improve(seq, deadline)
        if len(improved2) > len(seq):
            seq = improved2
    
    # Phase 4: Final greedy polish on the best sequence
    if time.time() < deadline - 2:
        final = _greedy_build(seq, deadline)
        if len(final) > len(seq):
            seq = final
    
    return seq
