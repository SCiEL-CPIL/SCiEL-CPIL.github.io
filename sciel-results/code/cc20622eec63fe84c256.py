import time
import random
from itertools import combinations, combinations_with_replacement

def _check_sidon(sets):
    """Check if a set is Sidon (all pair sums including a+a are unique)."""
    sums = set()
    for i, a in enumerate(sets):
        for b in sets[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_diffs(sets):
    """Get the set of nonzero differences."""
    diffs = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            d = abs(sets[j] - sets[i])
            diffs.add(d)
    return diffs

def _can_add(sets, x, diffs_set):
    """Check if x can be added to sets while maintaining Sidon and diff disjointness."""
    # Check Sidon: all pair sums involving x must be new
    existing_sums = set()
    for i in range(len(sets)):
        for j in range(i, len(sets)):
            existing_sums.add(sets[i] + sets[j])
    
    for a in sets:
        s = a + x
        if s in existing_sums:
            return False
        existing_sums.add(s)
    # x+x
    if x + x in existing_sums:
        return False
    
    # Check difference disjointness
    new_diffs = set()
    for a in sets:
        d = abs(x - a)
        new_diffs.add(d)
    if new_diffs & diffs_set:
        return False
    
    return True

def _greedy_construct(order, max_time):
    """Greedy construction given an ordering of candidates."""
    sets = [[], []]
    diffs = [set(), set()]
    
    for x in order:
        if time.time() > max_time:
            break
        # Try to add to the set that has fewer elements (balance)
        if len(sets[0]) <= len(sets[1]):
            side = 0
        else:
            side = 1
        
        if _can_add(sets[side], x, diffs[1 - side]):
            sets[side].append(x)
            # Update diffs
            for a in sets[side]:
                if a != x:
                    d = abs(x - a)
                    diffs[side].add(d)
        
        # Also try the other side if this one failed and the other has room
        if len(sets[side]) == 0 or not _can_add(sets[side], x, diffs[1 - side]):
            other = 1 - side
            if len(sets[other]) < 64:
                if _can_add(sets[other], x, diffs[side]):
                    sets[other].append(x)
                    for a in sets[other]:
                        if a != x:
                            d = abs(x - a)
                            diffs[other].add(d)
    
    return sets[0], sets[1]

def _local_improve(A, B, max_time):
    """Try to improve by adding skipped elements or swapping."""
    A = list(A)
    B = list(B)
    diffs_A = _get_diffs(A)
    diffs_B = _get_diffs(B)
    
    candidates = list(range(1, 257))
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() < max_time:
        improved = False
        for x in candidates:
            if time.time() > max_time:
                break
            # Try adding x to A
            if x not in A and len(A) < 64:
                new_diffs = set(abs(x - a) for a in A)
                if not (new_diffs & diffs_B):
                    # Check Sidon
                    sums = set()
                    valid = True
                    for i, a in enumerate(A):
                        for b in A[i:]:
                            s = a + b
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            break
                    if valid:
                        for a in A:
                            s = a + x
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if x + x in sums:
                            valid = False
                    if valid:
                        A.append(x)
                        diffs_A = _get_diffs(A)
                        improved = True
            # Try adding x to B
            if x not in B and len(B) < 64:
                new_diffs = set(abs(x - b) for b in B)
                if not (new_diffs & diffs_A):
                    sums = set()
                    valid = True
                    for i, b in enumerate(B):
                        for c in B[i:]:
                            s = b + c
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            break
                    if valid:
                        for b in B:
                            s = b + x
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if x + x in sums:
                            valid = False
                    if valid:
                        B.append(x)
                        diffs_B = _get_diffs(B)
                        improved = True
    
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_size = 0
    best_A, best_B = [1], [256]
    
    # Try multiple random restarts
    for restart in range(500):
        if time.time() > deadline:
            break
        
        order = list(range(1, 257))
        random.shuffle(order)
        
        A, B = _greedy_construct(order, deadline)
        
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A, best_B = A, B
        
        # Local improvement
        if time.time() < deadline - 5:
            A_imp, B_imp = _local_improve(A, B, deadline)
            size_imp = len(A_imp) + len(B_imp)
            if size_imp > best_size:
                best_size = size_imp
                best_A, best_B = A_imp, B_imp
    
    return [best_A, best_B]
