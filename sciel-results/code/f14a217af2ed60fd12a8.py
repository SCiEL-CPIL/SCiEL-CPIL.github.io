"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

def _validate_sidon(aset):
    """Check if aset is a valid Sidon set (all pair sums unique)."""
    if not aset:
        return True
    sums = []
    for a in aset:
        for b in aset:
            if a >= b:  # combinations_with_replacement equivalent
                sums.append(a + b)
    return len(sums) == len(set(sums))

def _diff_set(aset):
    """Compute nonzero difference set of aset."""
    diffs = set()
    for a in aset:
        for b in aset:
            if a != b:
                diffs.add(b - a)
    return diffs

def _greedy_order(order, deadline):
    """
    Greedy construction: iterate over elements in given order,
    add to A or B if constraints allow.
    Returns (A, B).
    """
    A = []
    B = []
    A_sums = set()
    B_sums = set()
    A_diffs = set()
    B_diffs = set()
    
    for x in order:
        if time.time() > deadline:
            break
        
        # Try adding to A
        can_add_A = True
        # Check Sidon condition for A
        for a in A:
            s1 = x + a
            s2 = x + x
            if s1 in A_sums or s2 in A_sums:
                can_add_A = False
                break
        if can_add_A:
            # Check difference disjointness
            for a in A:
                d = x - a
                if d in B_diffs:
                    can_add_A = False
                    break
                d2 = a - x
                if d2 in B_diffs:
                    can_add_A = False
                    break
        if can_add_A:
            # Update A structures
            A.append(x)
            for a in A:
                A_sums.add(x + a)
                if a != x:
                    A_diffs.add(x - a)
                    A_diffs.add(a - x)
            continue
        
        # Try adding to B
        can_add_B = True
        for b in B:
            s1 = x + b
            s2 = x + x
            if s1 in B_sums or s2 in B_sums:
                can_add_B = False
                break
        if can_add_B:
            for b in B:
                d = x - b
                if d in A_diffs:
                    can_add_B = False
                    break
                d2 = b - x
                if d2 in A_diffs:
                    can_add_B = False
                    break
        if can_add_B:
            B.append(x)
            for b in B:
                B_sums.add(x + b)
                if b != x:
                    B_diffs.add(x - b)
                    B_diffs.add(b - x)
    
    return A, B

def _local_search(A, B, deadline):
    """
    Try to improve by swapping elements or moving elements between sets.
    Simple local search: try removing one element from A and adding another, etc.
    Returns (A, B) with potentially improved size.
    """
    A = list(A)
    B = list(B)
    best_size = len(A) + len(B)
    best_A, best_B = A[:], B[:]
    
    used = set(A) | set(B)
    
    # Try to add unused elements
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if x in used:
            continue
        
        # Try adding to A
        A_new = A + [x]
        if _validate_sidon(A_new):
            A_diffs_new = _diff_set(A_new)
            B_diffs_cur = _diff_set(B)
            if not (A_diffs_new & B_diffs_cur):
                A = A_new
                used.add(x)
                best_size = len(A) + len(B)
                best_A, best_B = A[:], B[:]
                continue
        
        # Try adding to B
        B_new = B + [x]
        if _validate_sidon(B_new):
            B_diffs_new = _diff_set(B_new)
            A_diffs_cur = _diff_set(A)
            if not (A_diffs_cur & B_diffs_new):
                B = B_new
                used.add(x)
                best_size = len(A) + len(B)
                best_A, best_B = A[:], B[:]
    
    return best_A, best_B

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    
    best_A = []
    best_B = []
    best_size = 0
    
    # Multiple random orderings
    num_trials = 50
    rng = random.Random(42)
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Create a random ordering
        order = list(range(1, n + 1))
        rng.shuffle(order)
        
        A, B = _greedy_order(order, deadline)
        size = len(A) + len(B)
        
        if size > best_size:
            best_size = size
            best_A, best_B = A[:], B[:]
        
        # Local search improvement
        A_ls, B_ls = _local_search(A, B, deadline)
        size_ls = len(A_ls) + len(B_ls)
        
        if size_ls > best_size:
            best_size = size_ls
            best_A, best_B = A_ls[:], B_ls[:]
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    # Final validation
    if not _validate_sidon(best_A):
        best_A = [1]
    if not _validate_sidon(best_B):
        best_B = [2]
    
    A_diffs = _diff_set(best_A)
    B_diffs = _diff_set(best_B)
    if A_diffs & B_diffs:
        # Fallback to a simple valid pair
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
