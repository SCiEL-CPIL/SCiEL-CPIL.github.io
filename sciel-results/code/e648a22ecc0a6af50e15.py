# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def _pair_sums_unique(lst):
    """Check if all pair sums (including a+a) are unique."""
    sums = []
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sums.append(lst[i] + lst[j])
    return len(sums) == len(set(sums))

def _diff_set(lst):
    """Get the set of nonzero absolute differences."""
    s = set()
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            s.add(abs(lst[j] - lst[i]))
    return s

def _try_add(lst, x, used_diffs):
    """Try to add x to lst, checking Sidon and difference constraints."""
    if x in lst:
        return False
    new_lst = lst + [x]
    if not _pair_sums_unique(new_lst):
        return False
    new_diffs = _diff_set(new_lst)
    if new_diffs & used_diffs:
        return False
    return True

def _greedy_construct(order, deadline):
    """Greedy construction following a given order of candidates."""
    A = []
    B = []
    diffs_A = set()
    diffs_B = set()
    
    for x in order:
        if time.time() > deadline:
            break
        # Try adding to A
        if len(A) < 64 and _try_add(A, x, diffs_B):
            A.append(x)
            diffs_A = _diff_set(A)
        elif len(B) < 64 and _try_add(B, x, diffs_A):
            B.append(x)
            diffs_B = _diff_set(B)
    
    return A, B

def _local_search(A, B, deadline):
    """Try to improve by removing elements and re-adding others."""
    diffs_A = _diff_set(A)
    diffs_B = _diff_set(B)
    all_elements = set(range(1, 257))
    used = set(A) | set(B)
    candidates = list(all_elements - used)
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding a new element to A
        for x in candidates:
            if time.time() > deadline:
                break
            if len(A) < 64 and _try_add(A, x, diffs_B):
                A.append(x)
                diffs_A = _diff_set(A)
                candidates.remove(x)
                improved = True
                break
        
        # Try adding a new element to B
        if not improved:
            for x in candidates:
                if time.time() > deadline:
                    break
                if len(B) < 64 and _try_add(B, x, diffs_A):
                    B.append(x)
                    diffs_B = _diff_set(B)
                    candidates.remove(x)
                    improved = True
                    break
        
        # Try swap: remove from A, add to B or vice versa
        if not improved and len(A) > 1:
            for i, a in enumerate(A):
                if time.time() > deadline:
                    break
                A_new = A[:i] + A[i+1:]
                diffs_A_new = _diff_set(A_new)
                if _pair_sums_unique(A_new):
                    # Try to add a from A to B
                    if len(B) < 64 and _try_add(B, a, diffs_A_new):
                        B.append(a)
                        A = A_new
                        diffs_A = diffs_A_new
                        diffs_B = _diff_set(B)
                        improved = True
                        break
        
        if not improved and len(B) > 1:
            for i, b in enumerate(B):
                if time.time() > deadline:
                    break
                B_new = B[:i] + B[i+1:]
                diffs_B_new = _diff_set(B_new)
                if _pair_sums_unique(B_new):
                    if len(A) < 64 and _try_add(A, b, diffs_B_new):
                        A.append(b)
                        B = B_new
                        diffs_B = diffs_B_new
                        diffs_A = _diff_set(A)
                        improved = True
                        break
    
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    best_size = 0
    best_A, best_B = [1], [2]  # Default valid answer
    
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        order = list(range(1, 257))
        random.shuffle(order)
        
        A, B = _greedy_construct(order, deadline)
        size = len(A) + len(B)
        
        if size > best_size:
            best_size = size
            best_A, best_B = A, B
        
        # Local search to improve
        A, B = _local_search(A, B, deadline)
        size = len(A) + len(B)
        
        if size > best_size:
            best_size = size
            best_A, best_B = A, B
        
        # If we've found a good solution, do more restarts for polish
        if best_size >= 30 and num_restarts > 50:
            break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
