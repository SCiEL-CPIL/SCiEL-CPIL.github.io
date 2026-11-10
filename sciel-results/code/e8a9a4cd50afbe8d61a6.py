import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = 512
    # Build conflict graph: conflict[i][j] = True if i,j in same 4-AP
    # Two elements x < y conflict if y-x = d, 2d, or 3d for some 4-AP
    # i.e., there exists a 4-AP containing both x and y
    
    # For x and y to be in a 4-AP:
    # Case 1: x=a, y=a+d => d=y-x, need a+3d <= n and a >= 1
    #   a=x, d=y-x, need x+3(y-x) = 3y-2x <= n, and x>=1 (always true since x>=1)
    # Case 2: x=a, y=a+2d => d=(y-x)/2, need (y-x) even, and a+3d = x+3(y-x)/2 <= n
    # Case 3: x=a, y=a+3d => d=(y-x)/3, need (y-x) divisible by 3, and a+3d = y <= n (always true)
    # Case 4: x=a+d, y=a+2d => d=y-x, a=x-d=2x-y >= 1, need a+3d = 2x-y+3(y-x) = 2y-x <= n
    # Case 5: x=a+d, y=a+3d => d=(y-x)/2, need (y-x) even, a=x-d = x-(y-x)/2 = (3x-y)/2 >= 1
    # Case 6: x=a+2d, y=a+3d => d=y-x, a=x-d = 2x-y >= 1, need a >= 1
    
    # Actually, let me just check: for each pair (x,y), is there a 4-AP containing both?
    # A 4-AP is determined by (a,d). x and y are in it if x = a+id and y = a+jd for i,j in {0,1,2,3}, i≠j.
    # So y-x = (j-i)d for some d>0 and valid a.
    
    # Let me build the conflict set using bitsets for efficiency
    
    # conflict_mask[i] is a bitset of all j that conflict with i
    conflict = [0] * (n + 1)
    
    for a in range(1, n + 1):
        for d in range(1, n + 1):
            # Check if a, a+d, a+2d, a+3d are all in [1,n]
            if a + 3 * d > n:
                break
            terms = [a, a + d, a + 2 * d, a + 3 * d]
            # All pairs in this 4-AP conflict
            for i in range(4):
                for j in range(i + 1, 4):
                    x, y = terms[i], terms[j]
                    conflict[x] |= (1 << y)
                    conflict[y] |= (1 << x)
    
    # Now use greedy + local search
    best_set = None
    best_size = 0
    
    def greedy_construction(order):
        """Build a 4-AP-free set by processing elements in given order."""
        selected = set()
        selected_mask = 0
        for x in order:
            # Check if x conflicts with any selected element
            if selected_mask & conflict[x]:
                continue
            selected.add(x)
            selected_mask |= (1 << x)
        return selected
    
    def local_search(selected, deadline):
        """Try to improve the set by adding elements or swapping."""
        selected = set(selected)
        selected_mask = 0
        for x in selected:
            selected_mask |= (1 << x)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add elements
            for x in range(1, n + 1):
                if x not in selected:
                    if not (selected_mask & conflict[x]):
                        selected.add(x)
                        selected_mask |= (1 << x)
                        improved = True
            # Try swaps: remove one, add another
            if not improved:
                for x in list(selected):
                    for y in range(1, n + 1):
                        if y not in selected:
                            new_mask = (selected_mask & ~(1 << x)) | (1 << y)
                            if not (new_mask & conflict[y]):
                                # Check that removing x doesn't break anything else (it won't, since we're just removing)
                                selected.discard(x)
                                selected.add(y)
                                selected_mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
        return selected
    
    # Multiple random restarts
    random.seed(42)
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random ordering
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        selected = greedy_construction(order)
        size = len(selected)
        
        if size > best_size:
            best_size = size
            best_set = set(selected)
        
        # Local search
        if time.time() < deadline - 2:
            selected = local_search(selected, deadline - 2)
            size = len(selected)
            if size > best_size:
                best_size = size
                best_set = set(selected)
    
    # Final local search on best
    if best_set and time.time() < deadline - 1:
        best_set = local_search(best_set, deadline)
    
    if not best_set:
        # Fallback: use the greedy from seed
        selected = set()
        for x in range(1, n + 1):
            if all(not all(x - j * d in selected for j in (1, 2, 3)) for d in range(1, (x - 1) // 3 + 1)):
                selected.add(x)
        best_set = selected
    
    return sorted(best_set)
