import time
import random
from math import gcd

def construct():
    # Strict deadline to avoid timeout (90s limit, use 85s)
    deadline = time.time() + 85
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute bad triples as bitmasks
    # bad_mask[a][b] is a bitmask of c such that (a,b,c) is a bad triple
    # Note: (a,b,c) is bad if gcd(a,b)==gcd(a,c)==gcd(b,c)
    # We only need to store for a < b to save space, but symmetry is easy with full table or careful indexing.
    # Let's use a full N+1 x N+1 list of ints for simplicity and speed of access.
    bad_mask = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                    mask |= (1 << c)
            bad_mask[a][b] = mask
            bad_mask[b][a] = mask

    def get_forbidden_mask(selected):
        """
        Calculate the bitmask of all numbers x that cannot be added to 'selected'
        because they would form a bad triple with two elements in 'selected'.
        """
        forbidden = 0
        sel_list = list(selected)
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            for j in range(i + 1, n):
                b = sel_list[j]
                forbidden |= bad_mask[a][b]
        return forbidden

    def is_compatible(x, selected, forbidden_mask):
        """
        Check if x can be added to selected.
        x is compatible if x is not in forbidden_mask.
        Note: We assume x is not in selected.
        """
        return (forbidden_mask & (1 << x)) == 0

    def add_element(selected, x, forbidden_mask):
        """
        Add x to selected and return new selected and new forbidden_mask.
        """
        new_selected = selected | {x}
        # Update forbidden mask: OR in bad_mask[x][z] for all z in selected
        # We iterate over the OLD selected set
        new_forbidden = forbidden_mask
        for z in selected:
            new_forbidden |= bad_mask[x][z]
        return new_selected, new_forbidden

    best_set = set()
    best_size = 0
    
    # Run multiple restarts
    # Since each run is fast, we can do many.
    max_restarts = 100 
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        # 1. Greedy Construction
        # Randomize the order
        sorted_numbers = numbers[:]
        random.shuffle(sorted_numbers)
        
        selected = set()
        forbidden = 0
        
        for x in sorted_numbers:
            if time.time() > deadline - 2:
                break
            if is_compatible(x, selected, forbidden):
                selected, forbidden = add_element(selected, x, forbidden)
                
        # 2. Local Search
        # Phase A: Add any compatible number
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in numbers:
                if x not in selected:
                    if is_compatible(x, selected, forbidden):
                        selected, forbidden = add_element(selected, x, forbidden)
                        improved = True
                        break # Restart loop after addition
        
        # Phase B: 1-for-2 Swaps
        # Try to remove 1 element y and add 2 elements x1, x2
        ls_improved = True
        while ls_improved and time.time() < deadline - 2:
            ls_improved = False
            sel_list = list(selected)
            random.shuffle(sel_list)
            
            # Try removing each y in a random subset (or all if small)
            # To keep it fast, iterate through a few candidates
            for y in sel_list[:20]: 
                if time.time() > deadline - 2:
                    break
                
                temp_selected = selected - {y}
                temp_forbidden = get_forbidden_mask(temp_selected)
                
                # Find candidates x that are compatible with temp_selected
                candidates = []
                for x in numbers:
                    if x == y or x in selected:
                        continue
                    if is_compatible(x, temp_selected, temp_forbidden):
                        candidates.append(x)
                
                if len(candidates) < 2:
                    continue
                    
                # Try pairs from candidates
                # Limit pairs to avoid O(N^2) blowup
                random.shuffle(candidates)
                max_pairs_to_check = 100
                checked = 0
                
                for i in range(len(candidates)):
                    if checked >= max_pairs_to_check:
                        break
                    x1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        if checked >= max_pairs_to_check:
                            break
                        x2 = candidates[j]
                        checked += 1
                        
                        # Check if {x1, x2} is compatible with each other and temp_selected
                        # We know x1, x2 are compatible with temp_selected individually.
                        # We need to check if they form a bad triple with any z in temp_selected.
                        # And if x1, x2 form a bad triple with each other? No, need 3 distinct.
                        # So check: for all z in temp_selected, is (x1, x2, z) bad?
                        
                        valid = True
                        for z in temp_selected:
                            if bad_mask[x1][x2] & (1 << z):
                                valid = False
                                break
                            
                        if valid:
                            # Accept swap
                            selected = temp_selected | {x1, x2}
                            forbidden = get_forbidden_mask(selected)
                            ls_improved = True
                            break
                    if ls_improved:
                        break
                if ls_improved:
                    break
            
        # Update best
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
            
    return list(best_set)
