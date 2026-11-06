import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin before 90s kill
    
    N = 192
    universe = set(range(1, N + 1))
    
    # Precompute residues for fast lookup
    # residues[x][r] = sorted list of values in 1..N with value % x == r
    residues = {}
    for x in range(1, N + 1):
        residues[x] = {}
        for v in range(1, N + 1):
            r = v % x
            if r not in residues[x]:
                residues[x][r] = []
            residues[x][r].append(v)
    
    def is_valid(S):
        """Check if set S is valid: no element divides sum of any 3 distinct others."""
        S_list = sorted(S)
        K = len(S_list)
        if K < 4:
            return True
        
        for a in S_list:
            others = [v for v in S_list if v != a]
            K_others = len(others)
            if K_others < 3:
                continue
            
            # For fixed a, check if there exist distinct x,y,z in others with (x+y+z) % a == 0
            # Use pair sum approach with careful distinctness check
            pair_map = {}
            for i in range(K_others):
                xi = others[i]
                for j in range(i + 1, K_others):
                    xj = others[j]
                    s = (xi + xj) % a
                    if s not in pair_map:
                        pair_map[s] = []
                    pair_map[s].append((i, j))
            
            for z_idx, z in enumerate(others):
                target = (-z) % a
                if target in pair_map:
                    # Check if any pair (i, j) in pair_map[target] is distinct from z_idx
                    for i, j in pair_map[target]:
                        if i != z_idx and j != z_idx:
                            return False
        
        return True
    
    def can_add_incremental(a, selected):
        """
        Check if adding a to selected maintains validity.
        Only need to check:
        1. a does not divide sum of any 3 distinct elements in selected
        2. For each x in selected, x does not divide (a + b + c) for any distinct b,c in selected\{x,a}
        """
        sel = selected
        n = len(sel)
        
        # Check 1: a does not divide sum of any 3 distinct elements in sel
        if n >= 3:
            # Build residue set for quick lookup
            # For each pair (b,c) in sel, check if there exists d in sel, d!=b,c, with d % a == (-(b+c))%a
            # Use residues[a] for lookup
            # Create a set of values in sel for O(1) lookup
            sel_set = set(sel)
            # For each pair (b,c), compute target residue, then check if any value in sel with that residue
            # is distinct from b and c
            # To be efficient, precompute for each residue r mod a, how many elements in sel have that residue,
            # and which specific elements they are.
            
            # Build: for each residue r mod a, list of elements in sel with that residue
            res_to_elems = {}
            for v in sel:
                r = v % a
                if r not in res_to_elems:
                    res_to_elems[r] = []
                res_to_elems[r].append(v)
            
            # Now for each pair (b,c) in sel:
            for i in range(n):
                b = sel[i]
                for j in range(i + 1, n):
                    c = sel[j]
                    target_res = (-(b + c)) % a
                    if target_res in res_to_elems:
                        # Check if any element d in res_to_elems[target_res] is distinct from b and c
                        for d in res_to_elems[target_res]:
                            if d != b and d != c:
                                return False
        
        # Check 2: For each x in sel, x does not divide (a + b + c) for distinct b,c in sel\{x,a}
        for x in sel:
            if x == a:
                continue
            # Need b,c in sel, b != c, b != x, c != x, b != a, c != a
            # Such that (a + b + c) % x == 0
            # i.e., (b + c) % x == (-a) % x
            target_res = (-a) % x
            
            # Build residue map for elements in sel excluding x and a
            res_to_elems = {}
            for v in sel:
                if v == x or v == a:
                    continue
                r = v % x
                if r not in res_to_elems:
                    res_to_elems[r] = []
                res_to_elems[r].append(v)
            
            # For each pair (b,c) in the filtered list, check if (b+c) % x == target_res
            # But we can do it more efficiently: for each b in filtered list,
            # check if there exists c in filtered list, c != b, with c % x == (target_res - b%x) % x
            filtered = [v for v in sel if v != x and v != a]
            m = len(filtered)
            if m < 2:
                continue
            
            # Build residue counts and element lists for filtered
            res_to_elems2 = {}
            for v in filtered:
                r = v % x
                if r not in res_to_elems2:
                    res_to_elems2[r] = []
                res_to_elems2[r].append(v)
            
            for b in filtered:
                r_b = b % x
                r_c_needed = (target_res - r_b) % x
                if r_c_needed in res_to_elems2:
                    for c in res_to_elems2[r_c_needed]:
                        if c != b:
                            return False
        
        return True
    
    # Phase 1: Greedy from large to small with incremental check
    selected = []
    for x in range(N, 0, -1):
        if time.time() - start_time >= time_limit:
            break
        if can_add_incremental(x, selected):
            selected.append(x)
    
    S = set(selected)
    best_S = set(S)
    best_size = len(S)
    
    # Phase 2: Local search - try to add elements, then swaps
    max_iter = 2000
    for _ in range(max_iter):
        if time.time() - start_time >= time_limit:
            break
        
        U_not_S = universe - S
        if not U_not_S:
            break
        
        # Try to add one element
        added = False
        for x in sorted(U_not_S, reverse=True):
            if time.time() - start_time >= time_limit:
                break
            if can_add_incremental(x, list(S)):
                S.add(x)
                added = True
                break
        
        if added:
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            continue
        
        # If we cannot add, try a single swap
        if not S:
            break
        
        U_not_S_list = list(U_not_S)
        if not U_not_S_list:
            break
        
        # Sample a few swaps
        attempts = 50
        swapped = False
        for _ in range(attempts):
            if time.time() - start_time >= time_limit:
                break
            remove_candidate = random.choice(list(S))
            add_candidate = random.choice(U_not_S_list)
            new_S = (S - {remove_candidate}) | {add_candidate}
            if is_valid(new_S):
                S = new_S
                swapped = True
                # After a swap, immediately try to add again
                for x in sorted(universe - S, reverse=True):
                    if time.time() - start_time >= time_limit:
                        break
                    if can_add_incremental(x, list(S)):
                        S.add(x)
                        break
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                break
    
    # Final: return the best set found
    # Ensure we return a valid set
    if not is_valid(best_S):
        # Fallback: rebuild greedily
        selected = []
        for x in range(N, 0, -1):
            if can_add_incremental(x, selected):
                selected.append(x)
        best_S = set(selected)
    
    return list(best_S)
