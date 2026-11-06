import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def time_check():
        return time.time() > deadline

    # Helper: Check if a Cayley graph on Z_n with connection set D contains a C6
    # We check for a sequence v0=0, v1, ..., v5 such that:
    # v1 in D
    # v2-v1 in D
    # ...
    # v5-v4 in D
    # -v5 in D
    # And all v_i distinct.
    
    def has_c6_cayley(n, D):
        if len(D) < 3:
            return False
        D_set = set(D)
        # Precompute possible next steps? 
        # For small D, direct search is okay.
        # v0 = 0
        for v1 in D_set:
            if v1 == 0: continue
            for v2 in D_set:
                u2 = (v1 + v2) % n
                if u2 in (0, v1): continue
                for v3 in D_set:
                    u3 = (u2 + v3) % n
                    if u3 in (0, v1, u2): continue
                    for v4 in D_set:
                        u4 = (u3 + v4) % n
                        if u4 in (0, v1, u2, u3): continue
                        for v5 in D_set:
                            u5 = (u4 + v5) % n
                            if u5 in (0, v1, u2, u3, u4): continue
                            # Check if u5 is connected to 0
                            if (-u5) % n in D_set:
                                return True
        return False

    # Strategy:
    # 1. Try to find a valid 4-coloring of K_19 using cyclic symmetry Z_19.
    #    If found, return n=19.
    # 2. Try to find a valid 4-coloring of K_18. 
    #    K_18 doesn't have prime cyclic symmetry. We can try random search with strong pruning or use Z_17 + 1 vertex logic? 
    #    Actually, Z_17 is the known best. Let's focus on Z_19 first as it's a "jump" candidate.
    #    If Z_19 fails, fall back to known Z_17 construction.
    
    # --- Attempt 1: Z_19 Cyclic Search ---
    # In Z_19, pairs are defined by differences d in 1..9 (since (i,j) ~ (j,i) ~ diff d or 19-d).
    # We assign a color to each difference d in 1..9.
    # Total 4^9 = 262,144 possibilities.
    # For each, we check 4 Cayley graphs.
    
    n_try = 19
    found_n19 = None
    
    # Optimization: We can prune. 
    # If a color class D has size < 3, it can't have C6? No, C6 needs degree at least 3 in the cycle, but the graph can be sparse.
    # Actually, if |D| < 3, max degree is 2 (if symmetric) or less. C6 requires degree >= 2 everywhere in the cycle.
    # So if max degree < 2, no C6.
    # But we need to check all 4 colors.
    
    # Let's iterate through all assignments.
    # To speed up, we can check colors one by one.
    
    import sys
    sys.setrecursionlimit(10000)
    
    # We'll use a simple recursive backtracking with pruning for the 9 differences.
    # Or just simple loop since 262k is small.
    
    # Pre-compute nothing, just fast checks.
    
    def check_assignment_z19(assign):
        # assign: list of 9 colors for diffs 1..9
        diff_sets = [set() for _ in range(4)]
        for i, c in enumerate(assign):
            d = i + 1
            diff_sets[c].add(d)
            diff_sets[c].add(19 - d)
        
        for c in range(4):
            if has_c6_cayley(19, diff_sets[c]):
                return False
        return True

    # Simple loop
    # 4^9 = 262,144
    # Each check involves 4 * (|D|^5) operations roughly.
    # Worst case |D|=9, 9^5 = 59049. 4 * 59049 ~ 236k ops per assignment.
    # 262k * 236k = 61 Billion ops. Too slow for Python.
    
    # We need a faster check or smarter search.
    
    # Alternative: Use bitsets for the Cayley graph adjacency and BFS/DFS for C6?
    # Or, precompute all C6s in Z_19?
    # Number of C6s in K_19 is large.
    
    # Let's try a different approach for Z_19:
    # Randomized search with local improvement?
    # Or, restrict to "balanced" colorings?
    
    # Actually, let's look at the structure of the known Z_17 solution.
    # It's often based on quadratic residues or similar.
    
    # Let's try a smarter search for Z_19:
    # We can use the fact that if D contains 1, it's likely to have cycles.
    # Let's just do a random search with delta updates for Z_19?
    # No, the cyclic structure makes delta updates complex if we change one diff.
    # Changing diff d affects edges (i, i+d) for all i.
    # This changes the Cayley graph for color c_old and c_new.
    # We can maintain the count of C6s in each Cayley graph?
    # That's hard because C6s overlap.
    
    # Let's try a different structural idea:
    # Use a known valid coloring for n=17 and try to extend it to n=18 or n=19?
    # Extension is hard.
    
    # Let's stick to the Z_17 cyclic search but make it robust and fast, 
    # and use the remaining time to try a random search for n=18 or n=19 
    # using a different method: **Genetic Algorithm** or **Simulated Annealing on the full graph** 
    # but with a better initial state and faster delta calculation using NumPy?
    
    # The current program uses SA on full graph for n=17. 
    # The bottleneck is Python loops.
    
    # NEW STRATEGY:
    # 1. Fast Z_17 Cyclic Search (Guaranteed n=17).
    # 2. If time remains, use **NumPy-vectorized Simulated Annealing** for n=18.
    #    Represent colors as a 2D array? No, edges are 1D.
    #    We can pre-compute the list of cycles as a list of arrays.
    #    Use NumPy to compute mono counts?
    #    `colors` is an array of shape (E,).
    #    `all_cycles` is a list of arrays of shape (6,).
    #    For each cycle, `np.all(colors[cycle] == colors[cycle][0])` is vectorized?
    #    No, iterating over 742k cycles in Python is still slow.
    
    # Better: **Bitset representation of color classes**.
    # For each color c, maintain a bitset of edges of color c.
    # A C6 is monochromatic in c if all 6 edges are in the bitset.
    # This doesn't help with counting easily.
    
    # Let's try **Constraint Programming** style backtracking for n=18?
    # No, too hard.
    
    # Let's try **Randomized Cyclic Search for Z_19** with pruning.
    # We can prune if any color class already contains a C6.
    # We can check for C6 in a Cayley graph incrementally?
    # If we add a difference d to color c, does it create a C6?
    # We can check if the new edge (0, d) participates in any C6 with existing edges.
    # This is a local check.
    
    # Let's implement a backtracking search for Z_19.
    # Variables: diff 1..9.
    # Values: 0..3.
    # Constraint: For each color c, the set of diffs assigned to c must not generate a C6 in Z_19.
    
    # We can check the constraint for a partial assignment?
    # No, C6 requires 6 edges. If we have assigned k diffs to color c, we can check if they contain a C6.
    # If k < 3, impossible.
    # If k >= 3, we can run has_c6_cayley on the current set.
    
    # This is a "forward checking" approach.
    
    def solve_z19_backtracking():
        diffs = list(range(1, 10))
        current_assign = [None] * 9
        diff_sets = [set() for _ in range(4)]
        
        def backtrack(idx):
            if time_check():
                return None
            if idx == 9:
                # Check all 4 sets
                for c in range(4):
                    if has_c6_cayley(19, diff_sets[c]):
                        return None
                return current_assign[:]
            
            d = diffs[idx]
            # Heuristic: try colors that are "safe" first?
            # Or just 0..3.
            for c in range(4):
                # Add d and 19-d to diff_sets[c]
                diff_sets[c].add(d)
                diff_sets[c].add(19 - d)
                
                # Check if this color class now has a C6
                if not has_c6_cayley(19, diff_sets[c]):
                    current_assign[idx] = c
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                    current_assign[idx] = None
                else:
                    # Prune
                    pass
                
                diff_sets[c].discard(d)
                diff_sets[c].discard(19 - d)
            
            return None
        
        return backtrack(0)
    
    # This backtracking might be fast if pruning is effective.
    # has_c6_cayley is O(|D|^5). |D| grows up to 9.
    # 9^5 = 59049.
    # If we call this at each node, it might be slow.
    # But the tree is small (4^9 max).
    # Let's try it.
    
    # Also, we can try Z_17 backtracking as a fallback if Z_19 fails.
    # Z_17 has 8 diffs. 4^8 = 65536.
    
    # Let's run Z_19 first.
    sol_19 = solve_z19_backtracking()
    
    if sol_19:
        # Construct the coloring for n=19
        n = 19
        colors_list = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d > n // 2:
                    d = n - d
                # d is in 1..9
                c = sol_19[d - 1]
                colors_list.append(c)
        
        # Verify
        def verify_full(colors_list, n):
            # Check for C6 in each color class using bitsets or adjacency matrix
            # For n=19, 171 edges.
            adj = [0] * n # Bitsets
            for c in range(4):
                adj_c = [0] * n
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if colors_list[idx] == c:
                            adj_c[i] |= (1 << j)
                            adj_c[j] |= (1 << i)
                        idx += 1
                # Check for C6
                # v0..v5 distinct
                # We can iterate v0, v1, v2, v3, v4, v5
                # n=19, so 19^6 is too big.
                # Use the Cayley check? No, this is not necessarily cyclic.
                # But we constructed it cyclically, so it IS a Cayley graph.
                # So we can just use has_c6_cayley again?
                # Yes, because the coloring is defined by differences.
                # So if has_c6_cayley(19, diff_sets[c]) returned False, it's valid.
                # We already checked that in the solver.
                pass
            return True
        
        return {'n': 19, 'colors': colors_list}
    
    # If Z_19 failed, try Z_17 backtracking (known to exist)
    def solve_z17_backtracking():
        diffs = list(range(1, 9))
        current_assign = [None] * 8
        diff_sets = [set() for _ in range(4)]
        
        def backtrack(idx):
            if time_check():
                return None
            if idx == 8:
                for c in range(4):
                    if has_c6_cayley(17, diff_sets[c]):
                        return None
                return current_assign[:]
            
            d = diffs[idx]
            for c in range(4):
                diff_sets[c].add(d)
                diff_sets[c].add(17 - d)
                
                if not has_c6_cayley(17, diff_sets[c]):
                    current_assign[idx] = c
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                    current_assign[idx] = None
                diff_sets[c].discard(d)
                diff_sets[c].discard(17 - d)
            return None
        
        return backtrack(0)
    
    sol_17 = solve_z17_backtracking()
    
    if sol_17:
        n = 17
        colors_list = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d > n // 2:
                    d = n - d
                c = sol_17[d - 1]
                colors_list.append(c)
        return {'n': 17, 'colors': colors_list}
    
    # Fallback: Random search for n=17 or 18?
    # If backtracking failed (shouldn't for 17), use random SA.
    # But backtracking for 17 is very likely to succeed.
    
    # If all else fails, return n=9
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
