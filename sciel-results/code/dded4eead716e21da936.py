"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # The problem: Find a subset S of {1, ..., 128} such that for any distinct a, b, c in S,
    # gcd(a,b), gcd(a,c), gcd(b,c) are not all equal.
    # This is equivalent to saying that the graph where edges are colored by gcd(a,b)
    # contains no monochromatic triangle.
    
    # We use a backtracking search with MRV (Minimum Remaining Values) heuristic.
    # State:
    # - selected: list of currently selected numbers
    # - gcd_matrix: 2D array or dict storing gcd(i, j) for selected i, j
    # - conflict_check: For each new candidate x, we need to ensure that for any pair (a, b) in selected,
    #   gcd(a, x) != gcd(b, x) != gcd(a, b) is NOT the condition.
    #   The condition is: gcd(a, b), gcd(a, x), gcd(b, x) are NOT all equal.
    #   So, for a candidate x, and existing pair (a, b), if gcd(a, b) == gcd(a, x) == gcd(b, x), then x cannot be added.
    
    # Optimization:
    # Instead of checking all pairs (a,b) for each candidate x, we can precompute or maintain constraints.
    # However, 128 is small enough that O(N^2) check per insertion is fine if N is small, but N can be ~60-80.
    # 80^2 * 80 (candidates) = 512,000 ops per step. Very fast.
    
    # Let's try a randomized greedy + local search first, but with a stronger validity check.
    # The previous program did:
    # if all(len({gcd(a,b),gcd(a,x),gcd(b,x)}) > 1 for a,b in combinations(selected,2)):
    # This is correct.
    
    # Why did the previous program score 1.0?
    # It likely got stuck in a local optimum with a small set.
    # The inspirations mention "structural, prime-power-based construction" and "Simulated Annealing".
    
    # Let's implement a robust Simulated Annealing on a bitmask representation.
    # Representation: 128-bit integer mask.
    # Objective: Maximize popcount(mask) subject to validity.
    # Move: Flip a bit (add or remove a number).
    # Cost: Number of invalid triples. We want Cost == 0.
    # If Cost > 0, we minimize Cost. If Cost == 0, we maximize Size.
    
    # Precompute gcds for all pairs 1..128
    N = 128
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    # To check validity of a set quickly, we can't just sum costs easily without O(N^3) per step.
    # However, we can maintain the number of "bad triangles" incrementally.
    # Let's stick to a high-performance local search:
    # 1. Start with a random valid set (greedy).
    # 2. Try to add elements. If fails, try to remove one and add two? (2-opt style)
    # 3. Or simply Simulated Annealing on the mask.
    
    # Let's try a simpler, more effective approach:
    # Greedy construction in random order, repeated many times.
    # Then, local search: try to swap an element out for an element in.
    
    def is_valid_set(selected):
        # selected is a list of ints
        n = len(selected)
        for i in range(n):
            for j in range(i + 1, n):
                g_ij = gcd_table[selected[i]][selected[j]]
                for k in range(j + 1, n):
                    g_ik = gcd_table[selected[i]][selected[k]]
                    if g_ik != g_ij: continue
                    g_jk = gcd_table[selected[j]][selected[k]]
                    if g_jk == g_ij:
                        return False
        return True

    def greedy_construct(order):
        selected = []
        for x in order:
            # Check if x can be added to selected
            ok = True
            # For any pair (a, b) in selected, gcd(a,b), gcd(a,x), gcd(b,x) must not be all equal
            for i in range(len(selected)):
                a = selected[i]
                g_ax = gcd_table[a][x]
                for j in range(i + 1, len(selected)):
                    b = selected[j]
                    g_ab = gcd_table[a][b]
                    if g_ab == g_ax:
                        g_bx = gcd_table[b][x]
                        if g_bx == g_ab:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                selected.append(x)
        return selected

    # Strategy:
    # 1. Multiple random greedy restarts to find a good initial set.
    # 2. Local search: Iterate through all candidates not in set, try to add.
    #    If can't add, try to remove one element from set, then add the candidate.
    #    Accept if size increases or (size same and diversity improves? No, just size).
    #    Actually, standard hill climbing: Add if valid. If stuck, try 1-1 swap.
    
    best_size = 0
    best_set = []
    
    num_restarts = 50
    max_time_per_restart = 1.5
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.shuffle_order = list(range(1, N + 1))
        random.shuffle(random.shuffle_order)
        
        current_set = greedy_construct(random.shuffle_order)
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
            
        # Local Search Phase
        # We will try to improve current_set.
        # We maintain a set of "blocked" elements? No, just re-check.
        # Since N=128, checking validity of adding x is O(|S|^2).
        # |S| ~ 60. 60^2 = 3600. 128 candidates * 3600 = ~460k ops. Very fast.
        
        improved = True
        while improved and time.time() < start_time + max_time_per_restart + r*0.1: # rough budget
            improved = False
            # Try to add any element
            for x in range(1, N + 1):
                if x in current_set:
                    continue
                # Check if x can be added
                ok = True
                for i in range(len(current_set)):
                    a = current_set[i]
                    g_ax = gcd_table[a][x]
                    for j in range(i + 1, len(current_set)):
                        b = current_set[j]
                        g_ab = gcd_table[a][b]
                        if g_ab == g_ax:
                            g_bx = gcd_table[b][x]
                            if g_bx == g_ab:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    current_set.append(x)
                    improved = True
                    break # Restart loop with new set
            
            if not improved:
                # Try 1-1 swap: remove one, add one
                # To avoid O(N^3) or worse, we can just try random swaps
                # Or systematic: for each x not in set, for each y in set, try swap
                # This is O(N * |S| * |S|^2) which is too slow if done naively every step.
                # Let's do random swaps for a limited number of steps.
                for _ in range(100):
                    if time.time() > deadline:
                        return best_set
                    x = random.randint(1, N)
                    if x in current_set:
                        continue
                    y = random.choice(current_set)
                    
                    # Try swapping y with x
                    new_set = [z for z in current_set if z != y]
                    # Check if x can be added to new_set
                    ok = True
                    for i in range(len(new_set)):
                        a = new_set[i]
                        g_ax = gcd_table[a][x]
                        for j in range(i + 1, len(new_set)):
                            b = new_set[j]
                            g_ab = gcd_table[a][b]
                            if g_ab == g_ax:
                                g_bx = gcd_table[b][x]
                                if g_bx == g_ab:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok:
                        current_set = new_set + [x]
                        improved = True
                        break # Restart outer loop
                    
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
    # Final polish: Try to add any remaining
    for x in range(1, N + 1):
        if x in best_set:
            continue
        ok = True
        for i in range(len(best_set)):
            a = best_set[i]
            g_ax = gcd_table[a][x]
            for j in range(i + 1, len(best_set)):
                b = best_set[j]
                g_ab = gcd_table[a][b]
                if g_ab == g_ax:
                    g_bx = gcd_table[b][x]
                    if g_bx == g_ab:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            best_set.append(x)

    return best_set
# EVOLVE-BLOCK-END
