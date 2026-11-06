"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # Safety margin before 90s limit
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute lcm for all pairs using a 2D list for speed
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = i * j // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    def get_lcm(a, b):
        return lcm_table[a][b]
    
    # Precompute for each x, which pairs (a,b) in [1..N] would form a bad triple with x
    # i.e., lcm(a,x) == lcm(b,x) == lcm(a,b)
    # For each x, group elements by their LCM with x
    lcm_groups = {}
    for x in range(1, N + 1):
        groups = defaultdict(list)
        for y in range(1, N + 1):
            if y != x:
                groups[get_lcm(x, y)].append(y)
        lcm_groups[x] = groups
    
    def has_bad_triple_with_x(selected, x):
        """Check if adding x to selected creates a bad triple using precomputed groups."""
        sel_set = set(selected)
        # For each LCM value L, if two selected elements a,b have lcm(a,x)=L and lcm(b,x)=L
        # then we need lcm(a,b) == L
        groups = lcm_groups[x]
        for L, elems in groups.items():
            # Filter to only selected elements
            sel_in_group = [e for e in elems if e in sel_set]
            if len(sel_in_group) < 2:
                continue
            # Check all pairs in this group
            n = len(sel_in_group)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = sel_in_group[i], sel_in_group[j]
                    if get_lcm(a, b) == L:
                        return True
        return False
    
    def is_valid_set(selected):
        """Full validation that no bad triple exists in selected."""
        n = len(selected)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = selected[i], selected[j], selected[k]
                    if get_lcm(a, b) == get_lcm(a, c) == get_lcm(b, c):
                        return False
        return True
    
    def greedy(order):
        """Greedy construction with given element order."""
        selected = []
        selected_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x not in selected_set:
                if not has_bad_triple_with_x(selected, x):
                    selected.append(x)
                    selected_set.add(x)
        return selected
    
    # Precompute omega (total number of prime factors with multiplicity)
    def omega(x):
        c = 0
        n = x
        d = 2
        while d * d <= n:
            while n % d == 0:
                c += 1
                n //= d
            d += 1
        if n > 1:
            c += 1
        return c
    
    omega_vals = {x: omega(x) for x in nums}
    
    def is_pow2(x):
        return (x & (x - 1)) == 0
    
    # Try multiple orderings and keep the best
    best = []
    
    orderings = [
        ("asc", nums),
        ("desc", nums[::-1]),
        ("odd_first", sorted(nums, key=lambda x: (x & 1, x))),
        ("omega_asc", sorted(nums, key=omega_vals.get)),
        ("pow2_first", sorted(nums, key=lambda x: (not is_pow2(x), x))),
        ("odd_desc", sorted(nums, key=lambda x: (x & 1, -x))),
        ("omega_desc", sorted(nums, key=lambda x: -omega_vals[x])),
        ("largest_first", sorted(nums, reverse=True)),
        ("prime_like_first", sorted(nums, key=lambda x: (omega_vals[x] == 1, x))),
        ("composite_first", sorted(nums, key=lambda x: (omega_vals[x] > 1, x))),
    ]
    
    for name, order in orderings:
        if time.time() > deadline - 5:
            break
        s = greedy(order)
        if len(s) > len(best):
            best = list(s)
    
    if time.time() > deadline:
        return best if best else [1]
    
    # Local search: try adding remaining elements (multiple passes)
    improved = list(best)
    improved_set = set(improved)
    
    for _pass in range(3):
        if time.time() > deadline - 5:
            break
        remaining = [x for x in nums if x not in improved_set]
        added_any = False
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if not has_bad_triple_with_x(improved, x):
                improved.append(x)
                improved_set.add(x)
                added_any = True
        if not added_any:
            break
    
    if len(improved) > len(best):
        best = improved
    
    # Swap-based local search: try removing one element and adding two
    current_size = len(best)
    improvement_found = True
    while improvement_found and time.time() < deadline - 15:
        improvement_found = False
        best_set = set(best)
        
        for i in range(len(best)):
            if time.time() > deadline - 15:
                break
            removed = best[i]
            temp_list = best[:i] + best[i+1:]
            temp_set = best_set - {removed}
            
            for a in range(1, N + 1):
                if time.time() > deadline - 15:
                    break
                if a in temp_set:
                    continue
                if not has_bad_triple_with_x(temp_list, a):
                    temp_list_a = temp_list + [a]
                    temp_set_a = temp_set | {a}
                    
                    for b in range(a + 1, N + 1):
                        if time.time() > deadline - 15:
                            break
                        if b in temp_set_a:
                            continue
                        if not has_bad_triple_with_x(temp_list_a, b):
                            new_set = temp_list_a + [b]
                            if len(new_set) > current_size:
                                best = new_set
                                current_size = len(best)
                                best_set = set(best)
                                improvement_found = True
                                break
                    if improvement_found:
                        break
            if improvement_found:
                break
    
    # Single-swap local search: remove one, add one (net same size but may enable future additions)
    if time.time() < deadline - 10:
        best_set = set(best)
        for i in range(len(best)):
            if time.time() > deadline - 10:
                break
            removed = best[i]
            temp_list = best[:i] + best[i+1:]
            temp_set = best_set - {removed}
            
            # Try to add any element that fits
            for a in range(1, N + 1):
                if time.time() > deadline - 10:
                    break
                if a in temp_set:
                    continue
                if not has_bad_triple_with_x(temp_list, a):
                    # This swap doesn't increase size, but might help in combination
                    # Only keep if it allows more additions later - skip for now
                    pass
    
    # Final greedy pass on remaining
    if time.time() < deadline - 5:
        final_set = set(best)
        remaining = [x for x in nums if x not in final_set]
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if not has_bad_triple_with_x(best, x):
                best.append(x)
                final_set.add(x)
    
    # Final validation
    if not is_valid_set(best):
        # Fallback to greedy
        best = greedy(nums)
    
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
