"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from math import gcd

def construct():
    deadline = time.time() + 85.0  # 85s safety margin within 90s limit
    
    # Precompute gcd matrix
    n = 128
    gcd_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_mat[i][j] = gcd(i, j)
            gcd_mat[j][i] = gcd_mat[i][j]
    
    # Precompute for each pair (a,b), the set of x such that gcd(a,b)==gcd(a,x)==gcd(b,x)
    # A triple (a,b,x) is bad if gcd(a,b) == gcd(a,x) == gcd(b,x)
    # For each pair (a,b), precompute which x make it bad
    bad_x_for_pair = {}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd_mat[a][b]
            bad_xs = set()
            for x in range(1, n + 1):
                if x == a or x == b:
                    continue
                if gcd_mat[a][x] == g_ab and gcd_mat[b][x] == g_ab:
                    bad_xs.add(x)
            if bad_xs:
                bad_x_for_pair[(a, b)] = bad_xs
    
    # Also precompute: for each x, the set of pairs (a,b) that would form a bad triple with x
    bad_pairs_for_x = {x: set() for x in range(1, n + 1)}
    for (a, b), bad_xs in bad_x_for_pair.items():
        for x in bad_xs:
            bad_pairs_for_x[x].add((a, b))
    
    # Precompute conflict masks: for each element x, a bitmask of elements y
    # such that {x,y} is part of a bad triple with some third element z in [1,n]
    # Actually, we need: for a given set S, x can be added iff no pair (a,b) in S
    # is in bad_pairs_for_x[x].
    
    # More efficient: for each x, precompute the set of "conflicting pairs" as a frozenset
    # and check membership. But let's use a different approach:
    # For each x, store bad_pairs_for_x[x] as a set of (min(a,b), max(a,b)) tuples
    
    def is_valid_addition(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        bp = bad_pairs_for_x[x]
        selected_list = list(selected)
        for i in range(len(selected_list)):
            a = selected_list[i]
            for j in range(i + 1, len(selected_list)):
                b = selected_list[j]
                # Check if (a,b) or (b,a) is in bp
                if a < b:
                    if (a, b) in bp:
                        return False
                else:
                    if (b, a) in bp:
                        return False
        return True
    
    def greedy_add(start_order):
        """Greedy add elements in given order."""
        selected = set()
        for x in start_order:
            if is_valid_addition(x, selected):
                selected.add(x)
        return selected
    
    # Strategy 1: Greedy from full set removal
    def greedy_remove():
        """Start with all elements, remove those that create conflicts."""
        selected = set(range(1, n + 1))
        changed = True
        while changed:
            changed = False
            for x in list(selected):
                # Check if x conflicts with any pair in selected
                conflict_found = False
                bp = bad_pairs_for_x[x]
                selected_list = [y for y in selected if y != x]
                for i in range(len(selected_list)):
                    a = selected_list[i]
                    for j in range(i + 1, len(selected_list)):
                        b = selected_list[j]
                        if a < b:
                            if (a, b) in bp:
                                conflict_found = True
                                break
                        else:
                            if (b, a) in bp:
                                conflict_found = True
                                break
                    if conflict_found:
                        break
                if conflict_found:
                    selected.remove(x)
                    changed = True
        return selected
    
    # Get initial solution from greedy removal
    best_set = greedy_remove()
    best_size = len(best_set)
    
    # Try multiple greedy-add restarts with different orderings
    import random
    random.seed(42)
    
    for attempt in range(300):
        if time.time() > deadline - 10.0:
            break
        
        # Different ordering strategies
        if attempt % 4 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif attempt % 4 == 1:
            # Sort by number of bad pairs (least constrained first)
            order = sorted(range(1, n + 1), key=lambda x: len(bad_pairs_for_x[x]))
        elif attempt % 4 == 2:
            # Sort by number of bad pairs (most constrained first)
            order = sorted(range(1, n + 1), key=lambda x: -len(bad_pairs_for_x[x]))
        else:
            # Sort by value, sometimes reversed
            order = list(range(1, n + 1))
            if attempt % 8 == 3:
                order.reverse()
        
        candidate = greedy_add(order)
        if len(candidate) > best_size:
            best_set = candidate
            best_size = len(candidate)
    
    # Local search: 1-for-1 swaps (cheap)
    def local_search_1for1(current_set, max_iter=200):
        """Try 1-for-1 swaps to improve the set."""
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            current_list = list(current_set)
            for x in current_list:
                if time.time() > deadline - 8.0:
                    return current_set, False
                remaining = current_set - {x}
                # Try all elements not in current_set
                for y in range(1, n + 1):
                    if y in current_set:
                        continue
                    # Check if y can be added to remaining
                    if is_valid_addition(y, remaining):
                        new_set = remaining | {y}
                        # Verify it's actually an improvement (same size, but maybe better structure)
                        # For 1-for-1, size stays the same, so we need a different criterion
                        # Actually, 1-for-1 doesn't improve size. Let's skip this.
                        pass
        return current_set, improved
    
    # Better: 1-for-2 swaps to increase size
    if time.time() < deadline - 15.0:
        for _ in range(100):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            selected_list = list(best_set)
            
            for x in selected_list:
                if time.time() > deadline - 10.0:
                    break
                
                # Try removing x and adding two new elements
                remaining = best_set - {x}
                
                # Find candidates that can be added to remaining
                candidates = []
                for y in range(1, n + 1):
                    if y not in remaining and y != x:
                        if is_valid_addition(y, remaining):
                            candidates.append(y)
                
                # Try adding two candidates
                for i in range(len(candidates)):
                    if time.time() > deadline - 10.0:
                        break
                    y1 = candidates[i]
                    temp_set = remaining | {y1}
                    for j in range(i + 1, len(candidates)):
                        y2 = candidates[j]
                        if y2 == y1:
                            continue
                        if is_valid_addition(y2, temp_set):
                            new_set = temp_set | {y2}
                            if len(new_set) > best_size:
                                best_set = new_set
                                best_size = len(new_set)
                                improved = True
                                break
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    # Additional local search: try to add more elements after 1-for-2
    if time.time() < deadline - 8.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for y in range(1, n + 1):
                if y in best_set:
                    continue
                if is_valid_addition(y, best_set):
                    best_set.add(y)
                    best_size += 1
                    improved = True
                    break
            if not improved:
                break
    
    return list(best_set)
# EVOLVE-BLOCK-END
