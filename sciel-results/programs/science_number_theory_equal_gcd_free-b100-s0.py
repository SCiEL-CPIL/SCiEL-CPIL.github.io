import time
import random
from math import gcd

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    N = 128
    
    # Precompute GCD matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    def has_bad_triple_with_new(selected, new_elem):
        """
        Check if adding new_elem creates a bad triple with any pair in selected.
        """
        # Optimization: Precompute gcd of new_elem with all in selected
        g_new = {}
        for a in selected:
            g_new[a] = gcd_mat[new_elem][a]
            
        # Check all pairs in selected
        n = len(selected)
        for i in range(n):
            a = selected[i]
            g_an = g_new[a]
            for j in range(i + 1, n):
                b = selected[j]
                g_bn = g_new[b]
                if g_an == g_bn and g_an == gcd_mat[a][b]:
                    return True
        return False

    def is_valid_set(S):
        """
        Verify the entire set is valid. Used for final check or debugging.
        """
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                g_ab = gcd_mat[a][b]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                        return False
        return True

    def greedy_build(order):
        """Build set greedily following given order."""
        selected = []
        selected_set = set()
        for x in order:
            if x not in selected_set:
                if not has_bad_triple_with_new(selected, x):
                    selected.append(x)
                    selected_set.add(x)
        return selected

    def local_search(set_list, max_iters=1000):
        """
        Improve the set via additions, swaps, and 2-swaps.
        """
        set_list = list(set_list)
        set_list_set = set(set_list)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # 1. Try adding elements
            # Shuffle candidates to avoid bias
            candidates = [x for x in range(1, N + 1) if x not in set_list_set]
            random.shuffle(candidates)
            for x in candidates:
                if not has_bad_triple_with_new(set_list, x):
                    set_list.append(x)
                    set_list_set.add(x)
                    improved = True
                    break
            
            # 2. Try swaps: remove one, add another
            if not improved and time.time() < deadline:
                # Try a limited number of random swaps to save time
                swaps_tried = 0
                while swaps_tried < 50 and time.time() < deadline:
                    swaps_tried += 1
                    if not set_list: continue
                    i = random.randint(0, len(set_list) - 1)
                    removed = set_list[i]
                    candidate_set = set_list[:i] + set_list[i+1:]
                    
                    # Find a valid x to add
                    found = False
                    for x in range(1, N + 1):
                        if x in set_list_set:
                            continue
                        if not has_bad_triple_with_new(candidate_set, x):
                            set_list = candidate_set + [x]
                            set_list_set.remove(removed)
                            set_list_set.add(x)
                            improved = True
                            found = True
                            break
                    if found:
                        break

            # 3. Try 2-swaps: remove two, add two
            if not improved and time.time() < deadline and len(set_list) >= 2:
                # This is more expensive, so do it less often
                if random.random() < 0.1: # 10% chance per iteration
                    i1 = random.randint(0, len(set_list) - 1)
                    i2 = random.randint(0, len(set_list) - 1)
                    if i1 == i2:
                        continue
                    r1 = set_list[i1]
                    r2 = set_list[i2]
                    if r1 == r2:
                        continue
                    
                    # Create set without r1, r2
                    temp_set = [x for x in set_list if x != r1 and x != r2]
                    
                    # Try to add two elements
                    # This is O(N^2 * K^2) which is too slow if done naively.
                    # We will just try to add one, then the other.
                    # But we need to ensure the pair (x, y) doesn't form a bad triple with the rest.
                    # Actually, if we add x, then y, we check validity at each step.
                    # This might miss some 2-swaps where neither x nor y is valid alone, but together they are?
                    # No, if x is not valid alone, it forms a bad triple with the current set.
                    # Adding y won't remove that bad triple unless y is one of the elements in the bad triple?
                    # But y is new. The bad triple involves x and two elements from temp_set.
                    # So x must be valid alone (with respect to temp_set) for a sequential add to work?
                    # Wait. If x forms a bad triple (a,b,x) with a,b in temp_set, 
                    # can adding y fix it? No, the triple (a,b,x) still exists.
                    # So any element added must be valid with respect to the *current* set.
                    # Thus, 2-swaps can be decomposed into two 1-swaps?
                    # Not exactly. We remove r1, r2. Then we add x.
                    # x must be valid with temp_set.
                    # Then we add y. y must be valid with temp_set + {x}.
                    # This is equivalent to two sequential 1-swaps (remove r1, add x; remove r2, add y).
                    # So explicit 2-swaps don't add power if we allow sequential 1-swaps.
                    # The power of 2-swaps comes if we allow removing r1 and r2 and adding x and y 
                    # where x is NOT valid with temp_set, but the *set* temp_set + {x,y} is valid?
                    # No, if x is not valid with temp_set, there exist a,b in temp_set such that (a,b,x) is bad.
                    # This triple persists in temp_set + {x,y}.
                    # So x MUST be valid with temp_set.
                    # Therefore, 2-swaps are redundant if we can do 1-swaps.
                    pass

        return set_list

    def perturb(set_list, k):
        """
        Randomly remove k elements and greedily re-add them or others.
        """
        set_list = list(set_list)
        set_list_set = set(set_list)
        
        # Remove k random elements
        to_remove = random.sample(set_list, min(k, len(set_list)))
        for r in to_remove:
            set_list.remove(r)
            set_list_set.remove(r)
        
        # Try to add back removed elements or new ones greedily
        candidates = list(to_remove) + [x for x in range(1, N + 1) if x not in set_list_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if x not in set_list_set:
                if not has_bad_triple_with_new(set_list, x):
                    set_list.append(x)
                    set_list_set.add(x)
        
        return set_list

    best_set = []
    
    # Strategy 1: Greedy with natural order
    s1 = greedy_build(range(1, N + 1))
    s1 = local_search(s1)
    if len(s1) > len(best_set):
        best_set = s1
        
    # Strategy 2: Greedy with reverse order
    if time.time() < deadline:
        s2 = greedy_build(range(N, 0, -1))
        s2 = local_search(s2)
        if len(s2) > len(best_set):
            best_set = s2

    # Strategy 3: Iterated Local Search
    current_best = best_set
    current_best_len = len(current_best)
    
    # Use time to run ILS
    while time.time() < deadline - 2:
        # Perturb the current best
        # k=2 or 3 is a good perturbation size
        perturbed = perturb(current_best, 3)
        
        # Local search on perturbed
        improved_set = local_search(perturbed, max_iters=200)
        
        if len(improved_set) > current_best_len:
            current_best = improved_set
            current_best_len = len(improved_set)
            # Update global best
            if current_best_len > len(best_set):
                best_set = current_best
        
        # Occasionally restart from scratch with random order
        if random.random() < 0.1 and time.time() < deadline - 5:
            random_order = list(range(1, N + 1))
            random.shuffle(random_order)
            s_new = greedy_build(random_order)
            s_new = local_search(s_new, max_iters=200)
            if len(s_new) > current_best_len:
                current_best = s_new
                current_best_len = len(s_new)
                if current_best_len > len(best_set):
                    best_set = current_best

    # Final verification and return
    if not best_set:
        best_set = [1]
        
    # Optional: One last try to add any remaining elements
    # (Local search already does this, but just in case)
    final_set = list(best_set)
    final_set_set = set(final_set)
    for x in range(1, N + 1):
        if x not in final_set_set:
            if not has_bad_triple_with_new(final_set, x):
                final_set.append(x)
                final_set_set.add(x)
    
    return final_set
