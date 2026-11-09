"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0
    
    N = 4096
    
    def check_add(x, selected, triple_sums, pair_sums):
        """Check if x can be added to selected without violating triple-sum uniqueness.
        Returns (ok, new_sums) where new_sums is the list of new triple sums if ok."""
        n = len(selected)
        seen_new = set()
        
        # Triples with one x: x + a_i + a_j for i <= j
        # pair_sums contains all sums a_i + a_j for i <= j
        for ps in pair_sums:
            s = x + ps
            if s in seen_new or s in triple_sums:
                return False, None
            seen_new.add(s)
        
        # Triples with two x's: 2*x + a_i
        for i in range(n):
            s = 2 * x + selected[i]
            if s in seen_new or s in triple_sums:
                return False, None
            seen_new.add(s)
        
        # Triple with three x's
        s = 3 * x
        if s in seen_new or s in triple_sums:
            return False, None
        seen_new.add(s)
        
        return True, seen_new
    
    def build_greedy(order):
        """Greedy: add numbers in the given order, maintaining set of triple sums."""
        selected = []
        triple_sums = set()
        selected_set = set()
        pair_sums = []  # list of a_i + a_j for i <= j
        
        for x in order:
            if time.time() > DEADLINE - 5:
                break
            if x in selected_set:
                continue
            ok, new_sums = check_add(x, selected, triple_sums, pair_sums)
            if ok:
                # Update pair_sums: add x + a_i for all existing a_i, and x + x
                for a in selected:
                    pair_sums.append(x + a)
                pair_sums.append(2 * x)
                
                selected.append(x)
                selected_set.add(x)
                triple_sums.update(new_sums)
        
        return selected
    
    # Phase 1: Deterministic Greedies
    best_set = []
    
    # High to Low
    order_hl = range(N, 0, -1)
    set_hl = build_greedy(order_hl)
    if len(set_hl) > len(best_set):
        best_set = set_hl
    
    # Low to High
    order_lh = range(1, N + 1)
    set_lh = build_greedy(order_lh)
    if len(set_lh) > len(best_set):
        best_set = set_lh
    
    # Phase 2: Randomized Greedies with bias
    rng = random.Random(42)
    
    max_restarts = 200
    for i in range(max_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Mix strategies: pure random, biased high, biased low
        if i % 3 == 0:
            # Pure random
            order_rand = list(range(1, N + 1))
            rng.shuffle(order_rand)
        elif i % 3 == 1:
            # Bias towards high numbers: shuffle top half, then bottom half
            top = list(range(N // 2 + 1, N + 1))
            bottom = list(range(1, N // 2 + 1))
            rng.shuffle(top)
            rng.shuffle(bottom)
            order_rand = top + bottom
        else:
            # Bias towards low numbers
            top = list(range(N // 2 + 1, N + 1))
            bottom = list(range(1, N // 2 + 1))
            rng.shuffle(top)
            rng.shuffle(bottom)
            order_rand = bottom + top
        
        set_rand = build_greedy(order_rand)
        if len(set_rand) > len(best_set):
            best_set = set_rand
            
            if len(best_set) >= 41:
                return best_set

    # Phase 3: Local Search - try to add or swap
    if best_set and time.time() < DEADLINE - 10:
        best_set_sorted = sorted(best_set)
        best_set_set = set(best_set_sorted)
        
        # Build triple sums and pair sums
        triple_sums = set()
        pair_sums = []
        m = len(best_set_sorted)
        for i in range(m):
            for j in range(i, m):
                ps = best_set_sorted[i] + best_set_sorted[j]
                pair_sums.append(ps)
                triple_sums.add(ps + best_set_sorted[i])
                for k in range(j, m):
                    triple_sums.add(best_set_sorted[i] + best_set_sorted[j] + best_set_sorted[k])
        
        # Try to add any x not in best_set
        added_any = True
        while added_any and time.time() < DEADLINE - 5:
            added_any = False
            for x in range(1, N + 1):
                if time.time() > DEADLINE - 5:
                    break
                if x in best_set_set:
                    continue
                ok, new_sums = check_add(x, best_set_sorted, triple_sums, pair_sums)
                if ok:
                    for a in best_set_sorted:
                        pair_sums.append(x + a)
                    pair_sums.append(2 * x)
                    
                    best_set_sorted.append(x)
                    best_set_set.add(x)
                    triple_sums.update(new_sums)
                    added_any = True
                    break  # Restart scan after adding
            
            if len(best_set_sorted) >= 41:
                return best_set_sorted

        # Try swap-based local search: remove one element, try to add two
        if time.time() < DEADLINE - 8:
            for _ in range(20):  # Limited iterations
                if time.time() > DEADLINE - 5:
                    break
                
                improved = False
                m = len(best_set_sorted)
                
                for idx in range(m):
                    if time.time() > DEADLINE - 5:
                        break
                    
                    removed = best_set_sorted[idx]
                    # Rebuild without this element
                    temp_set = best_set_sorted[:idx] + best_set_sorted[idx+1:]
                    temp_set_set = set(temp_set)
                    
                    temp_triple_sums = set()
                    temp_pair_sums = []
                    tm = len(temp_set)
                    for i in range(tm):
                        for j in range(i, tm):
                            ps = temp_set[i] + temp_set[j]
                            temp_pair_sums.append(ps)
                            temp_triple_sums.add(ps + temp_set[i])
                            for k in range(j, tm):
                                temp_triple_sums.add(temp_set[i] + temp_set[j] + temp_set[k])
                    
                    # Try to add up to 2 new elements
                    added_count = 0
                    for x in range(1, N + 1):
                        if time.time() > DEADLINE - 5:
                            break
                        if x in temp_set_set:
                            continue
                        ok, new_sums = check_add(x, temp_set, temp_triple_sums, temp_pair_sums)
                        if ok:
                            for a in temp_set:
                                temp_pair_sums.append(x + a)
                            temp_pair_sums.append(2 * x)
                            temp_set.append(x)
                            temp_set_set.add(x)
                            temp_triple_sums.update(new_sums)
                            added_count += 1
                            if added_count >= 2:
                                break
                    
                    if added_count >= 2:
                        best_set_sorted = sorted(temp_set)
                        best_set_set = set(best_set_sorted)
                        improved = True
                        break
                
                if not improved:
                    break

    return best_set
# EVOLVE-BLOCK-END
