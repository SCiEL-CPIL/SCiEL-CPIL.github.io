"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0
    
    N = 4096
    
    def check_add(x, selected, triple_sums):
        """Check if x can be added to selected without violating triple-sum uniqueness.
        Returns (ok, new_sums) where new_sums is the list of new triple sums if ok."""
        n = len(selected)
        seen_new = set()
        
        # Triples with one x: x + a_i + a_j for i <= j
        for i in range(n):
            for j in range(i, n):
                s = x + selected[i] + selected[j]
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
        
        return True, list(seen_new)
    
    def build_greedy(order):
        """Greedy: add numbers in the given order, maintaining set of triple sums."""
        selected = []
        triple_sums = set()
        selected_set = set()
        
        for x in order:
            if time.time() > DEADLINE - 5:
                break
            if x in selected_set:
                continue
            ok, new_sums = check_add(x, selected, triple_sums)
            if ok:
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
    
    # Phase 2: Randomized Greedies
    # The theoretical max is ~41. We try to find it.
    rng = random.Random(42)
    
    max_restarts = 100
    for i in range(max_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Create a random permutation of a subset to save time?
        # Or just shuffle the full range.
        # Shuffling 4096 items is fast.
        order_rand = list(range(1, N + 1))
        rng.shuffle(order_rand)
        
        # Optimization: Bias towards larger numbers?
        # High-to-Low worked best theoretically.
        # Let's mix: 50% High-to-Low, 50% Random.
        if i % 2 == 0:
            # Slight randomization of High-to-Low
            # Keep top 50% in random order, bottom 50% in random order?
            # Or just shuffle the whole thing.
            pass 
            
        set_rand = build_greedy(order_rand)
        if len(set_rand) > len(best_set):
            best_set = set_rand
            
            # Early exit if we hit the theoretical max
            if len(best_set) >= 41:
                return best_set

    # Phase 3: Local Search / Polish
    # Try to add any remaining numbers to the best set
    if best_set and time.time() < DEADLINE - 5:
        best_set_sorted = sorted(best_set)
        best_set_set = set(best_set_sorted)
        
        # Rebuild sums for the best set
        triple_sums = set()
        m = len(best_set_sorted)
        for i in range(m):
            for j in range(i, m):
                for k in range(j, m):
                    triple_sums.add(best_set_sorted[i] + best_set_sorted[j] + best_set_sorted[k])
        
        # Try to add any x not in best_set
        for x in range(1, N + 1):
            if time.time() > DEADLINE - 5:
                break
            if x in best_set_set:
                continue
            ok, new_sums = check_add(x, best_set_sorted, triple_sums)
            if ok:
                best_set_sorted.append(x)
                best_set_set.add(x)
                triple_sums.update(new_sums)
                
                if len(best_set_sorted) >= 41:
                    return best_set_sorted

    return best_set
# EVOLVE-BLOCK-END
