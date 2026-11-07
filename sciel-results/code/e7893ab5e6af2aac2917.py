import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    total_masks = 1 << n  # 4096

    # Precompute comparability matrix
    # comp[a][b] = 1 if a and b are comparable (one is subset of other)
    # Using a list of bytearrays for memory and speed
    comp = [bytearray(total_masks) for _ in range(total_masks)]
    for a in range(total_masks):
        for b in range(a + 1, total_masks):
            if (a | b) == a or (a | b) == b:
                comp[a][b] = 1
                comp[b][a] = 1

    def count_comparables(family):
        cnt = 0
        for i in range(len(family)):
            a = family[i]
            ca = comp[a]
            for j in range(i + 1, len(family)):
                cnt += ca[family[j]]
        return cnt

    # Generate a random initial family of 40 distinct masks
    def random_family():
        all_masks = list(range(total_masks))
        random.shuffle(all_masks)
        return all_masks[:m]

    # Simulated Annealing
    def simulated_annealing(initial_family, time_limit):
        family = list(initial_family)
        current_score = count_comparables(family)
        best_family = list(family)
        best_score = current_score
        
        # Temperature schedule
        T = 10.0
        cooling_rate = 0.9995
        min_T = 0.01
        
        start_time = time.time()
        
        while T > min_T:
            if time.time() > time_limit:
                break
            
            # Pick a random index to swap
            i = random.randint(0, m - 1)
            old_mask = family[i]
            
            # Pick a random new mask not in family
            new_mask = random.randint(0, total_masks - 1)
            if new_mask in family:
                # Simple retry, or pick from a list of unused masks
                # For speed, we can just try again or pick a random different one
                # A more efficient way: maintain a set of unused masks? 
                # With 4096 masks and 40 used, collision is rare.
                # But checking `in family` is O(m). 
                # Let's just loop until we find one not in family.
                while new_mask in family:
                    new_mask = random.randint(0, total_masks - 1)
            
            # Calculate delta
            # delta = score(new_family) - score(old_family)
            # new_score = old_score - deg(old_mask) + deg(new_mask)
            # where deg(x) is number of comparable pairs with x in the family (excluding self)
            
            # Efficient delta calculation:
            # Remove old_mask, add new_mask.
            # Change in score = (comparables(new_mask, family\{old_mask})) - (comparables(old_mask, family\{old_mask}))
            
            # Note: if new_mask is comparable to old_mask, that pair is removed and not added?
            # No, the pair (old, new) is not in the family simultaneously in the "before" or "after" states directly in that way.
            # Before: family has old. Score includes pairs (old, x) for all x in family\{old}.
            # After: family has new. Score includes pairs (new, x) for all x in family\{old}.
            # The pair (old, new) is never present in either state.
            
            deg_old = 0
            deg_new = 0
            for j in range(m):
                if j == i:
                    continue
                other = family[j]
                deg_old += comp[old_mask][other]
                deg_new += comp[new_mask][other]
            
            delta = deg_new - deg_old
            
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                # Accept move
                family[i] = new_mask
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
            
            T *= cooling_rate
        
        return best_family, best_score

    # Run multiple restarts of SA
    best_overall_family = None
    best_overall_score = -1
    
    num_restarts = 20 # Adjust based on time
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initial family: mix of random and greedy?
        # Let's start with a random family for diversity
        init_fam = random_family()
        
        # Run SA with a time slice
        time_slice = (deadline - 5 - time.time()) / (num_restarts - r)
        # Ensure minimum time per run
        if time_slice < 1.0:
            time_slice = 1.0
            
        fam, score = simulated_annealing(init_fam, time.time() + time_slice)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_family = fam
            
        # If we have a very good score, we might reduce restarts or increase time for polishing
        # But for now, keep it simple.

    # Final Polish: Local Search on the best family found
    if best_overall_family is not None and time.time() < deadline - 2:
        fam = list(best_overall_family)
        score = count_comparables(fam)
        
        # Greedy Local Search
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(m):
                if time.time() > deadline - 1:
                    break
                old_mask = fam[i]
                # Find best replacement
                best_new = None
                best_delta = 0
                
                # Try random candidates and maybe some structured ones?
                # Random is fast.
                for _ in range(50):
                    new_mask = random.randint(0, total_masks - 1)
                    if new_mask in fam:
                        continue
                    
                    deg_old = 0
                    deg_new = 0
                    for j in range(m):
                        if j == i:
                            continue
                        other = fam[j]
                        deg_old += comp[old_mask][other]
                        deg_new += comp[new_mask][other]
                    
                    delta = deg_new - deg_old
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new_mask
                
                if best_new is not None and best_delta > 0:
                    fam[i] = best_new
                    score += best_delta
                    improved = True
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_family = fam

    # Convert to lists of ints
    result = []
    for mask in best_overall_family:
        lst = [int(i) for i in range(n) if (mask >> i) & 1]
        result.append(lst)
    
    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
