import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    N = 32
    MAXV = 512

    def make_mask(S):
        m = 0
        for x in S:
            m |= (1 << x)
        return m

    def count_diffs_from_mask(mask):
        """Count distinct d>0 such that exists a with a, a+d, a+2d all set."""
        diffs = set()
        # For each d, check if there exists a such that bits at a, a+d, a+2d are set
        # a ranges from 1 to MAXV-2d
        # a+d ranges from 2 to MAXV-d
        # a+2d ranges from 3 to MAXV
        # We can iterate over possible a values and check
        # Faster: for each pair (a, b) in set, d=b-a, check if a+2d is in set
        # But with bitmask, checking a+2d is just (mask >> (a+2d)) & 1
        # Let's iterate over elements in the set
        elems = [i for i in range(1, MAXV+1) if (mask >> i) & 1]
        for idx_a, a in enumerate(elems):
            for b in elems[idx_a+1:]:
                d = b - a
                if d > 0:
                    c = a + 2*d
                    if c <= MAXV and (mask >> c) & 1:
                        diffs.add(d)
        return len(diffs)

    def count_diffs(S):
        mask = make_mask(S)
        return count_diffs_from_mask(mask)

    def greedy_construct(seed):
        """Build a set greedily by adding elements that maximize progression differences."""
        rng = random.Random(seed)
        S = set()
        # Start with some random elements
        for _ in range(8):
            S.add(rng.randint(1, MAXV))
        
        best_score = count_diffs(S)
        
        # Try to add elements that create new progressions
        while len(S) < N:
            best_elem = None
            best_new_score = best_score
            candidates = [x for x in range(1, MAXV+1) if x not in S]
            rng.shuffle(candidates)
            # Test a subset of candidates for speed
            for cand in candidates[:200]:
                test_S = S | {cand}
                sc = count_diffs(test_S)
                if sc > best_new_score:
                    best_new_score = sc
                    best_elem = cand
            if best_elem is not None:
                S.add(best_elem)
                best_score = best_new_score
            else:
                S.add(rng.choice(candidates))
                best_score = count_diffs(S)
        return S, best_score

    def local_search(S, score, deadline, rng):
        """Improve set via targeted swaps that create new progression differences."""
        S = set(S)
        score = score
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
            improved = False
            elems = list(S)
            rng.shuffle(elems)
            
            for elem in elems:
                if time.time() > deadline - 5.0:
                    break
                
                # Strategy 1: Try to find a replacement that creates new progressions
                # For each d not yet a progression diff, try to create one
                mask = make_mask(S)
                current_diffs = set()
                elems_list = list(S)
                for i, a in enumerate(elems_list):
                    for b in elems_list[i+1:]:
                        d = b - a
                        if d > 0:
                            c = a + 2*d
                            if c <= MAXV and (mask >> c) & 1:
                                current_diffs.add(d)
                
                # Find missing small differences to try to add
                missing_diffs = [d for d in range(1, 100) if d not in current_diffs]
                rng.shuffle(missing_diffs)
                
                best_cand = None
                best_sc = score
                
                # Try candidates that might create new progressions
                for d in missing_diffs[:10]:
                    # To create progression with diff d: need a, a+d, a+2d
                    # If we remove elem and add cand, check if new set has more diffs
                    # Try a few candidate positions
                    for a_start in range(1, MAXV - 2*d + 1, max(1, (MAXV - 2*d) // 20)):
                        a = a_start
                        a_d = a + d
                        a_2d = a + 2*d
                        # Check which of these are in S (excluding elem)
                        in_set = sum(1 for x in [a, a_d, a_2d] if x in S and x != elem)
                        if in_set >= 1:
                            # Adding one of the missing ones might complete a progression
                            for cand in [a, a_d, a_2d]:
                                if cand in S or cand == elem:
                                    continue
                                test_S = set(S)
                                test_S.remove(elem)
                                test_S.add(cand)
                                sc = count_diffs(test_S)
                                if sc > best_sc:
                                    best_sc = sc
                                    best_cand = cand
                
                # Also try random candidates
                for _ in range(30):
                    cand = rng.randint(1, MAXV)
                    if cand in S:
                        continue
                    test_S = set(S)
                    test_S.remove(elem)
                    test_S.add(cand)
                    sc = count_diffs(test_S)
                    if sc > best_sc:
                        best_sc = sc
                        best_cand = cand
                
                if best_cand is not None:
                    S.remove(elem)
                    S.add(best_cand)
                    score = best_sc
                    improved = True
                    no_improve_count = 0
                    break
            
            if not improved:
                no_improve_count += 1
                # Random perturbation to escape local optimum
                if no_improve_count % 50 == 0:
                    elem = rng.choice(list(S))
                    cand = rng.randint(1, MAXV)
                    if cand not in S:
                        S.remove(elem)
                        S.add(cand)
                        score = count_diffs(S)
        
        return S, score

    # Multi-restart approach
    best_overall = None
    best_overall_score = 0
    
    # Phase 1: Greedy construction with multiple seeds
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 2718]
    for seed in seeds:
        if time.time() > deadline - 30.0:
            break
        S, sc = greedy_construct(seed)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = set(S)
    
    # Phase 2: Local search on the best greedy result
    if best_overall is not None:
        rng = random.Random(999)
        S, sc = local_search(best_overall, best_overall_score, deadline, rng)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = S
    
    # Phase 3: More local search with different perturbations
    if best_overall is not None and time.time() < deadline - 10.0:
        for trial in range(5):
            if time.time() > deadline - 10.0:
                break
            rng = random.Random(1000 + trial)
            S = set(best_overall)
            sc = best_overall_score
            # Perturb slightly
            for _ in range(3):
                elem = rng.choice(list(S))
                cand = rng.randint(1, MAXV)
                if cand not in S:
                    S.remove(elem)
                    S.add(cand)
            sc = count_diffs(S)
            S, sc = local_search(S, sc, deadline, rng)
            if sc > best_overall_score:
                best_overall_score = sc
                best_overall = S
    
    # Phase 4: Final intensive local search
    if best_overall is not None and time.time() < deadline - 5.0:
        rng = random.Random(777)
        S, sc = local_search(best_overall, best_overall_score, deadline, rng)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = S
    
    # Safety: ensure we return a valid set
    if best_overall is None or len(best_overall) != N:
        best_overall = set(range(1, N+1))
    
    return list(best_overall)
