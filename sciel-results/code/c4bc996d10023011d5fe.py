import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    n = 16
    all_points = list(range(n))
    all_triples = list(itertools.combinations(range(n), 3))

    # Precompute pairs and masks
    triple_pairs = {}
    triple_masks = {}
    for t in all_triples:
        pairs = (t[0], t[1]), (t[0], t[2]), (t[1], t[2])
        triple_pairs[t] = pairs
        m = 0
        for x in t:
            m |= (1 << x)
        triple_masks[t] = m

    # Index: pair -> list of triples containing that pair (for fast conflict check)
    pair_to_triples = defaultdict(list)
    for t in all_triples:
        for p in triple_pairs[t]:
            pair_to_triples[p].append(t)

    def is_linear(triple_set):
        seen_pairs = set()
        for t in triple_set:
            for p in triple_pairs[t]:
                if p in seen_pairs:
                    return False
                seen_pairs.add(p)
        return True

    def compute_alpha(triple_set, timeout=2.0):
        """
        Compute the independence number using branch-and-bound with bitmasks.
        Returns alpha or None on timeout.
        """
        if not triple_set:
            return n

        # Build point -> triples map
        point_triples = defaultdict(list)
        for t in triple_set:
            for x in t:
                point_triples[x].append(t)

        best_alpha = 0
        deadline_local = start_time + timeout
        node_count = [0]

        # Sort points by degree descending for better pruning
        degrees = {x: len(point_triples[x]) for x in range(n)}
        order = sorted(range(n), key=lambda x: -degrees[x])

        def backtrack(idx, current_mask, current_size):
            nonlocal best_alpha
            node_count[0] += 1
            if node_count[0] % 5000 == 0 and time.time() > deadline_local:
                raise TimeoutError()

            # Upper bound
            remaining = n - idx
            if current_size + remaining <= best_alpha:
                return

            if idx == n:
                if current_size > best_alpha:
                    best_alpha = current_size
                return

            p = order[idx]
            bit = 1 << p

            # Option 1: Exclude p
            backtrack(idx + 1, current_mask, current_size)

            # Option 2: Include p (if valid)
            new_mask = current_mask | bit
            valid = True
            for t in point_triples[p]:
                other_mask = triple_masks[t] & ~bit
                if (current_mask & other_mask) == other_mask:
                    valid = False
                    break
            if valid:
                backtrack(idx + 1, new_mask, current_size + 1)

        try:
            backtrack(0, 0, 0)
            return best_alpha
        except TimeoutError:
            return None

    # --- Inspiration 1 Strategy: STS(15) + Matching on 15 points ---
    # This construction guarantees a specific structure that allows for
    # extremely fast evaluation of alpha (specifically checking for independent sets of size 8).

    # 1. Generate STS(15) using PG(3,2) construction
    # Points are 1..15 (representing non-zero vectors in GF(2)^4), mapped to 0..14
    sts_triples = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            k = i ^ j
            if k > j:
                t = tuple(sorted([i - 1, j - 1, k - 1]))
                sts_triples.append(t)
    
    # Verify we have 35 triples
    assert len(sts_triples) == 35, "STS(15) generation failed"

    # Precompute pairs for each STS triple
    sts_triple_pairs = []
    for t in sts_triples:
        sts_triple_pairs.append([(t[0], t[1]), (t[0], t[2]), (t[1], t[2])])

    # 2. Precompute data for all 8-subsets of 16 points
    # We need to check if an 8-subset S is independent.
    # Final system = (STS triples not covered by M) U { (15, u, v) for (u,v) in M }
    # S is independent iff:
    # A) If 15 in S: No pair (u,v) in M is such that u in S and v in S.
    # B) For all STS triples t: If t subset S, then at least one pair of t is in M.
    
    subsets_8 = list(itertools.combinations(range(16), 8))
    subset_data = []
    for s in subsets_8:
        s_set = set(s)
        sts_idx_in_s = []
        for idx, t in enumerate(sts_triples):
            if t[0] in s_set and t[1] in s_set and t[2] in s_set:
                sts_idx_in_s.append(idx)
        subset_data.append((s, s_set, sts_idx_in_s))

    def evaluate_M_fast(M):
        """
        Returns the number of independent sets of size 8.
        If 0, alpha <= 7.
        """
        M_set = set(M)
        count = 0
        for s, s_set, sts_idx_in_s in subset_data:
            # Condition A: Check new triples involving 15
            if 15 in s_set:
                conflict = False
                for u, v in M:
                    if u in s_set and v in s_set:
                        conflict = True
                        break
                if conflict:
                    continue # Not independent
            
            # Condition B: Check STS triples
            # If there exists an STS triple in S that is NOT covered by M, then S contains a triple -> Not Independent.
            # So S is independent ONLY IF all STS triples in S are covered by M.
            conflict = False
            for idx in sts_idx_in_s:
                t_pairs = sts_triple_pairs[idx]
                covered = False
                for p in t_pairs:
                    if p in M_set:
                        covered = True
                        break
                if not covered:
                    conflict = True
                    break
            if not conflict:
                count += 1
        
        return count

    def get_final_triples(M):
        M_set = set(M)
        final = []
        # Add new triples with 15
        for u, v in M:
            final.append((15, u, v))
        # Add kept STS triples
        for i, t in enumerate(sts_triples):
            t_pairs = sts_triple_pairs[i]
            keep = True
            for p in t_pairs:
                if p in M_set:
                    keep = False
                    break
            if keep:
                final.append(t)
        return final

    # --- Optimization: Local Search on Matchings ---
    
    best_M = None
    best_score = 12870 # Max possible independent sets (C(16,8))
    best_triples = None
    
    # Initialize best_triples with a random linear packing as fallback
    def random_linear_packing(num_triples, seed=0):
        rng = random.Random(seed)
        triples = []
        used_pairs = set()
        candidates = all_triples[:]
        rng.shuffle(candidates)
        for t in candidates:
            if len(triples) >= num_triples:
                break
            ok = True
            for p in triple_pairs[t]:
                if p in used_pairs:
                    ok = False
                    break
            if ok:
                triples.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        return triples

    fallback_triples = random_linear_packing(20, seed=42)
    best_triples = fallback_triples

    def random_matching():
        pts = list(range(15))
        random.shuffle(pts)
        M = []
        for i in range(0, 14, 2):
            M.append(tuple(sorted((pts[i], pts[i + 1]))))
        return M

    # Simulated Annealing / Local Search on M
    # The "score" is the number of independent sets of size 8.
    # We want to minimize this score. If it hits 0, alpha <= 7.
    
    T = 10.0
    cooling = 0.99
    T_min = 0.1
    
    rng = random.Random(123)
    
    # Start with a few random restarts
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        M = random_matching()
        score = evaluate_M_fast(M)
        
        if score < best_score:
            best_score = score
            best_M = M[:]
            best_triples = get_final_triples(M)
            if best_score == 0:
                break
                
        # Local Search
        improved = True
        while improved:
            if time.time() > deadline - 2:
                break
            improved = False
            
            # Generate a neighbor: Swap one pair in M with a new pair
            # M has 7 pairs.
            # Pick a random pair to remove.
            idx_remove = rng.randrange(len(M))
            removed_pair = M.pop(idx_remove)
            
            # Find available points (not in remaining M)
            used_points = set()
            for u, v in M:
                used_points.add(u)
                used_points.add(v)
            
            available_points = [p for p in range(15) if p not in used_points]
            # available_points includes the 2 points from removed_pair and 1 other point (since 15-14=1)
            # Wait, M has 7 pairs -> 14 points used. 1 point unused.
            # If we remove 1 pair, we free 2 points. Total available = 3 points.
            # We need to pick 2 from 3.
            if len(available_points) >= 2:
                pair_candidates = list(itertools.combinations(available_points, 2))
                new_pair = tuple(sorted(pair_candidates[rng.randrange(len(pair_candidates))]))
                M.append(new_pair)
                
                new_score = evaluate_M_fast(M)
                
                # Accept if better or with probability
                if new_score < score or (score > 0 and rng.random() < pow(2.718281828, -(new_score - score) / max(T, 0.1))):
                    score = new_score
                    improved = True
                    if score < best_score:
                        best_score = score
                        best_M = M[:]
                        best_triples = get_final_triples(M)
                        if best_score == 0:
                            return [list(t) for t in best_triples]
                else:
                    # Revert
                    M.pop()
                    M.append(removed_pair)
            
            T *= cooling
            if T < T_min:
                break

    # If we didn't find alpha <= 7, check if we have a valid best_triples
    if best_triples is not None:
        # Verify linearity
        if is_linear(best_triples):
            # Optional: try to polish with exact alpha check if time permits?
            # The fast check says alpha <= 7 if score == 0.
            # If score > 0, alpha is at least 8.
            # We can try to verify the actual alpha of the best candidate found.
            # But for now, returning the best candidate is safe.
            return [list(t) for t in best_triples]

    # Fallback to random packing if STS strategy failed completely
    return [list(t) for t in fallback_triples]
