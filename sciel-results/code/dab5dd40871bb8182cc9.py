import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    n = 16
    all_points = list(range(n))

    # Precompute all C(16,3) = 560 possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # For each triple, precompute the 3 pairs it uses
    triple_pairs = {}
    for t in all_triples:
        pairs = ((t[0], t[1]), (t[0], t[2]), (t[1], t[2]))
        triple_pairs[t] = pairs

    # Index: pair -> list of triples containing that pair
    pair_to_triples = defaultdict(list)
    for t in all_triples:
        for p in triple_pairs[t]:
            pair_to_triples[p].append(t)

    # Bitmask helpers for independence check
    # Each triple is a 16-bit mask
    triple_masks = {}
    for t in all_triples:
        m = 0
        for x in t:
            m |= (1 << x)
        triple_masks[t] = m

    def is_linear(triple_set):
        """Check if no pair is shared by two triples."""
        seen_pairs = set()
        for t in triple_set:
            for p in triple_pairs[t]:
                if p in seen_pairs:
                    return False
                seen_pairs.add(p)
        return True

    def compute_alpha_exact(triple_set):
        """
        Compute the independence number (largest set containing no triple) 
        using exact bitmask enumeration over all 2^16 subsets.
        Returns the independence number.
        """
        if not triple_set:
            return n

        # Build a bitmask of "forbidden" subsets: a subset S is independent 
        # iff no triple is fully contained in S.
        # We can check this by: for each triple t, S must not contain all of t.
        
        # Efficient approach: iterate over all 2^16 subsets and check independence.
        # But 65536 checks is fast.
        
        # Alternative: use the fact that a set S is independent iff 
        # for every triple t, S is not a superset of t.
        
        # Let's use a different approach: find the maximum independent set
        # by checking all subsets in order of size (descending).
        
        # Actually, let's just iterate over all subsets and check.
        # For each subset, check if it contains any triple.
        
        best_alpha = 0
        
        # Precompute: for each point, the set of triples containing it
        point_triples = defaultdict(list)
        for t in triple_set:
            for x in t:
                point_triples[x].append(t)
        
        # For each subset (represented as bitmask), check if it's independent
        # A subset is independent if no triple is fully contained in it.
        
        # We can optimize: if a subset contains a triple, all supersets also contain it.
        # So we can use a recursive approach.
        
        # Simple approach: iterate over all subsets, check independence.
        # 65536 * (check cost) should be fine.
        
        # Check cost: for a subset, check if any triple is contained.
        # We can precompute for each triple its mask, then check if (subset_mask & triple_mask) == triple_mask.
        
        # Even faster: build a lookup. For each subset mask, determine if it's independent.
        # We can do this by: for each triple, mark all supersets as non-independent.
        # But that's 560 * 2^13 = 4.5M operations, which is fine.
        
        # Actually, let's just do the direct check for each subset.
        
        # Build list of triple masks for fast checking
        t_masks = [triple_masks[t] for t in triple_set]
        
        # Iterate over all subsets
        for mask in range(1 << n):
            # Check if this subset is independent
            independent = True
            for tm in t_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                popcount = bin(mask).count('1')
                if popcount > best_alpha:
                    best_alpha = popcount
                    # Early termination: if we found a set of size n, we can stop
                    if best_alpha == n:
                        break
        
        return best_alpha

    def compute_alpha_branch_and_bound(triple_set, timeout=1.0):
        """
        Compute the independence number using branch-and-bound with bitmasks.
        Returns alpha or None on timeout.
        """
        if not triple_set:
            return n

        # Build adjacency: for each point, which triples contain it
        point_triples = defaultdict(list)
        for t in triple_set:
            for x in t:
                point_triples[x].append(t)

        best_alpha = [0]
        deadline_local = start_time + timeout
        node_count = [0]

        # Sort points by degree (number of triples containing them) descending
        degrees = {x: len(point_triples[x]) for x in range(n)}
        order = sorted(range(n), key=lambda x: -degrees[x])

        # Reverse order for backtracking
        def backtrack(idx, current_mask, current_size):
            node_count[0] += 1
            if node_count[0] % 10000 == 0 and time.time() > deadline_local:
                raise TimeoutError()

            # Upper bound: current_size + remaining points
            remaining = n - idx
            if current_size + remaining <= best_alpha[0]:
                return

            if idx == n:
                if current_size > best_alpha[0]:
                    best_alpha[0] = current_size
                return

            p = order[idx]
            bit = 1 << p

            # Option 1: don't include p
            backtrack(idx + 1, current_mask, current_size)

            # Option 2: include p (if it doesn't complete a triple)
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
            return best_alpha[0]
        except TimeoutError:
            return None

    def greedy_independent_set(triple_set):
        """
        Greedy construction of an independent set.
        Returns the size of the constructed set (lower bound on alpha).
        """
        if not triple_set:
            return n

        current_set = set()
        remaining = set(range(n))
        
        # Sort points by degree (fewer constraints first)
        point_triples_count = defaultdict(int)
        for t in triple_set:
            for x in t:
                point_triples_count[x] += 1
        
        points_sorted = sorted(range(n), key=lambda x: point_triples_count[x])
        
        for p in points_sorted:
            if p not in remaining:
                continue
            # Check if adding p completes any triple
            can_add = True
            for t in triple_set:
                if p in t:
                    others = [x for x in t if x != p]
                    if all(o in current_set for o in others):
                        can_add = False
                        break
            if can_add:
                current_set.add(p)
                remaining.remove(p)
        
        return len(current_set)

    # --- Main: Simulated Annealing with exact alpha evaluation ---
    
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

    # Start with a moderate number of triples
    initial_count = 25
    current = random_linear_packing(initial_count, seed=42)
    
    # Verify linearity
    assert is_linear(current), "Initial packing not linear"
    
    # Compute initial alpha using exact method
    current_alpha = compute_alpha_exact(current)
    
    best = current[:]
    best_alpha = current_alpha
    
    T = 5.0  # initial temperature
    cooling = 0.995
    T_min = 0.1
    
    rng = random.Random(123)
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor
        action = rng.random()
        new_triples = current[:]
        
        if action < 0.35:
            # Add a random triple
            candidates = [t for t in all_triples if t not in new_triples]
            if candidates:
                t = rng.choice(candidates)
                # Check linearity
                ok = True
                for p in triple_pairs[t]:
                    for existing in new_triples:
                        if p in triple_pairs[existing]:
                            ok = False
                            break
                        if not ok:
                            break
                if ok:
                    new_triples.append(t)
        elif action < 0.70:
            # Remove a random triple
            if new_triples:
                idx = rng.randrange(len(new_triples))
                new_triples.pop(idx)
        elif action < 0.85:
            # Swap: remove one, add one
            if new_triples:
                removed = new_triples.pop(rng.randrange(len(new_triples)))
                candidates = [t for t in all_triples if t not in new_triples and t != removed]
                if candidates:
                    t = rng.choice(candidates)
                    ok = True
                    for p in triple_pairs[t]:
                        for existing in new_triples:
                            if p in triple_pairs[existing]:
                                ok = False
                                break
                            if not ok:
                                break
                    if ok:
                        new_triples.append(t)
                    else:
                        new_triples.append(removed)  # put it back
        else:
            # 2-opt: remove two, add two
            if len(new_triples) >= 2:
                idx1 = rng.randrange(len(new_triples))
                idx2 = rng.randrange(len(new_triples))
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % len(new_triples)
                r1 = new_triples.pop(idx1)
                r2 = new_triples.pop(idx2 if idx2 < idx1 else idx2 - 1)
                candidates = [t for t in all_triples if t not in new_triples and t != r1 and t != r2]
                if len(candidates) >= 2:
                    picks = rng.sample(candidates, 2)
                    ok = True
                    for t in picks:
                        for p in triple_pairs[t]:
                            for existing in new_triples:
                                if p in triple_pairs[existing]:
                                    ok = False
                                    break
                                if not ok:
                                    break
                        if not ok:
                            break
                    # Also check picks don't conflict with each other
                    if ok:
                        for p1 in triple_pairs[picks[0]]:
                            for p2 in triple_pairs[picks[1]]:
                                if p1 == p2:
                                    ok = False
                                    break
                            if not ok:
                                break
                    if ok:
                        new_triples.extend(picks)
                    else:
                        new_triples.extend([r1, r2])
                else:
                    new_triples.extend([r1, r2])

        if new_triples == current:
            continue
        
        # Evaluate using exact alpha (fast enough for 2^16 subsets)
        new_alpha = compute_alpha_exact(new_triples)
        
        delta = new_alpha - current_alpha
        
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / max(T, 0.01)):
            current = new_triples
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best = new_triples[:]
        
        T *= cooling
        
        # Periodic local search: try to remove triples that are "redundant" for independence
        if time.time() < deadline and len(current) > 5:
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                candidate = current[:]
                candidate.pop(i)
                a = compute_alpha_exact(candidate)
                if a <= current_alpha:
                    current = candidate
                    current_alpha = a
                    if a < best_alpha:
                        best_alpha = a
                        best = current[:]
                    break  # restart the loop

    # Final: verify best is valid
    if not is_linear(best):
        # Fallback
        best = random_linear_packing(15, seed=99)
    
    return [list(t) for t in best]
