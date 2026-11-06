import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Precompute all 4-subsets of [1,16] as bitmasks
    all_masks = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)

    N = len(all_masks)  # 1820
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}

    def is_sunflower(a, b, c):
        """Check if three 4-element sets (as bitmasks) form a sunflower."""
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc

    def can_add_fast(candidate, family_masks):
        """Check if candidate can be added to family without forming a sunflower."""
        for i in range(len(family_masks)):
            a = family_masks[i]
            for j in range(i + 1, len(family_masks)):
                b = family_masks[j]
                if is_sunflower(a, b, candidate):
                    return False
        return True

    def greedy_build_deterministic(deadline):
        """Build family greedily in lexicographic order."""
        family = []
        for idx in range(N):
            if time.time() > deadline:
                break
            mask = all_masks[idx]
            if can_add_fast(mask, family):
                family.append(mask)
        return family

    def greedy_build_random(seed_offset, deadline):
        """Build family greedily with a random starting order."""
        rng = random.Random(seed_offset)
        order = list(range(N))
        rng.shuffle(order)
        family = []
        for idx in order:
            if time.time() > deadline:
                break
            mask = all_masks[idx]
            if can_add_fast(mask, family):
                family.append(mask)
        return family

    def hypergraph_local_search(initial_family, deadline, max_steps=20000):
        """
        Reformulate as binary vector x in {0,1}^N.
        Objective: maximize sum(x) subject to no sunflower triple all being 1.
        
        We track for each pair (i,j) the number of "witness" sets k that would
        form a sunflower with i and j. If x_i = x_j = x_k = 1 for such a triple,
        it's invalid.
        
        Local search: randomly flip x_k (add or remove set k). Accept if the
        resulting configuration is still valid (no sunflower triple all-1).
        Since we only flip one bit at a time, we just need to check:
        - If adding k (0->1): no pair (i,j) both in family has is_sunflower(i,j,k)
        - If removing k (1->0): always valid (removing can't create a violation)
        
        To escape local optima, we also try 2-flips: remove one, add another.
        """
        family = initial_family[:]
        family_set = set(family)
        in_family = [False] * N
        for m in family:
            in_family[mask_to_idx[m]] = True
        
        best_size = len(family)
        no_improve = 0
        
        # Precompute conflict graph: for each set k, which pairs (i,j) in family
        # would form a sunflower with k. We'll recompute on the fly for speed.
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if no_improve > 300:
                break
            
            # Randomly choose a flip type
            r = random.random()
            
            if r < 0.6:
                # Single add: pick random k not in family, try to add
                k = random.randint(0, N - 1)
                if in_family[k]:
                    no_improve += 1
                    continue
                mk = all_masks[k]
                # Check if mk can be added
                valid = True
                for i in range(len(family)):
                    a = family[i]
                    for j in range(i + 1, len(family)):
                        b = family[j]
                        if is_sunflower(a, b, mk):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    family.append(mk)
                    family_set.add(mk)
                    in_family[k] = True
                    if len(family) > best_size:
                        best_size = len(family)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                    
            elif r < 0.85:
                # Swap: remove one random family member, try to add a random non-member
                if not family:
                    no_improve += 1
                    continue
                rem_idx = random.randint(0, len(family) - 1)
                rem_mask = family[rem_idx]
                rem_k = mask_to_idx[rem_mask]
                
                add_k = random.randint(0, N - 1)
                if in_family[add_k]:
                    no_improve += 1
                    continue
                add_mask = all_masks[add_k]
                
                # Temporarily remove rem_mask
                family.pop(rem_idx)
                family_set.remove(rem_mask)
                in_family[rem_k] = False
                
                # Try to add add_mask
                valid = True
                for i in range(len(family)):
                    a = family[i]
                    for j in range(i + 1, len(family)):
                        b = family[j]
                        if is_sunflower(a, b, add_mask):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    family.append(add_mask)
                    family_set.add(add_mask)
                    in_family[add_k] = True
                    # Size unchanged but structure may be better
                    # Try to greedily add more
                    added = 0
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        t = random.randint(0, N - 1)
                        if in_family[t]:
                            continue
                        mt = all_masks[t]
                        ok = True
                        for i in range(len(family)):
                            a = family[i]
                            for j in range(i + 1, len(family)):
                                b = family[j]
                                if is_sunflower(a, b, mt):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            family.append(mt)
                            family_set.add(mt)
                            in_family[t] = True
                            added += 1
                    if added > 0:
                        if len(family) > best_size:
                            best_size = len(family)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Restore
                    family.insert(rem_idx, rem_mask)
                    family_set.add(rem_mask)
                    in_family[rem_k] = True
                    no_improve += 1
                    
            else:
                # Remove one random member (to escape bad configurations)
                if not family:
                    no_improve += 1
                    continue
                rem_idx = random.randint(0, len(family) - 1)
                rem_mask = family[rem_idx]
                rem_k = mask_to_idx[rem_mask]
                family.pop(rem_idx)
                family_set.remove(rem_mask)
                in_family[rem_k] = False
                # Try to add back several
                added = 0
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    t = random.randint(0, N - 1)
                    if in_family[t]:
                        continue
                    mt = all_masks[t]
                    ok = True
                    for i in range(len(family)):
                        a = family[i]
                        for j in range(i + 1, len(family)):
                            b = family[j]
                            if is_sunflower(a, b, mt):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        family.append(mt)
                        family_set.add(mt)
                        in_family[t] = True
                        added += 1
                if added >= 1:
                    if len(family) > best_size:
                        best_size = len(family)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
        
        return family

    def try_add_all_remaining(family, deadline):
        """Try to add all remaining sets that can be added."""
        family = family[:]
        family_set = set(family)
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                mask = all_masks[idx]
                if mask not in family_set and can_add_fast(mask, family):
                    family.append(mask)
                    family_set.add(mask)
                    added_any = True
        return family

    best_family = []
    best_size = 0

    # Phase 1: Deterministic greedy
    family = greedy_build_deterministic(deadline - 30)
    if len(family) > best_size:
        best_size = len(family)
        best_family = family[:]

    # Phase 2: Multiple random greedy restarts (use remaining time budget)
    remaining = deadline - time.time()
    num_restarts = min(5, max(1, int(remaining / 10)))
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        rf = greedy_build_random(42 + r, deadline - 15)
        if len(rf) > best_size:
            best_size = len(rf)
            best_family = rf[:]

    # Phase 3: Hypergraph local search on best family
    if time.time() < deadline - 5:
        improved = hypergraph_local_search(best_family, deadline - 5, max_steps=20000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_family = improved[:]

    # Phase 4: Try to add any remaining sets
    if time.time() < deadline - 2:
        final = try_add_all_remaining(best_family, deadline - 2)
        if len(final) > best_size:
            best_size = len(final)
            best_family = final

    # Phase 5: One more local search pass on the final family
    if time.time() < deadline - 1:
        improved2 = hypergraph_local_search(best_family, deadline - 1, max_steps=5000)
        if len(improved2) > best_size:
            best_size = len(improved2)
            best_family = improved2[:]

    # Convert bitmasks back to sorted lists
    result = []
    for m in best_family:
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))

    return result
