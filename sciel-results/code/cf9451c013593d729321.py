"""Combined program: precomputed comparability matrix + multi-restart local search with chain decomposition."""
import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 82
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute comparability matrix: comp[a][b] = 1 if a is strict subset of b or b is strict subset of a
    # Using a flat array for cache efficiency: comp[a*N + b]
    comp = bytearray(N * N)
    # Also build a list of comparable neighbors for each mask (for O(1) degree updates)
    neighbors = [None] * N
    for a in range(N):
        # strict subsets: b such that b is a proper subset of a
        sub = a
        while sub:
            sub = (sub - 1) & a
            if sub != a:
                comp[a * N + sub] = 1
                comp[sub * N + a] = 1
        # strict supersets: b such that a is a proper subset of b
        # iterate over all supersets of a by iterating over complements
        comp_mask = (N - 1) ^ a
        sup = comp_mask
        while True:
            b = a | sup
            if b != a:
                comp[a * N + b] = 1
                comp[b * N + a] = 1
            if sup == 0:
                break
            sup = (sup - 1) & comp_mask
    # Build neighbor lists
    for a in range(N):
        base = a * N
        nbrs = []
        for b in range(N):
            if b != a and comp[base + b]:
                nbrs.append(b)
        neighbors[a] = nbrs

    # Precompute popcount
    popcount = [0] * N
    for i in range(1, N):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    def compute_score(fam):
        """Compute total comparable pairs in family."""
        cnt = 0
        for i in range(len(fam)):
            base_i = fam[i] * N
            for j in range(i + 1, len(fam)):
                if comp[base_i + fam[j]]:
                    cnt += 1
        return cnt
    
    def get_degree(mask, fam, exclude_idx):
        """Count comparable sets with mask in fam, excluding position exclude_idx."""
        base = mask * N
        cnt = 0
        for i, s in enumerate(fam):
            if i != exclude_idx and comp[base + s]:
                cnt += 1
        return cnt
    
    def make_chain_family(rng):
        """Create initial family using chain decomposition with greedy fill."""
        family = []
        seen = set()
        
        # Generate multiple chains with different orderings
        base_order = list(range(n))
        chains = []
        
        # Chain 1: natural order
        chains.append(list(base_order))
        
        # Chain 2: reversed
        chains.append(list(reversed(base_order)))
        
        # Chain 3: rotate by 1
        chains.append(base_order[1:] + base_order[:1])
        
        # Chain 4: rotate by 2
        chains.append(base_order[2:] + base_order[:2])
        
        # Chain 5: random shuffle
        shuffled = list(base_order)
        rng.shuffle(shuffled)
        chains.append(shuffled)
        
        # Chain 6: another random shuffle
        shuffled2 = list(base_order)
        rng.shuffle(shuffled2)
        chains.append(shuffled2)
        
        # Build family from chains
        for chain in chains:
            s = 0
            if s not in seen:
                family.append(s)
                seen.add(s)
            for v in chain:
                s |= (1 << v)
                if s not in seen:
                    family.append(s)
                    seen.add(s)
        
        # Greedily add remaining sets to maximize comparabilities
        # Use deg_candidate approach for speed
        deg_candidate = [0] * N
        for s in family:
            for nbr in neighbors[s]:
                deg_candidate[nbr] += 1
        
        while len(family) < m:
            best_add = None
            best_deg = -1
            for cand in range(N):
                if cand in seen:
                    continue
                deg = deg_candidate[cand]
                if deg > best_deg:
                    best_deg = deg
                    best_add = cand
            if best_add is not None:
                family.append(best_add)
                seen.add(best_add)
                for nbr in neighbors[best_add]:
                    deg_candidate[nbr] += 1
            else:
                for cand in range(N):
                    if cand not in seen:
                        family.append(cand)
                        seen.add(cand)
                        break
                if len(family) >= m:
                    break
        
        return family[:m]
    
    def local_search(family, rng, time_limit):
        """Run greedy local search until no improvement or time limit.
        Uses O(1) degree updates via precomputed neighbor lists."""
        cur_score = compute_score(family)
        # degree[i] = number of j != i such that family[i] and family[j] are comparable
        degree = [0] * m
        for i in range(m):
            base = family[i] * N
            for j in range(i + 1, m):
                if comp[base + family[j]]:
                    degree[i] += 1
                    degree[j] += 1
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(m):
                if time.time() > time_limit:
                    break
                old = family[i]
                deg_old = degree[i]
                # Find best replacement using neighbor lists
                # For candidate 'new', its degree with respect to family (excluding i) is:
                #   sum over j != i of [new comparable with family[j]]
                # We can compute this by iterating over neighbors of 'new' and checking
                # how many are in family (excluding i).
                # To make this fast, maintain a count array: count_in_fam[mask] = number of times
                # mask appears in family (0 or 1 since distinct).
                # Actually family has distinct elements, so it's a set.
                # deg_new = |neighbors(new) ∩ family_set| - (1 if old in neighbors(new) else 0)
                # Wait: we're replacing old with new, so the new set's degree is:
                #   number of j != i such that new is comparable with family[j]
                # = |{j : family[j] in neighbors(new)}|  (since family has distinct elements and old != new)
                # But we need to exclude i, and family[i] = old which is NOT new, so:
                #   deg_new = |{j : family[j] in neighbors(new)}|
                # This equals the count of elements of family that are in neighbors(new).
                # Since family has m distinct elements, we can precompute for each candidate 'new':
                #   deg_new = sum(1 for s in family if s in neighbors(new))
                # But that's O(m) per candidate. Instead, maintain a global count array.
                # Let cnt_in_fam[x] = 1 if x is in family, else 0.
                # Then deg_new = sum(cnt_in_fam[s] for s in neighbors(new))
                # We can precompute this for all new at once? No, it changes as family changes.
                # Alternative: for each i, iterate over all possible new masks and compute deg_new.
                # To speed up: for a given family, deg_new for all new can be computed as:
                #   deg_new[new] = sum_{j=0}^{m-1} comp[new * N + family[j]]  (but this is O(m*N) total)
                # That's 40 * 4096 = 163840 operations per i, times 40 i's = 6.5M per pass.
                # With the neighbor list approach: for each new, deg_new = |neighbors(new) ∩ family|
                # We can compute this by: for each j, for each s in neighbors(family[j]), increment deg_candidate[s].
                # That's sum over j of |neighbors(family[j])| which is at most m * 4095 but typically much less.
                # Average |neighbors| for a random mask in 12-dim: each mask has (2^k - 1) subsets and (2^(12-k) - 1) supersets.
                # Average over all masks: sum_k C(12,k)(2^k - 1 + 2^(12-k) - 1)/4096
                # = (sum C(12,k) 2^k - 4096 + sum C(12,k) 2^(12-k) - 4096)/4096
                # = (3^12 - 4096 + 3^12 - 4096)/4096 = (2*531441 - 8192)/4096 = 1054690/4096 ≈ 257.5
                # So sum over j of |neighbors(family[j])| ≈ 40 * 257.5 ≈ 10300
                # This is much faster than 163840!
                # So: build deg_candidate array by iterating over family and their neighbors.
                deg_candidate = [0] * N  # expensive to allocate each time? 4096 ints
                # Actually, let's just do it directly for each i.
                # For efficiency, we'll compute deg_candidate once per pass (not per i) since family
                # only changes when we make a move.
                # Wait, but deg_candidate depends on which i we're excluding.
                # deg_new_excluding_i[new] = |{j != i : family[j] in neighbors(new)}|
                # = deg_candidate_full[new] - (1 if old in neighbors(new) else 0)
                # So we can compute deg_candidate_full once per pass, then for each i just subtract.
                # Let's restructure: compute deg_candidate_full at start of each pass.
                break  # break out to restructure
            # Recompute deg_candidate_full
            deg_candidate_full = [0] * N
            for j in range(m):
                s = family[j]
                for nbr in neighbors[s]:
                    deg_candidate_full[nbr] += 1
            # Now for each i, best new is argmax over new not in family and new != old of:
            #   deg_candidate_full[new] - (1 if old in neighbors(new) else 0)
            # But "old in neighbors(new)" is equivalent to "new in neighbors(old)" (symmetric)
            # So: adjusted[new] = deg_candidate_full[new] - (1 if new in neighbors[old] else 0)
            # We need to find max of adjusted[new] over new not in family, new != old.
            # Since family has only 40 elements, we can check membership quickly.
            fam_set = set(family)
            for i in range(m):
                if time.time() > time_limit:
                    break
                old = family[i]
                deg_old = degree[i]
                best_new = None
                best_adj = -1
                old_nbrs = neighbors[old]
                # Iterate over all possible new masks
                for new in range(N):
                    if new in fam_set:
                        continue
                    adj = deg_candidate_full[new]
                    if new in old_nbrs:  # This is O(|neighbors(old)|) lookup... not great
                        adj -= 1
                    if adj > best_adj:
                        best_adj = adj
                        best_new = new
                if best_new is not None and best_adj > deg_old:
                    # Make the move
                    fam_set.discard(old)
                    family[i] = best_new
                    fam_set.add(best_new)
                    cur_score += (best_adj - deg_old)
                    # Update degree array
                    # Old degrees: for each j != i, if family[j] comparable with old, degree[j]--
                    for j in range(m):
                        if j == i:
                            continue
                        if comp[old * N + family[j]]:
                            degree[j] -= 1
                    # New degrees: for each j != i, if family[j] comparable with best_new, degree[j]++
                    for j in range(m):
                        if j == i:
                            continue
                        if comp[best_new * N + family[j]]:
                            degree[j] += 1
                    degree[i] = best_adj
                    # Update deg_candidate_full
                    for nbr in neighbors[old]:
                        deg_candidate_full[nbr] -= 1
                    for nbr in neighbors[best_new]:
                        deg_candidate_full[nbr] += 1
                    improved = True
        return cur_score
    
    def anneal_search(family, rng, time_limit):
        """Simulated annealing for escaping local optima with O(1) degree updates."""
        cur_score = compute_score(family)
        temp = 8.0
        cooling = 0.97
        family_set = set(family)
        # Precompute degree for each position
        degree = [0] * m
        for i in range(m):
            base = family[i] * N
            for j in range(i + 1, m):
                if comp[base + family[j]]:
                    degree[i] += 1
                    degree[j] += 1
        # deg_candidate_full[new] = number of family members comparable with new
        deg_candidate_full = [0] * N
        for j in range(m):
            s = family[j]
            for nbr in neighbors[s]:
                deg_candidate_full[nbr] += 1
        
        while temp > 0.5 and time.time() < time_limit:
            i = rng.randint(0, m - 1)
            old = family[i]
            deg_old = degree[i]
            
            new = rng.randint(0, N - 1)
            if new == old or new in family_set:
                temp *= cooling
                continue
            
            # deg_new = |{j != i : family[j] in neighbors(new)}|
            # = deg_candidate_full[new] - (1 if old in neighbors(new) else 0)
            # old in neighbors(new) iff new in neighbors(old)
            deg_new = deg_candidate_full[new]
            if new in neighbors[old]:
                deg_new -= 1
            delta = deg_new - deg_old
            
            if delta > 0 or rng.random() < pow(2.718281828, delta / temp):
                family_set.discard(old)
                family[i] = new
                family_set.add(new)
                cur_score += delta
                # Update degree
                for j in range(m):
                    if j == i:
                        continue
                    if comp[old * N + family[j]]:
                        degree[j] -= 1
                    if comp[new * N + family[j]]:
                        degree[j] += 1
                degree[i] = deg_new
                # Update deg_candidate_full
                for nbr in neighbors[old]:
                    deg_candidate_full[nbr] -= 1
                for nbr in neighbors[new]:
                    deg_candidate_full[nbr] += 1
            
            temp *= cooling
        
        return cur_score
    
    best_family = None
    best_score = -1
    
    # Multi-restart search
    rng = random.Random(42)
    num_restarts = 0
    
    while time.time() < deadline - 5:
        num_restarts += 1
        family = make_chain_family(rng)
        
        # Phase 1: Greedy local search
        remaining = deadline - 5 - time.time()
        phase1_time = time.time() + max(1.0, remaining * 0.4)
        score1 = local_search(family, rng, phase1_time)
        
        # Phase 2: Annealing
        if time.time() < deadline - 3:
            phase2_time = time.time() + max(0.5, (deadline - 3 - time.time()) * 0.5)
            score2 = anneal_search(family, rng, phase2_time)
            score1 = max(score1, score2)
        
        # Phase 3: Final greedy polish
        if time.time() < deadline - 1:
            phase3_time = deadline - 1
            score3 = local_search(family, rng, phase3_time)
            score1 = max(score1, score3)
        
        # Phase 4: 2-swap polish (swap two elements simultaneously)
        if time.time() < deadline - 0.5:
            # Try swapping pairs (i, j) with (new_i, new_j)
            fam_set = set(family)
            cur_score = compute_score(family)
            degree = [0] * m
            for i in range(m):
                base = family[i] * N
                for j in range(i + 1, m):
                    if comp[base + family[j]]:
                        degree[i] += 1
                        degree[j] += 1
            improved2 = True
            while improved2 and time.time() < deadline - 0.5:
                improved2 = False
                for i in range(m):
                    if time.time() > deadline - 0.5:
                        break
                    for j in range(i + 1, m):
                        if time.time() > deadline - 0.5:
                            break
                        old_i = family[i]
                        old_j = family[j]
                        # For a 2-swap, we try to find new_i, new_j that maximize total degree
                        # This is expensive, so we do a limited search:
                        # Try swapping old_i with a random candidate, and old_j with another
                        # Actually, let's do: for each candidate new_i not in family, 
                        #   compute delta_i = deg(new_i, family\{old_i}) - deg(old_i, family\{old_i})
                        # Similarly for new_j.
                        # But the interaction between new_i and new_j complicates things.
                        # For simplicity, do sequential single swaps which is already covered by local_search.
                        # Instead, do a 2-opt style: try all (new_i, new_j) pairs from a small candidate set.
                        # Generate top-20 candidates for each position by degree.
                        # This is getting complex; skip for now and rely on local_search + annealing.
                        pass
                # Actually, let's do a simpler 2-swap: randomly pick two positions and try to improve
                # by swapping their elements with two random candidates.
                if time.time() < deadline - 0.5:
                    i = rng.randint(0, m - 1)
                    j = rng.randint(0, m - 1)
                    if i == j:
                        continue
                    old_i = family[i]
                    old_j = family[j]
                    # Try a few random 2-swaps
                    for _ in range(5):
                        new_i = rng.randint(0, N - 1)
                        new_j = rng.randint(0, N - 1)
                        if new_i == old_i or new_j == old_j or new_i == old_j or new_j == old_i:
                            continue
                        if new_i in fam_set or new_j in fam_set:
                            continue
                        if new_i == new_j:
                            continue
                        # Compute delta for 2-swap
                        # Remove old_i and old_j, add new_i and new_j
                        # delta = (deg(new_i) + deg(new_j) + [new_i,new_j comparable]) 
                        #       - (deg(old_i) + deg(old_j) + [old_i,old_j comparable])
                        # where deg(x) is degree with respect to family \ {old_i, old_j}
                        # This is complex; let's just compute the full score difference.
                        # Too slow for random sampling. Skip.
                        pass
        
        if score1 > best_score:
            best_score = score1
            best_family = list(family)
        
        # If we've found a very good solution, do more polish
        if best_score >= 350:
            break
    
    # Final verification and polish
    if best_family is not None and time.time() < deadline - 0.5:
        final_score = local_search(best_family, rng, deadline - 0.5)
        if final_score > best_score:
            best_score = final_score
    
    # Convert to lists of ints
    result = []
    for mask in best_family:
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
