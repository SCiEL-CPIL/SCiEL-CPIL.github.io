"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

P = {'n': 12, 'm': 40}

def _calc_score_bitmask(family):
    """Calculate score for a family of bitmasks."""
    m = len(family)
    count = 0
    for i in range(m):
        si = family[i]
        for j in range(i + 1, m):
            sj = family[j]
            if si != sj:
                if (si & sj) == si or (si & sj) == sj:
                    count += 1
    return count

def _compute_comparability_matrix(family):
    """Compute pairwise comparability matrix. Returns (m, m) bool array where
    mat[i,j] is True if family[i] is a strict subset of family[j]."""
    m = len(family)
    mat = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i != j:
                if (family[i] & family[j]) == family[i]:
                    mat[i, j] = True
    return mat

def _count_comparabilities_from_matrix(mat):
    """Count comparable pairs from the upper triangle of the comparability matrix."""
    m = mat.shape[0]
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            if mat[i, j] or mat[j, i]:
                count += 1
    return count

def _build_chains_init(n, m, seed=42):
    """Build initial family using chain structures.
    
    Strategy: Build several chains of nested sets, then fill remaining slots
    with sets that maximize comparability.
    """
    rng = random.Random(seed)
    max_mask = 1 << n
    all_sets = list(range(max_mask))
    
    family = []
    used = set()
    
    # Build chains: start with empty set, keep adding elements
    # We want chains that are as long as possible
    # A chain of length L gives L*(L-1)/2 comparable pairs
    
    # Strategy: Build k chains, each of length ~m/k
    # But chains can share elements in a way that creates more comparabilities
    
    # Better: Build one or two long chains, then add sets that are comparable to many chain members
    
    # Build a long chain first
    chain = [0]
    used.add(0)
    current = 0
    remaining_elements = list(range(n))
    rng.shuffle(remaining_elements)
    
    # Build a chain of length ~20
    target_chain_len = min(20, m // 2 + 1)
    for _ in range(target_chain_len - 1):
        if not remaining_elements:
            break
        # Add a random unused element
        idx = rng.randint(0, len(remaining_elements) - 1)
        elem = remaining_elements.pop(idx)
        new_set = current | (1 << elem)
        if new_set not in used:
            chain.append(new_set)
            used.add(new_set)
            current = new_set
    
    family.extend(chain)
    
    # Build a second chain from the remaining elements
    if len(family) < m:
        chain2 = []
        current2 = 0
        remaining2 = list(range(n))
        rng.shuffle(remaining2)
        
        # Start from a set that's not in the first chain
        # Find a small set not used
        for s in range(1, max_mask):
            if s not in used and s.bit_count() <= 3:
                current2 = s
                chain2.append(s)
                used.add(s)
                break
        
        if chain2:
            for elem in remaining2:
                if len(chain2) >= target_chain_len - 1:
                    break
                new_set = current2 | (1 << elem)
                if new_set not in used:
                    chain2.append(new_set)
                    used.add(new_set)
                    current2 = new_set
                if len(family) + len(chain2) >= m:
                    break
            family.extend(chain2)
    
    # Fill remaining slots with sets that maximize comparability
    # Sort all remaining sets by a heuristic: prefer sets that are comparable to many existing sets
    remaining_sets = [s for s in range(max_mask) if s not in used]
    rng.shuffle(remaining_sets)
    
    # Score each remaining set by how many it's comparable to
    def comparability_score(s, fam):
        score = 0
        for t in fam:
            if s != t:
                if (s & t) == s or (s & t) == t:
                    score += 1
        return score
    
    # Greedily add sets with highest comparability score
    while len(family) < m and remaining_sets:
        # Sample a subset for efficiency
        sample_size = min(200, len(remaining_sets))
        sample = remaining_sets[:sample_size]
        
        best_set = None
        best_score = -1
        for s in sample:
            sc = comparability_score(s, family)
            if sc > best_score:
                best_score = sc
                best_set = s
        
        if best_set is not None:
            family.append(best_set)
            used.add(best_set)
            remaining_sets.remove(best_set)
        else:
            # Just take the first available
            s = remaining_sets.pop(0)
            family.append(s)
            used.add(s)
    
    return family[:m]

def _greedy_build(n, m, seed=42):
    """Greedy construction: add sets one at a time, each maximizing marginal gain."""
    rng = random.Random(seed)
    max_mask = 1 << n
    family = []
    used = set()
    
    # Start with empty set and full set (they're comparable to many)
    # Actually, let's be smarter
    
    for _ in range(m):
        if time.time() > 55:
            break
        
        # Sample candidates
        sample_size = min(500, max_mask - len(used))
        candidates = []
        for s in range(max_mask):
            if s not in used:
                candidates.append(s)
                if len(candidates) >= sample_size:
                    break
        
        rng.shuffle(candidates)
        
        best_set = None
        best_gain = -1
        
        for s in candidates:
            gain = 0
            for t in family:
                if s != t:
                    if (s & t) == s or (s & t) == t:
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_set = s
        
        if best_set is not None:
            family.append(best_set)
            used.add(best_set)
        else:
            # Fallback: pick any unused set
            for s in range(max_mask):
                if s not in used:
                    family.append(s)
                    used.add(s)
                    break
    
    return family

def _exact_delta(family, old_idx, new_set):
    """Compute exact change in score when replacing family[old_idx] with new_set."""
    m = len(family)
    old_set = family[old_idx]
    
    if new_set == old_set:
        return 0
    
    # Check new_set is not already in family (at another position)
    for j in range(m):
        if j != old_idx and family[j] == new_set:
            return -999999  # Invalid
    
    delta = 0
    for j in range(m):
        if j == old_idx:
            continue
        other = family[j]
        
        # Old contribution: was old_set comparable to other?
        old_comp = 0
        if old_set != other:
            if (old_set & other) == old_set or (old_set & other) == other:
                old_comp = 1
        
        # New contribution: is new_set comparable to other?
        new_comp = 0
        if new_set != other:
            if (new_set & other) == new_set or (new_set & other) == other:
                new_comp = 1
        
        delta += new_comp - old_comp
    
    return delta

def _local_search(family, n, m, deadline, verbose=False):
    """Local search with exact delta computation and smart candidate generation."""
    max_mask = 1 << n
    family = list(family)
    current_score = _calc_score_bitmask(family)
    best_family = list(family)
    best_score = current_score
    
    # Precompute: for each set in family, which other sets it's comparable to
    # This helps with candidate generation
    
    no_improvement = 0
    max_no_improvement = 5
    
    while time.time() < deadline:
        best_delta = 0
        best_idx = -1
        best_new_set = -1
        
        for idx in range(m):
            if time.time() >= deadline:
                break
            
            old_set = family[idx]
            
            # Generate smart candidates:
            # 1. Sets that are supersets of old_set (extend chains upward)
            # 2. Sets that are subsets of old_set (extend chains downward)  
            # 3. Sets that are "near" old_set in Hamming distance
            # 4. Random sets
            
            candidates = set()
            
            # Supersets of old_set (add 1-2 elements)
            for add1 in range(n):
                if not (old_set & (1 << add1)):
                    s = old_set | (1 << add1)
                    if s != old_set and s < max_mask:
                        candidates.add(s)
                    # Add 2 elements
                    for add2 in range(add1 + 1, n):
                        if not (old_set & (1 << add2)):
                            s2 = old_set | (1 << add1) | (1 << add2)
                            if s2 != old_set and s2 < max_mask:
                                candidates.add(s2)
            
            # Subsets of old_set (remove 1-2 elements)
            for rem1 in range(n):
                if (old_set & (1 << rem1)):
                    s = old_set & ~(1 << rem1)
                    if s != old_set:
                        candidates.add(s)
                    for rem2 in range(rem1 + 1, n):
                        if (old_set & (1 << rem2)):
                            s2 = old_set & ~(1 << rem1) & ~(1 << rem2)
                            if s2 != old_set:
                                candidates.add(s2)
            
            # Random candidates
            for _ in range(50):
                candidates.add(random.randint(0, max_mask - 1))
            
            # Also try the empty set and full set
            candidates.add(0)
            candidates.add(max_mask - 1)
            
            # Evaluate candidates
            for s in candidates:
                if s == old_set:
                    continue
                if s in family:
                    continue
                
                delta = _exact_delta(family, idx, s)
                if delta > best_delta:
                    best_delta = delta
                    best_idx = idx
                    best_new_set = s
        
        if best_delta > 0 and best_idx >= 0:
            family[best_idx] = best_new_set
            current_score += best_delta
            no_improvement = 0
            
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
                
            if verbose:
                print(f"  Improved: score={current_score}, idx={best_idx}, new_set={best_new_set}")
        else:
            no_improvement += 1
            
            if no_improvement >= max_no_improvement:
                # Perturbation: make a random move (even if it worsens)
                idx = random.randint(0, m - 1)
                # Try several random new sets
                for attempt in range(100):
                    new_set = random.randint(0, max_mask - 1)
                    if new_set in family:
                        continue
                    
                    delta = _exact_delta(family, idx, new_set)
                    # Accept with probability based on how bad it is
                    if delta >= 0:
                        family[idx] = new_set
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_family = list(family)
                        break
                    else:
                        # Accept bad moves with decreasing probability
                        if random.random() < math.exp(delta / 5.0):
                            family[idx] = new_set
                            current_score += delta
                            break
                
                no_improvement = 0
    
    return best_family, best_score

def _multi_start_local_search(n, m, deadline):
    """Run multiple initializations and local searches, return best result."""
    start_time = time.time()
    remaining = deadline - start_time
    
    # Multiple different initializations
    initializations = []
    
    # 1. Chain-based
    for seed in [42, 123, 456]:
        if time.time() - start_time > remaining * 0.3:
            break
        init = _build_chains_init(n, m, seed)
        initializations.append(init)
    
    # 2. Greedy
    for seed in [7, 99, 2024]:
        if time.time() - start_time > remaining * 0.4:
            break
        init = _greedy_build(n, m, seed)
        initializations.append(init)
    
    # 3. Pure random
    for seed in [1, 2, 3]:
        if time.time() - start_time > remaining * 0.5:
            break
        rng = random.Random(seed)
        init = rng.sample(range(1 << n), m)
        initializations.append(init)
    
    best_family = None
    best_score = -1
    
    # Allocate time to each initialization
    n_inits = len(initializations)
    time_per_init = (deadline - time.time()) / max(1, n_inits)
    
    for i, init in enumerate(initializations):
        init_time = time.time()
        init_deadline = init_time + time_per_init * 0.8  # Leave some buffer
        
        if time.time() >= deadline - 1:
            break
        
        fam, score = _local_search(init, n, m, init_deadline)
        
        if score > best_score:
            best_score = score
            best_family = fam
    
    if best_family is None:
        best_family = initializations[0]
        best_score = _calc_score_bitmask(best_family)
    
    return best_family, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin (90s limit, leave 5s)
    n = P['n']
    m = P['m']
    
    best_family, best_score = _multi_start_local_search(n, m, deadline)
    
    # Final polish: one more pass of local search with remaining time
    if time.time() < deadline - 2:
        final_deadline = deadline - 1
        best_family, best_score = _local_search(best_family, n, m, final_deadline)
    
    # Convert back to lists of ints
    result = []
    for s in best_family:
        lst = []
        for b in range(n):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
    
    # Verify: exactly 40 distinct lists
    assert len(result) == 40, f"Expected 40 sets, got {len(result)}"
    assert len(set(tuple(x) for x in result)) == 40, "Sets are not distinct"
    
    return result
# EVOLVE-BLOCK-END
