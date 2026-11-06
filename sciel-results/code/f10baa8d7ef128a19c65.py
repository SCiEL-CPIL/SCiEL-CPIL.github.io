import random
import itertools

def construct():
    n = 16
    points = list(range(n))
    
    # Precompute all 5-subsets as bitmasks
    # We want to hit all 5-subsets to ensure alpha <= 4
    target_size = 5
    all_5_sets = []
    for comb in itertools.combinations(points, target_size):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_5_sets.append(mask)
        
    num_5_sets = len(all_5_sets)
    
    # Precompute for each triple, which 5-sets it hits
    # triple_mask -> list of indices of 5-sets it hits
    triple_hits = {}
    all_triples = list(itertools.combinations(points, 3))
    for t in all_triples:
        t_mask = 0
        for p in t:
            t_mask |= (1 << p)
        hits = []
        for i, s_mask in enumerate(all_5_sets):
            if (t_mask & s_mask) == t_mask: # t is subset of s
                hits.append(i)
        triple_hits[t] = (t_mask, hits)
        
    best_E = []
    best_alpha = n # Start high
    
    # We will try to construct a system with alpha <= 4 (hits all 5-sets)
    # If we fail, we try alpha <= 5 (hits all 6-sets)? No, alpha <= 4 is stronger.
    # If we can't hit all 5-sets, we might have alpha=5.
    # Let's try to find a solution with alpha=4.
    
    def try_construct(target_alpha):
        # If target_alpha is 4, we must hit all 5-sets.
        # If target_alpha is 5, we must hit all 6-sets.
        # We will use target_size = target_alpha + 1
        
        t_size = target_alpha + 1
        if t_size > n:
            return None, target_alpha
            
        target_sets = []
        for comb in itertools.combinations(points, t_size):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            target_sets.append(mask)
            
        # Precompute hits for this target size
        t_hits = {}
        for t in all_triples:
            t_mask = 0
            for p in t:
                t_mask |= (1 << p)
            hits = []
            for i, s_mask in enumerate(target_sets):
                if (t_mask & s_mask) == t_mask:
                    hits.append(i)
            t_hits[t] = (t_mask, hits)
            
        # Randomized Greedy Set Cover with Linearity Constraint
        # We use a bitmask for uncovered sets. 
        # Since num sets can be large (C(16,5)=4368), we can use a list of bools or a bitset.
        # Python integers are arbitrary precision, so we can use a big int for the uncovered mask.
        
        # Map set index to bit position in the big int
        # uncovered_mask bit i is 1 if set i is uncovered
        
        for restart in range(50):
            uncovered = (1 << len(target_sets)) - 1
            E = []
            used_masks = [] # List of triple masks for linearity check
            
            # Heuristic: Pick triple that covers max uncovered sets
            # To speed up, we can just iterate all triples and calculate coverage
            # 560 triples * 50 restarts * 16 steps * 4368 checks is too slow if naive.
            # We need to update coverage efficiently.
            
            # Alternative: 
            # Maintain a count of how many times each triple is "good"
            # Actually, just picking the best available triple each step is O(T * S)
            # T=560, S=4368. 560*4368 ~ 2.4M ops per step. 16 steps ~ 40M ops per restart.
            # 50 restarts ~ 2B ops. Too slow in Python.
            
            # Optimization: 
            # Instead of checking all triples against all sets, we can precompute 
            # "which triples hit which set".
            # set_to_triples[i] = list of triples that hit set i.
            
            set_to_triples = [[] for _ in range(len(target_sets))]
            for t, (t_mask, hits) in t_hits.items():
                for h in hits:
                    set_to_triples[h].append(t)
                    
            # Now, greedy step:
            # Find the set with the fewest available covering triples? 
            # Or the set that is uncovered and has fewest options?
            # Standard Set Cover heuristic: pick set with min candidates, then pick best triple for it.
            
            success = True
            while uncovered:
                # Find an uncovered set with minimum number of valid covering triples
                min_count = float('inf')
                target_set_idx = -1
                best_triple_for_set = None
                max_new_coverage = -1
                
                # Iterate uncovered sets
                # To iterate uncovered sets efficiently from the big int:
                # We can just loop through indices if count is small, but 4368 is ok.
                # To make it faster, we can keep a list of uncovered indices.
                
                # Let's use a list of uncovered indices for simplicity, updating it is O(S)
                # Or just scan the big int.
                
                # Scanning 4368 bits is fast.
                u = uncovered
                while u:
                    # Get lowest set bit index
                    lsb = u & -u
                    idx = lsb.bit_length() - 1
                    u ^= lsb
                    
                    # Check if this set is still uncovered (it is, by definition of u)
                    # Get candidates for this set
                    candidates = set_to_triples[idx]
                    
                    # Filter candidates: must be linear with E
                    # And must not be already in E (implied by linearity if we check distinctness, 
                    # but we should ensure we don't add duplicate triples)
                    
                    valid_candidates = []
                    for t in candidates:
                        # Check linearity
                        t_mask = t_hits[t][0]
                        linear = True
                        for em in used_masks:
                            if (t_mask & em).bit_count() > 1:
                                linear = False
                                break
                        if linear:
                            # Check if t is already in E? 
                            # If t is in E, it would have covered this set already?
                            # Not necessarily, if the set was covered by another triple?
                            # No, if t is in E, and t hits idx, then idx would be covered.
                            # So if idx is uncovered, t is NOT in E.
                            valid_candidates.append(t)
                            
                    if not valid_candidates:
                        success = False
                        break
                        
                    # Pick the best triple among valid_candidates
                    # Best = covers most other uncovered sets
                    best_t = None
                    best_cov = -1
                    for t in valid_candidates:
                        hits = t_hits[t][1]
                        # Count how many of these hits are in uncovered
                        cov = 0
                        for h in hits:
                            if (uncovered >> h) & 1:
                                cov += 1
                        if cov > best_cov:
                            best_cov = cov
                            best_t = t
                            
                    if best_t is None:
                        success = False
                        break
                        
                    # Update best_triple_for_set logic:
                    # We want the set with min valid candidates.
                    if len(valid_candidates) < min_count:
                        min_count = len(valid_candidates)
                        target_set_idx = idx
                        best_triple_for_set = best_t
                        max_new_coverage = best_cov
                        
                        # If min_count is 1, we can break early? 
                        # Yes, we must pick the only option.
                        if min_count == 1:
                            break
                            
                if not success:
                    break
                    
                # Add the chosen triple
                chosen_t = best_triple_for_set
                E.append(chosen_t)
                used_masks.append(t_hits[chosen_t][0])
                
                # Update uncovered
                for h in t_hits[chosen_t][1]:
                    uncovered &= ~(1 << h)
                    
            if success:
                # Verify alpha
                # We hit all sets of size target_alpha+1.
                # So alpha <= target_alpha.
                return E, target_alpha
                
        return None, float('inf')
        
    # Try to get alpha = 4
    E, alpha = try_construct(4)
    if E is not None:
        return [list(e) for e in E]
        
    # Try to get alpha = 5
    E, alpha = try_construct(5)
    if E is not None:
        return [list(e) for e in E]
        
    # Fallback to random greedy if CSP fails (unlikely to be better, but safe)
    # Re-implement the old fast greedy as fallback
    full_mask = (1 << n) - 1
    def fast_calc_alpha(edges):
        if not edges:
            return n
        edge_masks = []
        for e in edges:
            m = 0
            for v in e:
                m |= (1 << v)
            edge_masks.append(m)
        is_indep = [True] * (1 << n)
        for em in edge_masks:
            complement = (~em) & full_mask
            sub = complement
            while True:
                s = em | sub
                is_indep[s] = False
                if sub == 0:
                    break
                sub = (sub - 1) & complement
        max_alpha = 0
        for i in range(1 << n):
            if is_indep[i]:
                pc = bin(i).count('1')
                if pc > max_alpha:
                    max_alpha = pc
        return max_alpha
        
    best_E = []
    best_alpha = n
    for restart in range(200):
        E = []
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        for edge in shuffled:
            edge_set = set(edge)
            linear = True
            for e in E:
                if len(edge_set & set(e)) > 1:
                    linear = False
                    break
            if linear:
                E.append(edge)
        alpha = fast_calc_alpha(E)
        if alpha < best_alpha:
            best_alpha = alpha
            best_E = E[:]
            
    return [list(e) for e in best_E]
