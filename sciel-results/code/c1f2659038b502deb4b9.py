import numpy as np
import time
import random

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        deadline = time.time() + 115.0
        N = q ** n  # 6561
        powers = [q ** i for i in range(n)]
        
        # Precompute neighbors: neighbors[w] is a list of words within distance 1
        neigh_size = 1 + n * (q - 1)
        neighbors = np.zeros((N, neigh_size), dtype=np.int32)
        
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    new_w = w + diff * powers[i]
                    neighbors[w, idx] = new_w
                    idx += 1
        
        # Precompute neighbor sets as Python sets for faster intersection
        neigh_sets = [set(neighbors[w, :].tolist()) for w in range(N)]

        def compute_coverage(code_set):
            cov = np.zeros(N, dtype=np.int32)
            for c in code_set:
                cov[neighbors[c, :]] += 1
            return cov

        def greedy_init(rng, deadline):
            uncovered = np.ones(N, dtype=np.bool_)
            code = set()
            while np.any(uncovered) and time.time() < deadline - 5:
                gains = uncovered[neighbors].sum(axis=1)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains == max_gain)[0]
                w = rng.choice(candidates)
                code.add(w)
                uncovered[neighbors[w, :]] = False
            return code

        def local_search(code_set, rng, deadline):
            code_set = set(code_set)
            no_improve_count = 0
            max_no_improve = 20
            
            while time.time() < deadline - 2:
                if no_improve_count >= max_no_improve:
                    break
                
                # 1. Compute Coverage
                cov = compute_coverage(code_set)
                
                # 2. Deletion Pass
                redundant = []
                for c in code_set:
                    if np.min(cov[neighbors[c, :]]) > 1:
                        redundant.append(c)
                
                if redundant:
                    for c in redundant:
                        code_set.remove(c)
                    no_improve_count = 0
                    continue # Recompute coverage
                
                # 3. Swap Pass (Single)
                improved = False
                c_list = list(code_set)
                rng.shuffle(c_list)
                
                for c in c_list:
                    if time.time() > deadline - 2:
                        break
                    
                    c_neighbors = neighbors[c, :]
                    essential_mask = (cov[c_neighbors] == 1)
                    essential_words = c_neighbors[essential_mask]
                    
                    if len(essential_words) == 0:
                        continue
                    
                    if len(essential_words) == 1:
                        candidates = neigh_sets[essential_words[0]]
                    else:
                        candidates = set(neigh_sets[essential_words[0]])
                        for ew in essential_words[1:]:
                            candidates &= neigh_sets[ew]
                            if not candidates:
                                break
                    
                    if not candidates:
                        continue
                    
                    valid_candidates = [d for d in candidates if d not in code_set]
                    if not valid_candidates:
                        continue
                    
                    d = rng.choice(valid_candidates)
                    code_set.remove(c)
                    code_set.add(d)
                    improved = True
                    no_improve_count = 0
                    break # Restart deletion pass
                
                if improved:
                    continue
                
                # 4. Double Swap Pass (if single swap failed)
                # Try to remove 2 codewords and add 1-2 new ones
                # This is more expensive, so only try if we have time
                if time.time() < deadline - 10:
                    double_improved = False
                    # Sample a few pairs
                    c_list = list(code_set)
                    rng.shuffle(c_list)
                    num_pairs_to_try = min(10, len(c_list) // 2)
                    
                    for _ in range(num_pairs_to_try):
                        if time.time() > deadline - 10:
                            break
                        if len(c_list) < 2:
                            break
                        c1, c2 = rng.sample(c_list, 2)
                        
                        # Essential words for c1 and c2
                        n1 = neighbors[c1, :]
                        n2 = neighbors[c2, :]
                        
                        # Words covered ONLY by c1 (in current set)
                        # We need to be careful: if a word is in N(c1) and N(c2), 
                        # and cov[word]==1, it's impossible since it would be covered by both? 
                        # No, cov[word] is count. If cov[word]==1, it's covered by exactly one.
                        # If it's in N(c1), it must be c1.
                        
                        e1 = n1[cov[n1] == 1]
                        e2 = n2[cov[n2] == 1]
                        
                        # If either has no essential words, it should have been deleted.
                        # But maybe cov changed? No, cov is fixed for this iteration.
                        # So if e1 is empty, c1 is redundant? 
                        # Wait, if c1 is redundant, it would be in `redundant` list from step 2.
                        # So e1 and e2 should be non-empty.
                        
                        if len(e1) == 0 or len(e2) == 0:
                            continue

                        # We need to find a set of new codewords D such that:
                        # 1. D covers E1 and E2 (essential words of c1 and c2)
                        # 2. D does not leave any other word uncovered (words with cov==1 that are NOT in E1 or E2? No, E1/E2 are ALL words with cov==1 in N(c1)/N(c2)).
                        #    Actually, removing c1 and c2 only affects words in N(c1) U N(c2).
                        #    Words in N(c1) U N(c2) with cov==1 are exactly E1 U E2 (disjoint? Not necessarily, but if a word is in both, cov would be >=2? 
                        #    If w in N(c1) and N(c2), and cov[w]==1, contradiction. So E1 and E2 are disjoint).
                        
                        # So we need D to cover E1 U E2.
                        # Also, D should be small (ideally 1 or 2).
                        
                        # Try adding 1 codeword:
                        # d must be in intersection of N(w) for all w in E1 U E2.
                        all_ess = np.unique(np.concatenate([e1, e2]))
                        if len(all_ess) <= 4: # Heuristic limit
                            cand_set = set(neigh_sets[all_ess[0]])
                            for ew in all_ess[1:]:
                                cand_set &= neigh_sets[ew]
                                if not cand_set:
                                    break
                            
                            valid_candidates = [d for d in cand_set if d not in code_set]
                            if valid_candidates:
                                d = rng.choice(valid_candidates)
                                code_set.remove(c1)
                                code_set.remove(c2)
                                code_set.add(d)
                                double_improved = True
                                no_improve_count = 0
                                break
                        
                        # Try adding 2 codewords:
                        # This is harder. We can try to split E1 U E2 into two sets, each coverable by one codeword.
                        # Or just pick a candidate for E1 and a candidate for E2 and see if union covers all.
                        # Let's try a simpler heuristic:
                        # Pick a candidate d1 for E1 (intersection of N(e) for e in E1)
                        # Pick a candidate d2 for E2 (intersection of N(e) for e in E2)
                        # Check if {d1, d2} covers E1 U E2.
                        
                        # Get candidates for E1
                        cand1_set = set(neigh_sets[e1[0]])
                        for ew in e1[1:]:
                            cand1_set &= neigh_sets[ew]
                        
                        # Get candidates for E2
                        cand2_set = set(neigh_sets[e2[0]])
                        for ew in e2[1:]:
                            cand2_set &= neigh_sets[ew]
                        
                        # Try a few pairs
                        if cand1_set and cand2_set:
                            cand1_list = list(cand1_set)
                            cand2_list = list(cand2_set)
                            rng.shuffle(cand1_list)
                            rng.shuffle(cand2_list)
                            
                            for d1 in cand1_list[:5]:
                                if d1 in code_set: continue
                                for d2 in cand2_list[:5]:
                                    if d2 in code_set: continue
                                    if d1 == d2: continue
                                    
                                    # Check if {d1, d2} covers E1 U E2
                                    # A word w in E1 U E2 is covered if it is in N(d1) or N(d2)
                                    # We can check this by seeing if N(d1) U N(d2) contains E1 U E2
                                    # Since E1 and E2 are small, we can just check each word.
                                    covered = True
                                    for ew in all_ess:
                                        if ew not in neigh_sets[d1] and ew not in neigh_sets[d2]:
                                            covered = False
                                            break
                                    
                                    if covered:
                                        # Also need to ensure we don't break coverage of other words?
                                        # Removing c1, c2 only affects N(c1) U N(c2).
                                        # Words in N(c1) U N(c2) with cov >= 2 are still covered.
                                        # Words with cov == 1 are E1 U E2, which we just checked.
                                        # So this is valid.
                                        
                                        code_set.remove(c1)
                                        code_set.remove(c2)
                                        code_set.add(d1)
                                        code_set.add(d2)
                                        double_improved = True
                                        no_improve_count = 0
                                        break
                                if double_improved:
                                    break
                            if double_improved:
                                break
                    
                    if double_improved:
                        continue
                
                # No improvement in this iteration
                no_improve_count += 1

            return code_set

        best_code = None
        best_size = float('inf')
        
        # Run multiple restarts
        for i in range(25):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.default_rng(42 + i)
            
            # Greedy Init
            code = greedy_init(rng, deadline)
            
            # Local Search
            code = local_search(code, rng, deadline)
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = code
                
            if best_size <= 486:
                break
        
        if best_code is None:
            # Fallback to a known structure if search fails
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            best_code = set()
            import itertools
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    val = 0
                    for i, d in enumerate(word):
                        val += d * (q ** i)
                    best_code.add(val)

        final_code = []
        for w in best_code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        
        return final_code
    
    else:
        N = q ** n
        deadline = time.time() + 110.0
        powers = [q ** i for i in range(n)]
        neigh_size = 1 + n * (q - 1)
        neighbors = np.zeros((N, neigh_size), dtype=np.int32)
        
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    new_w = w + diff * powers[i]
                    neighbors[w, idx] = new_w
                    idx += 1
        
        rng = np.random.default_rng()
        uncovered = np.ones(N, dtype=np.bool_)
        code = set()
        while np.any(uncovered) and time.time() < deadline:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = rng.choice(candidates)
            code.add(w)
            uncovered[neighbors[w, :]] = False
        
        final_code = []
        for w in code:
            word = []
            temp = w
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            final_code.append(word)
        return final_code
