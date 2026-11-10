import numpy as np
import time
import sys

def construct(q, n, radius):
    deadline = time.time() + 115.0  # Safety margin for 120s limit

    if q == 3 and n == 8 and radius == 1:
        N = 3 ** 8  # 6561
        powers = [3 ** i for i in range(n)]

        # Precompute neighbors for all words (1 + 2*n neighbors)
        # For q=3, n=8, radius=1, each word has 1 + 2*8 = 17 neighbors
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                diff1 = ((d + 1) % q) - d
                neighbors[w, idx] = w + diff1 * powers[i]
                idx += 1
                diff2 = ((d + 2) % q) - d
                neighbors[w, idx] = w + diff2 * powers[i]
                idx += 1

        def greedy_construct(rng, alpha=1.0):
            """Greedy set cover with randomized tie-breaking parameter alpha."""
            uncovered = np.ones(N, dtype=np.bool_)
            code = []
            while np.any(uncovered) and time.time() < deadline - 5:
                gains = uncovered[neighbors].sum(axis=1).astype(np.float64)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                # Select candidates with gain close to max
                threshold = max_gain * alpha
                candidates = np.where(gains >= threshold)[0]
                
                if alpha < 1.0 and len(candidates) > 1:
                    # Soft randomization: weight by gain
                    weights = gains[candidates]
                    w = rng.choice(candidates, p=weights / weights.sum())
                else:
                    w = rng.choice(candidates)
                
                code.append(w)
                uncovered[neighbors[w, :]] = False
            return code

        def reduce_code(code_set):
            """Remove redundant codewords via iterative coverage counting."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            changed = True
            while changed and time.time() < deadline - 3:
                changed = False
                # Iterate over a copy since we modify the set
                for c in list(code_set):
                    # Check if removing c leaves any word uncovered
                    # i.e., is there any neighbor of c with coverage == 1?
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        code_set.remove(c)
                        coverage[neighbors[c, :]] -= 1
                        changed = True
            return code_set

        def add_remove_search(code_set, max_iter=200):
            """
            Add-Remove local search targeting singly-covered words.
            1. Find words covered by exactly 1 codeword (bottlenecks).
            2. Add a codeword that covers many such bottlenecks.
            3. Try to remove any redundant codeword.
            4. If we remove a codeword different from the added one, we reduced size by 1.
            """
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)
            original_size = len(code_list)
            
            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break

                # Find singly-covered words (bottlenecks)
                single_mask = (coverage == 1)
                if not np.any(single_mask):
                    break  # No bottlenecks, can't improve via add-remove

                # For each codeword not in code, compute how many singly-covered words it covers
                # Vectorized: for each word w, count overlap with singly-covered set
                # gains[w] = sum of coverage==1 over neighbors[w]
                single_covered = np.where(single_mask)[0]
                
                # Compute gain for all potential new codewords
                # A new codeword w covers a singly-covered word s if s is in neighbors[w]
                # This is equivalent to: for each singly-covered s, add 1 to all neighbors of s
                # Then the gain for word w is the sum of these contributions at position w
                # More efficiently: gain[w] = number of singly-covered words in neighbors[w]
                
                # Build a count array: for each word, how many singly-covered words are its neighbors
                gain = np.zeros(N, dtype=np.int32)
                for s in single_covered:
                    gain[neighbors[s, :]] += 1

                # Filter out words already in code
                for c in code_list:
                    gain[c] = 0

                max_gain = np.max(gain)
                if max_gain == 0:
                    break

                # Select among top candidates
                threshold = max_gain * 0.8
                candidates = np.where(gain >= threshold)[0]
                if len(candidates) == 0:
                    candidates = np.where(gain == max_gain)[0]
                
                new_c = rng.choice(candidates)

                # Add new_c
                code_set.add(new_c)
                coverage[neighbors[new_c, :]] += 1
                code_list.append(new_c)
                
                # Try to remove redundant codewords
                removable_candidates = []
                for c in code_list:
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        removable_candidates.append(c)
                
                if removable_candidates:
                    old_removable = [c for c in removable_candidates if c != new_c]
                    if old_removable:
                        c_to_remove = rng.choice(old_removable)
                    else:
                        c_to_remove = new_c
                        
                    code_set.remove(c_to_remove)
                    coverage[neighbors[c_to_remove, :]] -= 1
                    code_list.remove(c_to_remove)
                    
                    if c_to_remove == new_c:
                        # No net change, continue
                        pass
                else:
                    # Nothing removable. Revert the addition.
                    code_set.remove(new_c)
                    coverage[neighbors[new_c, :]] -= 1
                    code_list.remove(new_c)
                    
            return code_set

        def try_swaps(code_set, max_iter=50):
            """Try 1-for-1 swaps to reduce code size, prioritizing codewords with most essential words."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)

            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Prioritize codewords with more essential (singly-covered) words
                # as they are harder to swap out but also more impactful
                essential_counts = np.array([
                    np.sum(coverage[neighbors[c, :]] == 1) for c in code_list
                ])
                
                # Pick from codewords with highest essential count (hardest to remove, but if we can swap, big gain)
                # Actually, pick randomly but weight toward those with fewer essentials (easier to swap)
                # Let's just pick randomly for diversity
                idx = rng.integers(0, len(code_list))
                c = code_list[idx]
                c_neigh = neighbors[c, :]

                # Words only covered by c
                essential = c_neigh[coverage[c_neigh] == 1]

                if len(essential) == 0:
                    # c is redundant
                    code_list.pop(idx)
                    coverage[neighbors[c, :]] -= 1
                    continue

                # Try to find a single replacement d that covers all essential words
                if len(essential) == 1:
                    candidates = neighbors[essential[0], :]
                else:
                    cand_set = set(neighbors[essential[0], :].tolist())
                    for ew in essential[1:]:
                        cand_set &= set(neighbors[ew, :].tolist())
                    if not cand_set:
                        continue
                    candidates = np.array(list(cand_set), dtype=np.int32)

                # Filter: d not already in code
                for d in candidates:
                    if d in code_set:
                        continue
                    # Check: after removing c and adding d, is everything covered?
                    # Only need to check words that were singly covered by c
                    d_neigh_set = set(neighbors[d, :].tolist())
                    bad = False
                    for w in essential:
                        if w not in d_neigh_set:
                            bad = True
                            break
                    if bad:
                        continue
                    # Accept the swap
                    code_set.remove(c)
                    code_set.add(d)
                    coverage[neighbors[c, :]] -= 1
                    coverage[neighbors[d, :]] += 1
                    code_list = list(code_set)
                    break

            return code_set

        def csp_polish(code_set, max_iter=300):
            """
            CSP-inspired polish: treat singly-covered words as constraints.
            For each bottleneck word, its only covering codeword is 'locked'.
            We try to find a replacement codeword that covers the bottleneck
            and enables removal of the locked codeword, using a backtracking-like
            search over candidate replacements ordered by their coverage gain.
            """
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)

            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break

                # Find all singly-covered words
                single_mask = (coverage == 1)
                single_words = np.where(single_mask)[0]
                if len(single_words) == 0:
                    break

                # For each singly-covered word, identify its unique covering codeword
                # and compute how many other singly-covered words that codeword covers
                # We want to find a codeword c such that replacing it with d
                # covers all of c's essential words AND enables removal of c.

                # Group essential words by their covering codeword
                essential_by_cw = {}
                for s in single_words:
                    # Find which codeword covers s
                    for c in code_list:
                        if coverage[neighbors[c, :]][neighbors[c, :] == s].sum() > 0:
                            if c not in essential_by_cw:
                                essential_by_cw[c] = []
                            essential_by_cw[c].append(s)
                            break

                if not essential_by_cw:
                    break

                # Sort codewords by number of essential words (most constrained first)
                sorted_cws = sorted(essential_by_cw.keys(), key=lambda c: len(essential_by_cw[c]), reverse=True)

                found_swap = False
                for c in sorted_cws:
                    if time.time() > deadline - 5:
                        break
                    if c not in code_set:
                        continue

                    essential_words = essential_by_cw[c]
                    if not essential_words:
                        continue

                    # Find candidate replacements: intersection of neighbors of all essential words
                    if len(essential_words) == 1:
                        candidates = neighbors[essential_words[0], :]
                    else:
                        cand_set = set(neighbors[essential_words[0], :].tolist())
                        for ew in essential_words[1:]:
                            cand_set &= set(neighbors[ew, :].tolist())
                        if not cand_set:
                            continue
                        candidates = np.array(list(cand_set), dtype=np.int32)

                    # Filter out existing codewords
                    valid_candidates = [d for d in candidates if d not in code_set]
                    if not valid_candidates:
                        continue

                    # Score candidates: how many total words they cover that are currently
                    # covered by exactly 2 codewords (these become more critical after swap)
                    # and how many words they cover that are currently uncovered (should be 0)
                    scored = []
                    for d in valid_candidates:
                        d_neigh = neighbors[d, :]
                        # Count how many of d's neighbors are currently covered by exactly 2
                        double_covered = np.sum(coverage[d_neigh] == 2)
                        # Count how many of d's neighbors are currently covered by exactly 1
                        # (these are other bottlenecks that d might help with)
                        single_covered_by_d = np.sum(coverage[d_neigh] == 1)
                        # Prefer candidates that cover more double-covered words (safer swap)
                        # and fewer single-covered words (less risk)
                        score = double_covered - single_covered * 0.5
                        scored.append((score, d))

                    if not scored:
                        continue

                    # Sort by score descending
                    scored.sort(key=lambda x: x[0], reverse=True)

                    # Try top candidates
                    for score, d in scored[:5]:
                        # Perform the swap and check if it enables removal of c
                        code_set.remove(c)
                        code_set.add(d)
                        coverage[neighbors[c, :]] -= 1
                        coverage[neighbors[d, :]] += 1

                        # Check if c is now removable (i.e., no word is singly covered by c)
                        # Actually, we already removed c, so check if the swap is valid
                        # and if any codeword can now be removed.
                        # For now, accept the swap if it doesn't create new bottlenecks
                        # that are worse than before.
                        
                        # Check coverage validity
                        valid = True
                        for w in range(N):
                            if coverage[w] == 0:
                                valid = False
                                break
                        if not valid:
                            # Revert
                            code_set.remove(d)
                            code_set.add(c)
                            coverage[neighbors[d, :]] -= 1
                            coverage[neighbors[c, :]] += 1
                            continue

                        # Try to remove any now-redundant codeword
                        removable = []
                        for cc in code_set:
                            if np.min(coverage[neighbors[cc, :]]) > 1:
                                removable.append(cc)
                        
                        if removable:
                            c_to_remove = rng.choice(removable)
                            code_set.remove(c_to_remove)
                            coverage[neighbors[c_to_remove, :]] -= 1
                            code_list = list(code_set)
                            found_swap = True
                            break
                        else:
                            # No removal possible, revert
                            code_set.remove(d)
                            code_set.add(c)
                            coverage[neighbors[d, :]] -= 1
                            coverage[neighbors[c, :]] += 1

                    if found_swap:
                        break

                if not found_swap:
                    # No improvement in this iteration, try a random perturbation
                    if len(code_list) > 0 and rng.random() < 0.3:
                        # Remove a random codeword and re-add a better one
                        idx = rng.integers(0, len(code_list))
                        c = code_list[idx]
                        code_set.remove(c)
                        coverage[neighbors[c, :]] -= 1
                        code_list.pop(idx)
                        
                        # Find best replacement
                        gains = np.zeros(N, dtype=np.int32)
                        for s in np.where(coverage == 1)[0]:
                            gains[neighbors[s, :]] += 1
                        for cc in code_list:
                            gains[cc] = 0
                        
                        max_gain = np.max(gains)
                        if max_gain > 0:
                            candidates = np.where(gains == max_gain)[0]
                            d = rng.choice(candidates)
                            code_set.add(d)
                            coverage[neighbors[d, :]] += 1
                            code_list.append(d)
                        else:
                            # No good replacement, put c back
                            code_set.add(c)
                            coverage[neighbors[c, :]] += 1
                            code_list.append(c)
                    else:
                        break

            return code_set

        # Run multiple greedy constructions with different parameters
        best_code = None
        best_size = float('inf')

        configs = [
            (42, 1.0), (123, 1.0), (456, 0.95), (789, 0.9),
            (101, 1.0), (202, 0.85), (303, 0.95), (404, 1.0),
            (505, 0.9), (606, 0.8), (707, 1.0), (808, 0.95),
            (909, 0.85), (111, 1.0), (222, 0.9), (333, 0.8),
            (444, 1.0), (555, 0.9), (666, 1.0), (777, 0.85),
            (888, 0.9), (999, 1.0), (1111, 0.95), (2222, 0.85)
        ]

        for seed, alpha in configs:
            if time.time() > deadline - 10:
                break

            rng = np.random.default_rng(seed)
            code_list = greedy_construct(rng, alpha)
            code_set = set(code_list)

            # Reduction phase
            code_set = reduce_code(code_set)
            
            # Swap phase
            code_set = try_swaps(code_set, max_iter=30)
            
            # Add-Remove phase
            code_set = add_remove_search(code_set, max_iter=150)
            
            # CSP polish phase (new)
            code_set = csp_polish(code_set, max_iter=100)
            
            # Second swap phase after add-remove
            code_set = try_swaps(code_set, max_iter=30)
            
            # Final reduction
            code_set = reduce_code(code_set)

            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set

            if best_size <= 486:
                break

        if best_code is None:
            # Fallback
            rng = np.random.default_rng(0)
            code_list = greedy_construct(rng, 1.0)
            best_code = set(reduce_code(code_list))

        # Final polish on the best code found
        if best_code:
            best_code = try_swaps(best_code, max_iter=40)
            best_code = add_remove_search(best_code, max_iter=100)
            best_code = csp_polish(best_code, max_iter=150)
            best_code = try_swaps(best_code, max_iter=40)
            best_code = reduce_code(best_code)

        # Convert to list of digit lists
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
        # Generic fallback for other (q, n, radius)
        N = q ** n
        powers = [q ** i for i in range(n)]
        
        # Correct generic neighbor generation
        neigh_count = 1 + n * (q - 1)
        neighbors = np.empty((N, neigh_count), dtype=np.int32)
        for w in range(N):
            neighbors[w, 0] = w
            idx = 1
            for i in range(n):
                d = (w // powers[i]) % q
                for delta in range(1, q):
                    new_d = (d + delta) % q
                    diff = new_d - d
                    neighbors[w, idx] = w + diff * powers[i]
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
