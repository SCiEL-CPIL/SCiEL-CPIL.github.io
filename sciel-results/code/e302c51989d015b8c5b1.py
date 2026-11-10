import numpy as np
import time

def construct(q, n, radius):
    deadline = time.time() + 115.0  # 150s wall, safety margin

    if q == 3 and n == 8 and radius == 1:
        N = 3 ** 8  # 6561
        powers = [3 ** i for i in range(n)]

        # Precompute neighbors for all words (17 neighbors: self + 16 at distance 1)
        neighbors = np.empty((N, 17), dtype=np.int32)
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

        # Build reverse neighbor map: for each word, which codeword positions cover it
        # Actually, let's precompute: for each word w, the list of words at distance <= 1
        # That's just neighbors[w]. We need the reverse: for each codeword c, which words does it cover.
        # That's neighbors[c]. Good.

        def greedy_construct(rng, alpha=1.0):
            """Greedy set cover with randomized tie-breaking."""
            uncovered = np.ones(N, dtype=np.bool_)
            code = []
            while np.any(uncovered) and time.time() < deadline - 5:
                gains = uncovered[neighbors].sum(axis=1).astype(np.float64)
                max_gain = np.max(gains)
                if max_gain == 0:
                    break
                candidates = np.where(gains >= max_gain * alpha)[0]
                if alpha < 1.0 and len(candidates) > 1:
                    w = rng.choice(candidates, p=gains[candidates] / gains[candidates].sum())
                else:
                    w = rng.choice(candidates)
                code.append(w)
                uncovered[neighbors[w, :]] = False
            return code

        def reduce_code(code_list):
            """Remove redundant codewords via iterative coverage counting."""
            code_set = set(code_list)
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            changed = True
            while changed and time.time() < deadline - 3:
                changed = False
                for c in list(code_set):
                    if np.min(coverage[neighbors[c, :]]) > 1:
                        code_set.remove(c)
                        coverage[neighbors[c, :]] -= 1
                        changed = True
            return code_set

        def try_swaps(code_set, max_iter=50):
            """Try 1-for-1 and 2-for-2 swaps to reduce code size."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)

            for _ in range(max_iter):
                if time.time() > deadline - 3:
                    break
                
                # Try 1-for-1 swap
                idx = rng.integers(0, len(code_list))
                c = code_list[idx]
                c_neigh = neighbors[c, :]

                essential = c_neigh[coverage[c_neigh] == 1]

                if len(essential) == 0:
                    code_list.pop(idx)
                    coverage[neighbors[c, :]] -= 1
                    continue

                if len(essential) == 1:
                    candidates = neighbors[essential[0], :]
                else:
                    cand_set = set(neighbors[essential[0], :].tolist())
                    for ew in essential[1:]:
                        cand_set &= set(neighbors[ew, :].tolist())
                    if not cand_set:
                        # Try 2-for-2 swap
                        if len(code_list) >= 2:
                            idx2 = rng.integers(0, len(code_list))
                            if idx2 == idx:
                                continue
                            c2 = code_list[idx2]
                            c2_neigh = neighbors[c2, :]
                            
                            # Find words that become uncovered if we remove both
                            temp_cov = coverage.copy()
                            temp_cov[c_neigh] -= 1
                            temp_cov[c2_neigh] -= 1
                            
                            uncovered_after = np.where(temp_cov == 0)[0]
                            if len(uncovered_after) == 0:
                                # Can remove both
                                code_set.remove(c)
                                code_set.remove(c2)
                                coverage[c_neigh] -= 1
                                coverage[c2_neigh] -= 1
                                code_list = list(code_set)
                                continue
                            
                            # Find candidates that cover all uncovered words
                            if len(uncovered_after) <= 2:
                                cand_set2 = set()
                                for uw in uncovered_after:
                                    cand_set2.update(neighbors[uw, :].tolist())
                                
                                for d in cand_set2:
                                    if d in code_set:
                                        continue
                                    # Check if adding d covers all uncovered
                                    d_neigh = neighbors[d, :]
                                    if all(w in d_neigh for w in uncovered_after):
                                        code_set.remove(c)
                                        code_set.remove(c2)
                                        code_set.add(d)
                                        coverage[c_neigh] -= 1
                                        coverage[c2_neigh] -= 1
                                        coverage[d_neigh] += 1
                                        code_list = list(code_set)
                                        break
                        continue
                    candidates = np.array(list(cand_set), dtype=np.int32)

                for d in candidates:
                    if d in code_set:
                        continue
                    bad = False
                    for w in c_neigh:
                        if coverage[w] == 1 and w not in neighbors[d, :]:
                            bad = True
                            break
                    if bad:
                        continue
                    code_set.remove(c)
                    code_set.add(d)
                    coverage[neighbors[c, :]] -= 1
                    coverage[neighbors[d, :]] += 1
                    code_list = list(code_set)
                    break

            return code_set

        def add_reduce_search(code_set, max_iter=100):
            """Add-Reduce local search: temporarily add a codeword, then reduce."""
            if not code_set:
                return code_set

            coverage = np.zeros(N, dtype=np.int32)
            for c in code_set:
                coverage[neighbors[c, :]] += 1

            rng = np.random.default_rng()
            code_list = list(code_set)

            for _ in range(max_iter):
                if time.time() > deadline - 3:
                    break

                # Try to find a codeword to remove
                removable = [c for c in code_list if np.min(coverage[neighbors[c, :]]) > 1]
                if not removable:
                    # No removable codeword; try adding one to enable removal
                    if np.any(coverage == 0):
                        # There's an uncovered word (shouldn't happen if valid, but just in case)
                        uncovered_words = np.where(coverage == 0)[0]
                        uw = rng.choice(uncovered_words)
                        candidates = neighbors[uw, :]
                        # Pick the candidate that covers the most currently uncovered words
                        gains = np.zeros(N, dtype=np.int32)
                        for c in code_list:
                            gains[neighbors[c, :]] += 1
                        # Actually, we want to find which candidate covers the most words that are barely covered
                        # Let's just pick randomly from candidates for now
                        new_c = rng.choice(candidates)
                        if new_c not in code_set:
                            code_set.add(new_c)
                            coverage[neighbors[new_c, :]] += 1
                            code_list.append(new_c)
                    else:
                        # All covered, try to add a codeword that creates maximum redundancy
                        # Find words with highest coverage (most redundant)
                        high_cov = np.where(coverage >= 3)[0]
                        if len(high_cov) > 0:
                            # Pick a word from high coverage area
                            uw = rng.choice(high_cov)
                            candidates = neighbors[uw, :]
                            # Pick candidate that is not already in code
                            available = [c for c in candidates if c not in code_set]
                            if available:
                                new_c = rng.choice(available)
                                code_set.add(new_c)
                                coverage[neighbors[new_c, :]] += 1
                                code_list.append(new_c)
                        else:
                            # No high coverage, try random
                            new_c = rng.integers(0, N)
                            if new_c not in code_set:
                                code_set.add(new_c)
                                coverage[neighbors[new_c, :]] += 1
                                code_list.append(new_c)
                    continue

                # Remove a random removable codeword
                c = rng.choice(removable)
                code_set.remove(c)
                coverage[neighbors[c, :]] -= 1
                code_list.remove(c)

            return code_set

        # Run multiple greedy constructions with different parameters
        best_code = None
        best_size = float('inf')

        configs = [
            (42, 1.0), (123, 1.0), (456, 0.95), (789, 0.9),
            (101, 1.0), (202, 0.85), (303, 0.95), (404, 1.0),
            (505, 0.9), (606, 0.8), (707, 1.0), (808, 0.95),
            (909, 0.85), (111, 1.0), (222, 0.9), (333, 0.8),
            (444, 0.75), (555, 0.9), (666, 0.85), (777, 0.95),
        ]

        for seed, alpha in configs:
            if time.time() > deadline - 20:
                break

            rng = np.random.default_rng(seed)
            code_list = greedy_construct(rng, alpha)
            code_set = set(code_list)

            # Reduction phase
            code_set = reduce_code(list(code_set))

            # Swap phase
            code_set = try_swaps(code_set, max_iter=50)

            # Add-reduce phase
            code_set = add_reduce_search(code_set, max_iter=60)

            # Final reduction
            code_set = reduce_code(list(code_set))

            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = code_set

            if best_size <= 486:
                break

        # Final reduction on best code with more aggressive search
        if best_code is not None:
            # Multiple rounds of refinement
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                best_code = reduce_code(list(best_code))
                best_code = try_swaps(best_code, max_iter=40)
                best_code = add_reduce_search(best_code, max_iter=40)
                best_code = reduce_code(list(best_code))

        if best_code is None:
            rng = np.random.default_rng(0)
            code_list = greedy_construct(rng, 1.0)
            best_code = set(reduce_code(code_list))

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
        neighbors = np.empty((N, 1 + 2 * n), dtype=np.int32)
        powers = [q ** i for i in range(n)]
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
