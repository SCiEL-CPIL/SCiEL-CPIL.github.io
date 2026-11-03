"""Combined approach: structural local search with critical-word swaps + vectorized redundancy."""
import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 142  # safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        total_space = 3 ** n  # 6561
        sphere_size = 1 + n * (q - 1)  # 17

        pow3 = [3 ** i for i in range(n + 1)]

        def int_to_word(idx):
            w = []
            for _ in range(n):
                w.append(idx % 3)
                idx //= 3
            return tuple(reversed(w))

        # Build sphere index matrix: shape (6561, 17)
        sphere_idx_2d = np.empty((total_space, sphere_size), dtype=np.int32)
        for idx in range(total_space):
            w = int_to_word(idx)
            sphere_idx_2d[idx, 0] = idx
            k = 1
            for pos in range(n):
                orig = w[pos]
                base = pow3[n - 1 - pos]
                for d in range(q):
                    if d != orig:
                        sphere_idx_2d[idx, k] = idx + (d - orig) * base
                        k += 1

        # Inverse mapping: for each word, the list of codewords that cover it.
        word_to_coders = np.empty((total_space, sphere_size), dtype=np.int32)
        for idx in range(total_space):
            w = int_to_word(idx)
            word_to_coders[idx, 0] = idx
            k = 1
            for pos in range(n):
                orig = w[pos]
                base = pow3[n - 1 - pos]
                for d in range(q):
                    if d != orig:
                        word_to_coders[idx, k] = idx + (d - orig) * base
                        k += 1

        # Precompute neighbor sets as Python sets for fast membership testing
        neighbor_sets = [set(sphere_idx_2d[w].tolist()) for w in range(total_space)]

        # ---------------- Phase 1: Vectorized greedy initialization ----------------
        covered = np.zeros(total_space, dtype=bool)
        gain = np.full(total_space, sphere_size, dtype=np.int32)
        codewords = []
        remaining = total_space
        code_set = set()

        greedy_deadline = start_time + 15
        while remaining > 0:
            if time.time() > greedy_deadline:
                break
            best_word = int(np.argmax(gain))
            best_gain = gain[best_word]
            if best_gain == 0:
                uncovered = np.where(~covered)[0]
                if len(uncovered) == 0:
                    break
                best_word = int(uncovered[0])

            codewords.append(best_word)
            code_set.add(best_word)

            for v in sphere_idx_2d[best_word]:
                v = int(v)
                if not covered[v]:
                    covered[v] = True
                    remaining -= 1
                    for w in sphere_idx_2d[v]:
                        gain[int(w)] -= 1

        m = len(codewords)
        sphere_idx_matrix = sphere_idx_2d[codewords]  # (m, 17)

        # Build coverage count
        cover_count = np.zeros(total_space, dtype=np.int32)
        in_code = np.zeros(total_space, dtype=bool)
        for ci in codewords:
            in_code[ci] = True
            for u in sphere_idx_2d[ci]:
                cover_count[int(u)] += 1

        # ---------------- Phase 2: Vectorized redundancy removal ----------------
        active = np.ones(m, dtype=bool)
        cleanup_deadline = start_time + 20
        for _ in range(5000):
            if time.time() > cleanup_deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            active_cover_counts = cover_count[sphere_idx_matrix[active_indices]]
            min_covers = np.min(active_cover_counts, axis=1)
            redundant_mask = min_covers >= 2
            if not np.any(redundant_mask):
                break
            redundant_local = np.where(redundant_mask)[0]
            redundant_global = active_indices[redundant_local]
            active[redundant_global] = False
            for ri in redundant_global:
                in_code[codewords[ri]] = False
                cover_count[sphere_idx_2d[codewords[ri]]] -= 1

        # ---------------- Phase 3: Critical-word structural local search ----------------
        # This is the key structural change: instead of SA-style random perturbations,
        # we target the most critical uncovered words and find swaps that fix them.
        best_code_size = int(np.sum(active))
        best_active = active.copy()
        best_cover_count = cover_count.copy()
        best_codewords = [codewords[i] for i in range(m) if active[i]]

        ls_deadline = deadline - 15
        no_improve_count = 0
        max_no_improve = 500

        while time.time() < ls_deadline and no_improve_count < max_no_improve:
            active_count = int(np.sum(active))
            if active_count < 2:
                break

            # Find words with minimum coverage (most critical)
            active_indices = np.where(active)[0]
            active_cover_counts = cover_count[sphere_idx_matrix[active_indices]]
            min_covers = np.min(active_cover_counts, axis=1)
            
            # Find codewords whose removal would create uncovered words
            critical_codeword_mask = min_covers == 1
            if not np.any(critical_codeword_mask):
                # No critical codewords, try random swap for perturbation
                if active_count > 2 and random.random() < 0.5:
                    idx1, idx2 = random.sample(range(len(active_indices)), 2)
                    c1 = codewords[active_indices[idx1]]
                    c2 = codewords[active_indices[idx2]]
                    
                    # Check if swapping maintains coverage
                    temp_count = cover_count.copy()
                    for u in sphere_idx_2d[c1]:
                        temp_count[int(u)] -= 1
                    for u in sphere_idx_2d[c2]:
                        temp_count[int(u)] += 1
                    
                    if np.min(temp_count) >= 1:
                        # Perform the swap
                        for u in sphere_idx_2d[c1]:
                            cover_count[int(u)] -= 1
                        for u in sphere_idx_2d[c2]:
                            cover_count[int(u)] += 1
                        in_code[c1] = False
                        in_code[c2] = True
                        active[active_indices[idx1]] = False
                        # Find or add c2
                        if c2 in code_set:
                            found = codewords.index(c2)
                            active[found] = True
                        else:
                            codewords.append(c2)
                            code_set.add(c2)
                            sphere_idx_matrix = np.vstack([sphere_idx_matrix, sphere_idx_2d[c2]])
                            active = np.append(active, True)
                        no_improve_count = 0
                        continue
                no_improve_count += 1
                continue

            # Pick a critical codeword to try removing
            crit_local = np.where(critical_codeword_mask)[0]
            remove_local = crit_local[random.randint(0, len(crit_local) - 1)]
            remove_global = active_indices[remove_local]
            remove_idx = codewords[remove_global]

            # Find which words become uncovered if we remove this codeword
            remove_sphere = sphere_idx_matrix[remove_global]
            newly_uncovered = []
            for u in remove_sphere:
                u = int(u)
                if cover_count[u] == 1:
                    newly_uncovered.append(u)

            if len(newly_uncovered) == 0:
                # Can remove without creating uncovered words
                active[remove_global] = False
                in_code[remove_idx] = False
                cover_count[remove_sphere] -= 1
                new_size = int(np.sum(active))
                if new_size < best_code_size:
                    best_code_size = new_size
                    best_active = active.copy()
                    best_cover_count = cover_count.copy()
                    best_codewords = [codewords[i] for i in range(len(active)) if active[i]]
                no_improve_count = 0
                continue

            # Find a candidate codeword that covers all newly_uncovered words
            # Start with coders of the first uncovered word
            candidate_set = set(neighbor_sets[newly_uncovered[0]])
            for u in newly_uncovered[1:]:
                candidate_set &= neighbor_sets[u]
                if not candidate_set:
                    break

            if not candidate_set:
                no_improve_count += 1
                continue

            # Filter out already-in-code and the removed codeword
            valid_cands = [c for c in candidate_set if c != remove_idx and not in_code[c]]
            if not valid_cands:
                # Try from existing codewords
                valid_cands = [c for c in candidate_set if c != remove_idx and in_code[c]]
                if not valid_cands:
                    no_improve_count += 1
                    continue

            # Pick the candidate that maximizes additional coverage
            best_cand = -1
            best_gain = -1
            for cand in valid_cands[:20]:  # Limit candidates for speed
                cand = int(cand)
                gain = int(np.sum(cover_count[sphere_idx_2d[cand]] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand

            if best_cand < 0:
                no_improve_count += 1
                continue

            # Perform the swap
            active[remove_global] = False
            in_code[remove_idx] = False
            cover_count[remove_sphere] -= 1

            if best_cand in code_set:
                found = codewords.index(best_cand)
                active[found] = True
                in_code[best_cand] = True
                cover_count[sphere_idx_2d[best_cand]] += 1
            else:
                codewords.append(best_cand)
                code_set.add(best_cand)
                sphere_idx_matrix = np.vstack([sphere_idx_matrix, sphere_idx_2d[best_cand]])
                active = np.append(active, True)
                in_code[best_cand] = True
                cover_count[sphere_idx_2d[best_cand]] += 1

            # Check if we improved
            new_size = int(np.sum(active))
            uncovered_after = int(np.sum(cover_count == 0))
            if uncovered_after == 0 and new_size < best_code_size:
                best_code_size = new_size
                best_active = active.copy()
                best_cover_count = cover_count.copy()
                best_codewords = [codewords[i] for i in range(len(active)) if active[i]]
                no_improve_count = 0
            else:
                # Revert if not valid or not improving
                # Undo the addition
                if best_cand in code_set:
                    found = codewords.index(best_cand)
                    if found != remove_global:
                        active[found] = False
                        in_code[best_cand] = False
                        cover_count[sphere_idx_2d[best_cand]] -= 1
                # Undo the removal
                active[remove_global] = True
                in_code[remove_idx] = True
                cover_count[remove_sphere] += 1
                no_improve_count += 1

        # Restore best state
        active = best_active.copy()
        cover_count = best_cover_count.copy()

        # ---------------- Phase 4: Final aggressive redundancy cleanup ----------------
        final_deadline = deadline - 3
        for _ in range(5000):
            if time.time() > final_deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            active_cover_counts = cover_count[sphere_idx_matrix[active_indices]]
            min_covers = np.min(active_cover_counts, axis=1)
            redundant_mask = min_covers >= 2
            if not np.any(redundant_mask):
                break
            redundant_local = np.where(redundant_mask)[0]
            redundant_global = active_indices[redundant_local]
            active[redundant_global] = False
            for ri in redundant_global:
                cover_count[sphere_idx_2d[codewords[ri]]] -= 1

        final_indices = [codewords[i] for i in range(len(active)) if active[i]]

        # Verify coverage
        verify_cover = np.zeros(total_space, dtype=bool)
        for idx in final_indices:
            verify_cover[sphere_idx_2d[idx]] = True
        if np.all(verify_cover):
            return [list(int_to_word(idx)) for idx in final_indices]

        # Fallback: return all words (valid but large)
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    else:
        # General case: tetracode-based construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
