"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

        # Inverse mapping: for each word, the list of codewords that cover it (its closed neighborhood).
        # Every word has exactly sphere_size potential coders, so criticality is uniform;
        # we keep the structure for the criticality-driven greedy ordering.
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

        # ---------------- Phase 1: criticality-driven greedy ----------------
        uncovered = np.ones(total_space, dtype=bool)
        cover_count = np.zeros(total_space, dtype=np.int32)
        code_indices = []
        code_set = set()

        greedy_deadline = start_time + 28
        while True:
            if not uncovered.any():
                break
            if time.time() > greedy_deadline:
                break

            # Pick an uncovered word with minimum criticality (fewest coders).
            unc_idx = np.where(uncovered)[0]
            crit = np.array([len(word_to_coders[w]) for w in unc_idx], dtype=np.int32)
            min_crit = int(crit.min())
            target_word = int(unc_idx[int(np.argmin(crit))])

            # Among codewords covering target_word, pick the one maximizing new coverage.
            coders = word_to_coders[target_word]
            best_c = -1
            best_gain = 0
            for c in coders:
                c = int(c)
                if c in code_set:
                    continue
                gain = int(np.sum(uncovered[sphere_idx_2d[c]]))
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c < 0:
                # Fallback: any codeword covering target_word
                for c in coders:
                    best_c = int(c)
                    break
            if best_c < 0 or best_gain <= 0:
                break

            code_indices.append(best_c)
            code_set.add(best_c)
            cover_count[sphere_idx_2d[best_c]] += 1
            uncovered[sphere_idx_2d[best_c]] = False

        m = len(code_indices)
        sphere_idx_matrix = sphere_idx_2d[code_indices]  # (m, 17)
        active = np.ones(m, dtype=bool)

        # ---------------- Phase 2: vectorized redundancy removal ----------------
        cleanup_deadline = start_time + 40
        for _ in range(2000):
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
                cover_count[sphere_idx_2d[code_indices[ri]]] -= 1

        # ---------------- Phase 3: 2-opt local search (remove one, add one) ----------------
        best_code_size = int(np.sum(active))
        best_active = active.copy()
        best_cover_count = cover_count.copy()

        T = 25.0
        cooling_rate = 0.999
        T_min = 0.05
        sa_deadline = deadline - 12

        while time.time() < sa_deadline and T > T_min:
            active_count = int(np.sum(active))
            if active_count < 2:
                break

            active_indices = np.where(active)[0]
            remove_pos = int(active_indices[random.randint(0, len(active_indices) - 1)])
            remove_idx = code_indices[remove_pos]

            # Remove it
            active[remove_pos] = False
            remove_sphere = sphere_idx_matrix[remove_pos]
            cover_count[remove_sphere] -= 1
            newly_uncovered = int(np.sum(cover_count == 0))

            if newly_uncovered > 0:
                unc_words = np.where(cover_count == 0)[0]
                best_gain = 0
                best_cand = -1

                # Sample up to 24 uncovered words; for each, examine its coders' gain.
                num_trials = min(24, len(unc_words))
                trial_set = random.sample(range(len(unc_words)), num_trials) if num_trials < len(unc_words) else range(len(unc_words))
                seen = set()
                for ti in trial_set:
                    uw = int(unc_words[ti])
                    for c in word_to_coders[uw]:
                        c = int(c)
                        if c in seen:
                            continue
                        seen.add(c)
                        gain = int(np.sum(cover_count[sphere_idx_2d[c]] == 0))
                        if gain > best_gain:
                            best_gain = gain
                            best_cand = c

                if best_cand >= 0 and best_gain > 0 and best_cand != remove_idx:
                    # Add the candidate
                    if best_cand in code_set:
                        found = code_indices.index(best_cand)
                        active[found] = True
                        cover_count[sphere_idx_2d[best_cand]] += 1
                    else:
                        code_indices.append(best_cand)
                        code_set.add(best_cand)
                        sphere_idx_matrix = np.vstack([sphere_idx_matrix, sphere_idx_2d[best_cand]])
                        active = np.append(active, True)
                        cover_count[sphere_idx_2d[best_cand]] += 1

                    new_uncovered_after = int(np.sum(cover_count == 0))
                    if new_uncovered_after == 0:
                        new_size = int(np.sum(active))
                        if new_size < best_code_size:
                            best_code_size = new_size
                            best_active = active.copy()
                            best_cover_count = cover_count.copy()
                    else:
                        delta_cost = new_uncovered_after - newly_uncovered
                        if not (delta_cost > 0 and random.random() < np.exp(-delta_cost / T)):
                            # Reject: undo the addition
                            if best_cand in code_indices:
                                found = code_indices.index(best_cand)
                                if found != remove_pos:
                                    active[found] = False
                                    cover_count[sphere_idx_2d[best_cand]] -= 1
                else:
                    # No useful candidate: revert removal
                    active[remove_pos] = True
                    cover_count[remove_sphere] += 1
            else:
                new_size = int(np.sum(active))
                if new_size < best_code_size:
                    best_code_size = new_size
                    best_active = active.copy()
                    best_cover_count = cover_count.copy()

            T *= cooling_rate

        # Restore best state
        active = best_active.copy()
        cover_count = best_cover_count.copy()

        # ---------------- Phase 4: final aggressive redundancy cleanup ----------------
        final_deadline = deadline - 3
        for _ in range(1000):
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
                cover_count[sphere_idx_2d[code_indices[ri]]] -= 1

        final_indices = [code_indices[i] for i in range(len(active)) if active[i]]

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
# EVOLVE-BLOCK-END
