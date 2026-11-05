"""Combined program: vectorized greedy with smooth gains + enhanced local search and polish."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists as numpy arrays of int32 for fast operations
    neighbors_list = []
    neighbor_sets = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbor_sets.append(set(nb))

    # ---------- Phase 1: Greedy Set-Cover with smooth gain objective ----------
    # Use a smooth surrogate: weight each uncovered point by 1/(1+coverage_count)
    # This creates a smoother landscape that prefers covering under-covered regions
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()
    coverage_counts = np.zeros(N, dtype=np.int32)

    while np.any(uncovered):
        if time.time() > deadline - 40:
            break

        # Compute smooth gains: for each candidate, sum of 1/(1+coverage) over its uncovered neighbors
        gains = np.zeros(N, dtype=np.float64)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            # Weight by inverse of current coverage to prefer under-covered areas
            weights = 1.0 / (coverage_counts[neighbors_list[idx]].astype(np.float64) + 1.0)
            gains[neighbors_list[idx]] += weights

        max_gain = float(np.max(gains))
        if max_gain == 0:
            break

        # Randomize tie-breaking among top candidates for diversity
        threshold = max_gain * 0.92
        best_indices = np.where(gains >= threshold)[0]
        if len(best_indices) > 1:
            weights = gains[best_indices]
            weights = weights / weights.sum()
            best_idx = int(best_indices[np.random.choice(len(best_indices), p=weights)])
        else:
            best_idx = int(best_indices[0])

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors_list[best_idx]] = False
        coverage_counts[neighbors_list[best_idx]] += 1

    greedy_time = time.time() - start_time
    # Allocate more local search time if greedy finished early
    ls_deadline = deadline - 5
    if greedy_time < 10:
        ls_deadline = deadline - 3  # More time for local search

    # ---------- Phase 2: Local Search ----------
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1

    best_size = len(code)
    best_code = list(code)

    def is_valid(code_list):
        """Check if a list of codewords covers the entire space."""
        temp_counts = np.zeros(N, dtype=np.int32)
        for idx in code_list:
            temp_counts[neighbors_list[idx]] += 1
        return np.all(temp_counts > 0)

    def greedy_repair(uncovered_set):
        """Given a set of uncovered points, greedily add codewords to cover them."""
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)

        while temp_uncovered:
            if time.time() > ls_deadline:
                break

            gains = np.zeros(N, dtype=np.int32)
            for idx in temp_uncovered:
                gains[neighbors_list[idx]] += 1

            max_gain = int(np.max(gains))
            if max_gain == 0:
                break

            best_indices = np.where(gains == max_gain)[0]
            if len(best_indices) > 1:
                best_idx = int(random.choice(best_indices))
            else:
                best_idx = int(best_indices[0])

            added.append(best_idx)

            for w in neighbors_list[best_idx]:
                temp_uncovered.discard(w)

        return added

    def try_remove():
        """Remove all redundant codewords. Uses random order to escape local minima."""
        removed = False
        shuffled_code = list(code)
        random.shuffle(shuffled_code)
        for c in shuffled_code:
            if c not in code_set:
                continue
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                removed = True
        return removed

    def try_swap(c):
        """Try to swap codeword c with a better-positioned codeword.
        Uses a smooth scoring: prefer replacements that maximize coverage of
        currently under-covered points while minimizing over-coverage."""
        if np.min(counts[neighbors_list[c]]) > 1:
            return False

        crit_points = [i for i in neighbors_list[c] if counts[i] == 1]

        if not crit_points:
            return False

        found_swap = False
        shuffled_crit = list(crit_points)
        random.shuffle(shuffled_crit)

        for p in shuffled_crit:
            if found_swap:
                break

            cands = [x for x in neighbor_sets[p] if x not in code_set]

            best_w = -1
            best_score = -1

            for w in cands:
                valid = True
                for k in crit_points:
                    if k not in neighbor_sets[w]:
                        valid = False
                        break

                if not valid:
                    continue

                # Smooth score: reward covering under-covered points, penalize over-coverage
                nb_w = neighbors_list[w]
                under_covered = int(np.sum((counts[nb_w] == 0)))
                well_covered = int(np.sum((counts[nb_w] <= 1)))
                over_covered = int(np.sum((counts[nb_w] >= 3)))
                score = under_covered * 2 + well_covered - over_covered * 2

                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1

                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors_list[best_w]] += 1

                found_swap = True
                return True

        # Strategy 2: Relocation
        temp_counts = counts.copy()
        temp_counts[neighbors_list[c]] -= 1

        would_be_uncovered = [i for i in neighbors_list[c] if temp_counts[i] == 0]

        if not would_be_uncovered:
            return False

        all_needed = set(crit_points) | set(would_be_uncovered)

        best_w = -1
        best_score = -1

        first_point = list(all_needed)[0]
        cands = [x for x in neighbor_sets[first_point] if x not in code_set]

        for w in cands:
            valid = True
            for k in all_needed:
                if k not in neighbor_sets[w]:
                    valid = False
                    break
            if not valid:
                continue

            nb_w = neighbors_list[w]
            under_covered = int(np.sum((counts[nb_w] == 0)))
            well_covered = int(np.sum((counts[nb_w] <= 1)))
            over_covered = int(np.sum((counts[nb_w] >= 3)))
            score = under_covered * 2 + well_covered - over_covered * 2

            if score > best_score:
                best_score = score
                best_w = w

        if best_w != -1:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1

            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors_list[best_w]] += 1
            return True

        return False

    def kick():
        """Remove a codeword with minimal unique coverage and try to repair."""
        if not code:
            return False

        old_size = len(code)

        # Find codeword with minimal unique coverage (most redundant)
        min_unique = float('inf')
        c_remove = None
        for c in code:
            unique = int(np.sum(counts[neighbors_list[c]] == 1))
            if unique < min_unique:
                min_unique = unique
                c_remove = c

        if c_remove is None:
            return False

        temp_coverage = counts.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            code.remove(c_remove)
            code_set.remove(c_remove)
            counts[neighbors_list[c_remove]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - 1 + len(added)

        if new_size < old_size:
            code.remove(c_remove)
            code_set.remove(c_remove)
            counts[neighbors_list[c_remove]] -= 1
            for c in added:
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1
            return True
        else:
            return False

    def kick2():
        """Remove two random codewords and try to repair."""
        if len(code) < 2:
            return False

        old_size = len(code)

        i1 = random.randint(0, len(code) - 1)
        i2 = random.randint(0, len(code) - 1)
        if i1 == i2:
            i2 = (i2 + 1) % len(code)

        c1 = code[i1]
        c2 = code[i2]

        temp_coverage = counts.copy()
        temp_coverage[neighbors_list[c1]] -= 1
        temp_coverage[neighbors_list[c2]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]
        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - 2 + len(added)

        if new_size < old_size:
            code.remove(c1)
            code_set.remove(c1)
            counts[neighbors_list[c1]] -= 1
            code.remove(c2)
            code_set.remove(c2)
            counts[neighbors_list[c2]] -= 1
            for c in added:
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1
            return True
        else:
            return False

    def random_subset_restart():
        """Try to find a smaller valid code by taking a random subset of the current code."""
        if len(code) < best_size + 5:
            return False

        reduction = random.randint(1, min(5, len(code) - best_size))
        target_size = len(code) - reduction

        if target_size < best_size:
            target_size = best_size

        subset = random.sample(code, target_size)

        if is_valid(subset):
            code.clear()
            code_set.clear()
            counts = np.zeros(N, dtype=np.int32)
            for idx in subset:
                code.append(idx)
                code_set.add(idx)
                counts[neighbors_list[idx]] += 1
            return True
        return False

    def final_polish(code_list):
        """High-precision polish: aggressive multi-pass peeling and targeted swaps."""
        if time.time() > ls_deadline:
            return code_list

        polish_code = list(code_list)
        polish_set = set(polish_code)
        polish_counts = np.zeros(N, dtype=np.int32)
        for idx in polish_code:
            polish_counts[neighbors_list[idx]] += 1

        # Multiple passes of aggressive peeling with randomized order
        for _pass in range(5):
            if time.time() > ls_deadline:
                break
            removed_any = True
            while removed_any:
                removed_any = False
                shuffled = list(polish_code)
                random.shuffle(shuffled)
                for c in shuffled:
                    if time.time() > ls_deadline:
                        break
                    if c not in polish_set:
                        continue
                    if np.min(polish_counts[neighbors_list[c]]) > 1:
                        polish_code.remove(c)
                        polish_set.remove(c)
                        polish_counts[neighbors_list[c]] -= 1
                        removed_any = True

        # Targeted swap: for each codeword, try to find a replacement that
        # covers all critical points AND reduces total coverage redundancy
        shuffled = list(polish_code)
        random.shuffle(shuffled)
        for c in shuffled:
            if time.time() > ls_deadline:
                break
            if c not in polish_set:
                continue
            crit_points = [i for i in neighbors_list[c] if polish_counts[i] == 1]
            if not crit_points:
                continue

            first_crit = crit_points[0]
            cands = [x for x in neighbor_sets[first_crit] if x not in polish_set]
            best_w = -1
            best_score = float('inf')

            for w in cands:
                valid = True
                for k in crit_points:
                    if k not in neighbor_sets[w]:
                        valid = False
                        break
                if not valid:
                    continue

                temp_counts = polish_counts.copy()
                temp_counts[neighbors_list[c]] -= 1
                temp_counts[neighbors_list[w]] += 1

                redundant_after = int(np.sum(temp_counts > 2))
                redundant_before = int(np.sum(polish_counts > 2))
                score = redundant_after - redundant_before

                if score < best_score:
                    best_score = score
                    best_w = w

            if best_w != -1 and best_score < 0:
                polish_code.remove(c)
                polish_set.remove(c)
                polish_counts[neighbors_list[c]] -= 1
                polish_code.append(best_w)
                polish_set.add(best_w)
                polish_counts[neighbors_list[best_w]] += 1

        # Verify validity
        final_counts = np.zeros(N, dtype=np.int32)
        for idx in polish_code:
            final_counts[neighbors_list[idx]] += 1
        if np.all(final_counts > 0):
            return polish_code
        return code_list

    # ---------- Main Local Search Loop ----------
    improved = True
    stagnation_counter = 0
    while improved and time.time() < ls_deadline:
        improved = False

        # Peeling phase
        while try_remove():
            improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if time.time() > ls_deadline:
            break

        # Swap phase
        shuffled_code = list(code)
        random.shuffle(shuffled_code)

        for c in shuffled_code:
            if time.time() > ls_deadline:
                break
            if c not in code_set:
                continue
            if try_swap(c):
                improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if time.time() > ls_deadline:
            break

        # Kick phase - try more kicks when close to best
        kicks_tried = 0
        max_kicks = 50 if stagnation_counter > 3 else 30
        while kicks_tried < max_kicks and time.time() < ls_deadline:
            if kick():
                improved = True
                kicks_tried = 0
            else:
                kicks_tried += 1

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if time.time() > ls_deadline:
            break

        if len(code) > 2:
            kicks2_tried = 0
            max_kicks2 = 20
            while kicks2_tried < max_kicks2 and time.time() < ls_deadline:
                if kick2():
                    improved = True
                    kicks2_tried = 0
                else:
                    kicks2_tried += 1

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
                stagnation_counter = 0
            else:
                stagnation_counter += 1

        # LLL-inspired restart: if stagnated, try random subset
        if stagnation_counter > 4:
            if random_subset_restart():
                improved = True
                stagnation_counter = 0

    # If we found a better code, restore it
    if len(best_code) < len(code):
        code = best_code
        code_set = set(best_code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

    # Final high-precision polish pass
    if time.time() < deadline - 3:
        polished = final_polish(code)
        if len(polished) < len(code):
            code = polished

    # Convert back to list of lists
    result = [idx_to_tuple(cw) for cw in code]

    return result
# EVOLVE-BLOCK-END
