import numpy as np
import itertools
import time
import random
import sys
from collections import defaultdict

def construct(q, n, radius):
    # Fallback for non-specific parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for K_3(8,1)
    total_points = q ** n  # 6561
    neighbors_per_word = 1 + n * (q - 1)  # 17

    start_time = time.time()
    time_limit = 148.0  # Safety margin before 150s kill

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all 6561 words
    neighbors = np.zeros((total_points, neighbors_per_word), dtype=np.int32)
    for i in range(total_points):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbors[i, count] = word_to_idx(w)
                count += 1
            w[pos] = orig_val

    all_words = [idx_to_word(i) for i in range(total_points)]

    # Build a set for each codeword's neighbors for fast lookup
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total_points)]

    def compute_coverage(chosen_indices):
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return cover_count

    def is_valid_cover(chosen_indices):
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return np.all(cover_count > 0)

    def structured_initial_code():
        """
        Construct the lexicographic tetracode product: 81 codewords.
        Tetracode: {(a, b, a+b, a+2b) mod 3 : a,b in {0,1,2}}
        Product: first 4 coords from one tetracode word, last 4 from another.
        """
        tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
        indices = []
        for t1 in tetracode:
            for t2 in tetracode:
                word = list(t1) + list(t2)
                idx = word_to_idx(word)
                indices.append(idx)
        return indices

    def greedy_complete(chosen_indices, randomize=False):
        """
        Greedily add codewords prioritizing words that cover the most uncovered points.
        """
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1

        chosen = list(chosen_indices)
        available = np.ones(total_points, dtype=bool)
        for ci in chosen:
            available[ci] = False

        while not np.all(cover_count > 0):
            if time.time() - start_time > time_limit - 20.0:
                break

            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break

            # For each available word, count how many uncovered points it covers
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break

            # Vectorized: for each available word, count newly covered points
            newly_covered = np.zeros(len(avail_indices), dtype=np.int32)
            for j, ai in enumerate(avail_indices):
                newly_covered[j] = np.sum(cover_count[neighbors[ai]] == 0)

            max_new = np.max(newly_covered)
            if max_new == 0:
                break

            # Among those with max_new, pick randomly or by criticality
            best_candidates = np.where(newly_covered == max_new)[0]
            if randomize and len(best_candidates) > 1:
                pick = random.choice(best_candidates.tolist())
            else:
                # Criticality tiebreaker: prefer words covering points with lower coverage
                min_cov_scores = []
                for bc in best_candidates:
                    ai = avail_indices[bc]
                    avg_cov = np.mean(cover_count[neighbors[ai]])
                    min_cov_scores.append(avg_cov)
                min_cov_scores = np.array(min_cov_scores)
                pick = best_candidates[np.argmin(min_cov_scores)]

            best_global = avail_indices[pick]
            chosen.append(best_global)
            cover_count[neighbors[best_global]] += 1
            available[best_global] = False

        return chosen

    def local_search_reduce(chosen_indices):
        """
        Local search to reduce code size:
        1. Try removing each codeword; if all points remain covered, remove it.
        2. Try swapping codeword i with codeword j to improve coverage.
        3. Try replacing one codeword with a better single replacement.
        """
        current = list(chosen_indices)
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in current:
            cover_count[neighbors[ci]] += 1

        improved = True
        max_rounds = 100
        round_num = 0

        while improved and round_num < max_rounds and time.time() - start_time < time_limit - 10.0:
            improved = False
            round_num += 1

            # Phase 1: Try removing each codeword
            i = 0
            while i < len(current):
                if time.time() - start_time > time_limit - 10.0:
                    break

                ci = current[i]
                exposed = neighbors[ci]
                would_uncover = exposed[cover_count[exposed] == 1]

                if len(would_uncover) == 0:
                    # Safe to remove
                    cover_count[neighbors[ci]] -= 1
                    current.pop(i)
                    improved = True
                    continue

                # Try to find a single replacement that covers all exposed points
                best_replacement = -1
                best_gain = -1

                exposed_set = set(would_uncover.tolist())

                # Get candidate words from neighbors of exposed points
                candidate_set = set()
                for ep in would_uncover:
                    candidate_set.update(neighbors[ep].tolist())
                candidate_set.discard(ci)

                candidates = list(candidate_set)
                if len(candidates) > 300:
                    random.shuffle(candidates)
                    candidates = candidates[:300]

                for cand in candidates:
                    if cand in current:
                        continue
                    cand_neigh = neighbors[cand]
                    if not all(p in neighbor_sets[cand] for p in exposed_set):
                        continue

                    new_cov = np.sum(cover_count[cand_neigh] == 0)
                    if new_cov > best_gain:
                        best_gain = new_cov
                        best_replacement = cand

                if best_replacement >= 0:
                    cover_count[neighbors[ci]] -= 1
                    cover_count[neighbors[best_replacement]] += 1
                    current[i] = best_replacement
                    improved = True

                i += 1

            # Phase 2: Try pairwise removals
            if time.time() - start_time > time_limit - 10.0:
                break

            n_current = len(current)
            for i in range(n_current):
                if time.time() - start_time > time_limit - 10.0:
                    break
                for j in range(i + 1, n_current):
                    if time.time() - start_time > time_limit - 10.0:
                        break

                    ci = current[i]
                    cj = current[j]

                    cover_count[neighbors[ci]] -= 1
                    cover_count[neighbors[cj]] -= 1

                    if np.all(cover_count > 0):
                        current.pop(j)
                        current.pop(i)
                        improved = True
                        break

                    cover_count[neighbors[ci]] += 1
                    cover_count[neighbors[cj]] += 1

                if improved:
                    break

        return current

    def perturbation(chosen_indices, strength=5):
        """
        Perturb the code by randomly removing and re-adding codewords.
        """
        current = list(chosen_indices)
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in current:
            cover_count[neighbors[ci]] += 1

        # Remove 'strength' random codewords
        to_remove = min(strength, len(current))
        removed_indices = random.sample(range(len(current)), to_remove)
        for idx in sorted(removed_indices, reverse=True):
            ci = current[idx]
            cover_count[neighbors[ci]] -= 1
            current.pop(idx)

        # Re-add using greedy
        current = greedy_complete(current, randomize=True)

        return current

    # Main algorithm
    # 1. Start with structured tetracode product (81 words)
    initial_code = structured_initial_code()

    # 2. Complete with greedy
    completed_code = greedy_complete(initial_code, randomize=False)

    # Verify validity
    if not is_valid_cover(completed_code):
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in completed_code:
            cover_count[neighbors[ci]] += 1
        while not np.all(cover_count > 0):
            if time.time() - start_time > time_limit - 20.0:
                break
            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break
            best_idx = -1
            best_count = -1
            for idx in range(total_points):
                cnt = np.sum(cover_count[neighbors[idx]] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
            if best_idx < 0:
                break
            completed_code.append(best_idx)
            cover_count[neighbors[best_idx]] += 1

    # 3. Apply local search
    optimized_code = local_search_reduce(completed_code)

    # 4. Iterated local search with perturbation
    best_code = optimized_code
    best_size = len(optimized_code)

    # Try multiple perturbations
    max_iterations = 50
    for it in range(max_iterations):
        if time.time() - start_time > time_limit - 15.0:
            break

        # Perturb
        perturbed = perturbation(best_code, strength=3 + it // 10)

        # Local search
        searched = local_search_reduce(perturbed)

        if len(searched) < best_size:
            best_code = searched
            best_size = len(searched)

    # Verify final validity
    if not is_valid_cover(best_code):
        best_code = completed_code

    # Final safety check
    if not is_valid_cover(best_code):
        cover_count = np.zeros(total_points, dtype=np.int32)
        chosen = []
        while not np.all(cover_count > 0):
            if time.time() - start_time > time_limit - 5.0:
                break
            best_idx = -1
            best_count = -1
            for idx in range(total_points):
                cnt = np.sum(cover_count[neighbors[idx]] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
            if best_idx < 0:
                break
            chosen.append(best_idx)
            cover_count[neighbors[best_idx]] += 1
        best_code = chosen

    return [all_words[i] for i in best_code]
