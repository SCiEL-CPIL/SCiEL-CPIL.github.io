import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = 118.0  # safety margin before 150s kill

    total_words = q ** n

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all words (radius-1 ball)
    neighbors = np.empty(total_words, dtype=object)
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[i] = new_val
                nbrs.append(word_to_idx(w))
            w[i] = orig
        neighbors[idx] = np.array(nbrs, dtype=np.int64)

    # Precompute a coverage bitset per word for fast operations
    # covered_mask: bool array total_words
    # For each word, neighbors[word] gives indices

    def covers(idx):
        return neighbors[idx]

    # --- Structured seed via Z_3 linear code ---
    # A linear code over Z_3 with generator matrix G (k x n) gives 3^k codewords.
    # For covering radius 1, we want a code whose cosets cover the space.
    # Try several random generator matrices and pick the best covering.

    def random_linear_code(k, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        # Random k x n matrix over Z_3, try to make rows independent
        rows = []
        attempts = 0
        while len(rows) < k and attempts < 200:
            attempts += 1
            row = [rng.randrange(q) for _ in range(n)]
            if all(r != 0 for r in row):
                # Check independence (over Z_3)
                independent = True
                for existing in rows:
                    # Check if row is a scalar multiple of existing
                    if existing[0] != 0:
                        factor = (row[0] * pow(existing[0], q - 2, q)) % q
                        if all((row[j] - factor * existing[j]) % q == 0 for j in range(n)):
                            independent = False
                            break
                    else:
                        # existing[0] == 0, check if row is multiple
                        found_pivot = False
                        for j in range(n):
                            if existing[j] != 0:
                                factor = (row[j] * pow(existing[j], q - 2, q)) % q
                                if all((row[a] - factor * existing[a]) % q == 0 for a in range(n)):
                                    independent = False
                                break
                        if not independent:
                            break
                if independent:
                    rows.append(row)
        if len(rows) < k:
            # Fill with standard basis if needed
            while len(rows) < k:
                row = [0] * n
                row[len(rows)] = 1
                rows.append(row)
        return rows

    def linear_code_codewords(G):
        k = len(G)
        codewords = []
        for coeffs in range(q ** k):
            c = coeffs
            cw = [0] * n
            for i in range(k):
                cw[i] = c % q
                c //= q
            word = [0] * n
            for j in range(n):
                s = 0
                for i in range(k):
                    s += cw[i] * G[i][j]
                word[j] = s % q
            codewords.append(word)
        return codewords

    def coverage_size(code_words):
        covered = np.zeros(total_words, dtype=bool)
        for w in code_words:
            idx = word_to_idx(w)
            covered[neighbors[idx]] = True
        return int(np.sum(covered))

    # Try several linear codes to find a good seed
    best_seed = None
    best_seed_coverage = -1
    for k in [3, 4]:
        for trial in range(30):
            if time.time() - start_time > DEADLINE - 60:
                break
            G = random_linear_code(k, seed=trial * 1000 + k)
            cw = linear_code_codewords(G)
            cov = coverage_size(cw)
            if cov > best_seed_coverage:
                best_seed_coverage = cov
                best_seed = cw
            if time.time() - start_time > DEADLINE - 60:
                break
        if best_seed_coverage >= total_words - 50:
            break

    # If no good linear seed, use greedy from scratch
    if best_seed is None:
        best_seed = []

    # --- Greedy set-cover to complete ---
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    for w in best_seed:
        idx = word_to_idx(w)
        code_indices.append(idx)
        covered[neighbors[idx]] = True

    # Greedy fill
    while not np.all(covered):
        if time.time() - start_time > DEADLINE - 20:
            break
        best_idx = -1
        best_new = -1
        for c in range(total_words):
            # Count uncovered neighbors
            nbrs = neighbors[c]
            new_cov = int(np.sum(~covered[nbrs]))
            if new_cov > best_new:
                best_new = new_cov
                best_idx = c
        if best_new <= 0:
            break
        code_indices.append(best_idx)
        covered[neighbors[best_idx]] = True

    # --- Local search polish ---
    # Try to swap a codeword for one that covers more uncovered, or remove redundant ones
    def is_covered_by(idx, code_list, exclude=None):
        # Check if idx is covered by some codeword in code_list (excluding 'exclude' index)
        for ci in code_list:
            if ci == exclude:
                continue
            nbrs = neighbors[ci]
            if idx in nbrs:
                return True
        return False

    def full_coverage_check(code_list):
        cov = np.zeros(total_words, dtype=bool)
        for ci in code_list:
            cov[neighbors[ci]] = True
        return cov

    # Local search: try removing redundant codewords, then swapping
    for iteration in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break

        # Try to remove redundant codewords
        removed = True
        while removed:
            removed = False
            for i in range(len(code_indices)):
                ci = code_indices[i]
                # Check if ci's coverage is redundant (all its neighbors covered by others)
                nbrs = neighbors[ci]
                # Build coverage without ci
                cov_without = np.zeros(total_words, dtype=bool)
                for j, cj in enumerate(code_indices):
                    if j == i:
                        continue
                    cov_without[neighbors[cj]] = True
                if np.all(cov_without[nbrs]):
                    code_indices.pop(i)
                    removed = True
                    break

        # Try swaps: replace one codeword with another to reduce total or maintain coverage
        improved = True
        while improved:
            improved = False
            current_cov = full_coverage_check(code_indices)
            for i in range(len(code_indices)):
                ci = code_indices[i]
                # Try removing ci and see what becomes uncovered
                cov_without = np.zeros(total_words, dtype=bool)
                for j, cj in enumerate(code_indices):
                    if j == i:
                        continue
                    cov_without[neighbors[cj]] = True
                uncovered_by_removal = ~cov_without
                if not np.any(uncovered_by_removal):
                    # ci is redundant, remove it
                    code_indices.pop(i)
                    improved = True
                    break
                # Try to find a replacement that covers all of uncovered_by_removal
                # and ideally covers more
                target = uncovered_by_removal
                for c in range(total_words):
                    if c in code_indices:
                        continue
                    nbrs = neighbors[c]
                    if np.all(cov_without[nbrs] | target[nbrs]):
                        # c covers everything ci covered (that wasn't covered by others)
                        # Actually we need: removing ci leaves target uncovered, adding c must cover target
                        if np.all(target[nbrs]):
                            code_indices[i] = c
                            improved = True
                            break
            if improved:
                continue

    # Final coverage verification
    final_cov = full_coverage_check(code_indices)
    if not np.all(final_cov):
        # Emergency fill
        covered = final_cov
        while not np.all(covered):
            best_idx = -1
            best_new = -1
            for c in range(total_words):
                new_cov = int(np.sum(~covered[neighbors[c]]))
                if new_cov > best_new:
                    best_new = new_cov
                    best_idx = c
            if best_new <= 0:
                break
            code_indices.append(best_idx)
            covered[neighbors[best_idx]] = True

    # Deduplicate
    seen = set()
    unique_code = []
    for idx in code_indices:
        if idx not in seen:
            seen.add(idx)
            unique_code.append(idx)

    return [idx_to_word(idx) for idx in unique_code]
