import numpy as np
import time
import random
from collections import Counter

def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = 118.0

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

    # Precompute neighbors for all words
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

    # Convert to list of lists for easier manipulation
    nbr_lists = [neighbors[i].tolist() for i in range(total_words)]

    # Precompute: for each word, which codewords cover it (inverse mapping)
    # This helps with fast redundancy checks

    # --- Greedy set-cover ---
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []

    # Compute initial coverage counts: how many codewords cover each position
    # Use a counter array for efficiency
    coverage_count = np.zeros(total_words, dtype=np.int32)

    while not np.all(covered):
        if time.time() - start_time > DEADLINE - 10:
            break

        best_idx = -1
        best_new = -1

        # Find the word that covers the most uncovered neighbors
        for c in range(total_words):
            nbrs = nbr_lists[c]
            # Count how many of c's neighbors are currently uncovered
            new_cov = sum(1 for nb in nbrs if not covered[nb])
            if new_cov > best_new:
                best_new = new_cov
                best_idx = c

        if best_new <= 0:
            break

        code_indices.append(best_idx)
        for nb in nbr_lists[best_idx]:
            covered[nb] = True

    # --- Local search: remove redundant codewords ---
    def is_redundant(ci, code_list):
        """Check if codeword ci is redundant (all its neighbors covered by others)."""
        nbrs = nbr_lists[ci]
        # For each neighbor of ci, check if it's covered by at least one other codeword
        for nb in nbrs:
            found = False
            for cj in code_list:
                if cj == ci:
                    continue
                if nb in nbr_lists[cj]:
                    found = True
                    break
            if not found:
                return False
        return True

    # Iteratively remove redundant codewords
    removed = True
    while removed and time.time() - start_time < DEADLINE - 20:
        removed = False
        for i in range(len(code_indices)):
            if is_redundant(code_indices[i], code_indices):
                code_indices.pop(i)
                removed = True
                break

    # --- Local search: try swaps to reduce size ---
    # For each codeword, try to find a replacement that allows removing another codeword
    for iteration in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break

        improved = False
        for i in range(len(code_indices)):
            ci = code_indices[i]
            # What would become uncovered if we remove ci?
            # Build coverage without ci
            cov_without = np.zeros(total_words, dtype=bool)
            for j, cj in enumerate(code_indices):
                if j == i:
                    continue
                for nb in nbr_lists[cj]:
                    cov_without[nb] = True

            uncovered = ~cov_without
            if not np.any(uncovered):
                # ci is redundant
                code_indices.pop(i)
                improved = True
                break

            # Try to find a codeword c (not in code_indices) that covers all uncovered positions
            # and whose addition allows us to also remove another codeword
            target = uncovered.copy()
            for c in range(total_words):
                if c in code_indices:
                    continue
                nbrs_c = nbr_lists[c]
                # Check if c covers all currently uncovered positions
                all_covered = True
                for nb in nbrs_c:
                    if target[nb]:
                        pass  # c covers this uncovered position, good
                # Actually check: does c cover ALL positions in target?
                for pos in range(total_words):
                    if target[pos] and pos not in nbrs_c:
                        all_covered = False
                        break
                if all_covered:
                    # Adding c covers all that ci covered. But can we remove another codeword too?
                    # Try removing each other codeword and see if coverage still holds
                    new_list = code_indices[:i] + code_indices[i+1:] + [c]
                    # Check if any codeword in new_list is redundant
                    for k, ck in enumerate(new_list):
                        if is_redundant(ck, new_list):
                            code_indices = new_list[:k] + new_list[k+1:]
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break

    # Final verification and emergency fill
    final_cov = np.zeros(total_words, dtype=bool)
    for ci in code_indices:
        for nb in nbr_lists[ci]:
            final_cov[nb] = True

    if not np.all(final_cov):
        covered = final_cov.copy()
        while not np.all(covered) and time.time() - start_time < DEADLINE - 1:
            best_idx = -1
            best_new = -1
            for c in range(total_words):
                nbrs = nbr_lists[c]
                new_cov = sum(1 for nb in nbrs if not covered[nb])
                if new_cov > best_new:
                    best_new = new_cov
                    best_idx = c
            if best_new <= 0:
                break
            code_indices.append(best_idx)
            for nb in nbr_lists[best_idx]:
                covered[nb] = True

    # Deduplicate
    seen = set()
    unique_code = []
    for idx in code_indices:
        if idx not in seen:
            seen.add(idx)
            unique_code.append(idx)

    return [idx_to_word(idx) for idx in unique_code]
