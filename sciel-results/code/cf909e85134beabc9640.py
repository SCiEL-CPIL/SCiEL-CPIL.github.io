"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(q, n, radius):
    """
    Constraint satisfaction approach: initialize with 486-word direct sum,
    then iteratively apply critical word heuristic and local search swaps
    to minimize codeword count.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Generate all words in {0,1,2}^n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    N = len(all_words)  # 6561

    # Build word-to-index mapping
    word_to_idx = {}
    for idx, w in enumerate(all_words):
        word_to_idx[tuple(w)] = idx

    # Precompute cover sets: for each word, the indices of words it covers
    # A word w covers itself and all words at distance 1
    covers = []
    for idx in range(N):
        w = all_words[idx]
        cover_set = [idx]
        for i in range(n):
            for alt in range(q):
                if alt == w[i]:
                    continue
                nw = list(w)
                nw[i] = alt
                cover_set.append(word_to_idx[tuple(nw)])
        covers.append(cover_set)

    covers_np = [np.array(c, dtype=np.int32) for c in covers]

    # Build incidence matrix: M[j, w] = 1 if word w is in covers[j]
    M = np.zeros((N, N), dtype=np.bool_)
    for j in range(N):
        M[j, covers_np[j]] = True

    # Initialize with direct sum construction (486 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))

    # Convert to indices
    selected_indices = []
    for cw in initial_codewords:
        idx = word_to_idx[tuple(cw)]
        selected_indices.append(idx)

    # Verify coverage
    covered = np.zeros(N, dtype=bool)
    for c_idx in selected_indices:
        covered[covers_np[c_idx]] = True

    if not covered.all():
        # Shouldn't happen, but fallback to greedy
        active = np.ones(N, dtype=bool)
        for c_idx in selected_indices:
            active[c_idx] = False
        while not covered.all() and time.time() < deadline:
            uncovered_mask = ~covered
            scores = M @ uncovered_mask.astype(np.int32)
            scores[~active] = -1
            best_idx = int(np.argmax(scores))
            if scores[best_idx] <= 0:
                break
            selected_indices.append(best_idx)
            active[best_idx] = False
            covered[covers_np[best_idx]] = True

    # Local search: try to remove redundant codewords and swap
    # First pass: greedy removal
    changed = True
    while changed and time.time() < deadline - 10.0:
        changed = False
        for c_idx in list(selected_indices):
            if time.time() > deadline - 10.0:
                break
            # Check if c_idx is redundant
            cover_words = covers_np[c_idx]
            other_selected = [s for s in selected_indices if s != c_idx]
            if len(other_selected) == 0:
                continue
            sel_arr = np.array(other_selected, dtype=np.int32)
            sub = M[np.ix_(sel_arr, cover_words)]
            col_sums = sub.sum(axis=0)
            if np.all(col_sums >= 1):
                selected_indices.remove(c_idx)
                changed = True
                # Recompute covered
                covered = np.zeros(N, dtype=bool)
                for c_idx2 in selected_indices:
                    covered[covers_np[c_idx2]] = True

    # Second pass: critical word heuristic
    # Identify words covered by only one codeword, try to swap
    for iteration in range(50):
        if time.time() > deadline - 15.0:
            break

        # Compute coverage counts
        coverage_count = np.zeros(N, dtype=np.int32)
        for c_idx in selected_indices:
            coverage_count[covers_np[c_idx]] += 1

        # Find critical words (covered by exactly one codeword)
        critical_words = np.where(coverage_count == 1)[0]
        if len(critical_words) == 0:
            break

        # For each critical word, try to find a better codeword
        for w_idx in critical_words[:100]:  # Limit to first 100 for speed
            if time.time() > deadline - 15.0:
                break

            # Find which codeword covers this word
            covering_codeword = None
            for c_idx in selected_indices:
                if M[c_idx, w_idx]:
                    covering_codeword = c_idx
                    break

            if covering_codeword is None:
                continue

            # Try to find an alternative codeword that covers w_idx
            # and has better overall coverage
            best_alt = None
            best_score = -1

            # Sample some alternatives
            for _ in range(50):
                alt_idx = np.random.randint(N)
                if alt_idx == covering_codeword:
                    continue
                if not M[alt_idx, w_idx]:
                    continue

                # Score: how many currently uncovered or singly-covered words does alt cover?
                alt_cover = covers_np[alt_idx]
                alt_score = np.sum(coverage_count[alt_cover] <= 1)

                if alt_score > best_score:
                    best_score = alt_score
                    best_alt = alt_idx

            if best_alt is not None and best_score > 0:
                # Check if swap is valid: removing covering_codeword and adding best_alt
                # still covers everything
                new_selected = [c for c in selected_indices if c != covering_codeword]
                new_selected.append(best_alt)

                # Check coverage
                new_covered = np.zeros(N, dtype=bool)
                for c_idx in new_selected:
                    new_covered[covers_np[c_idx]] = True

                if new_covered.all():
                    selected_indices = new_selected
                    # Recompute coverage
                    coverage_count = np.zeros(N, dtype=np.int32)
                    for c_idx in selected_indices:
                        coverage_count[covers_np[c_idx]] += 1

    # Third pass: final greedy removal
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        for c_idx in list(selected_indices):
            if time.time() > deadline - 5.0:
                break
            cover_words = covers_np[c_idx]
            other_selected = [s for s in selected_indices if s != c_idx]
            if len(other_selected) == 0:
                continue
            sel_arr = np.array(other_selected, dtype=np.int32)
            sub = M[np.ix_(sel_arr, cover_words)]
            col_sums = sub.sum(axis=0)
            if np.all(col_sums >= 1):
                selected_indices.remove(c_idx)
                changed = True

    # Final verification
    covered = np.zeros(N, dtype=bool)
    for c_idx in selected_indices:
        covered[covers_np[c_idx]] = True

    if not covered.all():
        # Fallback: add missing coverage
        active = np.ones(N, dtype=bool)
        for c_idx in selected_indices:
            active[c_idx] = False
        while not covered.all() and time.time() < deadline:
            uncovered_mask = ~covered
            scores = M @ uncovered_mask.astype(np.int32)
            scores[~active] = -1
            best_idx = int(np.argmax(scores))
            if scores[best_idx] <= 0:
                break
            selected_indices.append(best_idx)
            active[best_idx] = False
            covered[covers_np[best_idx]] = True

    # Convert to list of lists
    return [all_words[idx].tolist() for idx in selected_indices]
# EVOLVE-BLOCK-END
