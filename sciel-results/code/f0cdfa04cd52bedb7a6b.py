import time
import random
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin vs 150s kill

    total = q ** n

    # ---- Build all words and index helpers ----
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % q)
            idx //= q
        w.reverse()
        return w

    all_words = [idx_to_word(i) for i in range(total)]

    # ---- Build cover matrix: cover[i] = set of words within Hamming distance <= radius of i ----
    cover = np.zeros((total, total), dtype=bool)
    np.fill_diagonal(cover, True)

    if radius >= 1:
        powers = [q ** (n - 1 - pos) for pos in range(n)]
        for i in range(total):
            w = all_words[i]
            for pos in range(n):
                p = powers[pos]
                ov = w[pos]
                for dv in range(1, q):
                    j = i + dv * p
                    if j < total:
                        cover[i, j] = True
                    j2 = i - (q - dv) * p
                    if j2 >= 0:
                        cover[i, j2] = True

    # ---- Precompute unique coverage: uniq[i] = words covered ONLY by i (among all words) ----
    # A word w is in ball(i) iff cover[i, w] == True.
    # count[w] = number of i with cover[i, w] == True.
    # uniq[i] = { w : cover[i,w] and count[w] == 1 }
    # Note: cover is symmetric (distance is symmetric), so count[w] = number of i with cover[w, i]... 
    # Actually count[w] = sum_i cover[i, w] = sum_i cover[w, i] (symmetry) = row sum of cover[w].
    # So count[w] = cover[w].sum()
    row_sums = cover.sum(axis=1)  # count[w] for each w
    # uniq[i, w] = cover[i, w] AND (row_sums[w] == 1)
    uniq = cover & (row_sums == 1)[np.newaxis, :]

    # ---- Helper: coverage count over current code ----
    def compute_cov(code):
        cov = np.zeros(total, dtype=np.int32)
        for idx in code:
            cov += cover[idx]
        return cov

    # ---- Greedy set cover using unique-coverage gains ----
    # Gain of i = number of words in uniq[i] that are currently uncovered.
    # A word w is uncovered iff no chosen codeword covers it.
    # Since uniq[i] words are covered ONLY by i, they are uncovered iff i is not chosen.
    # So gain of i = len(uniq[i]) if i not chosen, else 0. That's not quite right for greedy
    # because we want to cover uncovered words, and a word w is covered by i iff cover[i,w].
    # But for greedy, the standard gain is |ball(i) \ covered|. 
    # However, words in ball(i) that are NOT in uniq[i] are also covered by other words, 
    # so they're already covered once ANY codeword covering them is chosen.
    # The truly "new" words i covers are those in ball(i) not covered by any previously chosen.
    # This is exactly: ball(i) \ union_{chosen} ball(chosen).
    # Using uniq is a lower bound but not exact. Let's use exact gain via matrix ops.
    
    def greedy_cover():
        uncovered = np.ones(total, dtype=bool)
        code = []
        cm = cover.astype(np.int8)
        
        while np.any(uncovered):
            if time.time() > deadline - 5.0:
                break
            # gain[i] = number of uncovered words covered by i
            gains = cm.dot(uncovered.astype(np.int8))
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best = int(candidates[random.randint(0, len(candidates) - 1)])
            code.append(best)
            uncovered[cover[best]] = False
        return code

    # ---- Local search: 1-opt (remove) + 1-swap + 2-opt (swap two) ----
    def local_search(code):
        if not code:
            return code
        current = code[:]
        cov = compute_cov(current)
        in_code = np.zeros(total, dtype=bool)
        for idx in current:
            in_code[idx] = True

        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5.0:
                break
            random.shuffle(current)
            n_cw = len(current)

            # --- 1-opt: try to remove each codeword ---
            for i in range(n_cw):
                if time.time() > deadline - 5.0:
                    break
                idx = current[i]
                mask = cover[idx]
                # Words uniquely covered by idx in current code:
                # w in ball(idx) where cov[w] == 1 (only idx covers it)
                unique_mask = mask & (cov == 1)
                if not np.any(unique_mask):
                    # Removable
                    cov -= mask
                    in_code[idx] = False
                    current.pop(i)
                    improved = True
                    break

            if improved:
                continue

            # --- 1-swap: replace idx with j not in code ---
            for i in range(len(current)):
                if time.time() > deadline - 5.0:
                    break
                idx = current[i]
                mask = cover[idx]
                unique_mask = mask & (cov == 1)
                if not np.any(unique_mask):
                    continue  # would have been removed
                unique_indices = np.where(unique_mask)[0]
                # Find j not in code such that cover[j] covers all unique_indices
                # and j covers some words not covered by idx (to be a net improvement, 
                # but since we're swapping 1-for-1, size stays same; we want to enable 
                # further removals later). Just find any valid swap.
                # cover_matrix[:, unique_indices].all(axis=1)
                candidates = np.where(cover[:, unique_indices].all(axis=1))[0]
                candidates = candidates[~in_code[candidates]]
                if len(candidates) > 0:
                    # Prefer candidate that covers most words not in mask
                    new_cand = candidates
                    # Score: number of words j covers that idx doesn't
                    # net_gain = (cover[j] & ~mask).sum()
                    # Pick best
                    best_j = -1
                    best_score = -1
                    for j in new_cand[:20]:  # limit to first 20 for speed
                        score = (cover[j] & ~mask).sum()
                        if score > best_score:
                            best_score = score
                            best_j = j
                    if best_j >= 0:
                        cov -= mask
                        cov += cover[best_j]
                        in_code[idx] = False
                        in_code[best_j] = True
                        current[i] = int(best_j)
                        improved = True
                        break

            if improved:
                continue

            # --- 2-opt: swap two codewords i, k with two new ones? Too expensive.
            # Instead, do a "move" step: try moving a codeword to a better position 
            # that allows a removal. Skip for now; 1-opt + 1-swap is the core.
            break

        return current

    # ---- Main: greedy + local search, with restarts ----
    best_code = greedy_cover()
    best_code = local_search(best_code)
    best_len = len(best_code)

    # Restart loop
    while time.time() < deadline - 10.0:
        cand = greedy_cover()
        cand = local_search(cand)
        if len(cand) < best_len:
            best_code = cand
            best_len = len(cand)

    # Final local search on best
    if time.time() < deadline - 5.0:
        best_code = local_search(best_code)

    return [all_words[i] for i in best_code]
