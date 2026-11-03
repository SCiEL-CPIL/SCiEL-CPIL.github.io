import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    N = q ** n
    ball_size = 1 + n * (q - 1)  # 17 for q=3,n=8,r=1

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    all_words = [idx_to_word(i) for i in range(N)]

    # Precompute neighbor lists (as numpy arrays for speed)
    neighbor_lists = []
    for c_idx in range(N):
        c_word = all_words[c_idx]
        neighbors = [c_idx]
        if radius == 1:
            for i in range(n):
                orig = c_word[i]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = c_word.copy()
                        new_word[i] = new_val
                        neighbors.append(word_to_idx(new_word))
        else:
            import itertools
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == c_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = c_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        neighbors.append(word_to_idx(new_word))
        neighbor_lists.append(np.array(neighbors, dtype=np.int32))

    # Precompute inverse: for each word, which codewords cover it
    word_to_coders = [np.array([], dtype=np.int32) for _ in range(N)]
    for c_idx in range(N):
        for w in neighbor_lists[c_idx]:
            word_to_coders[w] = np.append(word_to_coders[w], c_idx)
    word_to_coders = [wc for wc in word_to_coders]

    # Criticality: number of codewords that can cover each word
    criticality = np.array([len(wtc) for wtc in word_to_coders], dtype=np.int32)

    def greedy_cover_criticality(seed=None):
        """Greedy set cover with criticality-based ordering."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        uncovered = np.ones(N, dtype=bool)
        code = []
        code_set = set()

        while uncovered.any():
            if time.time() > deadline:
                break

            # Among uncovered words, find one with minimum criticality
            # Use a soft approach: weight by 1/criticality for stochastic selection
            uncov_idx = np.where(uncovered)[0]
            if len(uncov_idx) == 0:
                break

            # Pick the word with minimum criticality (with some randomization)
            crit_vals = criticality[uncov_idx].astype(np.float64)
            # Add small noise to break ties
            noise = rng.random(len(uncov_idx)) * 0.1
            scores = 1.0 / (crit_vals + noise + 1e-9)
            target_word = uncov_idx[np.argmax(scores)]

            # Find best codeword covering target_word
            candidates = word_to_coders[target_word]
            best_c = -1
            best_gain = -1

            for c_idx in candidates:
                if c_idx in code_set:
                    continue
                # Count new coverage
                nbrs = neighbor_lists[c_idx]
                gain = np.sum(uncovered[nbrs])
                if gain > best_gain:
                    best_gain = gain
                    best_c = c_idx

            if best_c == -1:
                # Fallback: pick any
                for c_idx in candidates:
                    if c_idx not in code_set:
                        best_c = c_idx
                        break
            if best_c == -1:
                break

            code.append(best_c)
            code_set.add(best_c)
            uncovered[neighbor_lists[best_c]] = False

        return code

    def local_search_remove(code):
        """Remove redundant codewords iteratively."""
        if not code:
            return code

        code = list(code)
        code_set = set(code)
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_counts[neighbor_lists[c]] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
                nbrs = neighbor_lists[c]
                # Check if any word is uniquely covered
                if np.any(cover_counts[nbrs] == 1):
                    continue
                # Remove
                code.pop(i)
                code_set.discard(c)
                cover_counts[nbrs] -= 1
                improved = True
                break

        return code

    def two_opt_swap(code):
        """Try to swap one codeword for another to enable further removals."""
        code = list(code)
        code_set = set(code)
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_counts[neighbor_lists[c]] += 1

        improved = True
        max_iters = 500
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iters:
            improved = False
            iter_count += 1

            for i in range(len(code)):
                if time.time() > deadline:
                    break

                c_remove = code[i]
                nbrs_remove = neighbor_lists[c_remove]

                # Find words uniquely covered by c_remove
                unique_mask = cover_counts[nbrs_remove] == 1
                unique_words = nbrs_remove[unique_mask]

                if len(unique_words) == 0:
                    # Can remove freely
                    code.pop(i)
                    code_set.discard(c_remove)
                    cover_counts[nbrs_remove] -= 1
                    improved = True
                    break

                # Find codewords that cover all unique_words
                # Intersect the coder sets of unique_words
                candidate_set = set(word_to_coders[unique_words[0]]) - code_set
                for w in unique_words[1:]:
                    candidate_set &= set(word_to_coders[w])
                    if not candidate_set:
                        break

                if not candidate_set:
                    continue

                best_add = -1
                best_net_gain = -1
                for c_add in candidate_set:
                    nbrs_add = neighbor_lists[c_add]
                    # Net gain: words newly covered by add that aren't covered by remove
                    # = |nbrs_add ∩ (uncovered after removing c_remove)|
                    # After removing c_remove, words with cover_count==1 become uncovered
                    # c_add must cover those, plus it may cover others
                    # Net change in code size: -1 (remove) + 1 (add) = 0, but enables future removals
                    # Better metric: how many words does c_add cover that c_remove doesn't?
                    # Or: total coverage after swap
                    new_cov = np.sum((cover_counts[nbrs_add] >= 1) | (nbrs_add[None, :] == nbrs_remove[:, None]))
                    # Simpler: count words in nbrs_add that have cover_count >= 1 (already covered by others)
                    # plus words in nbrs_add that are in nbrs_remove
                    already = np.sum(cover_counts[nbrs_add] >= 1)
                    in_remove = np.sum(np.isin(nbrs_add, nbrs_remove))
                    # The words that c_add covers that would remain covered:
                    # words in nbrs_add that are NOT uniquely covered by c_remove (i.e., cover_count >= 2 or not in nbrs_remove)
                    # Actually let's just compute: after removing c_remove and adding c_add,
                    # is everything still covered?
                    # Words in nbrs_remove with cover_count==1 must be in nbrs_add (we checked)
                    # Words in nbrs_add with cover_count==0: these become covered (good)
                    # Words in nbrs_remove with cover_count>=2: still covered by others
                    # So the swap is valid. Net gain in "slack" = words in nbrs_add with cover_count==0
                    slack = np.sum(cover_counts[nbrs_add] == 0)
                    if slack > best_net_gain:
                        best_net_gain = slack
                        best_add = c_add

                if best_add != -1:
                    # Perform swap
                    code.pop(i)
                    code_set.discard(c_remove)
                    cover_counts[nbrs_remove] -= 1

                    code.append(best_add)
                    code_set.add(best_add)
                    cover_counts[neighbor_lists[best_add]] += 1
                    improved = True
                    break

        return code

    def three_opt(code):
        """Try to remove 2 and add 1, or remove 1 and add 0 (handled by remove)."""
        # This is a heuristic: try removing pairs and see if we can add one
        code = list(code)
        code_set = set(code)
        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_counts[neighbor_lists[c]] += 1

        improved = True
        max_iters = 200
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iters:
            improved = False
            iter_count += 1

            # Try all pairs for removal
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(code)):
                    c1, c2 = code[i], code[j]
                    n1, n2 = neighbor_lists[c1], neighbor_lists[c2]

                    # Simulate removal
                    sim_counts = cover_counts.copy()
                    sim_counts[n1] -= 1
                    sim_counts[n2] -= 1

                    uncovered_after = np.where(sim_counts < 1)[0]
                    if len(uncovered_after) == 0:
                        # Can remove both
                        # Remove j first (higher index)
                        code.pop(j)
                        code_set.discard(c2)
                        cover_counts[n2] -= 1
                        code.pop(i)
                        code_set.discard(c1)
                        cover_counts[n1] -= 1
                        improved = True
                        break

                    # Try to find one codeword covering all uncovered_after
                    if len(uncovered_after) > 0 and len(uncovered_after) <= 20:
                        cand_set = set(word_to_coders[uncovered_after[0]]) - code_set
                        for w in uncovered_after[1:]:
                            cand_set &= set(word_to_coders[w])
                            if not cand_set:
                                break

                        if cand_set:
                            best_add = -1
                            best_slack = -1
                            for c_add in cand_set:
                                nbrs_add = neighbor_lists[c_add]
                                slack = np.sum(cover_counts[nbrs_add] == 0)
                                if slack > best_slack:
                                    best_slack = slack
                                    best_add = c_add
                            if best_add != -1:
                                # Perform: remove 2, add 1
                                # Remove j, then i
                                code.pop(j)
                                code_set.discard(c2)
                                cover_counts[n2] -= 1
                                code.pop(i)
                                code_set.discard(c1)
                                cover_counts[n1] -= 1

                                code.append(best_add)
                                code_set.add(best_add)
                                cover_counts[neighbor_lists[best_add]] += 1
                                improved = True
                                break
                if improved:
                    break
            if improved:
                break

        return code

    # Main search loop
    best_code = greedy_cover_criticality(seed=42)
    best_code = local_search_remove(best_code)
    best_code = two_opt_swap(best_code)
    best_code = local_search_remove(best_code)
    best_code = three_opt(best_code)
    best_code = local_search_remove(best_code)

    best_size = len(best_code)

    # Multiple restarts
    for restart in range(1, 50):
        if time.time() > deadline - 15:
            break

        seed = restart * 137 + 7
        code = greedy_cover_criticality(seed=seed)
        code = local_search_remove(code)
        code = two_opt_swap(code)
        code = local_search_remove(code)
        code = three_opt(code)
        code = local_search_remove(code)

        if len(code) < best_size:
            best_code = code
            best_size = len(code)

    # Final polish: more aggressive local search
    if time.time() < deadline - 5:
        best_code = two_opt_swap(best_code)
        best_code = local_search_remove(best_code)
        best_code = three_opt(best_code)
        best_code = local_search_remove(best_code)

    # Verification
    covered = np.zeros(N, dtype=bool)
    for c in best_code:
        covered[neighbor_lists[c]] = True
    if not covered.all():
        # Fix: add codewords for uncovered
        uncov = np.where(~covered)[0]
        while len(uncov) > 0 and time.time() < deadline:
            best_c = -1
            best_gain = -1
            for c in range(N):
                if c in set(best_code):
                    continue
                gain = np.sum(~covered[neighbor_lists[c]])
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            best_code.append(best_c)
            covered[neighbor_lists[best_c]] = True
            uncov = np.where(~covered)[0]

    result = [all_words[i] for i in best_code]
    return result
