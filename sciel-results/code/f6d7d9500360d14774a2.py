import numpy as np
import itertools
import time
import random
import heapq

def construct(q, n, radius):
    # Deadline guard: 150s wall-clock limit, keep 15s safety margin
    start_time = time.time()
    deadline = start_time + 135

    # Parameters
    N = q ** n

    # Helper functions
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

    # Precompute all words
    all_words = [idx_to_word(i) for i in range(N)]

    # Precompute neighbors for each word (words within Hamming distance <= radius)
    neighbor_lists = [[] for _ in range(N)]
    neighbor_masks = [0] * N

    for c_idx in range(N):
        c_word = all_words[c_idx]
        mask = 1 << c_idx
        neighbor_lists[c_idx].append(c_idx)

        if radius == 1:
            for i in range(n):
                orig = c_word[i]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = c_word.copy()
                        new_word[i] = new_val
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)
        else:
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == c_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = c_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        w_idx = word_to_idx(new_word)
                        mask |= (1 << w_idx)
                        neighbor_lists[c_idx].append(w_idx)

        neighbor_masks[c_idx] = mask

    # Precompute: for each word, which codewords cover it (inverse mapping)
    word_to_coders = [[] for _ in range(N)]
    for c_idx in range(N):
        for w in neighbor_lists[c_idx]:
            word_to_coders[w].append(c_idx)

    # Criticality-based greedy cover
    def greedy_cover_criticality():
        """Greedy that prioritizes uncovered words with fewest potential coders (criticality)"""
        uncovered = (1 << N) - 1
        code = []
        code_set = set()

        # Track coverage count for each word
        cover_count = np.zeros(N, dtype=np.int32)

        while uncovered:
            if time.time() > deadline:
                break

            # Find the uncovered word with minimum criticality (fewest potential coders)
            min_crit = N + 1
            target_word = -1

            # Get list of uncovered word indices
            uncovered_words = []
            temp = uncovered
            while temp:
                lsb = temp & (-temp)
                w_idx = lsb.bit_length() - 1
                uncovered_words.append(w_idx)
                temp ^= lsb

            if not uncovered_words:
                break

            # Find word with minimum criticality among uncovered
            for w_idx in uncovered_words:
                crit = len(word_to_coders[w_idx])
                if crit < min_crit:
                    min_crit = crit
                    target_word = w_idx
                if min_crit == 1:  # Can't get better
                    break

            if target_word == -1:
                break

            # Find the best codeword that covers target_word and maximizes new coverage
            best_c = -1
            best_new_count = -1

            for c_idx in word_to_coders[target_word]:
                if c_idx in code_set:
                    continue

                # Count how many new words this codeword would cover
                new_count = (uncovered & neighbor_masks[c_idx]).bit_count()

                if new_count > best_new_count:
                    best_new_count = new_count
                    best_c = c_idx

            if best_c == -1 or best_new_count == 0:
                # Fallback: pick any codeword that covers target_word
                for c_idx in word_to_coders[target_word]:
                    if c_idx not in code_set:
                        best_c = c_idx
                        break

            if best_c == -1:
                break

            code.append(best_c)
            code_set.add(best_c)
            uncovered &= ~neighbor_masks[best_c]

        return code

    # Local search: remove redundant codewords
    def local_search_remove(code):
        """Remove redundant codewords"""
        if not code:
            return code

        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
                # Check if any word is uniquely covered by c
                has_unique = False
                for w in neighbor_lists[c]:
                    if cover_counts[w] == 1:
                        has_unique = True
                        break

                if not has_unique:
                    code.pop(i)
                    for w in neighbor_lists[c]:
                        cover_counts[w] -= 1
                    improved = True
                    break

        return code

    # 2-opt local search: try to replace one codeword with another to enable removal of another
    def two_opt_local_search(code):
        """2-opt: try to replace one codeword with another to enable removal of another"""
        code = list(code)
        code_set = set(code)

        cover_counts = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in neighbor_lists[c]:
                cover_counts[w] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Try removing each codeword
            for i in range(len(code)):
                if time.time() > deadline:
                    break

                c_remove = code[i]

                # Find words that would become uncovered if we remove c_remove
                newly_uncovered = []
                for w in neighbor_lists[c_remove]:
                    if cover_counts[w] == 1:
                        newly_uncovered.append(w)

                if not newly_uncovered:
                    # Can remove without adding
                    code.pop(i)
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    code_set.discard(c_remove)
                    improved = True
                    break

                # Try to find a single codeword that covers all newly_uncovered
                # and is not already in code
                best_add = -1
                best_gain = -1

                # For each newly uncovered word, find codewords that cover it
                candidate_set = set()
                for w in newly_uncovered:
                    for c in word_to_coders[w]:
                        if c not in code_set:
                            candidate_set.add(c)

                # Try each candidate
                for c_add in candidate_set:
                    # Check if c_add covers all newly_uncovered
                    covers_all = True
                    for w in newly_uncovered:
                        if w not in neighbor_lists[c_add]:
                            covers_all = False
                            break

                    if covers_all:
                        # This swap is valid: remove c_remove, add c_add
                        # Count how many words c_add uniquely covers (potential for further removal)
                        gain = 0
                        for w in neighbor_lists[c_add]:
                            if cover_counts[w] == 0:
                                gain += 1

                        if gain > best_gain:
                            best_gain = gain
                            best_add = c_add

                if best_add != -1:
                    # Perform the swap
                    code[i] = best_add
                    code_set.discard(c_remove)
                    code_set.add(best_add)

                    # Update cover counts
                    for w in neighbor_lists[c_remove]:
                        cover_counts[w] -= 1
                    for w in neighbor_lists[best_add]:
                        cover_counts[w] += 1

                    improved = True
                    break

        return code

    # Simulated annealing with pair swaps
    def simulated_annealing(code, max_time=60):
        """Simulated annealing to escape local minima via pair swaps."""
        sa_start = time.time()
        sa_deadline = min(sa_start + max_time, deadline)

        code = list(code)
        code_set = set(code)

        # Precompute coverage counts
        cover_count = {}
        for c in code:
            for w in neighbor_lists[c]:
                cover_count[w] = cover_count.get(w, 0) + 1

        best_code = list(code)
        best_len = len(code)

        # Temperature schedule
        T = 10.0
        cooling_rate = 0.995
        min_T = 0.1

        while time.time() < sa_deadline and T > min_T:
            # Randomly pick two codewords to swap
            if len(code) < 2:
                break

            i, j = random.sample(range(len(code)), 2)
            c_i, c_j = code[i], code[j]

            # Try swapping: remove both, add two new random codewords
            new_candidates = []
            for _ in range(5):
                new_c = random.randint(0, N - 1)
                if new_c not in code_set and new_c not in new_candidates:
                    new_candidates.append(new_c)

            if len(new_candidates) < 2:
                T *= cooling_rate
                continue

            new_c1, new_c2 = random.sample(new_candidates, 2)

            # Check if this swap maintains coverage
            # Words only covered by c_i or c_j must be covered by new_c1 or new_c2
            only_i = set(neighbor_lists[c_i]) - set(neighbor_lists[c_j])
            only_j = set(neighbor_lists[c_j]) - set(neighbor_lists[c_i])

            new_cover = set(neighbor_lists[new_c1]) | set(neighbor_lists[new_c2])

            # Check coverage of words that were only covered by c_i
            uncovered_i = only_i - new_cover
            # Check coverage of words that were only covered by c_j
            uncovered_j = only_j - new_cover

            if len(uncovered_i) == 0 and len(uncovered_j) == 0:
                # Valid swap: accept it
                code[i] = new_c1
                code[j] = new_c2
                code_set.discard(c_i)
                code_set.discard(c_j)
                code_set.add(new_c1)
                code_set.add(new_c2)

                # Update cover_count
                for w in neighbor_lists[c_i]:
                    cover_count[w] -= 1
                for w in neighbor_lists[c_j]:
                    cover_count[w] -= 1
                for w in neighbor_lists[new_c1]:
                    cover_count[w] = cover_count.get(w, 0) + 1
                for w in neighbor_lists[new_c2]:
                    cover_count[w] = cover_count.get(w, 0) + 1

                # Try to remove redundant codewords
                improved = True
                while improved and time.time() < sa_deadline:
                    improved = False
                    for k in range(len(code)):
                        c_k = code[k]
                        is_redundant = True
                        for w in neighbor_lists[c_k]:
                            if cover_count.get(w, 0) == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            code.pop(k)
                            code_set.discard(c_k)
                            for w in neighbor_lists[c_k]:
                                cover_count[w] -= 1
                            improved = True
                            break

                if len(code) < best_len:
                    best_code = list(code)
                    best_len = len(code)

            T *= cooling_rate

        return best_code

    # Run greedy cover
    code = greedy_cover_criticality()

    # Run local search
    if time.time() < deadline:
        code = local_search_remove(code)

    # Run 2-opt local search
    if time.time() < deadline - 30:
        code = two_opt_local_search(code)

    # Run simulated annealing if time permits
    if time.time() < deadline - 20:
        code = simulated_annealing(code, max_time=min(40, deadline - time.time() - 10))

    # Final local search: try to remove redundant codewords again
    if time.time() < deadline:
        code = local_search_remove(code)

    # Verify the final code is valid
    covered = set()
    for c in code:
        covered.update(neighbor_lists[c])

    if len(covered) < N:
        # Fallback: add missing coverage
        while len(covered) < N and time.time() < deadline:
            best_code = None
            best_count = -1
            for c_idx in range(N):
                if c_idx in code:
                    continue
                count = sum(1 for w in neighbor_lists[c_idx] if w not in covered)
                if count > best_count:
                    best_count = count
                    best_code = c_idx
            if best_code is None:
                break
            code.append(best_code)
            covered.update(neighbor_lists[best_code])

    # Tetracode fallback if something went wrong
    if not code:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert indices to words
    result = [all_words[i] for i in code]
    return result
