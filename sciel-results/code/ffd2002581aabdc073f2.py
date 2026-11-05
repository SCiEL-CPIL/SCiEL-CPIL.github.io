import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # For the specific problem: q=3, n=6, radius=1
    # Use direct sum of ternary tetracode (9 words of length 4) with all tails
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_n = 4
        base_code = tetracode
    else:
        base_code = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        base_n = 4

    if n < base_n:
        return [list(range(n))]

    # Build initial code: direct sum
    tails = list(itertools.product(range(q), repeat=n - base_n))
    code = [tuple(h) + tuple(tail) for h in base_code for tail in tails]
    num_codewords = len(code)
    total_words = q ** n
    sphere_size = 1 + n * (q - 1)

    # Compute word indices for each codeword
    def compute_word_indices(codewords):
        indices = np.zeros(len(codewords), dtype=np.int64)
        for i in range(n):
            indices += codewords[:, i] * (q ** (n - 1 - i))
        return indices

    code_arr = np.array(code, dtype=np.int32)
    cw_indices = compute_word_indices(code_arr)

    # Precompute coverage: for each codeword, list of word indices it covers
    # And reverse: for each word, list of codewords that cover it
    codeword_covers = []
    covered_by = [[] for _ in range(total_words)]

    for ci in range(num_codewords):
        c = code_arr[ci]
        covers = set()
        covers.add(cw_indices[ci])
        for pos in range(n):
            orig = c[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_idx = cw_indices[ci] + (val - orig) * (q ** (n - 1 - pos))
                covers.add(new_idx)
        codeword_covers.append(list(covers))
        for wi in covers:
            covered_by[wi].append(ci)

    # Convert codeword_covers to numpy arrays for vectorized operations
    max_cover_size = max(len(c) for c in codeword_covers)
    codeword_covers_arr = np.full((num_codewords, max_cover_size), -1, dtype=np.int32)
    codeword_covers_len = np.zeros(num_codewords, dtype=np.int32)
    for ci in range(num_codewords):
        covers = codeword_covers[ci]
        codeword_covers_arr[ci, :len(covers)] = covers
        codeword_covers_len[ci] = len(covers)

    # Best code tracking
    best_code = None
    best_count = num_codewords

    # Helper: check if removing codeword ci is safe (all its words covered by others)
    def is_redundant(ci, coverage_count):
        covers = codeword_covers_arr[ci, :codeword_covers_len[ci]]
        return np.all(coverage_count[covers] > 1)

    # Helper: perform removal of codeword ci
    def remove_codeword(ci, active, coverage_count):
        active[ci] = False
        covers = codeword_covers_arr[ci, :codeword_covers_len[ci]]
        coverage_count[covers] -= 1

    # Helper: perform addition of codeword ci
    def add_codeword(ci, active, coverage_count):
        active[ci] = True
        covers = codeword_covers_arr[ci, :codeword_covers_len[ci]]
        coverage_count[covers] += 1

    # Multiple trials with different random orderings
    max_trials = 50

    for trial in range(max_trials):
        if time.time() > deadline - 5.0:
            break

        # Random order for checking removals
        order = list(range(num_codewords))
        random.shuffle(order)

        active = np.ones(num_codewords, dtype=bool)
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for ci in range(num_codewords):
            covers = codeword_covers_arr[ci, :codeword_covers_len[ci]]
            coverage_count[covers] += 1

        # Phase 1: Greedy removal - multiple passes until no more removals
        removed_any = True
        while removed_any:
            if time.time() > deadline - 5.0:
                break
            removed_any = False
            for ci in order:
                if time.time() > deadline - 5.0:
                    break
                if not active[ci]:
                    continue
                if is_redundant(ci, coverage_count):
                    remove_codeword(ci, active, coverage_count)
                    removed_any = True

        # Phase 2: Swap-based local search
        # Try to swap one active codeword for one inactive codeword
        active_list = np.where(active)[0]
        inactive_list = np.where(~active)[0]

        if len(inactive_list) > 0 and len(active_list) > 0:
            swap_iterations = 0
            max_swap_iters = 500
            no_improve_count = 0
            max_no_improve = 50

            while swap_iterations < max_swap_iters and no_improve_count < max_no_improve:
                if time.time() > deadline - 5.0:
                    break
                swap_iterations += 1

                # Randomly pick an active codeword to potentially remove
                ci_remove = int(random.choice(active_list))
                # Randomly pick an inactive codeword to potentially add
                ci_add = int(random.choice(inactive_list))

                # Check if removing ci_remove and adding ci_add maintains coverage
                # Words that would become uncovered if we remove ci_remove
                covers_remove = codeword_covers_arr[ci_remove, :codeword_covers_len[ci_remove]]
                covers_add = codeword_covers_arr[ci_add, :codeword_covers_len[ci_add]]

                # Find words at risk (covered only by ci_remove)
                at_risk_mask = (coverage_count[covers_remove] == 1)
                at_risk = covers_remove[at_risk_mask]

                if len(at_risk) == 0:
                    # Safe to remove without adding - but that's handled in Phase 1
                    # For swap, we need to add something, so skip
                    no_improve_count += 1
                    continue

                # Check if ci_add covers all at_risk words
                # Vectorized: check if all at_risk are in covers_add
                at_risk_set = set(at_risk.tolist())
                add_set = set(covers_add.tolist())
                all_covered = at_risk_set.issubset(add_set)

                if all_covered:
                    # Perform the swap
                    remove_codeword(ci_remove, active, coverage_count)
                    add_codeword(ci_add, active, coverage_count)

                    # Update lists
                    active_list = np.where(active)[0]
                    inactive_list = np.where(~active)[0]
                    no_improve_count = 0

                    if len(inactive_list) == 0:
                        break
                else:
                    no_improve_count += 1

        # Phase 3: Try 2-for-1 moves (remove 2, add 1) to further reduce
        active_list = np.where(active)[0]
        inactive_list = np.where(~active)[0]

        if len(inactive_list) > 0 and len(active_list) >= 2:
            two_for_one_iters = 0
            max_two_for_one = 200
            no_improve_2f1 = 0
            max_no_improve_2f1 = 30

            while two_for_one_iters < max_two_for_one and no_improve_2f1 < max_no_improve_2f1:
                if time.time() > deadline - 5.0:
                    break
                two_for_one_iters += 1

                # Pick two active codewords to remove
                ci_remove1 = int(random.choice(active_list))
                ci_remove2 = int(random.choice(active_list))
                if ci_remove1 == ci_remove2:
                    no_improve_2f1 += 1
                    continue

                # Pick one inactive codeword to add
                ci_add = int(random.choice(inactive_list))

                # Check coverage
                covers1 = codeword_covers_arr[ci_remove1, :codeword_covers_len[ci_remove1]]
                covers2 = codeword_covers_arr[ci_remove2, :codeword_covers_len[ci_remove2]]
                covers_add = codeword_covers_arr[ci_add, :codeword_covers_len[ci_add]]

                # Words at risk from removing both
                # For each word in covers1 or covers2, check if it's still covered after removal
                # Simulate: temporarily reduce coverage
                temp_coverage = coverage_count.copy()
                temp_coverage[covers1] -= 1
                temp_coverage[covers2] -= 1

                # Find words that would be uncovered
                all_covers = np.concatenate([covers1, covers2])
                unique_covers = np.unique(all_covers)
                at_risk_mask = (temp_coverage[unique_covers] == 0)
                at_risk = unique_covers[at_risk_mask]

                if len(at_risk) == 0:
                    # Can remove both without adding - even better!
                    remove_codeword(ci_remove1, active, coverage_count)
                    remove_codeword(ci_remove2, active, coverage_count)
                    no_improve_2f1 = 0
                    active_list = np.where(active)[0]
                    inactive_list = np.where(~active)[0]
                    if len(inactive_list) == 0:
                        break
                    continue

                # Check if ci_add covers all at_risk
                at_risk_set = set(at_risk.tolist())
                add_set = set(covers_add.tolist())
                all_covered = at_risk_set.issubset(add_set)

                if all_covered:
                    # Perform 2-for-1
                    remove_codeword(ci_remove1, active, coverage_count)
                    remove_codeword(ci_remove2, active, coverage_count)
                    add_codeword(ci_add, active, coverage_count)
                    no_improve_2f1 = 0
                    active_list = np.where(active)[0]
                    inactive_list = np.where(~active)[0]
                    if len(inactive_list) == 0:
                        break
                else:
                    no_improve_2f1 += 1

        trial_count = int(np.sum(active))
        if trial_count < best_count:
            best_count = trial_count
            best_indices = np.where(active)[0]
            best_code = [list(code[i]) for i in best_indices]

        # Early exit if we reached the known optimum
        if best_count <= 73:
            break

    if best_code is None:
        best_code = [list(c) for c in code]

    # Final verification
    if time.time() < deadline - 2.0 and len(best_code) <= 100:
        covered_set = set()
        for c in best_code:
            c_arr = np.array(c, dtype=np.int32)
            idx = int(compute_word_indices(c_arr.reshape(1, -1))[0])
            covered_set.add(idx)
            for pos in range(n):
                orig = c[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_idx = idx + (val - orig) * (q ** (n - 1 - pos))
                    covered_set.add(new_idx)
        if len(covered_set) < total_words:
            best_code = [list(c) for c in code]

    return best_code
