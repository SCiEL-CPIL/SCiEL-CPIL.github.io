"""Lagrangian relaxation + subgradient-guided greedy + 1-exchange local search for K_3(7,1)."""
import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n-4)]

    deadline = time.time() + 145.0
    rng = random.Random(12345)

    total = q ** n  # 2187

    # Precompute all words as indices
    # idx_to_word: list of lists
    idx_to_word = []
    for idx in range(total):
        w = []
        x = idx
        for _ in range(n):
            w.append(x % q)
            x //= q
        idx_to_word.append(w)

    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * q + v
        return idx

    # Precompute neighbor lists: for each codeword index, list of word indices it covers (radius 1)
    # neighbor_list[i] = list of indices j such that dist(idx_to_word[i], idx_to_word[j]) <= 1
    neighbor_list = []
    for i in range(total):
        w = idx_to_word[i]
        covered = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_w = w[:]
                    new_w[pos] = new_val
                    idx = word_to_idx(new_w)
                    covered.append(idx)
        neighbor_list.append(covered)

    # Convert to numpy arrays for speed
    # adj_matrix: total x total boolean, adj_matrix[i, j] = True if codeword i covers word j
    adj_matrix = np.zeros((total, total), dtype=bool)
    for i in range(total):
        for j in neighbor_list[i]:
            adj_matrix[i, j] = True

    # Also precompute reverse: for each word j, which codewords cover it
    # rev_matrix[j, i] = adj_matrix[i, j]
    rev_matrix = adj_matrix.T  # shape (total, total), rev_matrix[j, i] = True if codeword i covers word j

    def is_valid(chosen):
        """Check if chosen set of codeword indices covers all words."""
        if not chosen:
            return False
        covered = np.zeros(total, dtype=bool)
        for c in chosen:
            covered |= adj_matrix[c]
        return covered.all()

    def count_covered(chosen):
        """Return number of covered words."""
        if not chosen:
            return 0
        covered = np.zeros(total, dtype=bool)
        for c in chosen:
            covered |= adj_matrix[c]
        return covered.sum()

    # Lagrangian relaxation approach:
    # Primal: min sum(x_i) s.t. sum_{i in N(j)} x_i >= 1 for all j, x_i in {0,1}
    # Lagrangian: L(x, u) = sum(x_i) + sum_j u_j * (1 - sum_{i in N(j)} x_i)
    #             = sum_j u_j + sum_i x_i * (1 - sum_{j in N(i)} u_j)
    # For a given u, the Lagrangian is minimized by setting x_i = 1 only if (1 - sum_{j in N(i)} u_j) < 0
    # i.e., x_i = 1 if sum_{j in N(i)} u_j > 1
    #
    # Subgradient update: u_j += step * (1 - sum_{i in N(j)} x_i) for each j
    # where x is the solution to the Lagrangian relaxation for current u.
    #
    # After convergence or max iterations, we get a set of codewords.
    # Then we do local search (1-exchange) to improve.

    def lagrangian_greedy(max_iters=500, step_init=0.5):
        """Run Lagrangian relaxation with subgradient updates."""
        u = np.zeros(total, dtype=float)
        step = step_init
        best_chosen = None
        best_count = total + 1

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Compute reduced cost for each codeword: rc_i = 1 - sum_{j in N(i)} u_j
            # If rc_i < 0, codeword i is "profitable" to select
            # Sum over neighbors: for codeword i, sum_{j in N(i)} u_j
            # N(i) is the set of words covered by codeword i, which is neighbor_list[i]
            # So sum_{j in N(i)} u_j = sum of u[j] for j in neighbor_list[i]
            # We can compute this as: for each i, sum u over adj_matrix[i, :]
            # Using matrix-vector product: rc = 1 - adj_matrix @ u
            rc = 1.0 - adj_matrix @ u

            # Select codewords with rc < 0
            selected = np.where(rc < 0)[0].tolist()

            if not selected:
                # Fallback: select codeword with most negative rc
                min_idx = np.argmin(rc)
                selected = [int(min_idx)]

            # Check coverage
            covered = np.zeros(total, dtype=bool)
            for c in selected:
                covered |= adj_matrix[c]
            num_covered = covered.sum()

            if num_covered == total:
                # Full coverage achieved
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_chosen = selected[:]

            # Subgradient update
            # sg_j = 1 - sum_{i in N(j)} x_i where x_i = 1 if i in selected
            # sum_{i in N(j)} x_i = number of selected codewords that cover word j
            # = sum of x_i over i where adj_matrix[i, j] = True
            # = (adj_matrix @ x)_j where x is indicator of selected
            x = np.zeros(total, dtype=float)
            for c in selected:
                x[c] = 1.0
            coverage_count = adj_matrix @ x  # coverage_count[j] = num selected codewords covering word j
            sg = 1.0 - coverage_count

            # Step size: use diminishing step
            # step = step_init / (it + 1)
            u += step * sg
            u = np.maximum(u, 0.0)  # Keep dual variables non-negative
            step = step_init / (it + 1)

            # Occasionally, if coverage is full, try to reduce
            if num_covered == total and len(selected) < best_count:
                best_count = len(selected)
                best_chosen = selected[:]

        return best_chosen if best_chosen is not None else None

    # Initial solution: greedy set cover (fast, gives a valid starting point)
    def greedy_set_cover():
        """Standard greedy set cover."""
        uncovered = np.ones(total, dtype=bool)
        num_uncovered = total
        chosen = []
        chosen_set = set()

        # Initial marginal coverage for each candidate
        marginal = adj_matrix.sum(axis=1).astype(int)  # marginal[i] = num words covered by codeword i

        while num_uncovered > 0 and time.time() < deadline:
            # Find candidate with max marginal coverage
            # Among those not yet chosen
            best_idx = -1
            best_val = -1
            for i in range(total):
                if i in chosen_set:
                    continue
                if marginal[i] > best_val:
                    best_val = marginal[i]
                    best_idx = i

            if best_idx == -1 or best_val == 0:
                break

            chosen.append(best_idx)
            chosen_set.add(best_idx)

            # Update marginal coverage
            for j in neighbor_list[best_idx]:
                if uncovered[j]:
                    num_uncovered -= 1
                    uncovered[j] = False
                    # Decrease marginal for all candidates that cover j
                    # Candidates that cover j are those i where adj_matrix[i, j] = True
                    covering_i = np.where(rev_matrix[j])[0]
                    for i in covering_i:
                        if i != best_idx:
                            marginal[i] -= 1

        return chosen

    # Local search: 1-exchange moves
    def local_search(chosen, max_iters=200):
        """Try 1-exchange moves: remove one codeword, add another if it improves coverage or reduces count."""
        chosen = list(chosen)
        chosen_set = set(chosen)
        improved = True

        while improved and time.time() < deadline:
            improved = False
            # Try removing each codeword and see if we can add one to maintain coverage
            for i in range(len(chosen) - 1, -1, -1):
                if time.time() > deadline:
                    break
                # Try removing chosen[i]
                test_chosen = chosen[:i] + chosen[i+1:]
                test_set = set(test_chosen)

                # Check if coverage is maintained
                if test_set:
                    covered = np.zeros(total, dtype=bool)
                    for c in test_set:
                        covered |= adj_matrix[c]
                    if covered.all():
                        # Removal is valid
                        chosen = test_chosen
                        chosen_set = test_set
                        improved = True
                        break
                else:
                    # Empty set, not valid unless total == 0
                    pass

            # Try 1-exchange: remove one, add one
            if not improved:
                for i in range(len(chosen)):
                    if time.time() > deadline:
                        break
                    # Try removing chosen[i] and adding each other codeword
                    test_chosen = chosen[:i] + chosen[i+1:]
                    test_set = set(test_chosen)

                    # Find words not covered by test_set
                    if test_set:
                        covered = np.zeros(total, dtype=bool)
                        for c in test_set:
                            covered |= adj_matrix[c]
                        uncovered_words = np.where(~covered)[0]
                    else:
                        uncovered_words = np.arange(total)

                    if len(uncovered_words) == 0:
                        # Already covered, just remove
                        chosen = test_chosen
                        chosen_set = test_set
                        improved = True
                        break

                    # Try adding each candidate to cover the uncovered words
                    best_add = -1
                    best_gain = -1
                    for cand in range(total):
                        if cand in test_set:
                            continue
                        # How many uncovered words does cand cover?
                        gain = 0
                        for uw in uncovered_words:
                            if adj_matrix[cand, uw]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_add = cand

                    if best_add != -1 and best_gain == len(uncovered_words):
                        # Adding best_add covers all uncovered words
                        chosen = test_chosen + [best_add]
                        chosen_set = set(chosen)
                        improved = True
                        break

        return chosen

    # Main algorithm:
    # 1. Get initial solution via greedy
    initial_chosen = greedy_set_cover()

    # 2. Run Lagrangian relaxation to get a candidate solution
    lag_chosen = lagrangian_greedy(max_iters=300, step_init=0.5)

    # 3. Combine: take the better of initial and lagrangian, then local search
    candidates = []
    if initial_chosen and is_valid(initial_chosen):
        candidates.append(initial_chosen)
    if lag_chosen and is_valid(lag_chosen):
        candidates.append(lag_chosen)

    if not candidates:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n-4)]

    # Sort by length
    candidates.sort(key=len)
    best = candidates[0]

    # 4. Local search on the best candidate
    best = local_search(best, max_iters=500)

    # 5. Try more Lagrangian runs with different parameters
    for step_init in [0.3, 0.7, 1.0]:
        if time.time() > deadline:
            break
        lag_chosen = lagrangian_greedy(max_iters=200, step_init=step_init)
        if lag_chosen and is_valid(lag_chosen):
            if len(lag_chosen) < len(best):
                best = local_search(lag_chosen, max_iters=200)

    # 6. Final verification
    if not is_valid(best):
        # Fallback to greedy
        best = greedy_set_cover()
        best = local_search(best, max_iters=200)

    # Convert to codewords
    codewords = [idx_to_word[i] for i in best]

    return codewords
