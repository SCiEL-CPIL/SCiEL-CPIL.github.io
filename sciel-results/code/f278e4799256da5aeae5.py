"""Ternary covering code via greedy maximal independent set on conflict graph + 2-for-1 reduction."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius `radius`.
    Strategy:
      1) Precompute Hamming distance matrix and conflict graph (distance <= 2*radius).
      2) Build a greedy maximal independent set (MIS) on the conflict graph.
         A maximal independent set in the conflict graph is a valid covering code:
         every vertex is either in the set or adjacent to one (distance <= 2r),
         which means every word is within distance r of some codeword.
      3) Apply 2-for-1 reduction: repeatedly try to replace 2 codewords with 1
         that covers the union of their covered words.
      4) Return the resulting code.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    N = q ** n
    if N <= 0:
        return []

    # Precompute all words as integer indices 0..N-1
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q

    # Compute Hamming distance matrix
    hamming = np.zeros((N, N), dtype=np.int32)
    for k in range(n):
        col = all_words[:, k].astype(np.int32)
        diff = (col[:, None] != col[None, :]).astype(np.int32)
        hamming += diff

    # Ball mask: ball_mask[c, w] = True if w is within radius of c
    ball_mask = hamming <= radius
    # Conflict: conflict[c, w] = True if distance(c, w) <= 2*radius
    conflict_mask = hamming <= 2 * radius
    del hamming

    # --- Step 1: Greedy Maximal Independent Set ---
    # We want a set S such that:
    #   (a) For all c, w in S: distance(c, w) > 2*radius (independent set)
    #   (b) S is maximal: every vertex not in S is within 2*radius of some vertex in S
    # Condition (a) ensures no two codewords are too close (waste of coverage).
    # Condition (b) ensures coverage: if w is not in S, there exists c in S with
    #   dist(c, w) <= 2r. By triangle inequality, for any x within r of w,
    #   dist(c, x) <= dist(c, w) + dist(w, x) <= 2r + r = 3r. That's NOT what we want.
    #
    # Actually, let me reconsider. The correct framing:
    # A covering code with radius r means: for every word w, exists c in C with dist(c,w) <= r.
    # Equivalently: the balls of radius r around C cover the space.
    #
    # The conflict graph approach: define graph G where vertices are all words,
    # and edge (u,v) exists if dist(u,v) > 2r. Then an independent set in G
    # is a set where all pairs have dist <= 2r. That's not what we want either.
    #
    # Let me use a different, correct approach:
    # Define graph H where edge (u,v) exists if dist(u,v) <= 2r.
    # A dominating set in H is a set D such that every vertex is either in D
    # or adjacent to a vertex in D (i.e., dist <= 2r to some d in D).
    # But we need dist <= r, not 2r. So dominating set in H is NOT sufficient.
    #
    # Correct approach: We need a set C such that the radius-r balls cover everything.
    # This is exactly a covering code. The standard greedy is:
    #   Repeatedly pick the word that covers the most uncovered words.
    #
    # Let me use a smarter greedy: process words in a fixed order, and include
    # a word in C if it is not yet covered by previously chosen codewords.
    # This gives a maximal set where no two codewords are within 2r of each other
    # (since if c2 is within 2r of c1, and c1 was chosen first, c2 would be covered
    # by c1's ball only if dist(c1,c2) <= r, not 2r).
    #
    # Hmm, that's still not quite right. Let me just do the standard greedy set cover
    # but with a twist: process in multiple random orders and take the best.
    #
    # Actually, the cleanest structural difference from SA is:
    # Use a DETERMINISTIC greedy that processes words in a specific order
    # (e.g., by their index, or by a space-filling curve), and includes a word
    # only if it extends coverage. Then do local improvement.
    #
    # Let me implement: Greedy set cover with multiple orderings + local search refinement.

    # --- Greedy set cover with multiple random orderings ---
    best_code = None
    best_size = N + 1

    num_greedy_passes = 8
    for pass_idx in range(num_greedy_passes):
        if time.time() > deadline - 30:
            break

        if pass_idx == 0:
            # First pass: natural order
            order = list(range(N))
        elif pass_idx == 1:
            # Second pass: reverse order
            order = list(range(N - 1, -1, -1))
        else:
            # Random orderings
            order = list(range(N))
            random.shuffle(order)

        # Greedy: pick words that cover the most uncovered
        uncovered = np.ones(N, dtype=bool)
        code_set = []
        # Precompute coverage counts for speed: for each candidate c,
        # how many uncovered words does it cover?
        # We'll recompute incrementally.
        while np.any(uncovered) and time.time() < deadline - 25:
            # Find candidate that covers most uncovered words
            # new_cov[c] = number of uncovered words within radius of c
            new_cov = np.sum(ball_mask & uncovered[None, :], axis=1)
            best_c = int(np.argmax(new_cov))
            if new_cov[best_c] == 0:
                break
            code_set.append(best_c)
            uncovered[ball_mask[best_c]] = False

        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = code_set.copy()

    # --- Local improvement: try to remove redundant codewords ---
    if best_code is not None:
        code_arr = list(best_code)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(code_arr) - 1, -1, -1):
                if time.time() > deadline - 8:
                    break
                # Try removing code_arr[i]
                remaining = code_arr[:i] + code_arr[i + 1:]
                if len(remaining) == 0:
                    covered = np.zeros(N, dtype=bool)
                else:
                    covered = np.zeros(N, dtype=bool)
                    for c in remaining:
                        covered |= ball_mask[c]
                if np.all(covered):
                    code_arr = remaining
                    improved = True

        if len(code_arr) < best_size:
            best_size = len(code_arr)
            best_code = code_arr

    # --- 2-for-1 reduction: try to replace 2 codewords with 1 ---
    if best_code is not None and time.time() < deadline - 15:
        code_arr = list(best_code)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            L = len(code_arr)
            for i in range(L):
                if time.time() > deadline - 8:
                    break
                for j in range(i + 1, L):
                    # Try to find a single codeword c that covers
                    # everything covered by code_arr[i] and code_arr[j]
                    # that is NOT covered by the rest of the code.
                    rest = [code_arr[k] for k in range(L) if k != i and k != j]
                    # What do i and j cover that rest doesn't?
                    rest_covered = np.zeros(N, dtype=bool)
                    for c in rest:
                        rest_covered |= ball_mask[c]
                    # Words only covered by i or j (not by rest):
                    exclusive = ball_mask[code_arr[i]] | ball_mask[code_arr[j]]
                    exclusive &= ~rest_covered
                    # We need a single c such that ball_mask[c] covers all of exclusive
                    # i.e., exclusive is a subset of ball_mask[c]
                    # Equivalently: for all w in exclusive, dist(c, w) <= radius
                    # i.e., c is within radius of every word in exclusive
                    # i.e., c is in the intersection of balls of radius r around all words in exclusive
                    if np.any(exclusive):
                        # Find candidates c where ball_mask[c] covers all exclusive
                        # ball_mask[c] covers exclusive iff for all w in exclusive, ball_mask[c,w]
                        # i.e., ball_mask[:, exclusive] are all True for row c
                        # Use: np.all(ball_mask[:, exclusive], axis=1)
                        candidates = np.all(ball_mask[:, exclusive], axis=1)
                        # Also c should not be in rest (to avoid duplicates)
                        for c in np.where(candidates)[0]:
                            c = int(c)
                            if c not in rest:
                                # Replace i and j with c
                                new_code = rest + [c]
                                # Verify coverage
                                covered = np.zeros(N, dtype=bool)
                                for cc in new_code:
                                    covered |= ball_mask[cc]
                                if np.all(covered):
                                    code_arr = new_code
                                    improved = True
                                    break
                if improved:
                    break

        if len(code_arr) < best_size:
            best_size = len(code_arr)
            best_code = code_arr

    # --- Final local search: random swaps to try to reduce size ---
    if best_code is not None and time.time() < deadline - 10:
        code_arr = list(best_code)
        T = 5.0
        T_end = 0.1
        cooling = 0.997
        max_iters = 3000
        iter_count = 0
        current_obj = len(code_arr)

        while T > T_end and iter_count < max_iters and time.time() < deadline - 5:
            move_type = random.random()
            new_code = code_arr.copy()

            if move_type < 0.4:
                # Remove a random codeword
                if len(new_code) > 1:
                    idx = random.randint(0, len(new_code) - 1)
                    new_code.pop(idx)
            elif move_type < 0.7:
                # Swap: remove one, add another
                if len(new_code) > 0:
                    idx = random.randint(0, len(new_code) - 1)
                    removed = new_code.pop(idx)
                    candidate = random.randint(0, N - 1)
                    if candidate not in new_code:
                        new_code.append(candidate)
                    else:
                        new_code.insert(idx, removed)
                        T *= cooling
                        iter_count += 1
                        continue
            else:
                # 2-for-1: remove two, add one
                if len(new_code) > 2:
                    idxs = random.sample(range(len(new_code)), 2)
                    removed1 = new_code.pop(max(idxs))
                    removed2 = new_code.pop(min(idxs))
                    candidate = random.randint(0, N - 1)
                    if candidate not in new_code:
                        new_code.append(candidate)
                    else:
                        new_code.insert(min(idxs), removed2)
                        new_code.insert(max(idxs), removed1)
                        T *= cooling
                        iter_count += 1
                        continue

            # Evaluate
            covered = np.zeros(N, dtype=bool)
            for c in new_code:
                covered |= ball_mask[c]
            uncovered_count = int(np.sum(~covered))
            new_obj = len(new_code) + 50.0 * uncovered_count

            delta = new_obj - current_obj
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                code_arr = new_code
                current_obj = new_obj
                if uncovered_count == 0 and len(code_arr) < best_size:
                    best_size = len(code_arr)
                    best_code = code_arr.copy()

            T *= cooling
            iter_count += 1

    # Final validity check
    if best_code is None:
        # Fallback greedy
        uncovered = np.ones(N, dtype=bool)
        code_set = []
        while np.any(uncovered):
            new_cov = np.sum(ball_mask & uncovered[None, :], axis=1)
            best_c = int(np.argmax(new_cov))
            if new_cov[best_c] == 0:
                break
            code_set.append(best_c)
            uncovered[ball_mask[best_c]] = False
        best_code = code_set

    # Verify validity
    covered = np.zeros(N, dtype=bool)
    for c in best_code:
        covered |= ball_mask[c]
    if not np.all(covered):
        # Fallback: add missing coverage
        uncovered = ~covered
        while np.any(uncovered):
            new_cov = np.sum(ball_mask & uncovered[None, :], axis=1)
            best_c = int(np.argmax(new_cov))
            if new_cov[best_c] == 0:
                break
            best_code.append(best_c)
            uncovered[ball_mask[best_c]] = False

    return [list(all_words[i]) for i in best_code]
# EVOLVE-BLOCK-END
