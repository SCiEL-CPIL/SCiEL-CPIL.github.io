"""Combined program: vectorized greedy + MIS-style min-degree removal + swap local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before the 150s wall-clock kill

    # The concrete task: ternary covering code K_3(8,1).
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback: direct sum of the ternary tetracode (perfect code, radius 1) with
    # every word on the remaining coordinates -> 9 * 3**(n-4) words.
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    # Generic fallback (small spaces only).
    if q ** n <= 200000:
        total = q ** n
        weights = [q ** (n - 1 - i) for i in range(n)]

        def neighbors(w):
            out = [w]
            for pos in range(n):
                wt = weights[pos]
                digit = (w // wt) % q
                for nd in range(q):
                    if nd != digit:
                        out.append(w + (nd - digit) * wt)
            return out

        nb = [neighbors(w) for w in range(total)]
        cov = np.zeros(total, dtype=np.int32)
        sel = []
        for w in range(total):
            if cov[w] == 0:
                sel.append(w)
                cov[np.array(nb[w])] += 1
        return [[(w // weights[i]) % q for i in range(n)] for w in sel]

    return []


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute the 17-word ball (center + 16 neighbors) for every word.
    balls = np.zeros((total, 17), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            wt = weights[pos]
            digit = (w // wt) % 3
            for nd in range(q):
                if nd == digit:
                    continue
                balls[w, idx] = w + (nd - digit) * wt
                idx += 1

    # Ball membership: ball_membership[c, w] = True iff w is in ball(c).
    ball_membership = np.zeros((total, total), dtype=bool)
    for c in range(total):
        ball_membership[c, balls[c]] = True

    best_size = total
    best_set = None

    def greedy_init():
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while uncovered_mask.any() and time.time() < deadline - 6.0:
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = counts.max()
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            pick = candidates[np.random.randint(len(candidates))]
            selected.append(int(pick))
            cover_count[balls[pick]] += 1
            uncovered_mask = (cover_count == 0)
        return selected

    def local_search(codewords):
        current = list(dict.fromkeys(codewords))  # dedupe, keep order
        if not current:
            return []
        current_set = set(current)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current:
            cover_count[balls[c]] += 1

        improved = True
        while improved and time.time() < deadline - 6.0:
            improved = False
            # 1. Vectorized Removal Check
            curr_arr = np.array(current)
            if len(curr_arr) == 0:
                break
            unique_mask = (cover_count == 1)
            # Count unique words covered by each current codeword
            unique_counts = unique_mask[balls[curr_arr, :]].sum(axis=1)
            removable_indices = np.where(unique_counts == 0)[0]
            
            if len(removable_indices) > 0:
                # Remove all removable codewords at once
                for idx in removable_indices:
                    c = int(curr_arr[idx])
                    current_set.discard(c)
                    current.remove(c)
                    cover_count[balls[c]] -= 1
                improved = True
                continue # Restart loop to re-evaluate

            # 2. Swap Search (Sequential but optimized)
            random.shuffle(current)
            curr_arr = np.array(current)
            for c in current:
                if time.time() > deadline - 6.0:
                    break
                b = balls[c]
                U_c = b[cover_count[b] == 1]
                if U_c.size == 0:
                    # Should be caught above, but safety check
                    current_set.discard(c)
                    current.remove(c)
                    cover_count[b] -= 1
                    improved = True
                    break
                
                # Vectorized candidate search
                # ball_membership[:, U_c] is shape (total, len(U_c))
                # .all(axis=1) checks if all are True
                candidates = np.where(ball_membership[:, U_c].all(axis=1))[0]
                if c in candidates:
                    candidates = candidates[candidates != c]
                
                if len(candidates) > 0:
                    c_prime = int(candidates[np.random.randint(len(candidates))])
                    current_set.discard(c)
                    current.remove(c)
                    cover_count[b] -= 1
                    
                    current_set.add(c_prime)
                    current.append(c_prime)
                    cover_count[balls[c_prime]] += 1
                    
                    improved = True
                    break
                    
        return list(current_set)

    num_restarts = 40
    for _ in range(num_restarts):
        if time.time() > deadline - 8.0:
            break
        selected = greedy_init()
        if not selected:
            continue
        refined = local_search(selected)
        if refined and len(refined) < best_size:
            best_size = len(refined)
            best_set = refined

    if best_set is None:
        best_set = list(range(total))

    result = []
    for w in best_set:
        digits = []
        t = w
        for _ in range(n):
            digits.append(t % 3)
            t //= 3
        digits.reverse()
        result.append(digits)
    return result
# EVOLVE-BLOCK-END
