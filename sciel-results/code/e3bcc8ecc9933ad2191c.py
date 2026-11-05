import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Fallback for non-target parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # ---------- Target case: q=3, n=7, radius=1 ----------
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor (ball) lists and sets once
    neighbors_list = []
    neighbor_sets = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))
        neighbor_sets.append(set(nb))

    # ---------- Phase 1: Greedy Set-Cover with vectorized gain computation ----------
    # Gains are accumulated from uncovered points: for each uncovered point,
    # add 1 to all its neighbors. This touches only uncovered points, not all N.
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()

    while np.any(uncovered):
        if time.time() > deadline - 40:
            break

        gains = np.zeros(N, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            gains[neighbors_list[idx]] += 1

        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]
        if max_gain == 0:
            break

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors_list[best_idx]] = False

    # ---------- Phase 2: Local Search ----------
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1

    best_size = len(code)
    best_code = list(code)

    def greedy_repair(uncovered_set):
        """Given a set of uncovered points, greedily add codewords to cover them."""
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)
        while temp_uncovered:
            best_c = -1
            best_count = 0
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbor_sets[w])
            for c in candidate_pool:
                if c in code_set:
                    continue
                cnt = 0
                for x in neighbors_list[c]:
                    if x in temp_uncovered:
                        cnt += 1
                        if cnt > best_count:
                            best_count = cnt
                            best_c = c
                        if best_count == len(temp_uncovered):
                            break
                if best_count == len(temp_uncovered):
                    break
            if best_c == -1:
                break
            added.append(best_c)
            for w in neighbors_list[best_c]:
                temp_uncovered.discard(w)
        return added

    def try_remove():
        """Remove all redundant codewords (all neighbors covered by >1 codeword)."""
        removed = False
        for c in list(code):
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                removed = True
        return removed

    def try_swap(c):
        """Try to swap codeword c with a better-positioned codeword.
        Strategy 1: Find a replacement covering all critical points of c.
        Strategy 2: Relocation - move c to cover would-be-uncovered points.
        Returns True if a swap was made."""
        if np.min(counts[neighbors_list[c]]) > 1:
            return False

        crit_points = [i for i in neighbors_list[c] if counts[i] == 1]
        if not crit_points:
            return False

        # Strategy 1: Find replacement covering all critical points
        found_swap = False
        shuffled_crit = list(crit_points)
        random.shuffle(shuffled_crit)
        for p in shuffled_crit:
            if found_swap:
                break
            cands = [x for x in neighbor_sets[p] if x not in code_set]
            best_w = -1
            best_score = -1
            for w in cands:
                valid = True
                for k in crit_points:
                    if k not in neighbor_sets[w]:
                        valid = False
                        break
                if not valid:
                    continue
                score = int(np.sum(counts[neighbors_list[w]] == 1))
                if score > best_score:
                    best_score = score
                    best_w = w
            if best_w != -1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                code.append(best_w)
                code_set.add(best_w)
                counts[neighbors_list[best_w]] += 1
                found_swap = True
                return True

        # Strategy 2: Relocation
        temp_counts = counts.copy()
        temp_counts[neighbors_list[c]] -= 1
        would_be_uncovered = [i for i in neighbors_list[c] if temp_counts[i] == 0]
        if not would_be_uncovered:
            return False
        all_needed = set(crit_points) | set(would_be_uncovered)
        best_w = -1
        best_score = -1
        first_point = list(all_needed)[0]
        cands = [x for x in neighbor_sets[first_point] if x not in code_set]
        for w in cands:
            valid = True
            for k in all_needed:
                if k not in neighbor_sets[w]:
                    valid = False
                    break
            if not valid:
                continue
            score = int(np.sum(counts[neighbors_list[w]] == 1))
            if score > best_score:
                best_score = score
                best_w = w
        if best_w != -1:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1
            code.append(best_w)
            code_set.add(best_w)
            counts[neighbors_list[best_w]] += 1
            return True
        return False

    # Precompute ball indices for fast deletion/swap passes
    ball_idx = [np.argsort(neighbors_list[i]) for i in range(N)]

    def deletion_pass():
        """Repeatedly remove redundant codewords, tracking size."""
        nonlocal best_size, best_code
        progress = True
        while progress:
            progress = False
            for c in list(code):
                if np.min(counts[neighbors_list[c]]) > 1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbors_list[c]] -= 1
                    progress = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
        return progress

    def kick(size):
        """Perturb the code by removing 'size' random codewords and repairing."""
        nonlocal best_size, best_code
        if len(code) <= size:
            return False
        removed = random.sample(code, size)
        for c in removed:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1
        # Identify uncovered points
        uncovered_pts = [i for i in range(N) if counts[i] == 0]
        if not uncovered_pts:
            return False
        added = greedy_repair(set(uncovered_pts))
        for a in added:
            code.append(a)
            code_set.add(a)
            counts[neighbors_list[a]] += 1
        # Cleanup redundancies
        deletion_pass()
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
        return len(code) <= best_size

    # Main local-search loop
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Deletion pass
        if deletion_pass():
            improved = True
        # Swap pass over a shuffled order
        for c in random.sample(code, len(code)):
            if time.time() > deadline - 10:
                break
            if try_swap(c):
                improved = True
        # Kick to escape local minima (larger kick size 5 helps escape)
        if kick(5):
            improved = True

    # Use the best code found
    final_code = best_code

    # Convert indices to words
    return [idx_to_tuple(i) for i in final_code]
