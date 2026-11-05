"""Ternary covering code K_3(7,1) - greedy + local search with optimized repair."""
import time
import random
import itertools
import numpy as np
from collections import deque


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
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

    # Precompute neighbor lists as numpy arrays
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

            # Build candidate pool from neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbor_sets[w])

            # Find best candidate
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
        """Remove all redundant codewords."""
        removed = False
        for c in list(code):
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                removed = True
        return removed

    def try_swap(c):
        """Try to swap codeword c with a better-positioned codeword (numpy-accelerated)."""
        if np.min(counts[neighbors_list[c]]) > 1:
            return False

        crit_points = [i for i in neighbors_list[c] if counts[i] == 1]

        if not crit_points:
            return False

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

    def try_pair_replace():
        """Try to replace two codewords with one that covers all their critical points."""
        if len(code) < 2:
            return False

        # Find pairs of codewords whose critical points can be covered by a single new codeword
        crit_map = {}
        for c in code:
            cp = [i for i in neighbors_list[c] if counts[i] == 1]
            if cp:
                crit_map[c] = cp

        if len(crit_map) < 2:
            return False

        # Try all pairs (limit to avoid O(n^2) explosion)
        codes_with_crit = list(crit_map.keys())
        random.shuffle(codes_with_crit)
        max_pairs = min(50, len(codes_with_crit) * (len(codes_with_crit) - 1) // 2)
        pairs_tried = 0

        for i in range(len(codes_with_crit)):
            if pairs_tried >= max_pairs:
                break
            c1 = codes_with_crit[i]
            if c1 not in code_set:
                continue
            for j in range(i + 1, len(codes_with_crit)):
                if pairs_tried >= max_pairs:
                    break
                pairs_tried += 1
                c2 = codes_with_crit[j]
                if c2 not in code_set:
                    continue

                all_crit = set(crit_map[c1]) | set(crit_map[c2])

                # Find a codeword w that covers all critical points and is not in code
                first_point = list(all_crit)[0]
                cands = [x for x in neighbor_sets[first_point] if x not in code_set]

                best_w = -1
                best_score = -1

                for w in cands:
                    valid = True
                    for k in all_crit:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Score: how many currently singly-covered points does w cover?
                    score = int(np.sum(counts[neighbors_list[w]] == 1))
                    if score > best_score:
                        best_score = score
                        best_w = w

                if best_w != -1:
                    code.remove(c1)
                    code_set.remove(c1)
                    counts[neighbors_list[c1]] -= 1
                    code.remove(c2)
                    code_set.remove(c2)
                    counts[neighbors_list[c2]] -= 1

                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbors_list[best_w]] += 1
                    return True

        return False

    def kick():
        """Remove a random codeword and try to repair. Accept if size decreases."""
        if not code:
            return False

        old_size = len(code)

        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]

        temp_coverage = counts.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            code.remove(c_remove)
            code_set.remove(c_remove)
            counts[neighbors_list[c_remove]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - 1 + len(added)

        if new_size < old_size:
            code.remove(c_remove)
            code_set.remove(c_remove)
            counts[neighbors_list[c_remove]] -= 1
            for c in added:
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1
            return True
        else:
            return False

    def kick2():
        """Remove two random codewords and try to repair. Accept if size decreases."""
        if len(code) < 2:
            return False

        old_size = len(code)

        i1 = random.randint(0, len(code) - 1)
        i2 = random.randint(0, len(code) - 1)
        if i1 == i2:
            i2 = (i2 + 1) % len(code)

        c1 = code[i1]
        c2 = code[i2]

        temp_coverage = counts.copy()
        temp_coverage[neighbors_list[c1]] -= 1
        temp_coverage[neighbors_list[c2]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]
        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - 2 + len(added)

        if new_size < old_size:
            code.remove(c1)
            code_set.remove(c1)
            counts[neighbors_list[c1]] -= 1
            code.remove(c2)
            code_set.remove(c2)
            counts[neighbors_list[c2]] -= 1
            for c in added:
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1
            return True
        else:
            return False

    def kick3():
        """Remove three codewords (prioritizing redundant ones) and try to repair."""
        if len(code) < 3:
            return False

        old_size = len(code)

        # Score each codeword by redundancy (min count over its neighbors)
        # Prefer removing the most redundant ones
        scored = []
        for c in code:
            min_cnt = int(np.min(counts[neighbors_list[c]]))
            scored.append((min_cnt, c))
        scored.sort(reverse=True)

        # Take top 3 by redundancy, but add some randomness
        top3 = [c for _, c in scored[:3]]
        if random.random() < 0.5 and len(code) > 3:
            # Mix in a random one
            top3[random.randint(0, 2)] = random.choice(code)

        to_remove = top3[:3]

        temp_coverage = counts.copy()
        for c in to_remove:
            temp_coverage[neighbors_list[c]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]
        if len(uncovered) == 0:
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - 3 + len(added)

        if new_size < old_size:
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
            for c in added:
                code.append(c)
                code_set.add(c)
                counts[neighbors_list[c]] += 1
            return True
        else:
            return False

    # ---------- Main Local Search Loop ----------
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False

        # 1. Exhaustive removal pass
        while try_remove():
            improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 15:
            break

        # 2. Swap pass
        shuffled_code = list(code)
        random.shuffle(shuffled_code)

        for c in shuffled_code:
            if time.time() > deadline - 15:
                break
            if c not in code_set:
                continue
            if try_swap(c):
                improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 15:
            break

        # 2.5. Pair replacement pass (2-for-1)
        if try_pair_replace():
            improved = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

        if time.time() > deadline - 15:
            break

        # 3. Kick pass (1-removal)
        kicks_tried = 0
        max_kicks = 60
        while kicks_tried < max_kicks and time.time() < deadline - 15:
            if kick():
                improved = True
                kicks_tried = 0
            else:
                kicks_tried += 1

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 15:
            break

        # 4. 2-removal kick pass
        if len(code) > 2:
            kicks2_tried = 0
            max_kicks2 = 30
            while kicks2_tried < max_kicks2 and time.time() < deadline - 15:
                if kick2():
                    improved = True
                    kicks2_tried = 0
                else:
                    kicks2_tried += 1

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

        if time.time() > deadline - 15:
            break

        # 5. 3-removal kick pass (new - deeper exploration)
        if len(code) > 3:
            kicks3_tried = 0
            max_kicks3 = 15
            while kicks3_tried < max_kicks3 and time.time() < deadline - 15:
                if kick3():
                    improved = True
                    kicks3_tried = 0
                else:
                    kicks3_tried += 1

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

    # If we found a better code, restore it
    if len(best_code) < len(code):
        code = best_code
        code_set = set(best_code)
        counts = np.zeros(N, dtype=np.int32)
        for idx in code:
            counts[neighbors_list[idx]] += 1

    # Final polish: removal pass + swap pass
    if time.time() < deadline - 5:
        for c in list(code):
            if np.min(counts[neighbors_list[c]]) > 1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
        # Try a few swaps in the final polish
        shuffled = list(code)
        random.shuffle(shuffled)
        for c in shuffled[:20]:
            if time.time() > deadline - 2:
                break
            if c not in code_set:
                continue
            try_swap(c)

    # Convert back to list of lists
    result = [idx_to_tuple(cw) for cw in code]

    return result
