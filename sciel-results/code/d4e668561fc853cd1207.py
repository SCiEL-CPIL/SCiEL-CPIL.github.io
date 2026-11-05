"""Combined program: vectorized greedy + enhanced local search for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

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

    # Precompute neighbor lists as numpy arrays of int32 for fast operations
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

    # Precompute a lookup: for each word, which codewords can cover it
    # i.e., can_cover[word] = set of codeword indices that cover word
    # Actually, for each word w, the codewords that cover w are those in neighbor_sets[w]
    # So can_cover[w] = neighbor_sets[w] (already have this)

    # ---------- Phase 1: Greedy Set-Cover with vectorized gain computation ----------
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()

    while np.any(uncovered):
        if time.time() > deadline - 30:
            break

        # Compute gains: for each candidate codeword, how many uncovered points does it cover?
        gains = np.zeros(N, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            gains[neighbors_list[idx]] += 1

        max_gain = int(np.max(gains))
        if max_gain == 0:
            break

        # Randomize tie-breaking to explore different solutions
        best_indices = np.where(gains == max_gain)[0]
        if len(best_indices) > 1:
            best_idx = int(random.choice(best_indices))
        else:
            best_idx = int(best_indices[0])

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
            if time.time() > deadline - 5:
                break
            
            gains = np.zeros(N, dtype=np.int32)
            for idx in temp_uncovered:
                gains[neighbors_list[idx]] += 1
            
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
            
            best_indices = np.where(gains == max_gain)[0]
            if len(best_indices) > 1:
                best_idx = int(random.choice(best_indices))
            else:
                best_idx = int(best_indices[0])
            
            added.append(best_idx)
            
            for w in neighbors_list[best_idx]:
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
        """Try to swap codeword c with a better-positioned codeword."""
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

    def kick():
        """Remove a random codeword and try to repair."""
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
        """Remove two random codewords and try to repair."""
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
        """Remove three random codewords and try to repair."""
        if len(code) < 3:
            return False

        old_size = len(code)

        indices = random.sample(range(len(code)), 3)
        c1, c2, c3 = code[indices[0]], code[indices[1]], code[indices[2]]

        temp_coverage = counts.copy()
        temp_coverage[neighbors_list[c1]] -= 1
        temp_coverage[neighbors_list[c2]] -= 1
        temp_coverage[neighbors_list[c3]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]
        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)

        new_size = old_size - 3 + len(added)

        if new_size < old_size:
            for c in [c1, c2, c3]:
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
    while improved and time.time() < deadline - 10:
        improved = False

        while try_remove():
            improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 10:
            break

        shuffled_code = list(code)
        random.shuffle(shuffled_code)

        for c in shuffled_code:
            if time.time() > deadline - 10:
                break
            if c not in code_set:
                continue
            if try_swap(c):
                improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 10:
            break

        kicks_tried = 0
        max_kicks = 30
        while kicks_tried < max_kicks and time.time() < deadline - 10:
            if kick():
                improved = True
                kicks_tried = 0
            else:
                kicks_tried += 1

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        if time.time() > deadline - 10:
            break

        if len(code) > 2:
            kicks2_tried = 0
            max_kicks2 = 15
            while kicks2_tried < max_kicks2 and time.time() < deadline - 10:
                if kick2():
                    improved = True
                    kicks2_tried = 0
                else:
                    kicks2_tried += 1

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

        if len(code) > 3:
            kicks3_tried = 0
            max_kicks3 = 8
            while kicks3_tried < max_kicks3 and time.time() < deadline - 10:
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

    # Convert back to list of lists
    result = [idx_to_tuple(cw) for cw in code]

    return result
# EVOLVE-BLOCK-END
