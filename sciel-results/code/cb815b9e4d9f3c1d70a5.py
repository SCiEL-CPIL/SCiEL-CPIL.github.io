# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    if n == 0:
        return [[]]

    total_words = q ** n

    # For the specific hard case q=3, n=6, radius=1, use specialized group-structured ILS
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Generic fallback for other cases
    if total_words > 10000:
        return _generic_construct(q, n, radius, deadline)

    # Generic bitset-based solver for small spaces
    return _generic_bitset_construct(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = 729

    # Precompute neighbors as bitsets
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}

    # Precompute neighbor lists and bitset masks
    neighbor_indices = [[] for _ in range(total_words)]
    neighbors = [0] * total_words
    for i, t in enumerate(all_tuples):
        mask = 0
        mask |= (1 << i)
        neighbor_indices[i].append(i)
        for pos in range(n):
            orig = t[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                t_list = list(t)
                t_list[pos] = new_val
                nt = tuple(t_list)
                j = idx_map[nt]
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask

    full_mask = (1 << total_words) - 1

    # Precompute perturbation vectors in Z_3^6 for structured swaps
    # Single-coordinate changes
    perturbations_single = []
    for pos in range(n):
        for delta in (1, 2):
            p = [0] * n
            p[pos] = delta
            perturbations_single.append(p)

    # Two-coordinate changes
    perturbations_double = []
    for pos1 in range(n):
        for pos2 in range(pos1 + 1, n):
            for d1 in (1, 2):
                for d2 in (1, 2):
                    p = [0] * n
                    p[pos1] = d1
                    p[pos2] = d2
                    perturbations_double.append(p)

    def get_uncovered_mask(cover_count):
        mask = 0
        for i in range(total_words):
            if cover_count[i] == 0:
                mask |= (1 << i)
        return mask

    def tetracode_seed():
        """
        Start from the known 81-word tetracode direct-sum construction.
        Tetracode: 9 words of length 4, covering radius 1.
        Direct sum with all tails of length 2: 9 * 9 = 81 words.
        """
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code_set = set()
        cover_count = [0] * total_words
        for tail_a in range(3):
            for tail_b in range(3):
                for h in tetracode:
                    word = h + [tail_a, tail_b]
                    idx = idx_map[tuple(word)]
                    code_set.add(idx)
                    for nb in neighbor_indices[idx]:
                        cover_count[nb] += 1
        return code_set, cover_count

    def greedy_start():
        code_set = set()
        cover_count = [0] * total_words
        uncovered_mask = full_mask

        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i

            if best_idx == -1 or best_cnt == 0:
                break

            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]

        return code_set, cover_count, uncovered_mask

    def reduce_code(code_set, cover_count):
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                redundant = True
                for idx in neighbor_indices[c]:
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    code_set.remove(c)
                    for idx in neighbor_indices[c]:
                        cover_count[idx] -= 1
                    changed = True
                    break
        return code_set, cover_count

    def repair(code_set, cover_count, uncovered_mask):
        while uncovered_mask:
            best_idx = -1
            best_cnt = -1
            for i in range(total_words):
                if i in code_set:
                    continue
                new_covered = neighbors[i] & uncovered_mask
                cnt = new_covered.bit_count()
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_idx = i
            if best_idx == -1 or best_cnt == 0:
                break
            code_set.add(best_idx)
            for idx in neighbor_indices[best_idx]:
                cover_count[idx] += 1
            uncovered_mask &= ~neighbors[best_idx]
        return code_set, cover_count, uncovered_mask

    def local_search_1exchange(code_set, cover_count, rng, max_iters=200):
        """
        Try 1-exchange moves: replace codeword c with c+d for small d.
        Accept if the code remains valid and size doesn't increase.
        """
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            c = rng.choice(list(code_set))
            d = rng.choice(perturbations_single + perturbations_double)
            c_tuple = all_tuples[c]
            new_tuple = tuple((c_tuple[i] + d[i]) % q for i in range(n))
            new_idx = idx_map[new_tuple]

            if new_idx in code_set:
                continue

            # Try swap
            for idx in neighbor_indices[c]:
                cover_count[idx] -= 1
            code_set.remove(c)

            code_set.add(new_idx)
            for idx in neighbor_indices[new_idx]:
                cover_count[idx] += 1

            # Check validity using uncovered mask
            uncovered = 0
            for i in range(total_words):
                if cover_count[i] == 0:
                    uncovered |= (1 << i)
                    break

            if uncovered == 0:
                improved = True
            else:
                # Revert
                code_set.remove(new_idx)
                for idx in neighbor_indices[new_idx]:
                    cover_count[idx] -= 1
                code_set.add(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] += 1

        return code_set, cover_count, improved

    def local_search_2exchange(code_set, cover_count, rng, max_iters=50):
        """
        Try 2-exchange moves: remove two codewords, add two new ones.
        """
        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if len(code_set) < 2:
                break
            c1, c2 = rng.sample(list(code_set), 2)
            d1 = rng.choice(perturbations_single)
            d2 = rng.choice(perturbations_single)
            c1_tuple = all_tuples[c1]
            c2_tuple = all_tuples[c2]
            new1 = tuple((c1_tuple[i] + d1[i]) % q for i in range(n))
            new2 = tuple((c2_tuple[i] + d2[i]) % q for i in range(n))
            new_idx1 = idx_map[new1]
            new_idx2 = idx_map[new2]

            if new_idx1 in code_set or new_idx2 in code_set or new_idx1 == new_idx2:
                continue

            # Try swap
            for idx in neighbor_indices[c1]:
                cover_count[idx] -= 1
            for idx in neighbor_indices[c2]:
                cover_count[idx] -= 1
            code_set.remove(c1)
            code_set.remove(c2)

            code_set.add(new_idx1)
            code_set.add(new_idx2)
            for idx in neighbor_indices[new_idx1]:
                cover_count[idx] += 1
            for idx in neighbor_indices[new_idx2]:
                cover_count[idx] += 1

            # Check validity
            valid = True
            for i in range(total_words):
                if cover_count[i] == 0:
                    valid = False
                    break

            if valid:
                improved = True
            else:
                # Revert
                code_set.remove(new_idx1)
                code_set.remove(new_idx2)
                for idx in neighbor_indices[new_idx1]:
                    cover_count[idx] -= 1
                for idx in neighbor_indices[new_idx2]:
                    cover_count[idx] -= 1
                code_set.add(c1)
                code_set.add(c2)
                for idx in neighbor_indices[c1]:
                    cover_count[idx] += 1
                for idx in neighbor_indices[c2]:
                    cover_count[idx] += 1

        return code_set, cover_count, improved

    # Main ILS Loop
    rng = random.Random(42)

    # Start from tetracode direct-sum seed (81 words)
    code_set, cover_count = tetracode_seed()
    code_set, cover_count = reduce_code(code_set, cover_count)
    uncovered_mask = get_uncovered_mask(cover_count)

    best_size = len(code_set)
    best_code = list(code_set)

    # ILS with structured perturbations
    iteration = 0
    while time.time() < deadline - 2:
        iteration += 1

        # Local search first to settle
        code_set, cover_count, _ = local_search_1exchange(code_set, cover_count, rng, max_iters=30)

        # Perturb: structured swap or random removal
        r = rng.random()
        if r < 0.4:
            # 1-exchange perturbation
            code_set, cover_count, _ = local_search_1exchange(code_set, cover_count, rng, max_iters=20)
        elif r < 0.6:
            # 2-exchange perturbation
            code_set, cover_count, _ = local_search_2exchange(code_set, cover_count, rng, max_iters=10)
        else:
            # Random removal of k codewords
            k = rng.randint(2, 4)
            current_code = list(code_set)
            if len(current_code) < k:
                k = len(current_code)
            to_remove = rng.sample(current_code, k)
            for c in to_remove:
                code_set.remove(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] -= 1
            uncovered_mask = get_uncovered_mask(cover_count)
            code_set, cover_count, uncovered_mask = repair(code_set, cover_count, uncovered_mask)
            code_set, cover_count = reduce_code(code_set, cover_count)

        # Check improvement
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            if best_size == 73:
                break

    result = [list(all_tuples[i]) for i in best_code]
    return result

def _generic_bitset_construct(q, n, radius, deadline):
    total_words = q ** n
    all_tuples = list(itertools.product(range(q), repeat=n))
    idx_map = {t: i for i, t in enumerate(all_tuples)}

    neighbors = [0] * total_words
    neighbor_indices = [[] for _ in range(total_words)]
    for i, t in enumerate(all_tuples):
        mask = 0
        for j, t_j in enumerate(all_tuples):
            dist = sum(1 for k in range(n) if t[k] != t_j[k])
            if dist <= radius:
                mask |= (1 << j)
                neighbor_indices[i].append(j)
        neighbors[i] = mask

    full_mask = (1 << total_words) - 1

    code_set = set()
    cover_count = [0] * total_words
    uncovered_mask = full_mask

    while uncovered_mask and time.time() < deadline - 10:
        best_idx = -1
        best_cnt = -1
        for i in range(total_words):
            if i in code_set:
                continue
            new_covered = neighbors[i] & uncovered_mask
            cnt = new_covered.bit_count()
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = i
        if best_idx == -1: break
        code_set.add(best_idx)
        for idx in neighbor_indices[best_idx]:
            cover_count[idx] += 1
        uncovered_mask &= ~neighbors[best_idx]

    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for c in list(code_set):
            redundant = True
            for idx in neighbor_indices[c]:
                if cover_count[idx] == 1:
                    redundant = False
                    break
            if redundant:
                code_set.remove(c)
                for idx in neighbor_indices[c]:
                    cover_count[idx] -= 1
                changed = True
                break

    result = [list(all_tuples[i]) for i in code_set]
    return result

def _generic_construct(q, n, radius, deadline):
    all_words = list(itertools.product(range(q), repeat=n))
    def dist(a, b):
        return sum(1 for i in range(n) if a[i] != b[i])

    uncovered = set(all_words)
    code = []
    while uncovered and time.time() < deadline - 10:
        best = None
        best_cnt = -1
        for w in all_words:
            if w in set(code): continue
            cnt = sum(1 for u in uncovered if dist(w, u) <= radius)
            if cnt > best_cnt:
                best_cnt = cnt
                best = w
        if not best: break
        code.append(best)
        uncovered = {u for u in uncovered if dist(best, u) > radius}
    return [list(w) for w in code]

# EVOLVE-BLOCK-END
