"""
Maximize comparable pairs among 40 distinct subsets of {0..11}.
Strategy: use dense lattice structure (subsets of a 6-element set) for initialization,
then refine with local search.
"""
import time
import random
import itertools

# EVOLVE-BLOCK-START
N = 12
M = 40
NUM_MASKS = 1 << N  # 4096
ALL_MASKS = list(range(NUM_MASKS))
DEADLINE_MARGIN = 5.0


def _precompute_comparability():
    comp_mask = [0] * NUM_MASKS
    comp_count = [0] * NUM_MASKS

    for m1 in range(NUM_MASKS):
        for m2 in range(m1 + 1, NUM_MASKS):
            if (m1 & m2) == m1 and m1 != m2:
                comp_mask[m1] |= (1 << m2)
                comp_mask[m2] |= (1 << m1)
                comp_count[m1] += 1
                comp_count[m2] += 1
            elif (m1 & m2) == m2 and m1 != m2:
                comp_mask[m1] |= (1 << m2)
                comp_mask[m2] |= (1 << m1)
                comp_count[m1] += 1
                comp_count[m2] += 1

    return comp_mask, comp_count


def _count_comparable(family_masks, comp_mask):
    n = len(family_masks)
    count = 0
    for i in range(n):
        mi = family_masks[i]
        for j in range(i + 1, n):
            mj = family_masks[j]
            if (comp_mask[mi] >> mj) & 1:
                count += 1
    return count


def _greedy_construct(seed, comp_mask, comp_count):
    rng = random.Random(seed)
    in_family = [False] * NUM_MASKS
    family = []

    first = rng.choice(ALL_MASKS)
    family.append(first)
    in_family[first] = True

    scores = [0] * NUM_MASKS
    for y in ALL_MASKS:
        if y == first:
            continue
        if (comp_mask[first] >> y) & 1:
            scores[y] = 1

    for _ in range(M - 1):
        best_score = -1
        best_candidates = []
        for y in ALL_MASKS:
            if not in_family[y]:
                if scores[y] > best_score:
                    best_score = scores[y]
                    best_candidates = [y]
                elif scores[y] == best_score:
                    best_candidates.append(y)

        if not best_candidates:
            break

        chosen = rng.choice(best_candidates)
        family.append(chosen)
        in_family[chosen] = True

        for y in ALL_MASKS:
            if not in_family[y]:
                if (comp_mask[chosen] >> y) & 1:
                    scores[y] += 1

    if len(family) < M:
        remaining = [m for m in ALL_MASKS if not in_family[m]]
        rng.shuffle(remaining)
        family.extend(remaining[:M - len(family)])

    return family


def _local_search_1swap(family, comp_mask, deadline):
    family_set = set(family)
    current_count = _count_comparable(family, comp_mask)

    scores = [0] * NUM_MASKS
    for m in range(NUM_MASKS):
        if m in family_set:
            continue
        for f in family:
            if (comp_mask[m] >> f) & 1:
                scores[m] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False

        for i in range(len(family)):
            if time.time() > deadline:
                break

            removed = family[i]

            old_comp = 0
            for j in range(len(family)):
                if j == i:
                    continue
                if (comp_mask[removed] >> family[j]) & 1:
                    old_comp += 1

            best_gain = 0
            best_add = -1
            for m in range(NUM_MASKS):
                if m in family_set:
                    continue
                is_comp_with_removed = (comp_mask[m] >> removed) & 1
                gain = scores[m] - is_comp_with_removed - old_comp
                if gain > best_gain:
                    best_gain = gain
                    best_add = m

            if best_gain > 0 and best_add != -1:
                family[i] = best_add
                family_set.remove(removed)
                family_set.add(best_add)

                for m in range(NUM_MASKS):
                    if m in family_set:
                        continue
                    if (comp_mask[m] >> removed) & 1:
                        scores[m] -= 1
                    if (comp_mask[m] >> best_add) & 1:
                        scores[m] += 1

                current_count += best_gain
                improved = True

    return family, current_count


def _construct_from_six_elements(comp_mask, deadline):
    """
    Construct family from subsets of a 6-element set, then refine.
    """
    six_elements = list(range(6))
    all_six_subsets = []
    for r in range(4):
        for combo in itertools.combinations(six_elements, r):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            all_six_subsets.append(mask)

    # We have 42 masks. Pick 40 to maximize comparable pairs.
    # Compute comparable count for all 42
    comp_42 = 0
    for i in range(len(all_six_subsets)):
        for j in range(i + 1, len(all_six_subsets)):
            a, b = all_six_subsets[i], all_six_subsets[j]
            if ((a & b) == a or (a & b) == b) and a != b:
                comp_42 += 1

    # Try removing each pair
    best_40 = 0
    best_remove = (0, 1)
    for i in range(len(all_six_subsets)):
        for j in range(i + 1, len(all_six_subsets)):
            lost = 0
            for k in range(len(all_six_subsets)):
                if k == i or k == j:
                    continue
                a = all_six_subsets[k]
                if ((a & all_six_subsets[i]) == a or (a & all_six_subsets[i]) == all_six_subsets[i]) and a != all_six_subsets[i]:
                    lost += 1
                if ((a & all_six_subsets[j]) == a or (a & all_six_subsets[j]) == all_six_subsets[j]) and a != all_six_subsets[j]:
                    lost += 1
            a, b = all_six_subsets[i], all_six_subsets[j]
            if ((a & b) == a or (a & b) == b) and a != b:
                lost += 1

            count_40 = comp_42 - lost
            if count_40 > best_40:
                best_40 = count_40
                best_remove = (i, j)

    ri, rj = best_remove
    family = [m for idx, m in enumerate(all_six_subsets) if idx != ri and idx != rj]

    # Refine with local search
    family, count = _local_search_1swap(family, comp_mask, deadline)
    return family, count


def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN

    comp_mask, comp_count = _precompute_comparability()

    best_count = 0
    best_family = None

    # Phase 1: 6-element subset approach
    if time.time() < deadline - 10.0:
        fam1, cnt1 = _construct_from_six_elements(comp_mask, deadline - 10.0)
        if cnt1 > best_count:
            best_count = cnt1
            best_family = fam1

    # Phase 2: Greedy restarts
    num_restarts = 30
    restart_deadline = deadline - 5.0
    for i in range(num_restarts):
        if time.time() > restart_deadline:
            break
        family = _greedy_construct(i, comp_mask, comp_count)
        count = _count_comparable(family, comp_mask)
        if count > best_count:
            best_count = count
            best_family = list(family)

    # Phase 3: Final local search
    if best_family is not None:
        best_family, best_count = _local_search_1swap(best_family, comp_mask, deadline - 1.0)

    if best_family is None:
        vertices = list(range(6))
        all_sets = []
        for r in range(7):
            for combo in itertools.combinations(vertices, r):
                all_sets.append(list(combo))
        all_sets.sort(key=len)
        best_family = all_sets[:M]

    result = []
    for mask in best_family:
        result.append([b for b in range(N) if (mask >> b) & 1])

    return result
# EVOLVE-BLOCK-END
