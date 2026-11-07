import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit

    n = 12
    m = 40
    N = 1 << n
    full_mask = N - 1

    # Precompute comparability: comp[a] is a bitset of all b such that a and b are comparable
    # (i.e., a subset of b, or b subset of a, and a != b)
    comp = [0] * N
    for a in range(N):
        # All strict subsets of a
        sub = a
        while True:
            if sub != a:
                comp[a] |= (1 << sub)
            if sub == 0:
                break
            sub = (sub - 1) & a
        # All strict supersets of a
        comp_mask = full_mask ^ a
        sub = comp_mask
        while True:
            b = a | sub
            if b != a:
                comp[a] |= (1 << b)
            if sub == 0:
                break
            sub = (sub - 1) & comp_mask

    def is_subset(a, b):
        return (a & b) == a and a != b

    def count_comparable(family):
        count = 0
        len_fam = len(family)
        for i in range(len_fam):
            ai = family[i]
            for j in range(i + 1, len_fam):
                aj = family[j]
                if is_subset(ai, aj) or is_subset(aj, ai):
                    count += 1
        return count

    def greedy_start():
        """Greedy construction: add set that maximizes comparability with current family."""
        family = []
        used = set()
        for _ in range(m):
            best_mask = -1
            best_gain = -1
            for mask in range(N):
                if mask in used:
                    continue
                gain = 0
                for f in family:
                    if (mask & f) == mask or (f & mask) == f:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            if best_mask == -1:
                while True:
                    mask = random.randint(0, N - 1)
                    if mask not in used:
                        best_mask = mask
                        break
            family.append(best_mask)
            used.add(best_mask)
        return family

    def chain_based_start():
        """Create a family based on multiple chains."""
        family = []
        used = set()

        # Chain 1: {0}, {0,1}, {0,1,2}, ..., {0..11}
        for i in range(n + 1):
            mask = (1 << i) - 1
            if mask not in used:
                family.append(mask)
                used.add(mask)

        # Chain 2: different order
        order2 = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        mask = 0
        for i in range(n + 1):
            if i > 0:
                mask |= (1 << order2[i - 1])
            if mask not in used:
                family.append(mask)
                used.add(mask)

        # Chain 3: different order
        order3 = [2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        mask = 0
        for i in range(n + 1):
            if i > 0:
                mask |= (1 << order3[i - 1])
            if mask not in used:
                family.append(mask)
                used.add(mask)

        # Add more sets to reach 40
        while len(family) < m:
            mask = random.randint(0, N - 1)
            if mask not in used:
                family.append(mask)
                used.add(mask)

        return family[:m]

    def random_start():
        fam = []
        used = set()
        while len(fam) < m:
            mask = random.randint(0, N - 1)
            if mask not in used:
                used.add(mask)
                fam.append(mask)
        return fam

    def local_search(initial_family, deadline):
        family = initial_family[:]
        family_set = set(family)
        current_score = count_comparable(family)
        best_family = family[:]
        best_score = current_score
        temp = 10.0
        no_improve_count = 0

        iter_count = 0
        while time.time() < deadline:
            iter_count += 1

            if iter_count % 1000 == 0 and temp > 0.1:
                temp *= 0.995
                if no_improve_count > 5000:
                    no_improve_count = 0
                    temp = 5.0
                    continue

            idx = random.randint(0, m - 1)
            old_mask = family[idx]

            if random.random() < 0.6:
                new_mask = random.randint(0, N - 1)
            elif random.random() < 0.5:
                new_mask = old_mask ^ (1 << random.randint(0, n - 1))
            else:
                b1 = random.randint(0, n - 1)
                b2 = random.randint(0, n - 1)
                while b2 == b1:
                    b2 = random.randint(0, n - 1)
                new_mask = old_mask ^ (1 << b1) ^ (1 << b2)

            if new_mask == old_mask:
                continue
            if new_mask in family_set:
                continue

            lost = 0
            gained = 0
            for j in range(m):
                if j == idx:
                    continue
                other = family[j]
                if is_subset(old_mask, other) or is_subset(other, old_mask):
                    lost += 1
                if is_subset(new_mask, other) or is_subset(other, new_mask):
                    gained += 1

            delta = gained - lost

            accept = False
            if delta > 0:
                accept = True
            elif temp > 0.1:
                if random.random() < math.exp(delta / temp):
                    accept = True

            if accept:
                family[idx] = new_mask
                family_set.remove(old_mask)
                family_set.add(new_mask)
                current_score += delta
                no_improve_count = 0

                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
            else:
                no_improve_count += 1

        return best_family, best_score

    def swap_local_search(initial_family, deadline):
        """Local search that tries swapping pairs of sets."""
        family = initial_family[:]
        family_set = set(family)
        current_score = count_comparable(family)
        best_family = family[:]
        best_score = current_score

        while time.time() < deadline:
            improved = False
            for i in range(m):
                if time.time() > deadline:
                    return best_family, best_score
                for j in range(i + 1, m):
                    old_i = family[i]
                    old_j = family[j]

                    # Try swapping with random masks
                    new_i = random.randint(0, N - 1)
                    new_j = random.randint(0, N - 1)

                    if new_i == old_i and new_j == old_j:
                        continue
                    if new_i == new_j:
                        continue
                    if new_i in family_set and new_i != old_i and new_i != old_j:
                        continue
                    if new_j in family_set and new_j != old_i and new_j != old_j:
                        continue

                    lost = 0
                    gained = 0
                    for k in range(m):
                        if k == i or k == j:
                            continue
                        other = family[k]
                        if is_subset(old_i, other) or is_subset(other, old_i):
                            lost += 1
                        if is_subset(old_j, other) or is_subset(other, old_j):
                            lost += 1
                        if is_subset(new_i, other) or is_subset(other, new_i):
                            gained += 1
                        if is_subset(new_j, other) or is_subset(other, new_j):
                            gained += 1

                    # Also check if old_i and old_j were comparable to each other
                    if is_subset(old_i, old_j) or is_subset(old_j, old_i):
                        lost += 1
                    if is_subset(new_i, new_j) or is_subset(new_j, new_i):
                        gained += 1

                    delta = gained - lost

                    if delta > 0:
                        family[i] = new_i
                        family[j] = new_j
                        family_set.remove(old_i)
                        family_set.remove(old_j)
                        family_set.add(new_i)
                        family_set.add(new_j)
                        current_score += delta
                        improved = True

                        if current_score > best_score:
                            best_score = current_score
                            best_family = family[:]

            if not improved:
                break

        return best_family, best_score

    best_overall_family = None
    best_overall_score = 0

    # Try multiple initializations
    initializations = [
        greedy_start(),
        chain_based_start(),
        random_start(),
    ]

    for init in initializations:
        if time.time() > deadline - 10:
            break
        fam, score = local_search(init, deadline - 5)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_family = fam

    # Try swap local search on the best family found
    if best_overall_family is not None and time.time() < deadline - 5:
        fam, score = swap_local_search(best_overall_family, deadline)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_family = fam

    if best_overall_family is None:
        best_overall_family = list(range(m))

    result = []
    for mask in best_overall_family:
        result.append([b for b in range(n) if (mask >> b) & 1])

    return result
