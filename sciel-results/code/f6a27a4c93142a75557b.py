import time
import random
import math

def _family_set_check(family, exclude_idx):
    return set(f for i, f in enumerate(family) if i != exclude_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    m = 40
    total_sets = 1 << n
    
    # Precompute comparability matrix
    comp_matrix = [bytearray(total_sets) for _ in range(total_sets)]
    for s in range(total_sets):
        row = comp_matrix[s]
        for t in range(total_sets):
            if s != t and ((s & t) == s or (s & t) == t):
                row[t] = 1
    
    def count_comparables(family):
        count = 0
        k = len(family)
        for i in range(k):
            fi = family[i]
            for j in range(i + 1, k):
                if comp_matrix[fi][family[j]]:
                    count += 1
        return count

    def greedy_init():
        family = [0]
        family_set = {0}
        while len(family) < m:
            best_candidate = -1
            best_gain = -1
            for s in range(1, total_sets):
                if s in family_set:
                    continue
                gain = sum(1 for f in family if comp_matrix[f][s])
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = s
            if best_candidate != -1:
                family.append(best_candidate)
                family_set.add(best_candidate)
            else:
                for s in range(1, total_sets):
                    if s not in family_set:
                        family.append(s)
                        family_set.add(s)
                        break
        return family

    def greedy_init_randomized():
        family = [random.randint(0, total_sets - 1)]
        family_set = set(family)
        while len(family) < m:
            best_candidate = -1
            best_gain = -1
            for s in range(total_sets):
                if s in family_set:
                    continue
                gain = sum(1 for f in family if comp_matrix[f][s])
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = s
            if best_candidate != -1:
                family.append(best_candidate)
                family_set.add(best_candidate)
            else:
                for s in range(total_sets):
                    if s not in family_set:
                        family.append(s)
                        family_set.add(s)
                        break
        return family

    best_init = None
    best_init_score = -1

    fam = greedy_init()
    score = count_comparables(fam)
    if score > best_init_score:
        best_init_score = score
        best_init = fam[:]

    for _ in range(20):
        if time.time() > deadline - 10.0:
            break
        fam = greedy_init_randomized()
        score = count_comparables(fam)
        if score > best_init_score:
            best_init_score = score
            best_init = fam[:]

    family = best_init
    current_score = best_init_score

    T = 10.0
    T_min = 0.01
    alpha = 0.9995
    max_iter = 300000

    best_family = family[:]
    best_score = current_score

    for it in range(max_iter):
        if time.time() > deadline:
            break

        i = random.randint(0, m - 1)
        old_s = family[i]

        if random.random() < 0.7:
            ref = family[random.randint(0, m - 1)]
            if random.random() < 0.5:
                new_s = ref
                for _ in range(random.randint(1, n)):
                    new_s &= ~(1 << random.randint(0, n - 1))
            else:
                new_s = ref
                for _ in range(random.randint(1, n)):
                    new_s |= (1 << random.randint(0, n - 1))
        else:
            new_s = random.randint(0, total_sets - 1)

        if new_s == old_s:
            continue

        if new_s in _family_set_check(family, i):
            continue

        count_new = 0
        count_old = 0
        for j, f in enumerate(family):
            if j == i:
                continue
            if comp_matrix[f][new_s]:
                count_new += 1
            if comp_matrix[f][old_s]:
                count_old += 1

        delta = count_new - count_old

        if delta >= 0 or random.random() < math.exp(delta / T):
            family[i] = new_s
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_family = family[:]

        T *= alpha
        if T < T_min:
            T = T_min

    # Limited final polish
    for _ in range(2):
        if time.time() > deadline - 1.0:
            break
        improved = False
        for i in range(m):
            if time.time() > deadline - 1.0:
                break
            old_s = family[i]
            best_new = old_s
            best_delta = 0
            candidates = random.sample(range(total_sets), min(200, total_sets))
            for s in candidates:
                if s == old_s:
                    continue
                if s in _family_set_check(family, i):
                    continue
                count_new = 0
                count_old = 0
                for j, f in enumerate(family):
                    if j == i:
                        continue
                    if comp_matrix[f][s]:
                        count_new += 1
                    if comp_matrix[f][old_s]:
                        count_old += 1
                delta = count_new - count_old
                if delta > best_delta:
                    best_delta = delta
                    best_new = s
            if best_new != old_s:
                family[i] = best_new
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
        if not improved:
            break

    family = best_family

    result = []
    for s in family:
        subset = [i for i in range(n) if (s >> i) & 1]
        result.append(subset)
    return result
