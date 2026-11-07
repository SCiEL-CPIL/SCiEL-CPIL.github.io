import random
import time

def construct():
    n = 12
    m = 40
    total_masks = 1 << n

    def count_comparables(family_masks):
        count = 0
        for i in range(len(family_masks)):
            for j in range(i+1, len(family_masks)):
                a, b = family_masks[i], family_masks[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count

    def greedy_build(initial_mask):
        selected = [initial_mask]
        selected_set = {initial_mask}
        scores = [0] * total_masks

        for u in range(total_masks):
            if u == initial_mask:
                continue
            if (u & initial_mask) == u or (u & initial_mask) == initial_mask:
                scores[u] = 1

        for _ in range(m - 1):
            best_mask = -1
            best_score = -1
            for u in range(total_masks):
                if u in selected_set:
                    continue
                if scores[u] > best_score:
                    best_score = scores[u]
                    best_mask = u

            if best_mask == -1:
                break

            selected.append(best_mask)
            selected_set.add(best_mask)

            for u in range(total_masks):
                if u in selected_set:
                    continue
                if (u & best_mask) == u or (u & best_mask) == best_mask:
                    scores[u] += 1

        return selected

    def local_search_1swap(family, deadline):
        """Improve family via 1-swap moves using efficient delta computation."""
        family = list(family)
        fam_set = set(family)
        cur_score = count_comparables(family)

        # Precompute comparability matrix: comp[i][j] = 1 if family[i] and family[j] comparable
        comp = [[0]*m for _ in range(m)]
        for i in range(m):
            for j in range(i+1, m):
                a, b = family[i], family[j]
                if (a & b) == a or (a & b) == b:
                    comp[i][j] = 1
                    comp[j][i] = 1

        # For each family member i, its "contribution" to score is sum_j comp[i][j]
        # When we swap family[i] with new_mask u:
        #   gain = (sum_j comp(u, family[j]) for j != i) - (sum_j comp[i][j] for j != i)
        #        = (sum_j comp(u, family[j])) - (sum_j comp[i][j])
        # because comp(u, family[i]) doesn't matter (u replaces family[i])
        # Wait, actually: the new score = old_score - (comparables of family[i] with others) + (comparables of u with others excluding i)
        # Since u is not in family, comparables of u with others = sum_{j != i} comp(u, family[j])
        # And comparables of family[i] with others = sum_{j != i} comp[i][j] = sum_j comp[i][j] (since comp[i][i]=0)

        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(m):
                if time.time() > deadline:
                    break
                removed = family[i]
                # Current contribution of family[i] to score (with all other members)
                rem_comp = sum(comp[i])

                # Find best replacement
                best_new = -1
                best_new_comp = rem_comp
                for u in range(total_masks):
                    if u in fam_set:
                        continue
                    new_comp = 0
                    for j in range(m):
                        if j == i:
                            continue
                        a, b = u, family[j]
                        if (a & b) == a or (a & b) == b:
                            new_comp += 1
                    if new_comp > best_new_comp:
                        best_new_comp = new_comp
                        best_new = u

                if best_new != -1 and best_new_comp > rem_comp:
                    # Perform swap
                    gain = best_new_comp - rem_comp
                    # Update comp matrix
                    for j in range(m):
                        if j == i:
                            continue
                        old_val = comp[i][j]
                        a, b = best_new, family[j]
                        new_val = 1 if ((a & b) == a or (a & b) == b) else 0
                        comp[i][j] = new_val
                        comp[j][i] = new_val
                    family[i] = best_new
                    fam_set.remove(removed)
                    fam_set.add(best_new)
                    cur_score += gain
                    improved = True
                    break  # restart after change

        return family, cur_score

    best_family = []
    best_score = -1

    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Try several random initializations
    initial_candidates = [0, (1<<n)-1]
    rng = random.Random(42)
    for _ in range(30):
        initial_candidates.append(rng.randint(0, total_masks - 1))

    for idx, init in enumerate(initial_candidates):
        if time.time() > deadline - 5.0:
            break

        family = greedy_build(init)
        if len(family) < m:
            continue

        # Local search 1-swap
        family, score = local_search_1swap(family, deadline - 3.0)

        if score > best_score:
            best_score = score
            best_family = list(family)

    if not best_family:
        best_family = list(range(m))

    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        lst = []
        for b in range(n):
            if (mask >> b) & 1:
                lst.append(b)
        result.append(lst)

    return result
