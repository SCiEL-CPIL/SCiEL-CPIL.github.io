import time
import math

def construct():
    deadline = time.time() + 110  # safety margin under 150s

    # All moduli in [20, 200] (not just evens) -- more options to pack small 1/m
    moduli = list(range(20, 201))          # 181 moduli
    n = len(moduli)
    idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table (symmetric)
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute, for each modulus i, the set of residue values r % g that are
    # "available" (not conflicting with a given class set) -- we recompute cheaply
    # below, but precompute the gcd of each pair so per-check is O(1) arithmetic.

    def try_greedy(order):
        """Build a solution by adding moduli in the given order, first valid residue."""
        classes = []
        # residue_conflict[i][g_value] counts how many existing classes block residue class g_value mod g
        # Simpler: keep list of (mi, r) and check pairwise.
        for m in order:
            mi = idx.get(m)
            if mi is None:
                continue
            for r in range(m):
                valid = True
                for m2, r2 in classes:
                    g = gcd_table[mi][idx[m2]]
                    if r % g == r2 % g:
                        valid = False
                        break
                if valid:
                    classes.append([m, r])
                    break
        return classes

    best_classes = try_greedy(sorted(moduli))  # smallest-first greedy baseline
    best_score = sum(1.0 / m for m, _ in best_classes)

    # ---- Local search ----
    def score_of(classes):
        return sum(1.0 / m for m, _ in classes)

    def residue_ok(m, r, classes, skip_idx):
        mi = idx[m]
        for j, (m2, r2) in enumerate(classes):
            if j == skip_idx:
                continue
            g = gcd_table[mi][idx[m2]]
            if r % g == r2 % g:
                return False
        return True

    def local_search(classes, max_iters=4000):
        classes = [c[:] for c in classes]
        score = score_of(classes)
        present = set(m for m, _ in classes)
        missing = [m for m in moduli if m not in present]
        missing.sort()  # smallest first (highest 1/m)

        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False

            # Move A: swap in a smaller missing modulus (raises 1/m)
            # For each missing m (small first), try to find a residue that fits
            # after possibly reassigning one existing class's residue.
            for m in missing:
                mi = idx[m]
                # direct fit?
                for r in range(m):
                    if residue_ok(m, r, classes, -1):
                        classes.append([m, r])
                        present.add(m)
                        missing.remove(m)
                        score += 1.0 / m
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # Move B: reassign a residue to free a slot for a missing modulus
            # Try: for each missing m, for each existing class j, try changing
            # classes[j] residue so that m can be added.
            for m in missing:
                mi = idx[m]
                # find a residue r_m that conflicts with the fewest existing classes
                # We do a light search: try all r_m, count conflicts, pick min.
                best_rm = None
                best_conf = None
                for r_m in range(m):
                    conf = 0
                    for (m2, r2) in classes:
                        g = gcd_table[mi][idx[m2]]
                        if r_m % g == r2 % g:
                            conf += 1
                    if best_conf is None or conf < best_conf:
                        best_conf = conf
                        best_rm = r_m
                        if conf == 0:
                            break
                if best_conf is None:
                    continue
                # If best_conf == 0 already handled in Move A.
                # If best_conf > 1, we cannot resolve all conflicts by moving a single class.
                if best_conf > 1:
                    continue
                
                # Try to fix by reassigning one conflicting class j.
                # Identify conflicting classes for best_rm:
                conf_js = []
                for j, (m2, r2) in enumerate(classes):
                    g = gcd_table[mi][idx[m2]]
                    if best_rm % g == r2 % g:
                        conf_js.append(j)
                # For each conflicting j, try to reassign its residue to a value
                # that (a) doesn't conflict with other existing classes and
                # (b) doesn't conflict with best_rm mod gcd(m, m_j).
                for j in conf_js:
                    mj, rj = classes[j]
                    mj_i = idx[mj]
                    g_jm = gcd_table[mj_i][mi]
                    target = best_rm % g_jm  # residue j must NOT equal this mod g_jm
                    for new_r in range(mj):
                        if new_r == rj:
                            continue
                        if new_r % g_jm == target:
                            continue  # would still conflict with m
                        # check new_r against all other existing classes
                        ok = True
                        for k, (m2, r2) in enumerate(classes):
                            if k == j:
                                continue
                            g = gcd_table[mj_i][idx[m2]]
                            if new_r % g == r2 % g:
                                ok = False
                                break
                        if ok:
                            classes[j] = [mj, new_r]
                            classes.append([m, best_rm])
                            present.add(m)
                            missing.remove(m)
                            score += 1.0 / m
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        return classes, score

    def is_valid(classes):
        n = len(classes)
        for i in range(n):
            m1, r1 = classes[i]
            for j in range(i + 1, n):
                m2, r2 = classes[j]
                g = math.gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True

    best_classes, best_score = local_search(best_classes)
    if not is_valid(best_classes):
        # Fallback to a simple valid greedy if local search broke it
        best_classes = try_greedy(sorted(moduli))
        best_score = sum(1.0 / m for m, _ in best_classes)

    # A few randomized greedy restarts to find a better starting point, then local search
    import random
    rng = random.Random(42)
    for attempt in range(300):
        if time.time() > deadline - 8:
            break
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        weighted = sorted(shuffled, key=lambda m: (m * 0.5 + rng.random() * 60))
        cand = try_greedy(weighted)
        s = score_of(cand)
        if s > best_score:
            best_classes, best_score = local_search(cand)

    return best_classes
