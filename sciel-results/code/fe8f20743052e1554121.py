import time
import math
import random

def construct():
    deadline = time.time() + 110  # safety margin under 150s

    # Even moduli in [20, 200] -- structured gcds enable better packing
    moduli = list(range(20, 201, 2))  # 100 even moduli
    n = len(moduli)
    idx = {m: i for i, m in enumerate(moduli)}

    # Precompute gcd table (symmetric)
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def try_greedy(order):
        """Build a solution by adding moduli in the given order, first valid residue."""
        classes = []
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

    def local_search(classes, max_iters=6000):
        classes = [c[:] for c in classes]
        score = score_of(classes)
        present = set(m for m, _ in classes)
        missing = [m for m in moduli if m not in present]
        missing.sort()  # smallest first (highest 1/m)

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False

            # Move A: try to add a missing modulus directly
            for m in missing:
                mi = idx[m]
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

            # Move B: reassign one existing class's residue to enable adding a missing modulus
            for m in missing:
                mi = idx[m]
                # Find residue with minimum conflicts
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
                if best_conf is None or best_conf > 1:
                    continue

                # Identify conflicting classes
                conf_js = []
                for j, (m2, r2) in enumerate(classes):
                    g = gcd_table[mi][idx[m2]]
                    if best_rm % g == r2 % g:
                        conf_js.append(j)

                # Try to reassign each conflicting class
                for j in conf_js:
                    mj, rj = classes[j]
                    mj_i = idx[mj]
                    g_jm = gcd_table[mj_i][mi]
                    target = best_rm % g_jm
                    for new_r in range(mj):
                        if new_r == rj:
                            continue
                        if new_r % g_jm == target:
                            continue
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

    # Start with smallest-first greedy
    best_classes = try_greedy(sorted(moduli))
    best_score = score_of(best_classes)
    best_classes, best_score = local_search(best_classes)

    # Multiple randomized restarts with local search
    rng = random.Random(42)
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        shuffled = moduli[:]
        rng.shuffle(shuffled)
        # Weight towards smaller moduli
        weighted = sorted(shuffled, key=lambda m: (m * 0.3 + rng.random() * 80))
        cand = try_greedy(weighted)
        s = score_of(cand)
        if s > best_score:
            best_classes, best_score = local_search(cand)
        elif s >= best_score - 0.05:
            # Close enough, try local search anyway
            cand2, s2 = local_search(cand)
            if s2 > best_score:
                best_classes, best_score = cand2, s2

    # Final validation
    if not is_valid(best_classes):
        best_classes = try_greedy(sorted(moduli))
        best_score = score_of(best_classes)

    return best_classes
