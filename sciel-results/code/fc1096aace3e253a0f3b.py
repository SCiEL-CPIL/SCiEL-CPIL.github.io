import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s / 150s kill

    N = 51975

    # Generate divisors
    divisors = []
    for d in range(2, N + 1):
        if N % d == 0:
            divisors.append(d)
    divisors.sort()

    def omega(n):
        c = 0
        t = n
        for p in (3, 5, 7, 11):
            while t % p == 0:
                c += 1
                t //= p
        return c

    # ---- Greedy initializers (numpy-backed) ----
    def greedy(order):
        counts = np.zeros(N, dtype=np.int32)
        uncovered = N
        classes = []
        used = set()
        for m in order:
            Z = (counts == 0).astype(np.int32)
            best_r = 0
            best_gain = -1
            max_possible = N // m
            for r in range(m):
                gain = int(np.sum(Z[r::m]))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
                    if best_gain == max_possible:
                        break
            if best_gain > 0:
                classes.append([m, best_r])
                used.add(m)
                idx = np.arange(best_r, N, m)
                uncovered -= int(np.sum(counts[idx] == 0))
                counts[idx] += 1
        return classes, uncovered

    # Build candidate orderings
    orderings = []
    orderings.append(divisors.copy())                          # increasing
    orderings.append(divisors[::-1].copy())                    # decreasing
    orderings.append(sorted(divisors, key=lambda m: (N // m, m)))
    orderings.append(sorted(divisors, key=lambda m: (-omega(m), m)))
    orderings.append(sorted(divisors, key=lambda m: (m % 7, m)))
    for _ in range(6):
        p = divisors.copy()
        random.shuffle(p)
        orderings.append(p)

    best_classes = None
    best_uncovered = N
    for order in orderings:
        if time.time() > deadline - 60:
            break
        classes, unc = greedy(order)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = [c.copy() for c in classes]

    if best_classes is None:
        best_classes, best_uncovered = greedy(divisors.copy())

    # ---- Local search machinery on the best start ----
    counts = np.zeros(N, dtype=np.int32)
    uncovered = N
    for m, r in best_classes:
        idx = np.arange(r, N, m)
        uncovered -= int(np.sum(counts[idx] == 0))
        counts[idx] += 1

    classes = [c.copy() for c in best_classes]
    used_moduli = set(m for m, _ in classes)

    def update_add(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        dec = int(np.sum(counts[idx] == 0))
        counts[idx] += 1
        uncovered -= dec

    def update_remove(m, r):
        nonlocal uncovered
        idx = np.arange(r, N, m)
        inc = int(np.sum(counts[idx] == 1))
        counts[idx] -= 1
        uncovered += inc

    def find_best_residue(m):
        Z = (counts == 0).astype(np.int32)
        best_r = 0
        best_gain = -1
        max_possible = N // m
        for r in range(m):
            gain = int(np.sum(Z[r::m]))
            if gain > best_gain:
                best_gain = gain
                best_r = r
                if best_gain == max_possible:
                    break
        return best_r, best_gain

    current_uncovered = uncovered
    best_uncovered = uncovered
    best_classes = [c.copy() for c in classes]

    # Simulated annealing
    T = 8.0
    T_min = 0.05
    cooling = 0.9995

    while time.time() < deadline - 12:
        if T < T_min:
            break
        move_type = random.random()

        if move_type < 0.5 and classes:
            i = random.randint(0, len(classes) - 1)
            m, r = classes[i]
            new_r = random.randint(0, m - 1)
            if new_r == r:
                continue
            old_r = r
            update_remove(m, old_r)
            update_add(m, new_r)
            delta = uncovered - current_uncovered
            current_uncovered = uncovered
            classes[i][1] = new_r
            if delta > 0:
                if random.random() < np.exp(-delta / T):
                    pass
                else:
                    update_remove(m, new_r)
                    update_add(m, old_r)
                    classes[i][1] = old_r
                    current_uncovered = uncovered
            else:
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c.copy() for c in classes]

        elif move_type < 0.7 and len(classes) >= 2:
            i, j = random.sample(range(len(classes)), 2)
            m_i, r_i = classes[i]
            m_j, r_j = classes[j]
            update_remove(m_i, r_i)
            update_remove(m_j, r_j)
            update_add(m_i, r_j)
            update_add(m_j, r_i)
            delta = uncovered - current_uncovered
            current_uncovered = uncovered
            classes[i][1] = r_j
            classes[j][1] = r_i
            if delta > 0:
                if random.random() < np.exp(-delta / T):
                    pass
                else:
                    update_remove(m_i, r_j)
                    update_remove(m_j, r_i)
                    update_add(m_i, r_i)
                    update_add(m_j, r_j)
                    classes[i][1] = r_i
                    classes[j][1] = r_j
                    current_uncovered = uncovered
            else:
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c.copy() for c in classes]

        elif move_type < 0.85:
            unused = [m for m in divisors if m not in used_moduli]
            if unused:
                m = random.choice(unused)
                r = random.randint(0, m - 1)
                update_add(m, r)
                classes.append([m, r])
                used_moduli.add(m)
                delta = uncovered - current_uncovered
                current_uncovered = uncovered
                if delta > 0:
                    if random.random() < np.exp(-delta / T):
                        pass
                    else:
                        update_remove(m, r)
                        classes.pop()
                        used_moduli.remove(m)
                        current_uncovered = uncovered
                else:
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_classes = [c.copy() for c in classes]

        else:
            if classes:
                i = random.randint(0, len(classes) - 1)
                m, r = classes[i]
                update_remove(m, r)
                classes.pop(i)
                used_moduli.remove(m)
                delta = uncovered - current_uncovered
                current_uncovered = uncovered
                if delta > 0:
                    if random.random() < np.exp(-delta / T):
                        pass
                    else:
                        classes.insert(i, [m, r])
                        used_moduli.add(m)
                        update_add(m, r)
                        current_uncovered = uncovered
                else:
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_classes = [c.copy() for c in classes]

        T *= cooling

    # ---- Final exact polish: single-residue re-optimization sweep ----
    counts = np.zeros(N, dtype=np.int32)
    uncovered = N
    for m, r in best_classes:
        idx = np.arange(r, N, m)
        uncovered -= int(np.sum(counts[idx] == 0))
        counts[idx] += 1
    classes = [c.copy() for c in best_classes]
    used_moduli = set(m for m, _ in classes)

    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline - 1:
                break
            m, r = classes[i]
            update_remove(m, r)
            best_r, _ = find_best_residue(m)
            if best_r != r:
                update_add(m, best_r)
                classes[i][1] = best_r
                improved = True
                break
            else:
                update_add(m, r)

    return classes
