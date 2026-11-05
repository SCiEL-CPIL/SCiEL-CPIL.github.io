import math
import time
import random

def construct():
    deadline = time.time() + 112  # Safety margin before 150s kill

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def is_valid(selected, m, r):
        for (m2, r2) in selected:
            g = gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def find_valid_residue(selected, m):
        for r in range(m):
            if is_valid(selected, m, r):
                return r
        return None

    def score(selected):
        return sum(1.0 / m for m, r in selected)

    # ---------------- Beam search over ascending moduli ----------------
    beam_width = 80
    beam = [[]]  # each entry: list of (m, r)

    for m in range(20, 201):
        if time.time() > deadline - 15:
            break

        new_beam = []
        for sol in beam:
            # Branch 1: skip this modulus
            new_beam.append(sol)
            # Branch 2: add this modulus with a valid residue (diversity via a couple choices)
            added = False
            for r in range(m):
                if is_valid(sol, m, r):
                    new_beam.append(sol + [(m, r)])
                    added = True
                    break
            if not added:
                pass  # skip branch already present

        # Keep top diverse by score
        new_beam.sort(key=lambda x: score(x), reverse=True)
        beam = new_beam[:beam_width]

    # ---------------- Local search refinement (add / swap / residue) ----------------
    def local_search(pairs):
        pairs = [list(p) for p in pairs]
        current_score = score(pairs)
        best = [list(p) for p in pairs]
        best_score = current_score

        def try_add(cur, sc):
            have = {p[0] for p in cur}
            for m in range(20, 201):
                if m in have:
                    continue
                r = find_valid_residue(cur, m)
                if r is not None:
                    cur.append([m, r])
                    return cur, sc + 1.0 / m
            return cur, sc

        def try_swap(cur, sc):
            have = {p[0] for p in cur}
            for i in range(len(cur)):
                m_old = cur[i][0]
                for m_new in range(20, 201):
                    if m_new in have or m_new >= m_old:
                        continue
                    others = [cur[j] for j in range(len(cur)) if j != i]
                    r_new = find_valid_residue(others, m_new)
                    if r_new is not None:
                        ns = sc - 1.0 / m_old + 1.0 / m_new
                        if ns > sc + 1e-12:
                            cur[i] = [m_new, r_new]
                            return cur, ns
            return cur, sc

        improved = True
        stall = 0
        while improved and time.time() < deadline - 5:
            improved = False
            ns = try_add(pairs, current_score)
            if ns[1] > current_score + 1e-12:
                pairs, current_score = ns
                if current_score > best_score:
                    best_score = current_score
                    best = [list(p) for p in pairs]
                improved = True
                continue
            ns = try_swap(pairs, current_score)
            if ns[1] > current_score + 1e-12:
                pairs, current_score = ns
                if current_score > best_score:
                    best_score = current_score
                    best = [list(p) for p in pairs]
                improved = True
                continue
            # residue perturbation to enable further additions
            stall += 1
            if stall > 20:
                break
            if pairs:
                idx = random.randint(0, len(pairs) - 1)
                m, r = pairs[idx]
                for _ in range(15):
                    r_new = random.randint(0, m - 1)
                    ok = True
                    for j, (m2, r2) in enumerate(pairs):
                        if j == idx:
                            continue
                        if r_new % gcd(m, m2) == r2 % gcd(m, m2):
                            ok = False
                            break
                    if ok:
                        pairs[idx] = [m, r_new]
                        ns = try_add(pairs, current_score)
                        if ns[1] > current_score + 1e-12:
                            pairs, current_score = ns
                            if current_score > best_score:
                                best_score = current_score
                                best = [list(p) for p in pairs]
                            improved = True
                            stall = 0
                            break
        return best

    # Refine the top beam members; keep the best overall
    random.seed(1)
    best_pairs = []
    best_score = -1.0
    for sol in beam[:25]:
        if time.time() > deadline - 5:
            break
        polished = local_search(sol)
        s = score(polished)
        if s > best_score:
            best_score = s
            best_pairs = polished

    return [[m, r] for m, r in best_pairs]
