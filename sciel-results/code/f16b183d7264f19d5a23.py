"""Combined: greedy coprime set + local search + targeted small-modulus inclusion."""
import time
import random
import math
from fractions import Fraction

def _disjoint(m1, r1, m2, r2):
    """Check if residue classes are pairwise disjoint: r1 != r2 mod gcd(m1, m2)."""
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid(classes):
    """Check pairwise disjointness for a list of [m, r] pairs with distinct moduli."""
    if not classes:
        return True
    moduli = [c[0] for c in classes]
    if len(moduli) != len(set(moduli)):
        return False
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if not _disjoint(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                return False
    return True

def _score(classes):
    """Exact rational score, returned as float."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)

def _find_valid_residue(m, selected):
    """Find a residue r in [0, m) such that (m, r) is disjoint from all in selected.
    Returns r or None."""
    for r in range(m):
        ok = True
        for (m2, r2) in selected:
            if not _disjoint(m, r, m2, r2):
                ok = False
                break
        if ok:
            return r
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 150s kill

    best_classes = None
    best_score = 0.0

    # Strategy A: Greedy pairwise coprime set (baseline from current)
    def greedy_coprime_set(moduli_list):
        picked = []
        for m in moduli_list:
            ok = True
            for p in picked:
                if math.gcd(m, p) != 1:
                    ok = False
                    break
            if ok:
                picked.append(m)
        return picked

    moduli_range = list(range(20, 201))

    # Pure greedy coprime
    picked = greedy_coprime_set(moduli_range)
    classes = [[m, 0] for m in picked]
    if _is_valid(classes):
        sc = _score(classes)
        if sc > best_score:
            best_score = sc
            best_classes = classes[:]

    # Strategy B: Greedy inclusion with residue search (include non-coprime small moduli)
    # Process moduli in ascending order; for each, try to find a valid residue.
    # This is the key improvement: we don't restrict to coprime sets.

    for attempt in range(30):
        if time.time() > deadline - 15:
            break

        random.seed(42 + attempt)
        moduli_sorted = list(range(20, 201))

        # For later attempts, shuffle the larger moduli to explore different orderings
        if attempt > 0:
            # Keep small moduli (20-40) in order, shuffle the rest
            small = moduli_sorted[:21]  # 20 through 40
            rest = moduli_sorted[21:]
            random.shuffle(rest)
            moduli_sorted = small + rest

        selected = []  # list of [m, r]

        for m in moduli_sorted:
            if time.time() > deadline - 10:
                break
            r = _find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])

        if selected:
            if _is_valid(selected):
                sc = _score(selected)
                if sc > best_score:
                    best_score = sc
                    best_classes = selected[:]

    # Strategy C: Local search / improvement on the best found so far
    # Try to add more moduli or swap to improve score.
    if best_classes is not None:
        # Try adding unselected small moduli
        used_moduli = set(c[0] for c in best_classes)
        candidates = [m for m in range(20, 201) if m not in used_moduli]
        candidates.sort()  # Prefer small moduli (high value)

        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for m in candidates[:]:
                if time.time() > deadline - 10:
                    break
                r = _find_valid_residue(m, best_classes)
                if r is not None:
                    # Check: does adding m improve the score? It always does (1/m > 0)
                    best_classes.append([m, r])
                    used_moduli.add(m)
                    candidates.remove(m)
                    improved = True

    # Strategy D: Randomized greedy with different heuristics
    for attempt in range(40):
        if time.time() > deadline - 5:
            break

        random.seed(1000 + attempt)
        selected = []
        used_moduli = set()

        # Weighted random selection biased toward small moduli
        for _ in range(80):
            if time.time() > deadline - 5:
                break
            available = [m for m in range(20, 201) if m not in used_moduli]
            if not available:
                break

            # Weight by 1/m^alpha where alpha varies per attempt
            alpha = 0.5 + (attempt % 5) * 0.5
            weights = [(1.0 / (m ** alpha)) for m in available]
            tw = sum(weights)
            if tw <= 0:
                break
            r = random.random() * tw
            cumsum = 0
            chosen_m = available[-1]
            for m, w in zip(available, weights):
                cumsum += w
                if r <= cumsum:
                    chosen_m = m
                    break

            m = chosen_m
            res = _find_valid_residue(m, selected)
            if res is not None:
                selected.append([m, res])
                used_moduli.add(m)

        if selected:
            if _is_valid(selected):
                sc = _score(selected)
                if sc > best_score:
                    best_score = sc
                    best_classes = selected[:]

    # Strategy E: Try to optimize residues for a fixed set of moduli
    # Take the top-N smallest moduli and see if we can assign residues to make them all disjoint.
    # This is a constraint satisfaction problem.
    for N in [30, 40, 50, 60, 80, 100, 150]:
        if time.time() > deadline - 3:
            break

        target_moduli = list(range(20, 20 + N))
        if 20 + N > 201:
            target_moduli = list(range(20, 201))

        # Backtracking search for residue assignment
        # For small N this is feasible; for large N we use greedy with backtracking
        def backtrack(idx, assignment):
            if time.time() > deadline - 3:
                return None
            if idx == len(target_moduli):
                return assignment[:]

            m = target_moduli[idx]
            # Try residues in random order for diversity
            residues = list(range(m))
            random.shuffle(residues)

            for r in residues:
                ok = True
                for i in range(idx):
                    if not _disjoint(m, r, target_moduli[i], assignment[i]):
                        ok = False
                        break
                if ok:
                    assignment[idx] = r
                    result = backtrack(idx + 1, assignment)
                    if result is not None:
                        return result
            return None

        assignment = [0] * len(target_moduli)
        result = backtrack(0, assignment)
        if result is not None:
            classes = [[target_moduli[i], result[i]] for i in range(len(target_moduli))]
            if _is_valid(classes):
                sc = _score(classes)
                if sc > best_score:
                    best_score = sc
                    best_classes = classes[:]

    # Strategy F: Seed fallback
    seed_classes = [[20 * j, j - 1] for j in range(1, 11)]
    seed_score = _score(seed_classes)
    if seed_score > best_score:
        best_score = seed_score
        best_classes = seed_classes[:]

    if best_classes is None:
        best_classes = seed_classes

    return best_classes
