# EVOLVE-BLOCK-START
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    # Bit index per clause; we store unsat as an int bitset.
    # For each variable i, occ[i] is a bitset over clause positions where var i appears.
    # If the variable is False, the "currently satisfied-by-this-var" set is the
    # positions where it appears positively; if True, the negative positions.
    pos = [0] * n
    neg = [0] * n
    for ci, clause in enumerate(clauses):
        bit = 1 << ci
        for lit in clause:
            v = abs(lit) - 1
            if lit > 0:
                pos[v] |= bit
            else:
                neg[v] |= bit

    # Initial random assignment.
    rng = random.Random(0)
    asg = [rng.random() < 0.5 for _ in range(n)]

    # Build initial unsat bitset and per-variable unsat counts.
    unsat = 0
    counts = [0] * n
    for ci, clause in enumerate(clauses):
        ok = False
        for lit in clause:
            v = abs(lit) - 1
            if asg[v] == (lit > 0):
                ok = True
                break
        if not ok:
            unsat |= (1 << ci)
            for lit in clause:
                counts[abs(lit) - 1] += 1

    if unsat == 0:
        return asg

    # Precompute total occurrences per variable for the inverse-frequency prior.
    tot_occ = [0] * n
    for i in range(n):
        tot_occ[i] = bin(pos[i] | neg[i]).count("1")

    def active_mask(i):
        return pos[i] if asg[i] else neg[i]

    # Budget of flips.
    B = 300000
    for step in range(B):
        # Pick an unsatisfied clause (random among the set bits).
        c = (rng.getrandbits(63) % m)
        while not (unsat >> c) & 1:
            c = (c + 1) % m
        clause = clauses[c]

        # Candidate variables in this clause.
        cand = [abs(lit) - 1 for lit in clause]

        # Inverse-frequency weights: rare variables in the unsat set are
        # preferred, helping traverse flat regions.
        weights = []
        for v in cand:
            w = 1.0 / (1 + counts[v])
            weights.append(w)
        s = sum(weights)
        r = rng.random() * s
        acc = 0.0
        chosen = cand[0]
        for v, w in zip(cand, weights):
            acc += w
            if r <= acc:
                chosen = v
                break

        # Flip chosen variable, updating unsat bitset and counts.
        v = chosen
        asg[v] = not asg[v]
        unsat ^= active_mask(v)

        # Update counts for variables in clauses whose satisfied-status changed.
        changed = active_mask(v)  # clauses affected by this flip
        while changed:
            b = (changed & -changed).bit_length() - 1
            changed &= changed - 1
            ci = b
            clause = clauses[ci]
            sat = False
            for lit in clause:
                vv = abs(lit) - 1
                if asg[vv] == (lit > 0):
                    sat = True
                    break
            is_unsat = not sat
            for lit in clause:
                vv = abs(lit) - 1
                if is_unsat:
                    counts[vv] += 1
                else:
                    counts[vv] -= 1

        if unsat == 0:
            break

    return asg
# EVOLVE-BLOCK-END
