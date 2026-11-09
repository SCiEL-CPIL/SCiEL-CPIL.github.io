"""Rewritten: fast NumPy scoring + greedy/SA/local-search + coverage-removal."""
import time
import random
import numpy as np

BASE = 51975
_DIVS = []
for a in range(4):
    for b in range(3):
        for c in range(2):
            for d in range(2):
                v = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if v > 1:
                    _DIVS.append(v)
MODULI = sorted(_DIVS)
NMOD = len(MODULI)
COUNTS = np.array([BASE // m for m in MODULI], dtype=np.int64)

# For each modulus i, precompute the index arrays for each residue r: idx where (x % m) == r.
# That's memory-heavy; instead precompute the base template (residue 0) and use slicing by start=r step=m.
# We compute coverage by summing over active moduli: for residue r, positions are r, r+m, r+2m, ...
# Represent as boolean via (np.arange(BASE) % m) == r, but cache the % m array per modulus.
_ARANGE = np.arange(BASE, dtype=np.int64)
_MOD_ARR = [(_ARANGE % m) for m in MODULI]  # each length BASE, int64

DEADLINE = 142.0


def _coverage_from(assign, active):
    """Return coverage count array (int8) for given assign (len NMOD) and active (bool len NMOD)."""
    cov = np.zeros(BASE, dtype=np.int8)
    for i in np.nonzero(active)[0]:
        m = MODULI[i]
        r = int(assign[i])
        cov[_MOD_ARR[i] == r] += 1
    return cov


def _uncovered(cov):
    return int(np.count_nonzero(cov == 0))


def _score(assign, active):
    cov = _coverage_from(assign, active)
    return _uncovered(cov)


def _remove_redundant(assign, active):
    """Remove active classes whose removal does not increase uncovered count."""
    changed = True
    while changed:
        changed = False
        cov = _coverage_from(assign, active)
        base_unc = _uncovered(cov)
        for i in np.nonzero(active)[0]:
            active[i] = False
            cov2 = _coverage_from(assign, active)
            unc2 = _uncovered(cov2)
            if unc2 <= base_unc:
                base_unc = unc2
                changed = True
            else:
                active[i] = True
    return active


def _greedy_init():
    """Greedy: add moduli one by one, each choosing residue minimizing uncovered."""
    active = np.zeros(NMOD, dtype=bool)
    assign = np.zeros(NMOD, dtype=np.int64)
    # Start with the largest moduli (cover most at once) but choose residue well.
    # Use a heuristic order: larger moduli first to reduce uncovered faster.
    order = sorted(range(NMOD), key=lambda i: -MODULI[i])
    cov = np.zeros(BASE, dtype=np.int8)
    for i in order:
        m = MODULI[i]
        best_r = 0
        best_unc = BASE + 1
        # Evaluate residues; for large m there are many residues. Sample if too many.
        residues = np.arange(m)
        if m > 400:
            residues = residues[:: max(1, m // 400)]
        for r in residues:
            mask = _MOD_ARR[i] == r
            newcov = cov + mask.astype(np.int8)
            unc = int(np.count_nonzero(newcov == 0))
            if unc < best_unc:
                best_unc = unc
                best_r = int(r)
        # Refine around best_r among full residues near it
        lo = max(0, best_r - 5)
        hi = min(m, best_r + 6)
        for r in range(lo, hi):
            mask = _MOD_ARR[i] == r
            newcov = cov + mask.astype(np.int8)
            unc = int(np.count_nonzero(newcov == 0))
            if unc < best_unc:
                best_unc = unc
                best_r = r
        assign[i] = best_r
        active[i] = True
        cov[_MOD_ARR[i] == best_r] += 1
    # Remove redundant
    active = _remove_redundant(assign, active)
    return assign, active


def construct():
    t0 = time.time()
    random.seed(12345)
    np.random.seed(12345)

    assign, active = _greedy_init()
    best_unc = _score(assign, active)
    best_assign = assign.copy()
    best_active = active.copy()

    # Local search: single-residue moves on active moduli
    def local_search(assign, active, unc, max_iters=2000):
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
            improved = False
            act_idx = np.nonzero(active)[0]
            random.shuffle(act_idx.tolist())
            for i in act_idx:
                m = MODULI[i]
                old_r = int(assign[i])
                # Try a few candidate residues
                cands = [random.randrange(m) for _ in range(3)]
                for r in cands:
                    if r == old_r:
                        continue
                    assign[i] = r
                    cov = _coverage_from(assign, active)
                    u = _uncovered(cov)
                    if u < unc:
                        unc = u
                        improved = True
                        break
                    else:
                        assign[i] = old_r
            if not improved:
                break
        return assign, active, unc

    assign, active, best_unc = local_search(assign, active, best_unc)
    best_assign = assign.copy()
    best_active = active.copy()

    # Simulated annealing with restarts
    T = 200.0
    cool = 0.999
    while time.time() < DEADLINE - 5:
        # Restart from best
        assign = best_assign.copy()
        active = best_active.copy()
        unc = _score(assign, active)
        # Anneal
        for step in range(3000):
            if time.time() > DEADLINE - 2:
                break
            act_idx = np.nonzero(active)[0]
            if len(act_idx) == 0:
                break
            i = act_idx[random.randrange(len(act_idx))]
            m = MODULI[i]
            old_r = int(assign[i])
            new_r = random.randrange(m)
            if new_r == old_r:
                continue
            assign[i] = new_r
            cov = _coverage_from(assign, active)
            u = _uncovered(cov)
            d = u - unc
            if d <= 0 or random.random() < np.exp(-d / max(T, 1e-6)):
                unc = u
                if unc < best_unc:
                    best_unc = unc
                    best_assign = assign.copy()
                    best_active = active.copy()
            else:
                assign[i] = old_r
            T *= cool
        # Local search after each anneal
        assign, active, unc = local_search(assign, active, unc, max_iters=500)
        if unc < best_unc:
            best_unc = unc
            best_assign = assign.copy()
            best_active = active.copy()
        # Occasionally try adding a previously-inactive modulus
        if time.time() < DEADLINE - 10:
            inactive = np.nonzero(~best_active)[0]
            if len(inactive) > 0:
                i = inactive[random.randrange(len(inactive))]
                m = MODULI[i]
                # pick best residue for this modulus given current coverage
                cov = _coverage_from(best_assign, best_active)
                best_r = 0
                best_u = _uncovered(cov)
                residues = np.arange(m)
                if m > 300:
                    residues = residues[:: max(1, m // 300)]
                for r in residues:
                    newcov = cov + (_MOD_ARR[i] == r).astype(np.int8)
                    u = _uncovered(newcov)
                    if u < best_u:
                        best_u = u
                        best_r = int(r)
                if best_u < best_unc:
                    trial_active = best_active.copy()
                    trial_active[i] = True
                    trial_assign = best_assign.copy()
                    trial_assign[i] = best_r
                    trial_active = _remove_redundant(trial_assign, trial_active)
                    u2 = _score(trial_assign, trial_active)
                    if u2 < best_unc:
                        best_unc = u2
                        best_assign = trial_assign
                        best_active = trial_active

    # Final redundant removal and local search
    best_active = _remove_redundant(best_assign, best_active)
    best_assign, best_active, best_unc = local_search(best_assign, best_active, best_unc, max_iters=1000)
    best_active = _remove_redundant(best_assign, best_active)

    classes = []
    for i in np.nonzero(best_active)[0]:
        classes.append([int(MODULI[i]), int(best_assign[i])])
    return classes
