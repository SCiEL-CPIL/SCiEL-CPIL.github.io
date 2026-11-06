"""Minimum maximal no-isosceles set: MILP + local refinement."""
import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 16
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_POINTS = len(ALL_POINTS)
POINT_INDEX = {p: i for i, p in enumerate(ALL_POINTS)}

# Precompute all isosceles triples (a, b, c) where a is the apex
# and d(a,b) == d(a,c) with b != c.
# For each point a, group other points by squared distance.
# Each group of size k gives C(k,2) isosceles triples with apex a.
# But we only need the constraint: for each triple (a,b,c) with d(a,b)==d(a,c),
# at least one of a, b, c must be excluded.
# Equivalently: if b and c are both included, then a must be excluded.
# This is a 3-variable constraint: x_b + x_c - x_a <= 1

# Build the constraint matrix
# Each constraint: x_b + x_c - x_a <= 1, i.e., -x_a + x_b + x_c <= 1
# In standard form for milp: A_ub @ x <= b_ub

_constraints = []
for a in ALL_POINTS:
    # Group points by squared distance from a
    dist_groups = {}
    for b in ALL_POINTS:
        if b == a:
            continue
        d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        if d not in dist_groups:
            dist_groups[d] = []
        dist_groups[d].append(b)
    # For each group with >= 2 points, add constraints
    for d, group in dist_groups.items():
        if len(group) < 2:
            continue
        ia = POINT_INDEX[a]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ib = POINT_INDEX[group[i]]
                jc = POINT_INDEX[group[j]]
                # Constraint: x_b + x_c - x_a <= 1
                row = np.zeros(N_POINTS)
                row[ia] = -1
                row[ib] = 1
                row[jc] = 1
                _constraints.append(row)

_A_ub = np.array(_constraints) if _constraints else np.empty((0, N_POINTS))
_b_ub = np.ones(len(_constraints))

# Also need: x_i >= 0, x_i <= 1, integer
_integrality = np.ones(N_POINTS)
_lb = np.zeros(N_POINTS)
_ub = np.ones(N_POINTS)
_bounds = Bounds(_lb, _ub)

# Objective: minimize sum(x_i)
_c = np.ones(N_POINTS)

# Additional constraint: sum(x_i) >= 1 (trivial, but keep for safety)


def _check_valid(chosen):
    """Verify chosen is a valid no-isosceles set."""
    chosen_set = set(chosen)
    for a, b, c in itertools.combinations(chosen, 3):
        d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
        d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
        if d1 == d2 or d1 == d3 or d2 == d3:
            return False
    return True


def _check_maximal(chosen):
    """Check if no further point can be added without creating an isosceles triangle."""
    chosen_set = set(chosen)
    for p in ALL_POINTS:
        if p in chosen_set:
            continue
        # Try adding p
        valid = True
        for a, b in itertools.combinations(chosen, 2):
            d1 = (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
            d2 = (p[0] - b[0]) ** 2 + (p[1] - b[1]) ** 2
            d3 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d1 == d2 or d1 == d3 or d2 == d3:
                valid = False
                break
        if valid:
            return False  # Can add p, so not maximal
    return True


def _greedy_maximal():
    """Greedy construction to get a maximal valid set."""
    chosen = []
    for p in ALL_POINTS:
        if all(True for _ in []):  # placeholder
            pass
        # Check if adding p keeps no-isosceles property
        ok = True
        for a, b in itertools.combinations(chosen, 2):
            d1 = (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
            d2 = (p[0] - b[0]) ** 2 + (p[1] - b[1]) ** 2
            d3 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            if d1 == d2 or d1 == d3 or d2 == d3:
                ok = False
                break
        if ok:
            chosen.append(p)
    return chosen


def _try_remove(chosen, deadline):
    """Try to remove points while maintaining maximality. Returns smaller set or None."""
    chosen = list(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(chosen)):
            if time.time() >= deadline:
                break
            candidate = chosen[:i] + chosen[i + 1:]
            if not _check_valid(candidate):
                continue
            if _check_maximal(candidate):
                chosen = candidate
                improved = True
                break
    return chosen


def construct():
    deadline = time.time() + 280  # safety margin before 330s kill

    # Start with greedy maximal set
    best = _greedy_maximal()
    best_size = len(best)

    # Try MILP to find a minimum no-isosceles set (not necessarily maximal)
    try:
        result = milp(
            c=_c,
            integrality=_integrality,
            constraints=LinearConstraint(_A_ub, -np.inf, _b_ub),
            bounds=bounds,
            options={"time_limit": 120, "presolve": True},
        )
        if result.status == 0 and result.x is not None:
            x = result.x
            milp_set = [ALL_POINTS[i] for i in range(N_POINTS) if x[i] > 0.5]
            if _check_valid(milp_set):
                # Try to make it maximal by adding points
                milp_set = list(milp_set)
                for p in ALL_POINTS:
                    if time.time() >= deadline:
                        break
                    if p in milp_set:
                        continue
                    ok = True
                    for a, b in itertools.combinations(milp_set, 2):
                        d1 = (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
                        d2 = (p[0] - b[0]) ** 2 + (p[1] - b[1]) ** 2
                        d3 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                        if d1 == d2 or d1 == d3 or d2 == d3:
                            ok = False
                            break
                    if ok:
                        milp_set.append(p)
                if _check_valid(milp_set) and _check_maximal(milp_set):
                    if len(milp_set) < best_size:
                        best = milp_set
                        best_size = len(best)
    except Exception:
        pass

    # Local refinement: try to remove points
    if time.time() < deadline:
        refined = _try_remove(best, deadline)
        if refined is not None and len(refined) < best_size:
            best = refined
            best_size = len(refined)

    # Final verification
    if not _check_valid(best):
        # Fallback to greedy
        best = _greedy_maximal()

    return best
