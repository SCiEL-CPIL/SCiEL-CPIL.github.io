import itertools
import random
import time
import math
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
M = len(POINTS)  # 256

# ---------- Precompute ----------
def _build_cover():
    xs = np.array([p[0] for p in POINTS], dtype=np.int64)
    ys = np.array([p[1] for p in POINTS], dtype=np.int64)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    D2 = dx * dx + dy * dy
    A = np.arange(M)
    iu_a, iu_b = np.triu_indices(M, k=1)
    cover = []
    for p in range(M):
        d_pa = D2[p, A][:, None]
        d_pb = D2[p, A][None, :]
        iso = (d_pa == d_pb) | (d_pa == D2) | (d_pb == D2)
        a = iu_a; b = iu_b
        m = (a != p) & (b != p)
        a = a[m]; b = b[m]
        sel = iso[a, b]
        cover.append(list(zip(a[sel].tolist(), b[sel].tolist())))
    return cover

COVER = _build_cover()

# For fast compatibility check
COMPAT = [set(COVER[p]) for p in range(M)]

# Constraint density: number of isosceles constraints each point participates in
CONSTRAINT_DENSITY = [len(COVER[p]) for p in range(M)]


def _make_inS(S):
    inS = [False] * M
    for v in S:
        inS[v] = True
    return inS


def _compatible(v, inS):
    for a, b in COVER[v]:
        if inS[a] and inS[b]:
            return False
    return True


def _is_maximal(inS):
    for p in range(M):
        if inS[p]:
            continue
        for a, b in COVER[p]:
            if inS[a] and inS[b]:
                break
        else:
            return False
    return True


def _uncovered_count(inS):
    """Count points outside S that are not covered by any isosceles triple in S."""
    count = 0
    for p in range(M):
        if inS[p]:
            continue
        covered = False
        for a, b in COVER[p]:
            if inS[a] and inS[b]:
                covered = True
                break
        if not covered:
            count += 1
    return count


def _energy(inS):
    """Composite energy: |S| + 10 * (number of uncovered outside points).
    Lower is better. Maximality requires uncovered = 0."""
    size = sum(inS)
    uncovered = _uncovered_count(inS)
    return size + 10 * uncovered


def _greedy_constrained():
    """Greedy that prioritizes points with higher constraint density first.
    This tends to create denser coverage with fewer points."""
    S = set()
    inS = _make_inS(S)
    # Sort by constraint density descending, add randomness
    order = list(range(M))
    random.shuffle(order)
    order.sort(key=lambda p: -CONSTRAINT_DENSITY[p] + random.uniform(-0.5, 0.5))
    for v in order:
        if _compatible(v, inS):
            S.add(v)
            inS[v] = True
    return S


def _greedy_random():
    S = set()
    inS = _make_inS(S)
    order = list(range(M))
    random.shuffle(order)
    for v in order:
        if _compatible(v, inS):
            S.add(v)
            inS[v] = True
    return S


def _shrink(S):
    """Remove redundant points while maintaining maximality."""
    S = set(S)
    inS = _make_inS(S)
    if not _is_maximal(inS):
        return S  # Not maximal, can't shrink safely
    
    improved = True
    while improved:
        improved = False
        items = list(S)
        random.shuffle(items)
        for x in items:
            inS[x] = False
            S.discard(x)
            if _is_maximal(inS):
                improved = True
            else:
                S.add(x)
                inS[x] = True
    return S


def _repair_maximal(S):
    """Add points to make S maximal (if not already)."""
    S = set(S)
    inS = _make_inS(S)
    if _is_maximal(inS):
        return S
    order = list(range(M))
    random.shuffle(order)
    for v in order:
        if inS[v]:
            continue
        if _compatible(v, inS):
            S.add(v)
            inS[v] = True
    return S


def _simulated_annealing(initial_S, deadline):
    """Simulated annealing on set membership with composite energy."""
    S = set(initial_S)
    inS = _make_inS(S)
    current_energy = _energy(inS)
    
    best_S = set(S)
    best_energy = current_energy
    
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling = 0.995
    
    while time.time() < deadline:
        if T < T_min:
            break
        
        # Random move: add, remove, or swap a point
        move_type = random.random()
        if move_type < 0.3:
            # Add a random point
            candidates = [p for p in range(M) if not inS[p]]
            if not candidates:
                continue
            v = random.choice(candidates)
            if _compatible(v, inS):
                S.add(v)
                inS[v] = True
                new_energy = _energy(inS)
                delta = new_energy - current_energy
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_S = set(S)
                else:
                    S.discard(v)
                    inS[v] = False
        elif move_type < 0.6:
            # Remove a random point
            if not S:
                continue
            v = random.choice(list(S))
            S.discard(v)
            inS[v] = False
            new_energy = _energy(inS)
            delta = new_energy - current_energy
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_S = set(S)
            else:
                S.add(v)
                inS[v] = True
        else:
            # Swap: remove one, try to add another
            if not S:
                continue
            v_remove = random.choice(list(S))
            S.discard(v_remove)
            inS[v_remove] = False
            candidates = [p for p in range(M) if not inS[p]]
            if candidates:
                v_add = random.choice(candidates)
                if _compatible(v_add, inS):
                    S.add(v_add)
                    inS[v_add] = True
                    new_energy = _energy(inS)
                    delta = new_energy - current_energy
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_S = set(S)
                    else:
                        S.discard(v_add)
                        inS[v_add] = False
                        S.add(v_remove)
                        inS[v_remove] = True
            else:
                S.add(v_remove)
                inS[v_remove] = True
        
        T *= cooling
    
    return best_S


def construct():
    t0 = time.time()
    deadline = t0 + 290.0  # Safety margin
    
    # Phase 1: Generate diverse initial solutions using different strategies
    best = None
    best_size = M + 1
    
    # Use both constraint-density and random greedy
    for i in range(6):
        if i % 2 == 0:
            S = _greedy_constrained()
        else:
            S = _greedy_random()
        S = _shrink(S)
        if len(S) < best_size:
            best = S
            best_size = len(S)
    
    # Phase 2: Simulated annealing from best initial solution
    if time.time() < deadline - 30:
        sa_result = _simulated_annealing(best, deadline - 15)
        if len(sa_result) < best_size:
            best = sa_result
            best_size = len(sa_result)
    
    # Phase 3: Final shrink
    if time.time() < deadline - 5:
        best = _shrink(best)
    
    return [POINTS[i] for i in sorted(best)]
