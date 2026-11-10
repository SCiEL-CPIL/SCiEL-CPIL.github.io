import random
import time
import numpy as np

N = 16
M = N * (N - 1) // 2  # 120

# Precompute edge index: e(i,j) = i*(N-1) - i*(i+1)//2 + (j-i-1)
def eidx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-vertex subsets and their 3 cycles (each cycle = 3 edge indices)
CYCLES = []  # list of (4 vertices, [(e1,e2,e3), (e1,e2,e3), (e1,e2,e3)])
CYCLES_FOR_EDGE = [[] for _ in range(M)]

for a in range(N - 3):
    for b in range(a + 1, N - 2):
        for c in range(b + 1, N - 1):
            for d in range(c + 1, N):
                # Three perfect matchings of {a,b,c,d}
                matchings = [
                    (eidx(a, b), eidx(c, d), eidx(a, c)),  # (ab,cd) + (ac)
                    (eidx(a, c), eidx(b, d), eidx(a, b)),  # (ac,bd) + (ab)
                    (eidx(a, d), eidx(b, c), eidx(a, b)),  # (ad,bc) + (ab)
                ]
                # Actually, the three 4-cycles on 4 vertices:
                # Vertices a,b,c,d. The three cycles (up to reversal/rotation):
                # Cycle 1: a-b-c-d-a -> edges ab, bc, cd, da
                # Cycle 2: a-b-d-c-a -> edges ab, bd, dc, ca
                # Cycle 3: a-c-b-d-a -> edges ac, cb, bd, da
                # Each cycle has 4 edges.
                cycles = [
                    (eidx(a, b), eidx(b, c), eidx(c, d), eidx(d, a)),
                    (eidx(a, b), eidx(b, d), eidx(d, c), eidx(c, a)),
                    (eidx(a, c), eidx(c, b), eidx(b, d), eidx(d, a)),
                ]
                CYCLES.append((cycles))
                for cyc in cycles:
                    for e in cyc:
                        CYCLES_FOR_EDGE[e].append(len(CYCLES) - 1)

NCYC = len(CYCLES)  # C(16,4) = 1820

def count_colors(colors):
    return len(set(colors))

def check_cycle_ok(cyc, colors):
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    return not (len({c0, c1, c2, c3}) == 4)

def total_violations(colors):
    v = 0
    for cyc in CYCLES:
        if not check_cycle_ok(cyc, colors):
            v += 1
    return v

def construct():
    deadline = time.time() + 75.0  # safety margin for 90s limit
    rng = random.Random(12345)

    # Start with a random valid-ish coloring: use few colors initially
    # Strategy: start with 1 color (valid), then anneal to introduce more
    colors = [0] * M

    # Greedy initialization: try to introduce colors one at a time
    # Assign random colors but keep it mostly low-color
    # Better: start with a structured low-color assignment
    # Use a "block" structure: partition into groups, color within-group vs between-group
    # For now, start simple and let annealing do the work

    best_colors = colors[:]
    best_score = count_colors(colors)  # 1

    # Simulated annealing
    # Objective: maximize colors used, subject to 0 violations
    # We track: violations, and colors_used
    # Score function: colors_used - big_penalty * violations
    # But we want to maximize colors with 0 violations
    # Let's use: fitness = -violations * 1000 + colors_used

    def fitness(col):
        v = 0
        for cyc in CYCLES:
            if not check_cycle_ok(cyc, col):
                v += 1
        return -v * 1000 + count_colors(col)

    cur_fit = fitness(colors)
    best_fit = cur_fit

    # Phase 1: Greedy color introduction
    # Try to change each edge to a new color if it doesn't create violations
    for attempt in range(2000):
        if time.time() > deadline - 30:
            break
        # Pick a random edge
        e = rng.randrange(M)
        # Try a random new color (prefer unused ones)
        used = set(colors)
        unused = [c for c in range(120) if c not in used]
        if unused:
            new_c = rng.choice(unused)
        else:
            new_c = rng.randrange(120)
        if new_c == colors[e]:
            continue
        # Check if changing e to new_c keeps all affected cycles valid
        old_c = colors[e]
        ok = True
        for ci in CYCLES_FOR_EDGE[e]:
            cyc = CYCLES[ci]
            # Simulate: temporarily set
            colors[e] = new_c
            if not check_cycle_ok(cyc, colors):
                ok = False
                break
            colors[e] = old_c
        if ok:
            colors[e] = new_c
            cur_fit = fitness(colors)
            if cur_fit > best_fit:
                best_fit = cur_fit
                best_colors = colors[:]

    # Phase 2: Simulated annealing to escape local optima
    # Temperature schedule
    T = 50.0
    T_min = 0.01
    cooling = 0.999
    no_improve = 0

    while time.time() < deadline - 2:
        if T < T_min:
            break
        # Pick random edge
        e = rng.randrange(M)
        old_c = colors[e]
        # Choose new color: sometimes random, sometimes biased toward unused
        if rng.random() < 0.3:
            used = set(colors)
            unused = [c for c in range(120) if c not in used]
            if unused:
                new_c = rng.choice(unused)
            else:
                new_c = rng.randrange(120)
        else:
            new_c = rng.randrange(120)
        if new_c == old_c:
            continue

        # Compute delta fitness by checking only affected cycles
        # Old violations among affected cycles
        old_v = 0
        new_v = 0
        for ci in CYCLES_FOR_EDGE[e]:
            cyc = CYCLES[ci]
            if not check_cycle_ok(cyc, colors):
                old_v += 1
        colors[e] = new_c
        for ci in CYCLES_FOR_EDGE[e]:
            cyc = CYCLES[ci]
            if not check_cycle_ok(cyc, colors):
                new_v += 1
        # Color count change
        old_used = set(colors)
        # This is expensive; approximate
        # Actually let's just compute full fitness for simplicity in early stages
        # But that's slow. Let's track colors more carefully.

        # For speed, compute full fitness
        cur_fit_new = fitness(colors)
        delta = cur_fit_new - cur_fit
        if delta >= 0 or rng.random() < np.exp(delta / T):
            cur_fit = cur_fit_new
            if cur_fit > best_fit:
                best_fit = cur_fit
                best_colors = colors[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            colors[e] = old_c
            no_improve += 1

        T *= cooling
        if no_improve > 500:
            # Restart with perturbation
            no_improve = 0
            T = max(T, 10.0)
            # Perturb: change a few random edges
            for _ in range(5):
                pe = rng.randrange(M)
                colors[pe] = rng.randrange(120)
            cur_fit = fitness(colors)

    # Final: ensure we return the best valid coloring
    # Verify best_colors has 0 violations
    v = total_violations(best_colors)
    if v > 0:
        # Fallback: return a known valid low-color coloring
        return [0] * M

    return best_colors

# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Colors used: {len(set(res))}")
    v = total_violations(res)
    print(f"Violations: {v}")
