# EVOLVE-BLOCK-START
import time
import random

# Hex-lattice neighbors (unit-distance moves)
NEIGH = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))

def _edges_of(sel, nbr):
    """Count unit-distance pairs among the selected points."""
    c = 0
    for i in sel:
        for nb in nbr[i]:
            if nb > i and nb in sel:
                c += 1
    return c

def construct():
    start = time.time()
    DL = 140.0            # hard safety margin before the 150 s kill
    def dl():
        return time.time() < DL

    # ---- Build the search space: all lattice points inside a hexagon of radius R ----
    R = 6
    cand = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if max(abs(a), abs(b), abs(a + b)) <= R:
                cand.append((a, b))
    N = len(cand)
    idx = {p: i for i, p in enumerate(cand)}
    nbr = [[] for _ in range(N)]
    for i, (a, b) in enumerate(cand):
        for da, db in NEIGH:
            j = idx.get((a + da, b + db))
            if j is not None and j != i:
                nbr[i].append(j)

    NSEL = 32
    # ---- Deterministic, objective-guided construction of an initial 32-set ----
    core = set(i for i in range(N)
               if max(abs(cand[i][0]), abs(cand[i][1]), abs(cand[i][0] + cand[i][1])) <= 2)
    sel = set(core)
    while len(sel) < NSEL:
        best = None
        bg = -1
        for i in range(N):
            if i in sel:
                continue
            g = sum(1 for nb in nbr[i] if nb in sel)
            if g > bg:
                bg = g
                best = i
        if best is None:
            break
        sel.add(best)
    if len(sel) < NSEL:
        for i in range(N):
            if i not in sel:
                sel.add(i)
            if len(sel) == NSEL:
                break

    best_edges = _edges_of(sel, nbr)
    best_sel = set(sel)
    cur = best_edges

    # ---- Fast single-swap local search with random restarts ----
    random.seed(42)
    
    for restart in range(10): # Limit restarts to ensure time safety
        if not dl():
            break
            
        # If not first restart, perturb the current best solution
        if restart > 0:
            sel = set(best_sel)
            # Randomly swap 5 points to escape local optimum
            sel_list = list(sel)
            for _ in range(5):
                x = random.choice(sel_list)
                # Find a random y not in sel
                while True:
                    y = random.randint(0, N - 1)
                    if y not in sel:
                        break
                sel.remove(x)
                sel.add(y)
        
        cur = _edges_of(sel, nbr)
        if cur > best_edges:
            best_edges = cur
            best_sel = set(sel)
        
        improved = True
        while improved and dl():
            improved = False
            # Shuffle order of removals to avoid bias
            for x in random.sample(list(sel), len(sel)):
                if not dl():
                    break
                gx = sum(1 for nb in nbr[x] if nb in sel)
                for y in range(N):
                    if y in sel:
                        continue
                    gy = sum(1 for nb in nbr[y] if nb in sel)
                    ov = 0
                    for nb in nbr[x]:
                        if nb in sel and nb in nbr[y]:
                            ov += 1
                    d = gy + gx - 2 * ov
                    if d > 0:
                        sel.remove(x)
                        sel.add(y)
                        cur += d
                        improved = True
                        if cur > best_edges:
                            best_edges = cur
                            best_sel = set(sel)
                        if best_edges >= 76:
                            break
                if best_edges >= 76:
                    break
            if best_edges >= 76:
                break

    # ---- Finalize & validate ----
    sel = best_sel
    pts = [cand[i] for i in sorted(sel)]
    assert len(pts) == NSEL
    assert len(set(pts)) == NSEL
    for a, b in pts:
        assert abs(a) <= 64 and abs(b) <= 64
    return [[a, b] for a, b in pts]
# EVOLVE-BLOCK-END
