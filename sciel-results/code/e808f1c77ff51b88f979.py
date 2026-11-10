"""Local-search colorings of K16 maximizing distinct colors with no rainbow 4-cycle."""
import time
import random

N = 16
M = N * (N - 1) // 2  # 120 edges
MAX_COLOR = 120
DEADLINE = 78.0  # safety margin before 90s kill

# Precompute edge index for (i, j) with i < j
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Build all 4-cycles (each is 4 edges)
def build_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # 3 distinct 4-cycles on {a,b,c,d}
                    # cycle1: a-b-c-d-a
                    e1 = edge_idx(a, b)
                    e2 = edge_idx(b, c)
                    e3 = edge_idx(c, d)
                    e4 = edge_idx(d, a)
                    cycles.append((e1, e2, e3, e4))
                    # cycle2: a-b-d-c-a
                    e1 = edge_idx(a, b)
                    e2 = edge_idx(b, d)
                    e3 = edge_idx(d, c)
                    e4 = edge_idx(c, a)
                    cycles.append((e1, e2, e3, e4))
                    # cycle3: a-c-b-d-a
                    e1 = edge_idx(a, c)
                    e2 = edge_idx(c, b)
                    e3 = edge_idx(b, d)
                    e4 = edge_idx(d, a)
                    cycles.append((e1, e2, e3, e4))
    return cycles

CYCLES = build_cycles()
NCYC = len(CYCLES)

# For each edge, list of (cycle_index, position) so we can update rainbow counts fast
EDGE_TO_CYCLES = [[] for _ in range(M)]
for ci, cyc in enumerate(CYCLES):
    for pos, e in enumerate(cyc):
        EDGE_TO_CYCLES[e].append((ci, pos))

# Store cycle edges in a flat array for speed
CYCLE_EDGES = [cyc for cyc in CYCLES]

def color_count(colors):
    return len(set(colors))

def is_rainbow(cyc, colors):
    c0 = colors[cyc[0]]
    c1 = colors[cyc[1]]
    c2 = colors[cyc[2]]
    c3 = colors[cyc[3]]
    return len((c0, c1, c2, c3)) == 4

def count_rainbows(colors):
    cnt = 0
    for cyc in CYCLE_EDGES:
        c0 = colors[cyc[0]]
        c1 = colors[cyc[1]]
        c2 = colors[cyc[2]]
        c3 = colors[cyc[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            cnt += 1
    return cnt

def construct():
    start = time.time()
    rng = random.Random(12345)

    # Precompute edge endpoints for reporting (not needed for logic)
    # Initial coloring: assign each edge a random color in 0..119
    best_colors = None
    best_score = -1
    best_rainbows = None

    def evaluate_and_maybe_update(colors):
        nonlocal best_colors, best_score, best_rainbows
        cc = color_count(colors)
        rb = count_rainbows(colors)
        # score: we want high cc and rb==0. Use a composite.
        # Primary: number of colors. Secondary: minimize rainbows.
        score = cc * 1000 - rb
        if score > best_score:
            best_score = score
            best_colors = list(colors)
            best_rainbows = rb

    # We'll run multiple restarts with different strategies.
    for restart in range(200):
        if time.time() - start > DEADLINE:
            break

        # Initialize
        if restart == 0:
            # Start with a structured low-color pattern, then diversify
            colors = [0] * M
            # Give each edge a color based on something to break symmetry slightly
            for e in range(M):
                colors[e] = e % 5
        else:
            colors = [rng.randrange(MAX_COLOR) for _ in range(M)]

        evaluate_and_maybe_update(colors)

        # Local search: try single-edge color changes
        # Maintain rainbow count incrementally
        # For speed, recompute affected cycles only
        # We'll keep an array of rainbow flags per cycle
        rainbow = [0] * NCYC
        for ci, cyc in enumerate(CYCLE_EDGES):
            c0 = colors[cyc[0]]
            c1 = colors[cyc[1]]
            c2 = colors[cyc[2]]
            c3 = colors[cyc[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                rainbow[ci] = 1
        total_rb = sum(rainbow)

        # color frequency
        freq = [0] * MAX_COLOR
        for c in colors:
            freq[c] += 1

        order = list(range(M))
        rng.shuffle(order)

        # Iterate several passes
        for passnum in range(30):
            if time.time() - start > DEADLINE:
                break
            for e in order:
                if time.time() - start > DEADLINE:
                    break
                old = colors[e]
                # Decide candidate new color
                # Strategy: if currently many rainbows, try to pick a color that reduces rainbows.
                # Otherwise, try to introduce a new color.
                if total_rb > 0:
                    # Repair: choose a color that appears in one of the edges of a rainbow cycle containing e
                    # Collect candidate colors from cycles containing e that are rainbow
                    cand = set()
                    for (ci, pos) in EDGE_TO_CYCLES[e]:
                        if rainbow[ci]:
                            cyc = CYCLE_EDGES[ci]
                            for p in range(4):
                                if p != pos:
                                    cand.add(colors[cyc[p]])
                    if cand:
                        # pick one randomly
                        new = rng.choice(list(cand))
                    else:
                        new = old
                else:
                    # Valid: try to add a new color not used on this edge's cycles to increase diversity
                    # But must keep valid. Simpler: try a random new color and check.
                    # We'll just try a few random colors and accept first valid one that increases diversity.
                    new = old
                    # Try to pick a color not currently used at all (to increase count) if possible
                    # But that may create rainbows. Instead, try random colors.
                    # We'll do a small local probe.
                    improved = False
                    for _ in range(8):
                        nc = rng.randrange(MAX_COLOR)
                        if nc == old:
                            continue
                        # check validity of changing e to nc
                        # compute delta rainbow
                        ok = True
                        # We'll compute new rainbow for affected cycles
                        delta = 0
                        for (ci, pos) in EDGE_TO_CYCLES[e]:
                            cyc = CYCLE_EDGES[ci]
                            # old colors
                            old_c = [colors[cyc[p]] if p != pos else old for p in range(4)]
                            new_c = [colors[cyc[p]] if p != pos else nc for p in range(4)]
                            old_rb = 1 if len(set(old_c)) == 4 else 0
                            new_rb = 1 if len(set(new_c)) == 4 else 0
                            delta += new_rb - old_rb
                            if new_rb == 1:
                                ok = False
                        # We require no new rainbows (keep valid)
                        if delta == 0:
                            # accept if it increases color count
                            if freq[old] == 1 and freq[nc] == 0:
                                new = nc
                                improved = True
                                break
                            elif freq[old] > 1 and freq[nc] == 0:
                                new = nc
                                improved = True
                                break
                            elif freq[nc] == 0 and nc != old:
                                # introduces a new color even if old color still used
                                new = nc
                                improved = True
                                break
                    if not improved:
                        # try a random valid change to escape local minima
                        for _ in range(4):
                            nc = rng.randrange(MAX_COLOR)
                            if nc == old:
                                continue
                            ok = True
                            for (ci, pos) in EDGE_TO_CYCLES[e]:
                                cyc = CYCLE_EDGES[ci]
                                new_c = [colors[cyc[p]] if p != pos else nc for p in range(4)]
                                if len(set(new_c)) == 4:
                                    ok = False
                                    break
                            if ok:
                                new = nc
                                break

                if new == old:
                    continue

                # Apply change and update structures
                # Compute delta rainbow
                delta = 0
                for (ci, pos) in EDGE_TO_CYCLES[e]:
                    cyc = CYCLE_EDGES[ci]
                    new_c = [colors[cyc[p]] if p != pos else new for p in range(4)]
                    new_rb = 1 if len(set(new_c)) == 4 else 0
                    old_rb = rainbow[ci]
                    delta += new_rb - old_rb
                    rainbow[ci] = new_rb
                if delta == 0:
                    # accept (validity preserved)
                    freq[old] -= 1
                    freq[new] += 1
                    colors[e] = new
                    total_rb += delta
                    # occasionally evaluate global
                    if (e & 63) == 0:
                        evaluate_and_maybe_update(colors)
                else:
                    # revert rainbow flags
                    for (ci, pos) in EDGE_TO_CYCLES[e]:
                        cyc = CYCLE_EDGES[ci]
                        old_c = [colors[cyc[p]] if p != pos else old for p in range(4)]
                        rainbow[ci] = 1 if len(set(old_c)) == 4 else 0

        # End of passes: evaluate
        evaluate_and_maybe_update(colors)

        # If we have a valid coloring, record it
        if best_rainbows == 0:
            # try to further improve via more aggressive diversification
            pass

    # Ensure best_colors is valid; if not, fall back to all-zeros (valid, 1 color)
    if best_colors is None:
        best_colors = [0] * M
    else:
        # Final validation and repair
        rb = count_rainbows(best_colors)
        if rb > 0:
            # Repair: force colors to reduce rainbows. Simple: recolor edges in rainbow cycles to match a neighbor.
            # Iterate
            for _ in range(2000):
                if time.time() - start > DEADLINE:
                    break
                # find a rainbow cycle
                found = False
                for ci, cyc in enumerate(CYCLE_EDGES):
                    c0 = best_colors[cyc[0]]
                    c1 = best_colors[cyc[1]]
                    c2 = best_colors[cyc[2]]
                    c3 = best_colors[cyc[3]]
                    if len((c0, c1, c2, c3)) == 4:
                        # set edge cyc[3] to color of cyc[0]
                        best_colors[cyc[3]] = c0
                        found = True
                        break
                if not found:
                    break
            # If still invalid, fall back to all zeros
            if count_rainbows(best_colors) > 0:
                best_colors = [0] * M

    return best_colors
