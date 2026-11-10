import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Map pair (i,j), i<j, to edge index in lexicographic order
def _edge_index(i, j):
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Build edge list
EDGE_LIST = []
for i in range(N):
    for j in range(i + 1, N):
        EDGE_LIST.append((i, j))
EDGE_INDEX = {e: idx for idx, e in enumerate(EDGE_LIST)}

# Build the 3 cycles per 4-set
def _build_cycles():
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    cycles.append((_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(a, d)))
                    cycles.append((_edge_index(a, b), _edge_index(b, d), _edge_index(c, d), _edge_index(a, c)))
                    cycles.append((_edge_index(a, c), _edge_index(b, c), _edge_index(b, d), _edge_index(a, d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

# For each cycle, store the 4 edge indices as a numpy array
CYCLES_ARR = np.array(CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)

# For each edge, list of cycle indices containing it
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci in range(NUM_CYCLES):
    for e in CYCLES[ci]:
        EDGE_TO_CYCLES[e].append(ci)
EDGE_TO_CYCLES_ARR = [np.array(lst, dtype=np.int32) for lst in EDGE_TO_CYCLES]

def _is_rainbow(cyc, colors):
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3

def _structural_coloring(mode):
    """Return a valid base coloring with good structure."""
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    for idx, (i, j) in enumerate(EDGE_LIST):
        if mode == 0:
            colors[idx] = i
        elif mode == 1:
            colors[idx] = j
        elif mode == 2:
            colors[idx] = i + (j - i - 1) * 0  # same as i
        elif mode == 3:
            colors[idx] = (i * 2 + (j - i - 1)) % 15
        else:
            colors[idx] = i
    return colors

def _count_distinct(colors):
    return len(np.unique(colors))

def _is_move_valid(e, new_color, colors):
    """Check if setting edge e to new_color keeps all affected cycles non-rainbow."""
    cyc_list = EDGE_TO_CYCLES_ARR[e]
    if len(cyc_list) == 0:
        return True
    # For each affected cycle, check the new 4-tuple
    for ci in cyc_list:
        cyc = CYCLES[ci]
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if cyc[0] == e:
            c0 = new_color
        elif cyc[1] == e:
            c1 = new_color
        elif cyc[2] == e:
            c2 = new_color
        elif cyc[3] == e:
            c3 = new_color
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _local_search(colors, deadline, seed):
    """Hill-climb to maximize distinct colors, maintaining validity."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    colors = colors.copy()
    color_counts = np.zeros(MAX_COLOR + 1, dtype=np.int32)
    for c in colors:
        color_counts[c] += 1
    current_distinct = int(np.count_nonzero(color_counts))
    best_distinct = current_distinct
    best_colors = colors.copy()

    no_improve = 0
    max_no_improve = 4000

    while time.time() < deadline and no_improve < max_no_improve:
        e_idx = np_rng.randint(0, NUM_EDGES)
        old_color = int(colors[e_idx])

        # Choose new color: bias toward unused colors
        r = np_rng.random_sample()
        if r < 0.5:
            # pick an unused color if any
            unused = np.where(color_counts == 0)[0]
            if len(unused) > 0:
                new_color = int(unused[np_rng.randint(0, len(unused))])
            else:
                new_color = int(np_rng.randint(0, MAX_COLOR + 1))
        else:
            new_color = int(np_rng.randint(0, MAX_COLOR + 1))

        if new_color == old_color:
            continue

        if not _is_move_valid(e_idx, new_color, colors):
            no_improve += 1
            continue

        # Compute delta distinct
        color_counts[old_color] -= 1
        color_counts[new_color] += 1
        new_distinct = current_distinct
        if color_counts[old_color] == 0:
            new_distinct -= 1
        if color_counts[new_color] == 1:
            new_distinct += 1

        if new_distinct >= current_distinct:
            # accept (improvement or tie)
            colors[e_idx] = new_color
            current_distinct = new_distinct
            no_improve = 0
            if new_distinct > best_distinct:
                best_distinct = new_distinct
                best_colors = colors.copy()
        else:
            # reject, revert
            color_counts[old_color] += 1
            color_counts[new_color] -= 1
            no_improve += 1

    return best_colors, best_distinct

def _perturb(colors, k, deadline, seed):
    """Randomly reassign up to k edges to unused/other colors, keeping validity."""
    rng = np.random.RandomState(seed)
    colors = colors.copy()
    color_counts = np.zeros(MAX_COLOR + 1, dtype=np.int32)
    for c in colors:
        color_counts[c] += 1
    edges = list(range(NUM_EDGES))
    rng.shuffle(edges)
    for e in edges[:k]:
        new_color = int(rng.randint(0, MAX_COLOR + 1))
        if _is_move_valid(e, new_color, colors):
            old = int(colors[e])
            color_counts[old] -= 1
            color_counts[new_color] += 1
            colors[e] = new_color
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    best_overall = None
    best_score = -1

    # Try multiple structural bases and local search from each
    for mode in range(4):
        if time.time() > deadline - 5.0:
            break
        base = _structural_coloring(mode)
        # local search
        sub_deadline = min(deadline - 5.0, start_time + 12.0)
        res_colors, res_distinct = _local_search(base, sub_deadline, seed=1000 + mode)
        if res_distinct > best_score:
            best_score = res_distinct
            best_overall = res_colors

    # Multi-restart: perturb best and re-climb
    remaining_restarts = 0
    while time.time() < deadline - 2.0:
        base = _perturb(best_overall, k=8, deadline=deadline, seed=5000 + remaining_restarts)
        sub_deadline = min(deadline - 2.0, time.time() + 8.0)
        res_colors, res_distinct = _local_search(base, sub_deadline, seed=9000 + remaining_restarts)
        if res_distinct > best_score:
            best_score = res_distinct
            best_overall = res_colors
        remaining_restarts += 1
        if remaining_restarts > 200:
            break

    # Final: verify best_overall
    if best_overall is None:
        best_overall = _structural_coloring(0)

    # Verify validity
    for cyc in CYCLES:
        if _is_rainbow(cyc, best_overall):
            # fallback to structural
            best_overall = _structural_coloring(0)
            break

    return best_overall.tolist()
