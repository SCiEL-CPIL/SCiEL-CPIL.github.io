import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of 4 edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = (edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a))
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = (edge_index(a, b), edge_index(b, d), edge_index(d, c), edge_index(c, a))
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = (edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a))
                    cycles.append(e3)
    return cycles

_CYCLES = build_four_cycles()
_CYCLES_BY_EDGE = [[] for _ in range(NUM_EDGES)]
for cyc in _CYCLES:
    for e in cyc:
        _CYCLES_BY_EDGE[e].append(cyc)

def validate(colors):
    """Check that no 4-cycle has 4 distinct colors. Returns True if valid."""
    for cyc in _CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if c0 == c1 or c0 == c2 or c0 == c3:
            continue
        if c1 == c2 or c1 == c3:
            continue
        if c2 == c3:
            continue
        return False
    return True

def count_colors(colors):
    return len(set(colors))

def generate_vertex_based(rng, K_range):
    """Color edge (i,j) based on vertex coloring: color = (v[i] + v[j]) mod K."""
    K = rng.randint(*K_range)
    v = [rng.randint(0, K - 1) for _ in range(N)]
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_index(i, j)] = (v[i] + v[j]) % K
    return colors

def generate_modular(rng):
    """Color edge (i,j) = (i * a + j * b) mod K."""
    K = rng.randint(10, 60)
    a = rng.randint(1, K - 1)
    b = rng.randint(1, K - 1)
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_index(i, j)] = (i * a + j * b) % K
    return colors

def generate_block(rng):
    """Split vertices into blocks."""
    block_size = rng.choice([2, 3, 4, 5])
    block_of = [i // block_size for i in range(N)]
    K = rng.randint(15, 80)
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            if block_of[i] == block_of[j]:
                colors[edge_index(i, j)] = rng.randint(0, K - 1)
            else:
                colors[edge_index(i, j)] = (block_of[i] * 7 + block_of[j] * 13) % K
    return colors

def generate_random_sparse(rng):
    """Mostly one color, with a few random edges getting unique colors."""
    base = rng.randint(0, MAX_COLOR)
    colors = [base] * NUM_EDGES
    n_unique = rng.randint(20, 60)
    for _ in range(n_unique):
        e = rng.randint(0, NUM_EDGES - 1)
        colors[e] = rng.randint(0, MAX_COLOR)
    return colors

def generate_greedy_fill(rng):
    """Greedy: assign each edge a random color that keeps validity."""
    colors = [0] * NUM_EDGES
    edge_order = list(range(NUM_EDGES))
    rng.shuffle(edge_order)
    for e in edge_order:
        # Try random colors
        for _ in range(50):
            c = rng.randint(0, MAX_COLOR)
            colors[e] = c
            valid = True
            for cyc in _CYCLES_BY_EDGE[e]:
                c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
                if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                    valid = False
                    break
            if valid:
                break
    return colors

def local_improve(colors, rng, deadline):
    """Try to increase the number of distinct colors via local search."""
    colors = colors[:]
    best_count = count_colors(colors)
    color_set = set(colors)
    
    while time.time() < deadline:
        improved = False
        for _ in range(500):
            if time.time() >= deadline:
                break
            e = rng.randint(0, NUM_EDGES - 1)
            old = colors[e]
            
            # Try to assign a new color not in the set
            if len(color_set) < MAX_COLOR + 1:
                # Find a color not in the set
                new_c = rng.randint(0, MAX_COLOR)
                if new_c in color_set:
                    # Try to find an unused color
                    for _ in range(20):
                        new_c = rng.randint(0, MAX_COLOR)
                        if new_c not in color_set:
                            break
                    else:
                        new_c = rng.randint(0, MAX_COLOR)
            
            colors[e] = new_c
            # Check only cycles containing edge e
            valid = True
            for cyc in _CYCLES_BY_EDGE[e]:
                c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
                if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                    valid = False
                    break
            
            if valid:
                if new_c not in color_set:
                    color_set.add(new_c)
                    improved = True
                # Keep the change even if not new, for diversity
                break
            else:
                colors[e] = old
        
        cur = count_colors(colors)
        if cur > best_count:
            best_count = cur
        if not improved and cur == best_count:
            # Try a few more rounds
            pass
    
    return colors, best_count

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 90s limit
    
    rng = random.Random(12345)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Generate diverse candidates
    generators = [
        lambda: generate_vertex_based(rng, (20, 120)),
        lambda: generate_modular(rng),
        lambda: generate_block(rng),
        lambda: generate_random_sparse(rng),
        lambda: generate_greedy_fill(rng),
    ]
    
    candidates = []
    for gen in generators:
        for _ in range(10):
            if time.time() >= deadline - 15:
                break
            c = gen()
            if validate(c):
                cnt = count_colors(c)
                candidates.append((cnt, c))
    
    # Also try random starts
    for _ in range(10):
        if time.time() >= deadline - 15:
            break
        c = [rng.randint(0, MAX_COLOR) for _ in range(NUM_EDGES)]
        if validate(c):
            cnt = count_colors(c)
            candidates.append((cnt, c))
    
    # Sort by count descending
    candidates.sort(key=lambda x: -x[0])
    top_candidates = candidates[:15]
    
    if top_candidates:
        best_count = top_candidates[0][0]
        best_colors = top_candidates[0][1][:]
    
    # Local improvement on top candidates
    for cnt, c in top_candidates:
        if time.time() >= deadline - 5:
            break
        improved_c, improved_cnt = local_improve(c, rng, deadline - 2)
        if improved_cnt > best_count:
            best_count = improved_cnt
            best_colors = improved_c
    
    # Final validation
    if not validate(best_colors):
        best_colors = [0] * NUM_EDGES
        best_count = 1
    
    return best_colors
