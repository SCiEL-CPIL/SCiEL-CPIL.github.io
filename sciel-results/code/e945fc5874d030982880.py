import time
import random
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_data():
    """Precompute all 4-cycles and adjacency."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three 4-cycles on {a,b,c,d}
                    cycles.append((edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)))
                    cycles.append((edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a)))
                    cycles.append((edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a)))
    
    cycles_by_edge = [[] for _ in range(NUM_EDGES)]
    for idx, cycle in enumerate(cycles):
        for e in cycle:
            cycles_by_edge[e].append(idx)
    
    return cycles, cycles_by_edge

def count_colors(colors):
    return len(set(colors))

def is_valid_move(colors, e, new_c, cycles_by_edge, cycles):
    """Check if recoloring edge e to new_c is valid (no 4-cycle gets 4 distinct colors)."""
    for ci in cycles_by_edge[e]:
        cyc = cycles[ci]
        c0, c1, c2, c3 = None, None, None, None
        cnt = 0
        for oe in cyc:
            if oe == e:
                continue
            c = colors[oe]
            if c is not None:
                if c0 is None:
                    c0 = c
                elif c1 is None:
                    if c != c0:
                        c1 = c
                    else:
                        cnt = 1  # duplicate, but we track distinct
                        continue
                elif c2 is None:
                    if c != c0 and c != c1:
                        c2 = c
                    else:
                        continue
                else:
                    if c != c0 and c != c1 and c != c2:
                        return False  # 4 distinct already without new_c
                cnt += 1
        # Now check adding new_c
        # Distinct existing colors in cycle: count them
        existing = set()
        for oe in cyc:
            if oe != e:
                existing.add(colors[oe])
        if new_c not in existing and len(existing) == 3:
            return False
    return True

def greedy_init(cycles_by_edge, cycles, rng):
    """Greedy initialization with random edge order."""
    colors = [None] * NUM_EDGES
    used = set()
    order = list(range(NUM_EDGES))
    rng.shuffle(order)
    
    for e in order:
        # Try to assign a new color first
        found = False
        # Try new colors
        for c in range(NUM_EDGES - 1, -1, -1):
            if c in used:
                continue
            if is_valid_move(colors, e, c, cycles_by_edge, cycles):
                colors[e] = c
                used.add(c)
                found = True
                break
        if not found:
            # Try existing colors
            for c in sorted(used, reverse=True):
                if is_valid_move(colors, e, c, cycles_by_edge, cycles):
                    colors[e] = c
                    found = True
                    break
        if not found:
            # Fallback: use color 0
            colors[e] = 0
    
    return colors

def algebraic_init():
    """Deterministic algebraic construction: vertices in Z_4 x Z_4, color by sum/diff hash."""
    # Map vertex i to (i//4, i%4)
    # Edge color: hash of ((u1+v1)%4, (u2+v2)%4, (u1-v1)%4, (u2-v2)%4)
    # But we need colors in 0..119, so use a compact hash
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            u1, u2 = i // 4, i % 4
            v1, v2 = j // 4, j % 4
            s1, s2 = (u1 + v1) % 4, (u2 + v2) % 4
            d1, d2 = (u1 - v1) % 4, (u2 - v2) % 4
            # Pack into a small number
            h = s1 * 64 + s2 * 16 + d1 * 4 + d2
            colors[edge_index(i, j)] = h % 120
    return colors

def simulated_annealing(colors, cycles_by_edge, cycles, rng, deadline):
    """Simulated annealing to maximize distinct colors."""
    cur_colors = colors[:]
    cur_count = count_colors(cur_colors)
    best_colors = cur_colors[:]
    best_count = cur_count
    
    start_temp = 5.0
    end_temp = 0.1
    t = start_temp
    total_steps = 0
    max_steps = 2000000
    
    while time.time() < deadline and total_steps < max_steps:
        total_steps += 1
        # Pick random edge
        e = rng.randint(0, NUM_EDGES - 1)
        old_c = cur_colors[e]
        
        # Pick new color: prefer new colors
        if rng.random() < 0.4:
            # Try a new color
            used = set(cur_colors)
            candidates = [c for c in range(NUM_EDGES) if c not in used]
            if not candidates:
                new_c = rng.randint(0, NUM_EDGES - 1)
            else:
                new_c = rng.choice(candidates)
        else:
            # Try an existing color (for exploration)
            new_c = rng.randint(0, NUM_EDGES - 1)
        
        if new_c == old_c:
            continue
        
        if not is_valid_move(cur_colors, e, new_c, cycles_by_edge, cycles):
            continue
        
        cur_colors[e] = new_c
        new_count = count_colors(cur_colors)
        
        if new_count >= cur_count:
            cur_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_colors = cur_colors[:]
        else:
            # Reject with probability exp(-(new_count - cur_count)/t)
            # Since new_count < cur_count, delta = new_count - cur_count < 0
            delta = new_count - cur_count  # negative
            if rng.random() > pow(2.718281828, delta / t):
                # Reject
                cur_colors[e] = old_c
            else:
                cur_count = new_count
        
        # Cool
        t *= 0.99999
        if t < end_temp:
            t = end_temp
    
    return best_colors, best_count

def construct():
    deadline = time.time() + 55  # safety margin
    
    cycles, cycles_by_edge = build_data()
    rng = random.Random(12345)
    
    best_colors = None
    best_count = 0
    
    # Phase 1: Algebraic baseline
    alg_colors = algebraic_init()
    alg_count = count_colors(alg_colors)
    if alg_count > best_count:
        best_count = alg_count
        best_colors = alg_colors[:]
    
    # Phase 2: Multiple restarts with greedy init + simulated annealing
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Greedy init
        g_colors = greedy_init(cycles_by_edge, cycles, rng)
        g_count = count_colors(g_colors)
        
        # Simulated annealing
        sa_colors, sa_count = simulated_annealing(g_colors, cycles_by_edge, cycles, rng, deadline)
        
        if sa_count > best_count:
            best_count = sa_count
            best_colors = sa_colors[:]
        
        # Also try starting from algebraic + SA
        if time.time() < deadline:
            sa2_colors, sa2_count = simulated_annealing(alg_colors[:], cycles_by_edge, cycles, rng, deadline)
            if sa2_count > best_count:
                best_count = sa2_count
                best_colors = sa2_colors[:]
    
    if best_colors is None:
        best_colors = [0] * NUM_EDGES
    
    return best_colors
