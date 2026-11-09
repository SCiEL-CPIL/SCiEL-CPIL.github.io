import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def build_edge_index():
    edge_to_idx = {}
    idx_to_edge = []
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(idx_to_edge)
            idx_to_edge.append((i, j))
    return edge_to_idx, idx_to_edge

EDGE_TO_IDX, IDX_TO_EDGE = build_edge_index()

# Precompute for each edge the list of (o1, o2, o3) triples for each 4-cycle it belongs to
def build_cycle_others():
    cycle_others = [[] for _ in range(NUM_EDGES)]
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        e12 = EDGE_TO_IDX[(v1, v2)]
        e13 = EDGE_TO_IDX[(v1, v3)]
        e14 = EDGE_TO_IDX[(v1, v4)]
        e23 = EDGE_TO_IDX[(v2, v3)]
        e24 = EDGE_TO_IDX[(v2, v4)]
        e34 = EDGE_TO_IDX[(v3, v4)]
        
        # Cycle 1: v1-v2-v3-v4-v1 -> edges e12, e23, e34, e14
        cycle_others[e12].append((e23, e34, e14))
        cycle_others[e23].append((e12, e34, e14))
        cycle_others[e34].append((e12, e23, e14))
        cycle_others[e14].append((e12, e23, e34))
        
        # Cycle 2: v1-v2-v4-v3-v1 -> edges e12, e24, e34, e13
        cycle_others[e12].append((e24, e34, e13))
        cycle_others[e24].append((e12, e34, e13))
        cycle_others[e34].append((e12, e24, e13))
        cycle_others[e13].append((e12, e24, e34))
        
        # Cycle 3: v1-v3-v2-v4-v1 -> edges e13, e23, e24, e14
        cycle_others[e13].append((e23, e24, e14))
        cycle_others[e23].append((e13, e24, e14))
        cycle_others[e24].append((e13, e23, e14))
        cycle_others[e14].append((e13, e23, e24))
    return cycle_others

CYCLE_OTHERS = build_cycle_others()

def is_valid(colors):
    for v1, v2, v3, v4 in itertools.combinations(range(N), 4):
        e12 = EDGE_TO_IDX[(v1, v2)]
        e13 = EDGE_TO_IDX[(v1, v3)]
        e14 = EDGE_TO_IDX[(v1, v4)]
        e23 = EDGE_TO_IDX[(v2, v3)]
        e24 = EDGE_TO_IDX[(v2, v4)]
        e34 = EDGE_TO_IDX[(v3, v4)]
        
        c = colors
        # Cycle 1
        if len({c[e12], c[e23], c[e34], c[e14]}) == 4:
            return False
        # Cycle 2
        if len({c[e12], c[e24], c[e34], c[e13]}) == 4:
            return False
        # Cycle 3
        if len({c[e13], c[e23], c[e24], c[e14]}) == 4:
            return False
    return True

def can_take_color(colors, edge, new_color):
    """Check if assigning new_color to edge is valid."""
    for o1, o2, o3 in CYCLE_OTHERS[edge]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        # If the three others are all distinct, new_color must be one of them
        if c1 != c2 and c1 != c3 and c2 != c3:
            if new_color != c1 and new_color != c2 and new_color != c3:
                return False
    return True

def get_allowed_colors(colors, edge):
    """Return set of colors that can be assigned to edge."""
    allowed = set(range(MAX_COLOR))
    for o1, o2, o3 in CYCLE_OTHERS[edge]:
        c1 = colors[o1]
        c2 = colors[o2]
        c3 = colors[o3]
        if c1 != c2 and c1 != c3 and c2 != c3:
            allowed.intersection_update({c1, c2, c3})
            if not allowed:
                return set()
    return allowed

def hierarchical_construct(seed_rng):
    """Build a valid coloring via hierarchical clustering.
    Each merge of two clusters assigns a new unique color to all edges between them."""
    colors = [0] * NUM_EDGES
    # Start with 16 singleton clusters
    clusters = [set([i]) for i in range(N)]
    active = list(range(N))
    next_color = 0
    
    while len(active) > 1:
        # Pick two random active clusters to merge
        if seed_rng is not None:
            i = seed_rng.randint(0, len(active) - 1)
            j = seed_rng.randint(0, len(active) - 1)
            while j == i:
                j = seed_rng.randint(0, len(active) - 1)
        else:
            i = 0
            j = 1
        
        ci = active[i]
        cj = active[j]
        
        # Assign new color to all edges between cluster ci and cj
        for u in clusters[ci]:
            for v in clusters[cj]:
                a, b = (u, v) if u < v else (v, u)
                e = EDGE_TO_IDX[(a, b)]
                colors[e] = next_color
        
        # Merge: add cj's vertices to ci, remove cj from active
        clusters[ci] |= clusters[cj]
        clusters[cj] = set()
        # Remove j from active (after i, so adjust index if j < i)
        if j < i:
            i = i - 1
        active.pop(j)
        if i < len(active):
            active.pop(i)
        
        next_color += 1
    
    return colors

def local_search(colors, deadline):
    colors = list(colors)
    used_colors = set(colors)
    best_score = len(used_colors)
    best_colors = list(colors)
    
    no_improve_count = 0
    max_no_improve = 80
    
    all_colors = list(range(MAX_COLOR))
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Kick: randomly recolor several edges
            num_kicks = random.randint(5, 15)
            for _ in range(num_kicks):
                e = random.randint(0, NUM_EDGES - 1)
                allowed = get_allowed_colors(colors, e)
                if allowed:
                    colors[e] = random.choice(list(allowed))
            
            used_colors = set(colors)
            no_improve_count = 0
            max_no_improve = random.randint(50, 150)
            continue

        # Try to improve: find an edge that can take an unused color
        improved = False
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        for e in edges:
            if time.time() >= deadline:
                break
            
            unused = [c for c in range(MAX_COLOR) if c not in used_colors]
            if not unused:
                break
            
            random.shuffle(unused)
            
            allowed = get_allowed_colors(colors, e)
            if not allowed:
                continue
            
            # Try unused colors that are allowed
            found = False
            for nc in unused:
                if nc in allowed:
                    colors[e] = nc
                    used_colors.add(nc)
                    improved = True
                    found = True
                    break
            
            if found:
                no_improve_count = 0
                if len(used_colors) > best_score:
                    best_score = len(used_colors)
                    best_colors = list(colors)
            else:
                no_improve_count += 1
        
        if not improved:
            no_improve_count += 5

    return best_colors, best_score

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    # Strategy 1: Hierarchical clustering (guaranteed valid, ~15 colors)
    rng = random.Random(42)
    hier_colors = hierarchical_construct(rng)
    if is_valid(hier_colors):
        best_colors = hier_colors
        best_score = len(set(hier_colors))
    
    # Multiple restarts with different hierarchical seeds + local search
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() >= deadline:
            break
        
        # Different random seeds for diversity
        seed = 1000 + r * 7919
        rng = random.Random(seed)
        colors = hierarchical_construct(rng)
        
        # Perturb slightly
        for _ in range(5):
            e = random.randint(0, NUM_EDGES - 1)
            allowed = get_allowed_colors(colors, e)
            if allowed:
                colors[e] = random.choice(list(allowed))
        
        if not is_valid(colors):
            continue
        
        # Allocate time for local search
        remaining = deadline - time.time()
        ls_deadline = time.time() + remaining / (num_restarts - r)
        
        improved_colors, score = local_search(colors, ls_deadline)
        
        if score > best_score:
            best_score = score
            best_colors = improved_colors
        
        if best_score >= 40:
            break
    
    # Final verification
    if not is_valid(best_colors):
        return [0] * NUM_EDGES
    
    return best_colors

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
