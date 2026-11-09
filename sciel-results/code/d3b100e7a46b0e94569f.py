"""Rewritten program: maximize distinct colors in a C4-free (rainbow-free) edge coloring of K16."""
import time
import random
from itertools import combinations

N = 16
M = N * (N - 1) // 2  # 120 edges

# Precompute edge index mapping
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles (as sets of 4 edge indices)
# A 4-cycle on vertices a,b,c,d (a<b<c<d) has 3 possible pairings:
#   (a,b,c,d) -> edges ab,bc,cd,da
#   (a,b,d,c) -> edges ab,bd,dc,ca
#   (a,c,b,d) -> edges ac,cb,bd,da
def _precompute_c4s():
    c4s = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        c4s.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
        # Cycle 2: a-b-d-c-a
        c4s.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
        # Cycle 3: a-c-b-d-a
        c4s.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))
    return c4s

_C4S = _precompute_c4s()
_NUM_C4S = len(_C4S)

def _is_valid(coloring):
    """Check if coloring has no rainbow 4-cycle."""
    for e1, e2, e3, e4 in _C4S:
        c1, c2, c3, c4 = coloring[e1], coloring[e2], coloring[e3], coloring[e4]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            return False
    return True

def _count_colors(coloring):
    return len(set(coloring))

def _check_edge_change(coloring, e, new_color):
    """Check if changing edge e to new_color would create a rainbow C4.
    Only need to check C4s containing edge e."""
    old_color = coloring[e]
    if old_color == new_color:
        return True
    coloring[e] = new_color
    ok = True
    for e1, e2, e3, e4 in _C4S:
        if e in (e1, e2, e3, e4):
            c1, c2, c3, c4 = coloring[e1], coloring[e2], coloring[e3], coloring[e4]
            if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                ok = False
                break
    coloring[e] = old_color
    return ok

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # Strategy: Start with a valid coloring using few colors, then try to increase
    # the number of distinct colors by recoloring edges.
    
    # Initial coloring: use color = min(i,j) which gives 15 colors and is valid.
    # Actually let's use a simpler valid base: all zeros, then greedily assign.
    
    # Better: use the min(i,j) strategy as a known valid base with 15 colors.
    # edge (i,j) with i<j gets color i (range 0..14)
    coloring = [0] * M
    for i in range(N):
        for j in range(i+1, N):
            coloring[edge_idx(i, j)] = i
    
    best_coloring = coloring[:]
    best_count = _count_colors(coloring)
    
    # Precompute for each edge, which C4s it belongs to (for fast checking)
    edge_c4s = [[] for _ in range(M)]
    for idx, (e1, e2, e3, e4) in enumerate(_C4S):
        for e in (e1, e2, e3, e4):
            edge_c4s[e].append(idx)
    
    def _check_edge_change_fast(coloring, e, new_color):
        """Fast check using precomputed C4 list for edge e."""
        old_color = coloring[e]
        if old_color == new_color:
            return True
        coloring[e] = new_color
        ok = True
        for c4_idx in edge_c4s[e]:
            e1, e2, e3, e4 = _C4S[c4_idx]
            c1, c2, c3, c4 = coloring[e1], coloring[e2], coloring[e3], coloring[e4]
            if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                ok = False
                break
        coloring[e] = old_color
        return ok
    
    # Greedy improvement: repeatedly try to increase distinct colors
    # by recoloring edges to new colors or merging/splitting.
    
    # Phase 1: Try to assign each edge a unique color if possible (ambitious)
    # Phase 2: Local search to maximize distinct colors
    
    # Local search: for each edge, try changing its color to a new value
    # that increases the total distinct color count without violating C4 constraint.
    
    used_colors = set(coloring)
    
    # Try to introduce new colors
    # For each edge, if its current color is used by multiple edges, try giving it a new color
    # to increase distinct count.
    
    # More systematic: iterate edges, try all possible new colors (0..119)
    # Accept if it increases distinct color count and is valid.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e in range(M):
            if time.time() > deadline:
                break
            old_c = coloring[e]
            # Try to give this edge a new unique color not yet used
            # Find a color not in used_colors
            # But we need to be strategic: only do this if old_c is shared
            color_count = coloring.count(old_c)
            if color_count <= 1:
                continue  # already unique, no benefit
            
            # Try to assign a new color
            # Find smallest unused color
            for new_c in range(120):
                if new_c in used_colors:
                    continue
                if _check_edge_change_fast(coloring, e, new_c):
                    coloring[e] = new_c
                    used_colors.add(new_c)
                    improved = True
                    break
            # Also try: if old_c is used by others, maybe we can still keep it
            # but give another edge a new color - that's handled in next iteration
    
    # Phase 2: More aggressive - try all color assignments for edges
    # that might increase distinct count
    for e in range(M):
        if time.time() > deadline:
            break
        old_c = coloring[e]
        # Try each possible color
        for new_c in range(120):
            if new_c == old_c:
                continue
            # Only accept if it increases distinct color count
            if new_c not in used_colors:
                # This would add a new color
                if _check_edge_change_fast(coloring, e, new_c):
                    coloring[e] = new_c
                    used_colors.add(new_c)
                    break
            elif old_c not in used_colors - {old_c} and new_c in used_colors:
                # This would remove a color (bad) - skip
                pass
            else:
                # Both colors already used, no change in distinct count
                # But might enable future improvements - skip for now
                pass
    
    # Phase 3: Random restarts with different initial colorings
    rng = random.Random(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random valid coloring: start with few colors, build up
        c = [0] * M
        # Assign random small number of colors initially
        num_init_colors = rng.randint(2, 8)
        for e in range(M):
            c[e] = rng.randint(0, num_init_colors - 1)
        
        # Fix violations: if a C4 is rainbow, recolor one of its edges
        # to match another edge in that C4
        valid = False
        for _ in range(1000):
            if time.time() > deadline:
                break
            violated = False
            for e1, e2, e3, e4 in _C4S:
                c1, c2, c3, c4 = c[e1], c[e2], c[e3], c[e4]
                if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
                    violated = True
                    # Recolor one edge to break the rainbow
                    # Pick a random edge and set it to match another
                    pick = rng.choice([e1, e2, e3, e4])
                    others = [x for x in (e1, e2, e3, e4) if x != pick]
                    c[pick] = c[rng.choice(others)]
                    break
            if not violated:
                valid = True
                break
        
        if not valid:
            continue
        
        # Now try to increase distinct colors
        uc = set(c)
        for _ in range(200):
            if time.time() > deadline:
                break
            # Pick a random edge whose color is shared
            shared_edges = [e for e in range(M) if c.count(c[e]) > 1]
            if not shared_edges:
                break
            e = rng.choice(shared_edges)
            # Try a new color
            new_c = rng.randint(0, 119)
            if new_c in uc:
                continue
            if _check_edge_change_fast(c, e, new_c):
                c[e] = new_c
                uc.add(new_c)
        
        cnt = len(uc)
        if cnt > best_count:
            best_count = cnt
            best_coloring = c[:]
    
    return best_coloring

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
