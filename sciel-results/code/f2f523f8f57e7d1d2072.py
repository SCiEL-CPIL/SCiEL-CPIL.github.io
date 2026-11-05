# EVOLVE-BLOCK-START
import random, time, math, sys
from itertools import combinations

BOX = 100
N = 20

def _now():
    return time.time()

def _collinear3(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]) == 0

def _triple_count(pts):
    c = 0
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                if _collinear3(pts[i], pts[j], pts[k]):
                    c += 1
    return c

def _has_four_collinear(pts):
    n = len(pts)
    if n < 4:
        return False
    for i in range(n):
        for j in range(i+1, n):
            cnt = 0
            pi, pj = pts[i], pts[j]
            for k in range(n):
                if k == i or k == j: continue
                if _collinear3(pi, pj, pts[k]):
                    cnt += 1
                    if cnt >= 2:
                        return True
    return False

def _valid(pts):
    if len(set(map(tuple, pts))) != N:
        return False
    if any(abs(x) > BOX or abs(y) > BOX for x, y in pts):
        return False
    return not _has_four_collinear(pts)

def _max_3free(pts):
    """
    Exact branch-and-bound for largest 3-free subset (Max Independent Set in 3-uniform hypergraph).
    Equivalent to N - Min Vertex Cover.
    Uses iterative B&B branching on edges to avoid recursion depth issues.
    """
    n = N
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear3(pts[i], pts[j], pts[k]):
                    triples.append((i, j, k))
    
    if not triples:
        return n
    
    # Precompute for each vertex, the list of triple indices it belongs to
    vertex_triples = [[] for _ in range(n)]
    for idx, (i, j, k) in enumerate(triples):
        vertex_triples[i].append(idx)
        vertex_triples[j].append(idx)
        vertex_triples[k].append(idx)
        
    # For B&B, we track uncovered triples.
    # State: (uncovered_mask, current_vc_size)
    # We want to minimize current_vc_size.
    
    best_vc = [len(triples)]
    start_time = time.time()
    time_limit = 2.0 # 2 seconds per evaluation to allow many evaluations
    
    # Use a stack for DFS
    # Stack items: (uncovered_mask, vc_size)
    # We use a bitmask for triples if len(triples) <= 63? No, can be up to 1140.
    # Use a set of indices or a tuple of indices?
    # For speed, if len(triples) < 64, use int mask. Else, use tuple/set.
    # Let's assume len(triples) is manageable with a frozenset or list for simplicity, 
    # but int mask is faster. 
    # Max triples for 20 pts is 1140. Python int can handle 1140 bits.
    
    initial_mask = (1 << len(triples)) - 1
    
    # Lower bound function: Max matching in the hypergraph of uncovered triples?
    # Simpler LB: Greedy packing of disjoint triples.
    def get_lower_bound(mask):
        if mask == 0:
            return 0
        lb = 0
        m = mask
        while m:
            # Pick lowest set bit
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            # Remove this triple and any triple sharing a vertex with it? 
            # No, for LB, we just need disjoint triples.
            # If we pick triple idx, we must remove all triples that share a vertex with idx.
            # This gives a valid packing (matching).
            t = triples[idx]
            # Find all triples that intersect with t
            conflict_mask = 0
            for v in t:
                for t_idx in vertex_triples[v]:
                    conflict_mask |= (1 << t_idx)
            m &= ~conflict_mask
            lb += 1
        return lb

    stack = [(initial_mask, 0)]
    
    while stack:
        if time.time() - start_time > time_limit:
            break
            
        mask, vc_size = stack.pop()
        
        if mask == 0:
            if vc_size < best_vc[0]:
                best_vc[0] = vc_size
            continue
            
        # Prune
        if vc_size + get_lower_bound(mask) >= best_vc[0]:
            continue
            
        # Pick an edge (triple) from mask
        # Heuristic: pick the triple with highest "density" or just the first one
        # To be deterministic and fast, pick the first set bit
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        t = triples[idx]
        
        # Branch on each vertex in the triple
        # We must cover this triple, so at least one of its vertices must be in the cover.
        seen = set()
        for v in t:
            if v in seen:
                continue
            seen.add(v)
            
            # Add v to cover
            new_vc = vc_size + 1
            # Remove all triples containing v from mask
            new_mask = mask
            for t_idx in vertex_triples[v]:
                new_mask &= ~(1 << t_idx)
                
            # Check prune before pushing
            if new_vc + get_lower_bound(new_mask) < best_vc[0]:
                stack.append((new_mask, new_vc))
                
    # If best_vc is still initial (len(triples)), it means we didn't find a better one or timed out with bad LB.
    # Fallback to greedy if time ran out and best_vc is high?
    # The B&B is exact if it finishes. If it times out, it returns the best found so far.
    
    return n - best_vc[0]

def _construct_line_packed():
    """Construct 20 points packed on lines of 3 to maximize structural density."""
    random.seed(42)
    pts = []
    used = set()
    
    # Strategy: Use a 4x5 grid, but perturb to remove 4-collinear lines.
    # A 4x5 grid has rows of 5 (4 collinear) and cols of 4 (4 collinear).
    # We need to break these.
    # Let's try a specific set of points known to have many 3-collinearities.
    
    # Base grid
    for x in range(-2, 3):
        for y in range(-2, 3):
            pts.append([x, y])
            
    # This is 5x5=25 points. We need 20.
    # Remove 5 points to break the 4-collinear lines?
    # Actually, let's just generate random points and check validity, 
    # but bias towards "line-like" structures.
    
    # Better: Use the previous manual construction but ensure validity.
    pts = []
    used = set()
    
    # Line 1: y=0
    for x in [-30, 0, 30]:
        p = (x, 0)
        pts.append([x, 0])
        used.add(p)
        
    # Line 2: y=10
    for x in [-20, 0, 20]:
        p = (x, 10)
        if p not in used:
            pts.append([x, 10])
            used.add(p)
            
    # Line 3: y=-10
    for x in [-20, 0, 20]:
        p = (x, -10)
        if p not in used:
            pts.append([x, -10])
            used.add(p)
            
    # Line 4: x=0 (vertical)
    for y in [-30, 30]:
        p = (0, y)
        if p not in used:
            pts.append([x, y]) # BUG: x is from previous loop
            # Fix:
            # p = (0, y)
            # if p not in used: pts.append([0, y]); used.add(p)
            
    # Let's restart the construction cleanly in the code below.
    return pts

def _construct_valid_line_packed():
    """
    Construct a valid set of 20 points with many collinear triples.
    We use a grid and then move points to break 4-collinearities.
    """
    random.seed(42)
    # Start with a 4x5 grid
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            pts.append([i, j])
            
    # We have 25 points. Remove 5 to get 20.
    # Which 5 to remove? 
    # Removing points from the center might break lines.
    # Let's try to find a subset of 20 from the 5x5 grid that has no 4 collinear.
    # If none exists, we perturb.
    
    # Check if the full 5x5 grid has 4 collinear. Yes.
    # We need a subset of 20.
    
    # Alternative: Generate points on a "distorted" grid.
    # x = i, y = j + epsilon * i^2
    # This breaks horizontal lines (rows become curves) but might keep verticals?
    # Verticals: x=i. Points (i, j + eps*i^2). 
    # Are 4 points (i, j1+..), (i, j2+..), (i, j3+..), (i, j4+..) collinear?
    # They all have x=i, so they are on the vertical line x=i.
    # So verticals are still 4-collinear.
    # We need to break verticals too.
    # x = i + delta * j^2, y = j.
    # Then rows are broken. Cols: x depends on j.
    # Points (i+d*j1^2, j1), (i+d*j2^2, j2), ...
    # Are 4 collinear? Unlikely for small d.
    
    d = 1
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + d * (j * j)
            y = j
            pts.append([x, y])
            
    # Check validity
    if _valid(pts):
        return pts
        
    # If not valid, try different d or perturb randomly
    for seed in range(100):
        random.seed(seed)
        pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                x = i + random.randint(-2, 2)
                y = j + random.randint(-2, 2)
                pts.append([x, y])
        if _valid(pts):
            return pts
            
    # Fallback: Random
    return _construct_random()

def _construct_random():
    """Random construction with 4-collinear avoidance."""
    random.seed(random.randint(0, 1000000))
    pts = []
    used = set()
    attempts = 0
    while len(pts) < N and attempts < 10000:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        p = (x, y)
        if p in used:
            attempts += 1
            continue
        pts.append([x, y])
        if _has_four_collinear(pts):
            pts.pop()
        else:
            used.add(p)
        attempts += 1
    if len(pts) < N:
        pts = [[x, x*x] for x in range(-10, 10)]
    return pts[:N]

def construct():
    t0 = time.time()
    DEADLINE = t0 + 110.0  # 10s safety margin
    
    # Generate candidates
    candidates = []
    
    # 1. Distorted Grid
    dg_pts = _construct_valid_line_packed()
    if _valid(dg_pts):
        candidates.append(dg_pts)
        
    # 2. Random points
    for seed in [1, 2, 3]:
        random.seed(seed)
        r_pts = _construct_random()
        if _valid(r_pts):
            candidates.append(r_pts)
            
    # 3. Parabola (worst case, high score)
    par_pts = [[x, x*x] for x in range(-10, 10)]
    if _valid(par_pts):
        candidates.append(par_pts)
        
    if not candidates:
        return par_pts
        
    # Evaluate candidates
    best_score = float('inf')
    best_pts = None
    for cand in candidates:
        if time.time() > DEADLINE - 20:
            break
        s = _max_3free(cand)
        if s < best_score:
            best_score = s
            best_pts = [list(p) for p in cand]
            
    if best_pts is None:
        return par_pts
        
    # Local Search / Simulated Annealing
    pts = best_pts
    cur_score = best_score
    T = 2.0
    last_improve = 0
    it = 0
    
    while time.time() < DEADLINE - 5:
        i = random.randrange(N)
        np_ = [list(p) for p in pts]
        
        # Random move
        if random.random() < 0.5:
            np_[i] = [random.randint(-BOX, BOX), random.randint(-BOX, BOX)]
        else:
            # Perturb
            np_[i][0] += random.randint(-3, 3)
            np_[i][1] += random.randint(-3, 3)
            np_[i][0] = max(-BOX, min(BOX, np_[i][0]))
            np_[i][1] = max(-BOX, min(BOX, np_[i][1]))
            
        if not _valid(np_):
            continue
            
        new_score = _max_3free(np_)
        
        if new_score < cur_score:
            pts = np_
            cur_score = new_score
            if cur_score < best_score:
                best_score = cur_score
                best_pts = [list(p) for p in pts]
            last_improve = it
        elif random.random() < math.exp(-(new_score - cur_score) / T):
            pts = np_
            cur_score = new_score
            
        T = max(0.5, T * 0.99)
        it += 1
        
        if it - last_improve > 50 and time.time() < DEADLINE - 10:
            # Restart
            random.seed(random.randint(0, 1000000))
            new_pts = _construct_random()
            if _valid(new_pts):
                new_sc = _max_3free(new_pts)
                if new_sc < best_score:
                    best_score = new_sc
                    best_pts = [list(p) for p in new_pts]
                pts = [list(p) for p in new_pts]
                cur_score = new_sc
                last_improve = it
                T = 2.0
                
        if best_score <= 12:
            break
            
    return best_pts
# EVOLVE-BLOCK-END
