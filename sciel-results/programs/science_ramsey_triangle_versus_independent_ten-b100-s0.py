"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _has_triangle_circulant(n, S):
    """Check if circulant graph on Z_n with connection set S has a triangle.
    Triangle exists if there exist a, b, c in Z_n with b-a, c-b, c-a all in S or -S.
    Equivalently, for some x, y in S (with x != y, x != -y), x+y in S or -S.
    """
    S_set = set(S)
    neg_S = {(-x) % n for x in S}
    for x in S:
        for y in S:
            if x == y:
                continue
            z = (x + y) % n
            if z in S_set or z in neg_S:
                # Check: vertices 0, x, x+y form triangle if x in S, y in S, and (x+y) in S or -S
                # But we need all three edges: 0-x (x in S), 0-(x+y) ((x+y) in S or -S), x-(x+y) (y in S)
                # So yes, triangle if z in S or z in neg_S
                return True
    return False

def _max_independent_set_circulant(n, S):
    """Compute alpha of circulant graph using branch and bound with bitmasks."""
    S_set = set(S)
    # Build adjacency: vertex i connected to i+s mod n for s in S
    adj_mask = [0] * n
    for i in range(n):
        for s in S:
            j = (i + s) % n
            adj_mask[i] |= (1 << j)
    
    # Non-neighbors (including self excluded)
    comp_mask = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_mask[i] = full_mask ^ (1 << i) ^ adj_mask[i]
    
    best = 0
    def dfs(candidates, current_size):
        nonlocal best
        if current_size + bin(candidates).count('1') <= best:
            return
        if candidates == 0:
            if current_size > best:
                best = current_size
            return
        # Pick least significant bit
        v = (candidates & -candidates).bit_length() - 1
        # Include v
        dfs(candidates & comp_mask[v], current_size + 1)
        # Exclude v
        dfs(candidates & ~(1 << v), current_size)
    
    dfs(full_mask, 0)
    return best

def _verify_circulant(n, S):
    """Verify circulant graph is triangle-free and alpha < 10."""
    if _has_triangle_circulant(n, S):
        return False
    alpha = _max_independent_set_circulant(n, S)
    if alpha >= 10:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Try circulant graphs for various n
    # Connection sets should be symmetric: if s in S, then -s in S
    # To avoid triangles: no 3-term AP in S ∪ {-S}
    
    for n in range(40, 25, -1):  # Try larger n first
        if time.time() > deadline:
            break
        
        # Generate candidate connection sets
        # S must be symmetric: S = -S
        # So we only need to choose from 1..floor(n/2)
        # And avoid 3-term APs
        
        candidates = []
        
        # Try various densities
        for density in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            target_size = int(n * density)
            # Try random symmetric sets
            import random
            random.seed(n * 1000 + int(density * 1000))
            for attempt in range(20):
                if time.time() > deadline:
                    break
                
                # Choose symmetric set
                S = set()
                half = n // 2
                available = list(range(1, half + 1))
                random.shuffle(available)
                
                # Pick elements ensuring symmetry
                for x in available:
                    if len(S) >= target_size:
                        break
                    # Check if adding x and -x creates a triangle
                    new_S = S | {x, (-x) % n}
                    # Quick triangle check
                    S_set = set(new_S)
                    neg_S = {(-y) % n for y in new_S}
                    has_tri = False
                    for a in new_S:
                        for b in new_S:
                            if a == b:
                                continue
                            c = (a + b) % n
                            if c in S_set or c in neg_S:
                                has_tri = True
                                break
                        if has_tri:
                            break
                    if not has_tri:
                        S = new_S
                
                if len(S) > 0:
                    candidates.append(S)
        
        # Also try structured sets: multiples of small numbers
        for base in range(2, 8):
            S = set()
            for k in range(1, n // base + 1):
                x = k * base
                if x < n and x != 0:
                    S.add(x)
                    S.add((-x) % n)
            if S:
                candidates.append(S)
        
        # Try arithmetic progressions with gaps
        for start in range(1, 5):
            for step in range(2, 8):
                S = set()
                x = start
                while x < n:
                    S.add(x)
                    S.add((-x) % n)
                    x += step
                if S:
                    candidates.append(S)
        
        # Test candidates
        for S in candidates:
            if time.time() > deadline:
                break
            if _verify_circulant(n, S):
                # Build edge list
                edges = []
                S_set = set(S)
                for i in range(n):
                    for s in S:
                        j = (i + s) % n
                        if j > i:
                            edges.append([i, j])
                best_n = n
                best_edges = edges
                break
        
        if best_n >= n:
            break  # Found valid graph for this n, move to next
    
    # Fallback: known valid 26-vertex Andrásfai graph
    if best_n < 26:
        n0 = 26
        best_edges = [[i, j] for i in range(n0) for j in range(i + 1, n0) if (j - i) % n0 % 3 == 1]
        best_n = n0
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
