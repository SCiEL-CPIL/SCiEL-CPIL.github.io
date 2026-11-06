"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all K4s (4-vertex subsets) and their edges
_K4S = list(combinations(range(16), 4))
_K4_EDGES = []
for quad in _K4S:
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            u = quad[i]
            v = quad[j]
            # index into the 120-element list: pairs (i,j) with i<j
            # index = sum_{k=0}^{i-1} (15-k) + (j-i-1) = i*15 - i*(i-1)//2 + j - i - 1
            # Let's compute: for vertex u, the number of pairs (u, w) with w < u is u
            # Actually, lexicographic order: (0,1),(0,2),...,(0,15),(1,2),...
            # index(u,v) where u<v:
            # number of pairs with first coord < u: sum_{k=0}^{u-1} (15-k) = 15*u - u*(u-1)//2
            # within first coord u, pairs are (u,u+1),...,(u,v),... so offset is v-u-1
            idx = 15*u - u*(u-1)//2 + (v - u - 1)
            edges.append(idx)
    _K4_EDGES.append(edges)

# Build adjacency: for each edge index, list of (k4_idx, pos_in_k4)
_EDGE_TO_K4S = [[] for _ in range(120)]
for k4_idx, edges in enumerate(_K4_EDGES):
    for pos, e_idx in enumerate(edges):
        _EDGE_TO_K4S[e_idx].append((k4_idx, pos))


def _count_colours_in_k4(coloring, k4_idx):
    edges = _K4_EDGES[k4_idx]
    cols = set()
    for e_idx in edges:
        cols.add(coloring[e_idx])
    return len(cols)


def _check_valid(coloring):
    """Return True if every K4 has at least 5 distinct colours."""
    for k4_idx in range(len(_K4S)):
        if _count_colours_in_k4(coloring, k4_idx) < 5:
            return False
    return True


def _count_violations(coloring):
    """Return number of K4s with fewer than 5 colours."""
    count = 0
    for k4_idx in range(len(_K4S)):
        if _count_colours_in_k4(coloring, k4_idx) < 5:
            count += 1
    return count


def _num_colours(coloring):
    return len(set(coloring))


def _try_additive_prime():
    """Try (i + j) mod p for various primes p >= 17."""
    best = None
    best_count = 17  # seed uses 17 colours
    for p in [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if p > 120:
            break
        coloring = [(i + j) % p for i in range(16) for j in range(i + 1, 16)]
        if _check_valid(coloring):
            nc = _num_colours(coloring)
            if nc < best_count:
                best_count = nc
                best = coloring[:]
    return best, best_count


def _try_multiplicative():
    """Try (i * j) mod p and related constructions."""
    best = None
    best_count = 17
    for p in [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if p > 120:
            break
        # (i*j) mod p
        coloring = [(i * j) % p for i in range(16) for j in range(i + 1, 16)]
        if _check_valid(coloring):
            nc = _num_colours(coloring)
            if nc < best_count:
                best_count = nc
                best = coloring[:]
        # (i*j + i + j) mod p
        coloring = [(i * j + i + j) % p for i in range(16) for j in range(i + 1, 16)]
        if _check_valid(coloring):
            nc = _num_colours(coloring)
            if nc < best_count:
                best_count = nc
                best = coloring[:]
    return best, best_count


def _try_z5_product():
    """Map 16 vertices to Z5 x {0,1} and colour by cyclic distance in Z5."""
    # Assign vertices: for a in 0..4, b in {0,1}, vertex = 2*a + b
    # Colour edge (u,v) by min(|a_u - a_v|, 5 - |a_u - a_v|) which is in {0,1,2}
    # But that gives only 3 colours. We need more structure.
    # Alternative: colour by (a_u - a_v) mod 5, giving 5 colours {0,1,2,3,4}
    best = None
    best_count = 17
    for offset in range(5):
        coloring = [0] * 120
        for i in range(16):
            for j in range(i + 1, 16):
                ai = i // 2
                aj = j // 2
                c = (aj - ai + offset) % 5
                idx = 15*i - i*(i-1)//2 + (j - i - 1)
                coloring[idx] = c
        if _check_valid(coloring):
            nc = _num_colours(coloring)
            if nc < best_count:
                best_count = nc
                best = coloring[:]
    return best, best_count


def _try_z5_product_extended():
    """Extended Z5 x {0,1} with more colour structure."""
    best = None
    best_count = 17
    # Try colouring by (a_u - a_v) mod 5 * 2 + (b_u XOR b_v)
    # This gives colours in {0,1,2,3,4} * 2 + {0,1} = 10 colours
    for use_xor in [0, 1]:
        coloring = [0] * 120
        for i in range(16):
            for j in range(i + 1, 16):
                ai = i // 2
                bi = i % 2
                aj = j // 2
                bj = j % 2
                base = (aj - ai) % 5
                if use_xor:
                    c = base * 2 + (bi ^ bj)
                else:
                    c = base * 2 + (bi + bj) % 2
                idx = 15*i - i*(i-1)//2 + (j - i - 1)
                coloring[idx] = c
        if _check_valid(coloring):
            nc = _num_colours(coloring)
            if nc < best_count:
                best_count = nc
                best = coloring[:]
    return best, best_count


def _hill_climb(coloring, max_iters=50000, time_limit=100.0):
    """Hill-climbing: randomly change one edge colour to reduce violations and colour count."""
    start_time = time.time()
    current = coloring[:]
    current_violations = _count_violations(current)
    current_nc = _num_colours(current)
    best = current[:]
    best_violations = current_violations
    best_nc = current_nc
    best_score = best_violations * 1000 + best_nc  # minimize violations first, then colours
    
    rng = random.Random(42)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random edge
        e_idx = rng.randint(0, 119)
        old_c = current[e_idx]
        
        # Try a new colour
        new_c = rng.randint(0, 119)
        if new_c == old_c:
            continue
        
        current[e_idx] = new_c
        
        # Compute change in violations
        # Only K4s containing edge e_idx are affected
        delta_viol = 0
        for k4_idx, pos in _EDGE_TO_K4S[e_idx]:
            # Before change
            before = _count_colours_in_k4(current, k4_idx)
            # We already changed, so compute before by temporarily reverting
            # Actually let's compute properly
            # Revert temporarily
            current[e_idx] = old_c
            before = _count_colours_in_k4(current, k4_idx)
            current[e_idx] = new_c
            after = _count_colours_in_k4(current, k4_idx)
            
            # Violation count: 1 if < 5 colours
            before_v = 1 if before < 5 else 0
            after_v = 1 if after < 5 else 0
            delta_viol += after_v - before_v
        
        new_violations = current_violations + delta_viol
        new_nc = _num_colours(current)
        new_score = new_violations * 1000 + new_nc
        
        if new_score <= best_score:
            best_score = new_score
            best = current[:]
            best_violations = new_violations
            best_nc = new_nc
            current_violations = new_violations
            current_nc = new_nc
        else:
            # Revert
            current[e_idx] = old_c
    
    return best, best_nc


def _simulated_annealing(coloring, time_limit=100.0):
    """Simulated annealing to find valid colouring with fewer colours."""
    start_time = time.time()
    current = coloring[:]
    current_violations = _count_violations(current)
    current_nc = _num_colours(current)
    best = current[:]
    best_violations = current_violations
    best_nc = current_nc
    best_score = best_violations * 1000 + best_nc
    
    rng = random.Random(123)
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if T < 0.01:
            break
        
        e_idx = rng.randint(0, 119)
        old_c = current[e_idx]
        new_c = rng.randint(0, 119)
        if new_c == old_c:
            continue
        
        current[e_idx] = new_c
        
        delta_viol = 0
        for k4_idx, pos in _EDGE_TO_K4S[e_idx]:
            current[e_idx] = old_c
            before = _count_colours_in_k4(current, k4_idx)
            current[e_idx] = new_c
            after = _count_colours_in_k4(current, k4_idx)
            before_v = 1 if before < 5 else 0
            after_v = 1 if after < 5 else 0
            delta_viol += after_v - before_v
        
        new_violations = current_violations + delta_viol
        new_nc = _num_colours(current)
        new_score = new_violations * 1000 + new_nc
        old_score = current_violations * 1000 + current_nc
        delta = new_score - old_score
        
        if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
            current_violations = new_violations
            current_nc = new_nc
            if new_score < best_score:
                best_score = new_score
                best = current[:]
                best_violations = new_violations
                best_nc = new_nc
        else:
            current[e_idx] = old_c
        
        T *= cooling
    
    return best, best_nc


def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Start with the seed: (i+j) mod 17, uses 17 colours
    coloring = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    best = coloring[:]
    best_nc = _num_colours(best)  # should be 17
    
    # Try systematic constructions
    if time.time() < deadline - 10:
        c, nc = _try_additive_prime()
        if c is not None and nc < best_nc:
            best = c
            best_nc = nc
    
    if time.time() < deadline - 10:
        c, nc = _try_multiplicative()
        if c is not None and nc < best_nc:
            best = c
            best_nc = nc
    
    if time.time() < deadline - 10:
        c, nc = _try_z5_product()
        if c is not None and nc < best_nc:
            best = c
            best_nc = nc
    
    if time.time() < deadline - 10:
        c, nc = _try_z5_product_extended()
        if c is not None and nc < best_nc:
            best = c
            best_nc = nc
    
    # Hill climbing from best
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        c, nc = _hill_climb(best, time_limit=min(remaining, 60.0))
        if nc < best_nc:
            best = c
            best_nc = nc
    
    # Simulated annealing from best
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        c, nc = _simulated_annealing(best, time_limit=min(remaining, 60.0))
        if nc < best_nc:
            best = c
            best_nc = nc
    
    # Final hill climb polish
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 3
        c, nc = _hill_climb(best, time_limit=min(remaining, 30.0))
        if nc < best_nc:
            best = c
            best_nc = nc
    
    # Ensure validity: if best is not valid, fall back to seed
    if not _check_valid(best):
        best = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Ensure all colours are in 0..119
    best = [c % 120 for c in best]
    
    return best
# EVOLVE-BLOCK-END
