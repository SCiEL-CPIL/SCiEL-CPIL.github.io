"""Linear triple systems on 16 points with small independence number."""
import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START

def _validate(triples):
    """Check that all triples are valid and the system is linear."""
    if not triples:
        return True
    seen = set()
    for t in triples:
        if len(t) != 3:
            return False
        if any(p < 0 or p > 15 for p in t):
            return False
        if len(set(t)) != 3:
            return False
        key = tuple(sorted(t))
        if key in seen:
            return False
        seen.add(key)
    # Check linearity: any two triples share at most one point
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            inter = set(triples[i]) & set(triples[j])
            if len(inter) > 1:
                return False
    return True

def _independence_number_fast(triples):
    """
    Compute independence number via ILP (scipy.optimize.milp).
    Maximizes sum(x_i) subject to x_i + x_j + x_k <= 2 for each triple {i,j,k}.
    Returns the maximum number of points with no triple fully contained.
    """
    if not triples:
        return 16
    
    n = 16
    m = len(triples)
    
    # c = -1 for each variable (minimize -sum = maximize sum)
    c = -np.ones(n)
    
    # A: m x n matrix, each row has 1s at the triple positions
    A = np.zeros((m, n), dtype=int)
    for idx, t in enumerate(triples):
        A[idx, t[0]] = 1
        A[idx, t[1]] = 1
        A[idx, t[2]] = 1
    
    # Constraints: A @ x <= 2 for each triple
    constraint = LinearConstraint(A, lb=-np.inf, ub=np.ones(m))
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
    # Integrality: all variables integer
    integrality = np.ones(n)
    
    result = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
    
    if result.success:
        # Maximize sum, so alpha = -result.fun
        return int(round(-result.fun))
    else:
        # Fallback: use bitmask brute force for small cases
        triple_masks = []
        for t in triples:
            mm = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            triple_masks.append(mm)
        
        best = 0
        for mask in range(1 << 16):
            independent = True
            for tm in triple_masks:
                if (tm & mask) == tm:
                    independent = False
                    break
            if independent:
                cnt = bin(mask).count('1')
                if cnt > best:
                    best = cnt
        return best

def _independence_number(triples):
    """Compute the independence number: largest set of points containing no triple."""
    return _independence_number_fast(triples)

def _greedy_construct(n_points=16, n_iter=500, seed=42, deadline=None):
    """Greedy construction of linear triple system with small independence number."""
    rng = random.Random(seed)
    best_alpha = 16
    best_triples = []
    
    # Precompute all triples
    all_triples = list(combinations(range(n_points), 3))
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Build a linear triple system greedily
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        
        triples = []
        pair_used = set()  # frozenset pairs for fast lookup
        
        for t in shuffled:
            # Check linearity: no pair from t has appeared in any previous triple
            ok = True
            for a, b in combinations(t, 2):
                if (min(a,b), max(a,b)) in pair_used:
                    ok = False
                    break
            if ok:
                triples.append(list(t))
                for a, b in combinations(t, 2):
                    pair_used.add((min(a,b), max(a,b)))
        
        # Compute independence number
        alpha = _independence_number_fast(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
            if alpha <= 5:
                break  # Can't do better than 5 for 16 points
    
    return best_triples, best_alpha

def _local_search(triples, n_points=16, max_iter=200, deadline=None):
    """Try to improve by adding triples that reduce independence number."""
    if not triples:
        return triples, _independence_number(triples)
    
    best_alpha = _independence_number_fast(triples)
    current = [list(t) for t in triples]
    all_triples = list(combinations(range(n_points), 3))
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if best_alpha <= 5:
            break
        
        improved = False
        rng = random.Random(it * 7919 + 13)
        shuffled = all_triples[:]
        rng.shuffle(shuffled)
        
        # Build pair_used set for fast linearity check
        pair_used = set()
        for t in current:
            for a, b in combinations(t, 2):
                pair_used.add((min(a,b), max(a,b)))
        
        # Try adding triples
        for t in shuffled:
            if len(current) > 120:
                break
            # Check linearity via pair_used
            ok = True
            for a, b in combinations(t, 2):
                if (min(a,b), max(a,b)) in pair_used:
                    ok = False
                    break
            if ok:
                current.append(list(t))
                for a, b in combinations(t, 2):
                    pair_used.add((min(a,b), max(a,b)))
                new_alpha = _independence_number_fast(current)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    improved = True
                else:
                    current.pop()
                    for a, b in combinations(t, 2):
                        pair_used.discard((min(a,b), max(a,b)))
        
        if not improved and best_alpha <= 5:
            break
    
    return current, best_alpha

def _algebraic_construction():
    """
    Deterministic construction using affine geometry over F_4^2.
    
    Points of AG(2,4) are 16 elements of F_4^2.
    Lines in AG(2,4) have 4 points each. We need triples, not 4-sets.
    
    Alternative: Use the fact that F_4 = {0, 1, ω, ω^2} where ω^2 + ω + 1 = 0.
    We can construct triples from parallel classes of lines, taking 3 of the 4 points.
    
    Actually, let's use a different approach: construct a 3-uniform linear hypergraph
    from the affine plane by selecting specific triples.
    
    Better approach: Use the complete graph K_16 and decompose into triangles carefully.
    Or use a known construction: the 16 points as binary vectors of length 4,
    and triples as sets of 3 vectors with specific properties.
    
    Let me try: points are 0..15 (4-bit vectors). Define triples based on
    Hamming distance or linear algebra over F_2.
    
    For a simpler deterministic approach: use the structure of the 4-dimensional
    cube. Each "face" (2-dimensional subcube) has 4 vertices. We can form triples
    from these faces.
    
    Actually, let's use a greedy but structured approach:
    1. Start with all 4-bit vectors as points
    2. For each pair of points at Hamming distance 1, they form an edge
    3. For each edge, find a third point to form a "path of length 2" or "V-shape"
    4. Ensure linearity
    
    Let me try a concrete construction based on the affine geometry AG(4,2):
    - Points: 16 elements of F_2^4
    - We want triples that are "lines" in some sense
    - In AG(4,2), lines have 2 points, so that doesn't directly work
    
    Alternative: Use the 2-(16,4,2) design (affine plane of order 4) and
    select triples from each block.
    
    The affine plane AG(2,4) has 16 points, 20 lines (each with 4 points),
    and 5 parallel classes of 4 lines each.
    
    From each line of 4 points, we can form 4 triples (remove one point).
    If we select triples carefully to maintain linearity, we might get a good system.
    
    Let me implement a specific deterministic construction:
    """
    # F_4 arithmetic: elements are 0, 1, 2, 3 where 2=ω, 3=ω^2
    # Addition: (0,1,2,3) -> (0,1,3,2) i.e., 2+2=0, 2+3=1, 3+3=0
    # Multiplication: 2*2=3, 2*3=2, 3*3=2, 2*2=3, etc.
    # Actually: ω^2 = ω+1, so:
    # 0+0=0, 0+1=1, 0+2=2, 0+3=3
    # 1+1=0, 1+2=3, 1+3=2
    # 2+2=0, 2+3=1
    # 3+3=0
    # Multiplication:
    # 0*x=0
    # 1*x=x
    # 2*2=3 (ω^2), 2*3=2 (ω*ω^2=ω^3=1... wait)
    # ω^3 = 1, so ω*ω^2 = ω^3 = 1
    # So 2*3 = 1
    # 3*3 = ω^4 = ω = 2
    
    # Let me just hardcode the affine plane of order 4
    # Points: (x,y) where x,y in F_4, identified with 0..15
    # Line with slope m through (x0,y0): y = m(x-x0) + y0
    # Plus vertical lines x = c
    
    # F_4 tables
    add_table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0]
    ]
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    
    def f4_add(a, b):
        return add_table[a][b]
    
    def f4_mul(a, b):
        return mul_table[a][b]
    
    def f4_sub(a, b):
        # In F_4, subtraction = addition (characteristic 2)
        return f4_add(a, b)
    
    # Generate all lines in AG(2,4)
    # Each line has 4 points
    # Points are indexed as x*4 + y where x,y in {0,1,2,3}
    
    lines = []
    
    # Vertical lines: x = c
    for c in range(4):
        line = [c*4 + y for y in range(4)]
        lines.append(line)
    
    # Lines with slope m for m in {0, 1, 2, 3}
    for m in range(4):
        for b in range(4):  # intercept
            line = []
            for x in range(4):
                y = f4_add(f4_mul(m, x), b)
                line.append(x*4 + y)
            lines.append(line)
    
    # Now we have 5 parallel classes of 4 lines each = 20 lines total
    # Each line has 4 points
    # We need to select triples from these lines
    # To maintain linearity, we should be careful about which triples we select
    
    # Strategy: For each line, select one triple (remove one point)
    # But we need to ensure that no pair of points appears in more than one triple
    # This is equivalent to: the selected triples form a linear 3-graph
    
    # Let's try: for each line, remove the point with the smallest index
    # and check if the resulting system is linear
    # If not, try other removals
    
    best_triples = []
    best_alpha = 16
    
    # Try different strategies for selecting triples from lines
    strategies = [
        "smallest",   # remove smallest point
        "largest",    # remove largest point  
        "first",      # remove first in line
        "last",       # remove last in line
        "random1",    # random with seed 1
        "random2",    # random with seed 2
    ]
    
    for strategy in strategies:
        triples = []
        pair_used = set()
        
        for line in lines:
            if strategy == "smallest":
                remove_idx = line.index(min(line))
            elif strategy == "largest":
                remove_idx = line.index(max(line))
            elif strategy == "first":
                remove_idx = 0
            elif strategy == "last":
                remove_idx = 3
            elif strategy == "random1":
                remove_idx = (line[0] + line[1]) % 4
            elif strategy == "random2":
                remove_idx = (line[2] + line[3]) % 4
            
            triple = [line[i] for i in range(4) if i != remove_idx]
            triple_set = set(triple)
            
            # Check linearity
            ok = True
            for a, b in combinations(triple, 2):
                pair = (min(a,b), max(a,b))
                if pair in pair_used:
                    ok = False
                    break
            if ok:
                triples.append(triple)
                for a, b in combinations(triple, 2):
                    pair_used.add((min(a,b), max(a,b)))
        
        alpha = _independence_number_fast(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = [list(t) for t in triples]
    
    return best_triples, best_alpha

def construct():
    """Construct a linear triple system on 16 points minimizing independence number."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Phase 0: Try deterministic algebraic construction
    alg_triples, alg_alpha = _algebraic_construction()
    best_triples = alg_triples
    best_alpha = alg_alpha
    
    if time.time() > deadline:
        return best_triples
    
    # Phase 1: Greedy construction with multiple restarts
    greedy_triples, greedy_alpha = _greedy_construct(
        n_points=16, 
        n_iter=500, 
        seed=12345,
        deadline=deadline
    )
    
    if greedy_alpha < best_alpha:
        best_triples = greedy_triples
        best_alpha = greedy_alpha
    
    if time.time() > deadline:
        return best_triples
    
    # Phase 2: Local search improvement
    improved_triples, improved_alpha = _local_search(
        best_triples,
        n_points=16,
        max_iter=200,
        deadline=deadline
    )
    
    if improved_alpha < best_alpha:
        best_triples = improved_triples
        best_alpha = improved_alpha
    
    if best_alpha <= 5:
        return best_triples
    
    # Phase 3: More greedy restarts with different strategies
    for seed in [7, 42, 99, 2024, 31415, 1234, 5678, 9999]:
        if time.time() > deadline:
            break
        t, a = _greedy_construct(
            n_points=16,
            n_iter=200,
            seed=seed,
            deadline=deadline
        )
        if a < best_alpha:
            best_triples = t
            best_alpha = a
            if best_alpha <= 5:
                break
    
    if time.time() > deadline:
        return best_triples
    
    # Final local search
    final_triples, final_alpha = _local_search(
        best_triples,
        n_points=16,
        max_iter=100,
        deadline=deadline
    )
    
    if final_alpha < best_alpha:
        best_triples = final_triples
    
    return best_triples

# EVOLVE-BLOCK-END
