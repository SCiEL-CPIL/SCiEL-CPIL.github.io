import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def generate_sts15():
    # Points 0..14
    # Lines of PG(3,2)
    # We can hardcode or generate
    # Generation:
    # Points are non-zero vectors in GF(2)^4
    # We can just use a known list of STS(15) or generate it.
    
    # Simple generator for STS(15)
    # Actually, let's just use a standard construction.
    # Or generate all triples and pick a Steiner system?
    # Easier: Use the fact that STS(15) is unique up to isomorphism? No, there are many.
    # But any STS(15) works.
    
    # Let's generate one using the affine geometry AG(4,2) minus infinity?
    # No, PG(3,2) is easier.
    
    points = []
    for i in range(1, 16):
        points.append(i-1)
        
    # We need the lines.
    # A line is {u, v, u+v} in GF(2)^4.
    # Let's map 0..14 to non-zero vectors.
    # 1: 0001, 2: 0010, 4: 0100, 8: 1000
    # 3: 0011, 5: 0101, 6: 0110, 7: 0111
    # 9: 1001, 10: 1010, 11: 1011, 12: 1100
    # 13: 1101, 14: 1110, 15: 1111
    # This mapping is not 1-to-1 with indices 0..14 directly if we use 1..15.
    # Let's just use indices 0..14 and define vectors.
    
    vecs = []
    for i in range(15):
        # Use binary representation of i+1
        v = 0
        x = i + 1
        for j in range(4):
            if x & (1 << j):
                v |= (1 << j)
        vecs.append(v)
        
    lines = []
    seen = set()
    for i in range(15):
        for j in range(i+1, 15):
            u = vecs[i]
            v = vecs[j]
            w = u ^ v
            # Find index of w
            # w is non-zero since u!=v
            k = vecs.index(w)
            # The line is {i, j, k}
            # Sort to unique
            line = tuple(sorted([i, j, k]))
            if line not in seen:
                seen.add(line)
                lines.append(line)
                
    # There should be 35 lines
    return [list(l) for l in lines]

def compute_alpha(triples):
    """
    Compute the independence number (max size of subset containing no triple)
    using scipy.optimize.milp as an exact ILP solver.
    
    Formulation: Maximize sum(x_i) subject to:
      For each triple {a,b,c}: x_a + x_b + x_c <= 2
      x_i in {0, 1} for i = 0..15
    """
    n_points = 16
    n_triples = len(triples)
    
    if n_triples == 0:
        return n_points
    
    # Objective: maximize sum(x_i) => minimize -sum(x_i)
    c = np.full(n_points, -1.0)
    
    # Constraints: for each triple, sum of its 3 vertices <= 2
    # A matrix: n_triples x n_points
    A = np.zeros((n_triples, n_points))
    for i, t in enumerate(triples):
        for p in t:
            A[i, p] = 1.0
    
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(n_points)
    ub = np.ones(n_points)
    
    # Integrality: all variables are integer
    integrality = np.ones(n_points)
    
    # Solve ILP
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, ub=np.full(n_triples, 2.0), lb=-np.inf),
            integrality=integrality,
            bounds=Bounds(lb, ub)
        )
        
        if res.status == 0:  # Optimal
            # res.fun is the minimized value, which is -max_size
            max_size = int(round(-res.fun))
            return max_size
        else:
            # Fallback to bitmask if ILP fails
            return _compute_alpha_bitmask(triples)
    except Exception:
        return _compute_alpha_bitmask(triples)


def _compute_alpha_bitmask(triples):
    """Fallback: exhaustive bitmask enumeration."""
    triple_masks = [0] * len(triples)
    for i, t in enumerate(triples):
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks[i] = m
    
    max_size = 0
    for mask in range(1 << 16):
        is_independent = True
        for m in triple_masks:
            if (mask & m) == m:
                is_independent = False
                break
        if is_independent:
            size = bin(mask).count("1")
            if size > max_size:
                max_size = size
    return max_size

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    sts15 = generate_sts15()
    base_triples = [list(t) for t in sts15]
    
    best_alpha = 16
    best_triples = None
    
    # Evaluate the base STS(15) with point 15 isolated
    alpha = compute_alpha(base_triples)
    best_alpha = alpha
    best_triples = base_triples[:]
    
    # Map pair to line index in sts15
    pair_to_line = {}
    for i, line in enumerate(sts15):
        for a in range(3):
            for b in range(a + 1, 3):
                p1, p2 = line[a], line[b]
                pair_to_line[(min(p1, p2), max(p1, p2))] = i
    
    points_15 = list(range(15))
    
    # Strategy: Systematically explore configurations where we replace
    # some STS lines with triples involving point 15.
    # Since ILP is fast, we can afford more iterations.
    
    # First, try deterministic small-k sweeps
    for k in range(1, 8):
        if time.time() > deadline:
            break
        
        # Try all combinations of k disjoint pairs (sampled if too many)
        # Number of ways to choose k disjoint pairs from 15 points
        # For k=1: C(15,2)=105
        # For k=2: ~105*93/2 ≈ 4800 (rough)
        # We'll sample randomly for larger k
        
        max_iter = 500 if k <= 2 else 200
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            available = set(points_15)
            selected_pairs = []
            while len(selected_pairs) < k and len(available) >= 2:
                p1, p2 = random.sample(sorted(available), 2)
                pair = (min(p1, p2), max(p1, p2))
                selected_pairs.append(pair)
                available.discard(p1)
                available.discard(p2)
            
            if len(selected_pairs) < k:
                continue
            
            # Construct new triples
            new_triples = []
            removed_indices = set()
            for pair in selected_pairs:
                idx = pair_to_line.get(pair)
                if idx is not None:
                    removed_indices.add(idx)
                new_triples.append([pair[0], pair[1], 15])
            
            for i, line in enumerate(sts15):
                if i not in removed_indices:
                    new_triples.append(line)
            
            alpha = compute_alpha(new_triples)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = new_triples[:]
    
    # Random search for remaining time
    while time.time() < deadline:
        k = random.randint(1, 7)
        available = set(points_15)
        selected_pairs = []
        while len(selected_pairs) < k and len(available) >= 2:
            p1, p2 = random.sample(sorted(available), 2)
            pair = (min(p1, p2), max(p1, p2))
            selected_pairs.append(pair)
            available.discard(p1)
            available.discard(p2)
        
        if len(selected_pairs) < k:
            continue
        
        new_triples = []
        removed_indices = set()
        for pair in selected_pairs:
            idx = pair_to_line.get(pair)
            if idx is not None:
                removed_indices.add(idx)
            new_triples.append([pair[0], pair[1], 15])
        
        for i, line in enumerate(sts15):
            if i not in removed_indices:
                new_triples.append(line)
        
        alpha = compute_alpha(new_triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = new_triples[:]
    
    return best_triples
