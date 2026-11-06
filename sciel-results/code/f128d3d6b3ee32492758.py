"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    n = 34
    start_time = time.time()
    time_limit = 115  # Leave buffer for overhead
    
    # Use a deterministic seed for reproducibility
    rng = random.Random(42)
    
    # Initial coloring: use a structured algebraic-based approach
    # Color based on weighted sum modulo a prime, with careful weighting
    # to break symmetries that cause monochromatic cliques
    weights_a = [1, 2, 4, 8, 16, 5, 10, 3, 6, 12, 13, 15, 14, 11, 7, 9, 17]  # 17 weights for Z_17
    # Map vertices 0..33 to (i, j) where i in Z_17, j in {0,1}
    
    # Generate all triples in lexicographic order
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    
    # Initial coloring: use a hash-like function based on vertex positions
    # to create a pseudo-random but deterministic initial state
    colors = [0] * num_triples
    for idx, (a, b, c) in enumerate(triples):
        # Map to Z_17 x Z_2 representation
        a1, a2 = a % 17, a // 17
        b1, b2 = b % 17, b // 17
        c1, c2 = c % 17, c // 17
        
        # Compute a structured value
        # Use multiple moduli to avoid simple patterns
        s1 = (a1 + 2*b1 + 4*c1) % 17
        s2 = (a2 + b2 + c2) % 2
        s3 = (a1*b1 + b1*c1 + c1*a1) % 5
        s4 = (a + 3*b + 9*c) % 7
        
        # Combine to get a color
        combined = (s1 * 7 + s2 * 11 + s3 * 13 + s4 * 17) % 29
        colors[idx] = 0 if combined < 14 else 1  # Roughly balanced
    
    # Helper: get color of triple (a,b,c)
    triple_index = {}
    for idx, (a, b, c) in enumerate(triples):
        triple_index[(a, b, c)] = idx
    
    def get_color(a, b, c):
        return colors[triple_index[(a, b, c)]]
    
    def set_color(a, b, c, col):
        colors[triple_index[(a, b, c)]] = col
    
    # Helper: check if a set of vertices forms a monochromatic clique
    def is_red_k4(vertices):
        """Check if 4 vertices have all 4 triples red"""
        v = sorted(vertices)
        for combo in combinations(v, 3):
            if get_color(*combo) != 0:
                return False
        return True
    
    def is_blue_k5(vertices):
        """Check if 5 vertices have all 10 triples blue"""
        v = sorted(vertices)
        for combo in combinations(v, 3):
            if get_color(*combo) != 1:
                return False
        return True
    
    def count_violations():
        """Count red K4 and blue K5 violations"""
        red_violations = 0
        blue_violations = 0
        # Sample some 4-sets and 5-sets for speed
        # Full check is too slow, so we do partial checks
        sample_4sets = list(combinations(range(n), 4))
        rng.shuffle(sample_4sets)
        for quad in sample_4sets[:5000]:
            if is_red_k4(quad):
                red_violations += 1
        
        sample_5sets = list(combinations(range(n), 5))
        rng.shuffle(sample_5sets)
        for quint in sample_5sets[:2000]:
            if is_blue_k5(quint):
                blue_violations += 1
        
        return red_violations, blue_violations
    
    # Phase 1: Local search to fix violations
    max_iterations = 5000
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Find a violation
        found_violation = False
        
        # Check for red K4 violations (sample)
        for _ in range(100):
            quad = tuple(sorted(rng.sample(range(n), 4)))
            if is_red_k4(quad):
                # Flip a random edge in this K4 to blue
                edges = list(combinations(quad, 3))
                edge_to_flip = rng.choice(edges)
                set_color(*edge_to_flip, 1)
                found_violation = True
                break
        
        if found_violation:
            continue
        
        # Check for blue K5 violations (sample)
        for _ in range(50):
            quint = tuple(sorted(rng.sample(range(n), 5)))
            if is_blue_k5(quint):
                # Flip a random edge in this K5 to red
                edges = list(combinations(quint, 3))
                edge_to_flip = rng.choice(edges)
                set_color(*edge_to_flip, 0)
                found_violation = True
                break
    
    # Phase 2: Final verification and cleanup
    # Do a more thorough check and fix remaining violations
    for _ in range(3):
        if time.time() - start_time > time_limit:
            break
        
        # Check all 4-sets for red K4 (this is C(34,4)=46376, feasible)
        for quad in combinations(range(n), 4):
            if time.time() - start_time > time_limit:
                break
            if is_red_k4(quad):
                # Flip all 4 edges to break the K4
                for edge in combinations(quad, 3):
                    set_color(*edge, 1)
        
        # Check all 5-sets for blue K5 (C(34,5)=278256, might be slow)
        # Only do partial check if time permits
        if time.time() - start_time < time_limit - 20:
            for quint in combinations(range(n), 5):
                if time.time() - start_time > time_limit:
                    break
                if is_blue_k5(quint):
                    # Flip all 10 edges to break the K5
                    for edge in combinations(quint, 3):
                        set_color(*edge, 0)
    
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
