"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def construct():
    # D11 lattice roots: vectors with exactly two non-zero entries of ±1
    # Norm squared = 2 for all roots
    # Minimum pairwise distance condition: ||v_i - v_j|| >= max_norm = sqrt(2)
    # So ||v_i - v_j||^2 >= 2
    
    # For two roots v, w with norm^2 = 2:
    # ||v - w||^2 = ||v||^2 + ||w||^2 - 2<v,w> = 4 - 2<v,w>
    # We need 4 - 2<v,w> >= 2 => <v,w> <= 1
    
    # The inner product <v,w> for roots of D11 can be:
    # -2: if v = w (not allowed, we pick distinct)
    # -1: if they share one coordinate with opposite signs, other coords different
    #  0: if they are orthogonal (no shared coordinates, or shared with canceling)
    #  1: if they share one coordinate with same signs
    #  2: if v = w (same vector)
    
    # We need <v,w> <= 1, so we must avoid pairs with <v,w> = 2 (identical, already excluded)
    # But also need to check <v,w> = 1 cases
    
    # Actually, let's be more careful. For D11 roots:
    # v has non-zero entries at positions i,j with values a,b (each ±1)
    # w has non-zero entries at positions k,l with values c,d (each ±1)
    
    # <v,w> = sum of v_m * w_m over all m
    
    # If {i,j} and {k,l} are disjoint: <v,w> = 0 ✓
    # If they share exactly one coordinate, say i=k:
    #   <v,w> = a*c (from shared coordinate) + 0 + 0 + 0 = a*c ∈ {-1, 1}
    #   So if a*c = 1, <v,w> = 1 ✓ (allowed, since we need <v,w> <= 1)
    #   If a*c = -1, <v,w> = -1 ✓
    # If {i,j} = {k,l} (same support):
    #   <v,w> = a*c + b*d
    #   If (a,b) = (c,d): <v,w> = 2 ✗ (same vector, excluded)
    #   If (a,b) = (-c,-d): <v,w> = -2 ✓
    #   If (a,b) = (c,-d): <v,w> = 0 ✓
    #   If (a,b) = (-c,d): <v,w> = 0 ✓
    
    # So the constraint <v,w> <= 1 is automatically satisfied for all distinct pairs of D11 roots!
    # The only forbidden case is identical vectors, which we don't include.
    
    # Therefore, ALL D11 roots form a valid kissing configuration!
    # Number of D11 roots = C(11,2) * 4 = 55 * 4 = 220
    
    vectors = []
    for i, j in combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
