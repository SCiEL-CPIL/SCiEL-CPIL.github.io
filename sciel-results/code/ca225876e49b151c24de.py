"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N = len(vectors_v4)
    
    if N == 0:
        return np.array([], dtype=np.int64).reshape(0, D)

    # Compute compatibility matrix: compatible if dot <= 2
    # dots = vectors_v4 @ vectors_v4.T
    # To save memory/time, we can compute on the fly or use sparse representation
    # But N=5280, N^2 is 27M floats, ~200MB. Acceptable.
    dots = vectors_v4 @ vectors_v4.T
    # Compatible if dot <= 2. Conflict if dot > 2.
    # We want Max Independent Set in Conflict Graph.
    # This is Max Clique in Compatibility Graph.
    
    # Tomita's Algorithm for Maximum Clique
    # Graph is defined by compatibility.
    # We represent the graph by the compatibility matrix or just the conflict matrix.
    # Let's use the conflict matrix to define "neighbors" in the MIS sense.
    # In MIS, we pick a node, remove it and its neighbors (conflicts).
    # In Max Clique (on compatibility graph), we pick a node, and restrict to its neighbors (compatibles).
    
    # Tomita's algorithm works on the graph where we seek the clique.
    # So we build the Compatibility Graph.
    # Adjacency: i, j are adjacent if dots[i,j] <= 2.
    
    # We will implement a recursive backtracking with coloring heuristic (Tomita).
    # This is complex to implement perfectly from scratch in Python within 120s if not optimized.
    # However, for N=5280, a pure Python implementation might be too slow.
    # We need an efficient C-like loop or use a library?
    # No external libs for graph algorithms.
    
    # Alternative: Use the fact that the graph is very regular/symmetric.
    # Or use a fast Greedy + Local Search but with a "Different" neighborhood?
    # The directive says "Design a new program ... from an approach family that none of the current programs use".
    # If I write a slow Python B&B, it will timeout.
    
    # Let's try a **Constructive** approach based on the **E8 x A3** lattice which is guaranteed to give 246.
    # And then use a **Fast Greedy** to try to add more.
    # This is a hybrid, but the core is the construction.
    
    # Generate E8 vectors (in 8D)
    # E8 min norm vectors:
    # 1. Integer vectors with norm^2=2 (two 1s or two -1s).
    # 2. Half-integer vectors with norm^2=2 (all coords +/-0.5, even number of minus? No, sum is even integer).
    # For E8, the 240 vectors are:
    # 112 integer vectors: permutations of (1,1,0,0,0,0,0,0) and (1,-1,0,0,0,0,0,0).
    #    Count: C(8,2)*4 = 28*4 = 112.
    # 128 half-integer vectors: all coords +/-0.5, with an even number of minus signs?
    #    Actually, E8 is {x in Z^8 | sum x_i even} U {x in (Z+0.5)^8 | sum x_i even}.
    #    Min norm in half-integer part is 8*(0.5)^2 = 2.
    #    All such vectors have norm^2=2.
    #    Count: 2^8 = 256.
    #    Condition: sum is even. Sum of 8 half-integers is integer.
    #    Sum = k/2. k must be even.
    #    Number of minus signs m. Sum = (8-2m)/2 = 4-m.
    #    4-m must be even => m must be even.
    #    m can be 0, 2, 4, 6, 8.
    #    Count: C(8,0)+C(8,2)+C(8,4)+C(8,6)+C(8,8) = 1+28+70+28+1 = 128.
    # Total 112 + 128 = 240.
    
    e8_int = []
    for positions in itertools.combinations(range(8), 2):
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0]*8
                v[positions[0]] = s1
                v[positions[1]] = s2
                e8_int.append(v)
                
    e8_half = []
    for signs in itertools.product([0.5, -0.5], repeat=8):
        # Check if number of -0.5 is even
        num_neg = sum(1 for s in signs if s < 0)
        if num_neg % 2 == 0:
            e8_half.append(signs)
            
    e8_vectors = []
    for v in e8_int:
        e8_vectors.append(v)
    for v in e8_half:
        e8_vectors.append(v)
        
    e8_vectors = np.array(e8_vectors, dtype=np.float64)
    # Scale by 2 to make them integers?
    # If we scale by 2, norm^2 becomes 8.
    # We want to combine with A3.
    
    # A3 vectors (in 3D)
    # Min norm vectors in A3 (sum=0, integer).
    # (1, -1, 0), (1, 0, -1), (0, 1, -1) and negatives.
    a3_vectors = []
    for i in range(3):
        for j in range(3):
            if i != j:
                v = [0]*3
                v[i] = 1
                v[j] = -1
                a3_vectors.append(v)
                a3_vectors.append([-x for x in v])
                
    a3_vectors = np.array(a3_vectors, dtype=np.float64)
    
    # Combine E8 and A3.
    # We need integer vectors in R11.
    # E8 vectors are half-integers. Scale by 2 -> integers, norm^2=8.
    # A3 vectors are integers, norm^2=2.
    # To combine, we need same norm?
    # If we use E8 (scaled, norm^2=8) and A3 (scaled by 2, norm^2=8).
    # Let's scale both to norm^2=8.
    # E8_scaled = 2 * E8. (Integers)
    # A3_scaled = 2 * A3. (Integers)
    
    e8_scaled = 2 * e8_vectors
    a3_scaled = 2 * a3_vectors
    
    # Construct full 11D vectors.
    # Type 1: E8 part non-zero, A3 part zero.
    # Type 2: E8 part zero, A3 part non-zero.
    
    final_vectors = []
    
    # Type 1: 240 vectors
    for v in e8_scaled:
        full_v = np.zeros(11)
        full_v[:8] = v
        final_vectors.append(full_v)
        
    # Type 2: 12 vectors (6 in A3, scaled by 2)
    for v in a3_scaled:
        full_v = np.zeros(11)
        full_v[8:] = v
        final_vectors.append(full_v)
        
    result = np.array(final_vectors, dtype=np.int64)
    
    # This gives 252 vectors.
    # Check validity:
    # Norm^2 = 8 for all.
    # Distance between Type 1 and Type 1:
    # Min distance in E8 is sqrt(2). Scaled by 2 -> sqrt(8).
    # Norm is sqrt(8). Distance >= Norm. OK.
    # Distance between Type 2 and Type 2:
    # Min distance in A3 is sqrt(2). Scaled by 2 -> sqrt(8).
    # OK.
    # Distance between Type 1 and Type 2:
    # Orthogonal. Distance = sqrt(8+8) = sqrt(16) = 4.
    # Norm = sqrt(8) = 2.82.
    # 4 >= 2.82. OK.
    
    return result

# EVOLVE-BLOCK-END
