import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 16
    # Proper edge coloring of K_16 using 15 colors
    # Vertices 0..15
    # Colors 0..14
    # Edge (i, j) gets color (i + j) % 15 if i != j? No, (i+j)%15 is not proper for even n?
    # For K_{2k}, a proper coloring with 2k-1 colors is given by (i+j) mod (2k-1) for i!=j?
    # Let's check K_4 (n=4, 3 colors).
    # (0,1)->1, (0,2)->2, (0,3)->0
    # (1,2)->3%3=0. (0,3) is 0, (1,2) is 0. Disjoint. OK.
    # (1,3)->4%3=1. (0,1) is 1. Adjacent? No, (0,1) and (1,3) share vertex 1.
    # Color of (0,1) is 1. Color of (1,3) is 1. This is NOT proper.
    
    # Correct proper coloring for K_{2k}:
    # Use colors 0..2k-2.
    # Edge (i, j) where i, j in 0..2k-2: (i+j) mod (2k-1)
    # Edge (i, 2k-1): i
    # Edge (2k-1, j): j
    # Let's verify K_4 (k=2, colors 0,1,2).
    # Vertices 0,1,2,3.
    # (0,1): 1
    # (0,2): 2
    # (1,2): 3%3=0
    # (0,3): 0
    # (1,3): 1
    # (2,3): 2
    # Check adjacency:
    # Vertex 0: (0,1)=1, (0,2)=2, (0,3)=0. Distinct.
    # Vertex 1: (0,1)=1, (1,2)=0, (1,3)=1. Wait. (0,1) and (1,3) both 1.
    # (1,3) color is 1. (0,1) color is 1. They share vertex 1.
    # So this formula is wrong for the last vertex.
    
    # Standard 1-factorization based coloring:
    # Colors 0..14.
    # For color c, the matching is:
    # If c < 15:
    #   Edge (i, j) such that (i+j) % 15 == c? No.
    #   Let's use the group Z_15 x {0,1}? No.
    
    # Simplest:
    # Colors 0..14.
    # For i in 0..14, j in i+1..14: color = (i+j) % 15
    # For i in 0..14: edge (i, 15) has color i
    # Let's check vertex 15:
    # Edges (0,15)=0, (1,15)=1, ..., (14,15)=14. All distinct.
    # Check vertex i (0..14):
    # Edges to 15: color i.
    # Edges to j (j != i, j != 15): color (i+j) % 15.
    # We need (i+j) % 15 != i for all j != i, 15.
    # (i+j) % 15 = i => j % 15 = 0 => j=0.
    # So if j=0, color is (i+0)%15 = i.
    # Edge (i, 0) has color i. Edge (i, 15) has color i.
    # They share vertex i. So this is NOT proper.
    
    # Fix:
    # Use colors 0..14.
    # For i < j < 15: color = (i+j) % 15
    # For j=15: color = i
    # Conflict at j=0 for i>0.
    # (i,0) color i. (i,15) color i.
    # To fix, shift the colors for the first 15 vertices?
    # Let's use a known 1-factorization.
    # K_16 vertices 0..15.
    # 15 factors F_0..F_14.
    # F_c consists of edges (i, j) such that (i+j) mod 15 = c, for i,j in 0..14, i!=j.
    # Plus edge (15, ?). No, F_c has 8 edges in 0..14? No, 15 vertices is odd.
    # 0..14 is 15 vertices. (i+j)%15=c has 7 pairs and 1 self-loop? No.
    # In Z_15, equation x+y=c.
    # If c is even, 7 pairs and 0 self? No, 15 is odd.
    # x+y=c. If x=c/2, self loop.
    # So for each c, there is 1 vertex v_c = c*inv(2) mod 15?
    # 2*8=16=1. So inv(2)=8.
    # v_c = 8c mod 15.
    # The other 14 vertices form 7 pairs.
    # So F_c has 7 edges in 0..14, plus edge (v_c, 15).
    # Total 8 edges per factor. 15 factors * 8 = 120 edges. Correct.
    
    colors = [0] * 120
    idx = 0
    for i in range(16):
        for j in range(i+1, 16):
            if j == 15:
                # Edge (i, 15) is in factor i
                colors[idx] = i
            else:
                # Edge (i, j) with i,j < 15
                # Find factor c such that (i+j) % 15 == c
                colors[idx] = (i + j) % 15
            idx += 1
            
    return colors

# EVOLVE-BLOCK-END
