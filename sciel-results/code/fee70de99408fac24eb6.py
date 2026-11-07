"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 10}

def construct():
    n = P['n']
    deadline = time.time() + 50  # 50s safety margin within 60s limit
    
    # L1: (i + j) % 10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find a second square L2 orthogonal to L1.
    # We will use a randomized search with backtracking or just try specific forms.
    # Given the difficulty, we try a few specific non-linear forms first.
    
    # Form: (i + a*j + b*i*j) % n ? No, must be Latin.
    
    # Let's try to find L2 by permuting rows and columns of a candidate.
    # A candidate Latin square can be L1 itself, but we need orthogonality.
    # We can generate random Latin squares and check orthogonality.
    # Generation of random Latin squares is slow.
    
    # Instead, we can use the known fact that 2 MOLS exist.
    # I will hardcode a known pair if possible, or use a fast search.
    
    # Fast search:
    # Fix L1.
    # L2 must be a Latin square.
    # We can try to construct L2 row by row.
    # Row 0 of L2 must be a permutation of 0..9.
    # Let's fix L2[0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (identity) to reduce symmetry?
    # No, L2[0] can be any permutation.
    
    # Actually, there is a simpler construction for n=10.
    # L1(i,j) = (i+j) % 10
    # L2(i,j) = (i + 3*j) % 10 is NOT orthogonal.
    
    # Let's try to return 1 square if search fails, but try hard for 2.
    
    # Heuristic: Try a few specific "twisted" squares.
    # One known pair for n=10:
    # S1 = (i+j) % 10
    # S2 = ...
    
    # I will use a simple randomized search for L2.
    # Since n=10 is small, we can try to complete a partial Latin square.
    
    # To be safe and fast, I will return [L1] if I can't find L2 quickly.
    # But I want to maximize.
    
    # Let's try a specific construction:
    # L2[i][j] = (i + j + (i*j % 2) * 5) % 10 ? No.
    
    # I will implement a basic backtracking search for L2 orthogonal to L1.
    # This might be too slow for 10x10 in Python without optimization.
    
    # Alternative: Use the product of MOLS of order 2 and 5.
    # MOLS of order 2: L2_1(i,j) = i+j % 2.
    # MOLS of order 5: L5_1(i,j) = i+j % 5, L5_2(i,j) = i+2j % 5, etc.
    # We can combine them.
    
    # The product construction:
    # Let M = 2, N = 5.
    # Symbols are pairs (a,b) where a in 0..1, b in 0..4.
    # Map (a,b) to a*5 + b.
    # L_M(i,j) = i+j % 2.
    # L_N(i,j) = i+j % 5.
    # L_10(i,j) = 5 * L_M(i,j) + L_N(i,j).
    # This gives a Latin square of order 10.
    # Is it orthogonal to L1 = (i+j) % 10?
    # L1(i,j) = i+j.
    # L_10(i,j) = 5*(i+j % 2) + (i+j % 5).
    # Let's check orthogonality.
    
    def check_orthogonal(La, Lb):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (La[i][j], Lb[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    # Construct L_10 from MOLS of 2 and 5.
    # L2(i,j) = (i+j) % 2
    # L5(i,j) = (i+j) % 5
    # L_prod(i,j) = 5 * L2(i,j) + L5(i,j)
    
    L2_part = [[(i + j) % 2 for j in range(n)] for i in range(n)]
    L5_part = [[(i + j) % 5 for j in range(n)] for i in range(n)]
    
    L2 = [[5 * L2_part[i][j] + L5_part[i][j] for j in range(n)] for i in range(n)]
    
    # Verify L2 is a Latin square
    is_latin = True
    for i in range(n):
        if len(set(L2[i])) != n:
            is_latin = False
            break
    if is_latin:
        for j in range(n):
            col = [L2[i][j] for i in range(n)]
            if len(set(col)) != n:
                is_latin = False
                break
    
    if is_latin:
        if check_orthogonal(L1, L2):
            return [L1, L2]
            
    # If the simple product didn't work, try another combination.
    # L5_2(i,j) = (i + 2*j) % 5
    L5_part2 = [[(i + 2 * j) % 5 for j in range(n)] for i in range(n)]
    L2b = [[5 * L2_part[i][j] + L5_part2[i][j] for j in range(n)] for i in range(n)]
    
    is_latin = True
    for i in range(n):
        if len(set(L2b[i])) != n:
            is_latin = False
            break
    if is_latin:
        for j in range(n):
            col = [L2b[i][j] for i in range(n)]
            if len(set(col)) != n:
                is_latin = False
                break
                
    if is_latin:
        if check_orthogonal(L1, L2b):
            return [L1, L2b]
            
    # If all fails, return L1
    return [L1]

# EVOLVE-BLOCK-END
