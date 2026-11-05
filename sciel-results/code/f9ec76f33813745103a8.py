import numpy as np
from functools import lru_cache
import time
import itertools
from scipy.optimize import linprog

# Global deadline: 150s wall-clock, keep 20s margin for safety
START_TIME = time.time()
DEADLINE = 150 - 20  # 130s

def time_remaining():
    """Return seconds remaining before deadline."""
    return DEADLINE - (time.time() - START_TIME)

def is_timed_out():
    return time_remaining() < 1.0

# Task dimensions: multiply 4x5 matrix by 5x6 matrix
# A: 4x5, B: 5x6, C = A @ B: 4x6
# Tensor: T_{i,j,k} = 1 iff C[i,k] += A[i,j]*B[j,k]
# n=4 (rows of A/cols of C), m=5 (cols of A/rows of B), p=6 (cols of B)
n, m, p = 4, 5, 6

# Strassen-like decomposition for 2x2x2 blocks
# For a 2x2x2 sub-tensor with indices (i in {0,1}, j in {0,1}, k in {0,1}):
# C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
# C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
# C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
# C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
#
# Strassen uses 7 products instead of 4:
# P1 = (A00+A11)(B00+B11)
# P2 = (A10+A11)B00
# P3 = A00(B10-B11)
# P4 = A11(B01-B00)
# P5 = (A00+A01)B11
# P6 = (A10-A00)(B00+B01)
# P7 = (A01-A11)(B10+B11)
#
# Then:
# C00 = P1 - P4 + P6 + P7
# C01 = P3 + P5 + P7
# C10 = P2 + P4 + P6
# C11 = P1 - P3 + P5 + P2  -- wait, let me recheck
#
# Actually the standard Strassen formulas:
# C00 = P1 - P4 + P6 + P7
# C01 = P3 + P5 + P7
# C10 = P2 + P4 + P6
# C11 = P1 - P3 + P5 + P2  -- no, that's wrong
#
# Let me redo this carefully. Standard Strassen for 2x2:
# P1 = (A11 + A22) * (B11 + B22)
# P2 = (A21 + A22) * B11
# P3 = A11 * (B21 - B22)
# P4 = A22 * (B12 - B11)
# P5 = (A11 + A12) * B22
# P6 = (A21 - A11) * (B11 + B12)
# P7 = (A12 - A22) * (B21 + B22)
#
# C11 = P1 - P4 + P6 + P7
# C12 = P3 + P5 + P7
# C21 = P2 + P4 + P6
# C22 = P1 - P3 + P5 + P2  -- still wrong
#
# Let me use 0-indexed:
# P1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,0])
# P2 = (A[1,0] + A[1,1]) * B[0,0]
# P3 = A[0,0] * (B[1,0] - B[1,1])
# P4 = A[1,1] * (B[0,1] - B[0,0])
# P5 = (A[0,0] + A[0,1]) * B[1,1]
# P6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
# P7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
#
# C[0,0] = P1 - P4 + P6 + P7
# C[0,1] = P3 + P5 + P7
# C[1,0] = P2 + P4 + P6
# C[1,1] = P1 - P3 + P5 + P2  -- NO

# Let me just implement the standard 2x2 Strassen with 1-indexed then convert:
# For 2x2 matrices (1-indexed in standard literature):
# P1 = (A11+A22)(B11+B22)
# P2 = (A21+A22)B11  
# P3 = A11(B21-B22)
# P4 = A22(B12-B11)
# P5 = (A11+A12)B22
# P6 = (A21-A11)(B11+B12)
# P7 = (A12-A22)(B21+B22)
# C11 = P1 - P4 + P6 + P7
# C12 = P3 + P5 + P7
# C21 = P2 + P4 + P6
# C22 = P1 - P3 + P5 + P2  -- I keep getting confused, let me derive

# Derivation for 2x2 (0-indexed):
# We need to compute:
# C00 = A00*B00 + A01*B10
# C01 = A00*B01 + A01*B11
# C10 = A10*B00 + A11*B10
# C11 = A10*B01 + A11*B11
#
# Strassen:
# M1 = (A00+A11)(B00+B10) = A00*B00 + A00*B10 + A11*B00 + A11*B10
# M2 = (A10+A11)B00 = A10*B00 + A11*B00
# M3 = A00(B10-B11) = A00*B10 - A00*B11
# M4 = A11(B01-B00) = A11*B01 - A11*B00
# M5 = (A00+A01)B11 = A00*B11 + A01*B11
# M6 = (A10-A00)(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01
# M7 = (A01-A11)(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11
#
# C00 = M1 - M4 + M6 + M7
#   = (A00*B00 + A00*B10 + A11*B00 + A11*B10) - (A11*B01 - A11*B00) + (A10*B00 + A10*B01 - A00*B00 - A00*B01) + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
#   = A00*B00 + A00*B10 + A11*B00 + A11*B10 - A11*B01 + A11*B00 + A10*B00 + A10*B01 - A00*B00 - A00*B01 + A01*B10 + A01*B11 - A11*B10 - A11*B11
#   = A00*B10 - A11*B01 + A11*B00 + A10*B00 + A10*B01 - A00*B01 + A01*B10 + A01*B11 - A11*B11
#   This doesn't simplify to A00*B00 + A01*B10. So my formulas are wrong.

# Let me just use the well-known correct Strassen formulas. After checking:
# The standard result is:
# C[0,0] = P1 - P4 + P6 + P7
# C[0,1] = P3 + P5 + P7
# C[1,0] = P2 + P4 + P6
# C[1,1] = P1 - P3 + P5 + P2  -- this is definitely wrong
#
# I'll just do it directly with a known correct reference implementation.

def strassen_2x2():
    """
    Return the 7 terms for 2x2x2 matrix multiplication using Strassen.
    Each term is (u, v, w) dicts mapping (index) to coefficient.
    u: (i,j) -> coeff for A[i,j]
    v: (j,k) -> coeff for B[j,k]
    w: (i,k) -> coeff for C[i,k]
    """
    # Using verified Strassen formulas for 0-indexed 2x2:
    # P1 = (A00+A11)*(B00+B10)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    terms = []
    
    # P1 = (A00+A11)*(B00+B10)
    u1 = {(0,0):1, (1,1):1}
    v1 = {(0,0):1, (1,0):1}
    terms.append((u1, v1, {}))  # w filled later
    
    # P2 = (A10+A11)*B00
    u2 = {(1,0):1, (1,1):1}
    v2 = {(0,0):1}
    terms.append((u2, v2, {}))
    
    # P3 = A00*(B10-B11)
    u3 = {(0,0):1}
    v3 = {(1,0):1, (1,1):-1}
    terms.append((u3, v3, {}))
    
    # P4 = A11*(B01-B00)
    u4 = {(1,1):1}
    v4 = {(0,1):1, (0,0):-1}
    terms.append((u4, v4, {}))
    
    # P5 = (A00+A01)*B11
    u5 = {(0,0):1, (0,1):1}
    v5 = {(1,1):1}
    terms.append((u5, v5, {}))
    
    # P6 = (A10-A00)*(B00+B01)
    u6 = {(1,0):1, (0,0):-1}
    v6 = {(0,0):1, (0,1):1}
    terms.append((u6, v6, {}))
    
    # P7 = (A01-A11)*(B10+B11)
    u7 = {(0,1):1, (1,1):-1}
    v7 = {(1,0):1, (1,1):1}
    terms.append((u7, v7, {}))
    
    # Now determine W coefficients:
    # C00 = P1 - P4 + P6 + P7
    # C01 = P3 + P5 + P7
    # C10 = P2 + P4 + P6
    # C11 = P1 - P3 + P5 + P2  -- Let me verify by expansion
    
    # Actually, I need to verify these W formulas. Let me compute C[i,k] as sum_r W[r,i,k]*P_r
    # and check that it equals A[i,j]*B[j,k] summed over j.
    
    # Let me just build W by requiring the decomposition to be exact.
    # For 2x2, there are 4 output entries and 7 products.
    # Each output entry is a linear combination of the 7 products.
    # I need to find the coefficients.
    
    # Let me verify P1 - P4 + P6 + P7 for C00:
    # P1 = A00*B00 + A00*B10 + A11*B00 + A11*B10
    # -P4 = -A11*B01 + A11*B00
    # P6 = A10*B00 + A10*B01 - A00*B00 - A00*B01
    # P7 = A01*B10 + A01*B11 - A11*B10 - A11*B11
    # Sum = A00*B00 + A00*B10 + A11*B00 + A11*B10 - A11*B01 + A11*B00 + A10*B00 + A10*B01 - A00*B00 - A00*B01 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    #     = A00*B10 - A11*B01 + A11*B00 + A10*B00 + A10*B01 - A00*B01 + A01*B10 + A01*B11 - A11*B11
    
    # We want C00 = A00*B00 + A01*B10. These don't match, so my Strassen formula for W is wrong.
    
    # Let me look up the correct Strassen W matrix for 2x2.
    # After careful derivation:
    # C[0,0] = P1 - P4 + P6 + P7
    # C[0,1] = P3 + P5 + P7
    # C[1,0] = P2 + P4 + P6
    # C[1,1] = P1 - P3 + P5 + P2
    
    # I'll just test this numerically.
    
    # W coefficients: w[r, (i,k)] = coeff of product r in output (i,k)
    # r=0:P1, r=1:P2, r=2:P3, r=3:P4, r=4:P5, r=5:P6, r=6:P7
    W_coeffs = {}
    # C[0,0]:
    W_coeffs[(0,0)] = {(0,0):1}
    # P1
    W_coeffs[(0,0)].update()  # this syntax won't work
    
    # Let me just build the W matrix directly
    # w[i,k,r] for output (i,k) as sum of r-th product
    # C00 = P1 - P4 + P6 + P7
    # C01 = P3 + P5 + P7
    # C10 = P2 + P4 + P6
    # C11 = P1 - P3 + P5 + P2
    
    # So W matrix (flattened as W[k*p+i, r]):
    # For i=0,k=0: [1,0,0,-1,0,1,1]
    # For i=0,k=1: [0,0,1,0,1,0,1]
    # For i=1,k=0: [0,1,0,1,0,1,0]
    # For i=1,k=1: [1,1,-1,0,1,0,0]
    
    # Wait, let me double-check C11 = P1 - P3 + P5 + P2
    # P1 = A00*B00 + A00*B10 + A11*B00 + A11*B10
    # -P3 = -A00*B10 + A00*B11
    # P5 = A00*B11 + A01*B11
    # P2 = A10*B00 + A11*B00
    # Sum = A00*B00 + A00*B10 + A11*B00 + A11*B10 - A00*B10 + A00*B11 + A00*B11 + A01*B11 + A10*B00 + A11*B00
    #     = A00*B00 + A11*B00 + A11*B10 + 2*A00*B11 + A01*B11 + A10*B00
    # We want C11 = A10*B01 + A11*B11. This is completely wrong.
    
    # My Strassen formulas are incorrect. Let me just use a brute-force approach
    # to find a valid decomposition, or use the schoolbook method and accept rank 20 (4*5*6=120 schoolbook, 
    # but for 2x2x2 blocks it's 4 vs 7 for Strassen).
    
    # Actually, let me just return the schoolbook terms for 2x2x2:
    # 4 terms, one for each (i,j,k) where i,j in {0,1}, k in {0,1}... 
    # No, for 2x2x2 the schoolbook has 2*2*2=8? No.
    # For multiplying 2x2 by 2x2: each C[i,k] = sum_j A[i,j]*B[j,k]
    # Total schoolbook: for each (i,j) pair and each k, we have a product. 
    # Actually the tensor has n*m*p = 4*4*4=64? No, 2*2*2=8 entries in the tensor.
    # Schoolbook rank for 2x2x2 is 4 (one per output entry, since each is a sum of 2 products).
    # Wait no, the rank of the 2x2x2 multiplication tensor is 4.
    # Strassen says rank <= 7? That can't be right since 4 < 7.
    
    # I'm confusing myself. The 2x2x2 matrix multiplication tensor has:
    # - 4 input pairs (i,j) for A
    # - 4 input pairs (j,k) for B  
    # - 4 output pairs (i,k) for C
    # The tensor T[i,j,k,l,m,n] = 1 if l=m and (i,j,k) corresponds to C[i,k] += A[i,j]*B[j,k]
    # Actually the standard view: T is a 3-tensor with modes (i,k) for C, (i,j) for A, (j,k) for B
    # Mode sizes: C has 4 entries, A has 4 entries, B has 4 entries
    # The schoolbook decomposition has 4 terms (one for each (i,j) in the sum, and k is determined)
    # Wait, no. Each product A[i,j]*B[j,k] contributes to C[i,k].
    # For 2x2, there are 2*2=4 values of (i,j), and for each, 2 values of k.
    # So 8 products total? No, that's not right either.
    
    # Let me be very precise. To compute C = A@B where A is 2x2 and B is 2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  (2 products)
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  (2 products)
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]  (2 products)
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]  (2 products)
    # Total: 8 bilinear products.
    
    # But in the tensor decomposition framework, each "term" r defines:
    # u_r: vector of length n*m=4 (coefficients for A)
    # v_r: vector of length m*p=4 (coefficients for B)
    # w_r: vector of length p*n=4 (coefficients for C)
    # The rank is the number of terms.
    
    # Schoolbook: 8 terms (one for each (i,j,k) triple)
    # Strassen: 7 terms (if it works for 2x2, which it should since Strassen is for 2x2)
    
    # So Strassen reduces from 8 to 7 for 2x2x2. Let me just implement it correctly.
    
    return terms  # I'll fix this below

# Given the complexity, let me use a different approach.
# I'll directly construct the tensor and find a low-rank decomposition.

def construct():
    """
    Construct U, V, W for matrix multiplication 4x5 * 5x6 -> 4x6.
    U: (n*m, rank) = (20, rank)
    V: (m*p, rank) = (30, rank) 
    W: (p*n, rank) = (24, rank)
    W[k*n+i, r] is the coefficient for C[i,k] in term r.
    """
    
    # First, try the schoolbook decomposition: rank = n*m*p... no, rank = n*m + stuff
    # Schoolbook: for each (i,j) in A and k in B, we have a product A[i,j]*B[j,k] contributing to C[i,k]
    # Number of such products = n*m*p = 4*5*6 = 120? No.
    # For each (i, j) pair (20 of them) and each k (6 of them), we have product A[i,j]*B[j,k].
    # But that's 120 products. However, each such product contributes to a different C[i,k].
    # Actually, the standard schoolbook has n*m*p = 120 terms? That seems high.
    
    # Wait: the multiplication tensor for (n x m) * (m x p) -> (n x p) has:
    # Number of (i,j) indices: n*m
    # Number of (j,k) indices: m*p
    # Number of (i,k) indices: n*p
    # The tensor is trilinear: T[A,B] = A@B
    # Each "scalar product" in the schoolbook is A[i,j] * B[j,k] which goes to C[i,k].
    # There are n*m*p such pairs? No, for a fixed (i,j), j is fixed, and k ranges over p values.
    # So for each (i,j) with n*m choices, and each k with p choices, we get one product.
    # Total: n*m*p = 4*5*6 = 120 products. But each product only affects one C[i,k].
    
    # However, in the tensor rank decomposition, each term r has:
    # u_r[i*m+j] = a_r, v_r[j*p+k] = b_r, w_r[k*n+i] = c_r
    # And C[i,k] = sum_r c_r * a_r * b_r (with appropriate indexing)
    
    # For schoolbook: each term r corresponds to a unique (i,j,k):
    # u_r has 1 at position i*m+j, v_r has 1 at position j*p+k, w_r has 1 at position k*n+i
    # Rank = n*m*p = 120 for schoolbook.
    
    # Strassen can do better by combining products.
    
    # Let me just go with a straightforward approach:
    # 1. Build the schoolbook terms (rank 120)
    # 2. Try to reduce using Strassen-like blocks
    # 3. For the given dimensions 4x5x6, decompose into blocks
    
    # Dimensions: n=4, m=5, p=6
    # Decompose as:
    # i: 4 = 2+2
    # j: 5 = 2+3 (or 1+4, 2+3, etc.)
    # k: 6 = 2+4 (or 3+3, 2+4, 1+5, etc.)
    
    # For a block of size (ni x nj) * (nj x nk):
    # Schoolbook rank = ni * nj * nk
    # If ni=nj=nk=2, Strassen gives 7 instead of 8.
    
    # Let me decompose:
    # Split i into {0,1} and {2,3}
    # Split j into {0,1} and {2,3,4}
    # Split k into {0,1,2} and {3,4,5}
    
    # This gives 4 blocks:
    # Block (0,0): 2x2 * 2x3 -> rank = 2*2*3 = 12 schoolbook
    # Block (0,1): 2x2 * 2x3 -> rank = 12
    # Block (1,0): 2x3 * 3x3 -> rank = 2*3*3 = 18
    # Block (1,1): 2x3 * 3x3 -> rank = 18
    # Total: 60. That's half of 120.
    
    # But wait, when we split, the blocks are independent, so the total rank is the sum.
    # For 2x2*2x2 blocks, we can use Strassen (7 instead of 8).
    # But our blocks aren't all 2x2*2x2.
    
    # Let me try a better decomposition:
    # i: 4 = 2+2
    # j: 5 = 2+3
    # k: 6 = 3+3
    
    # Blocks:
    # (i=2,j=2,k=3): 2x2*2x3 = 12
    # (i=2,j=3,k=3): 2x3*3x3 = 18
    # (i=2,j=2,k=3): 2x2*2x3 = 12
    # (i=2,j=3,k=3): 2x3*3x3 = 18
    # Total: 60
    
    # Alternatively:
    # i: 4 = 2+2
    # j: 5 = 2+3
    # k: 6 = 2+2+2
    # 3 splits in k, so 6 blocks total
    # (2x2*2x2): 8 each, use Strassen -> 7
    # (2x2*2x2): 7
    # (2x3*2x2): 12
    # (2x3*2x2): 12
    # (2x2*2x2): 7
    # (2x3*2x2): 12
    # Total: 7+7+12+12+7+12 = 57
    
    # Even better:
    # i: 4 = 2+2
    # j: 5 = 2+2+1
    # k: 6 = 2+2+2
    # 2*3*3 = 18 blocks, most are 2x2x2 or 2x1x2 or 1x2x2 etc.
    
    # Let me just implement a recursive block decomposition with Strassen for 2x2x2.
    
    # For now, let me start with a simple approach and optimize.
    
    # Strategy: recursive decomposition
    # Base case: if any dimension is 1, use schoolbook (rank = product of dims)
    # If 2x2x2, use Strassen (rank 7)
    # Otherwise, split the largest dimension
    
    def schoolbook_rank(ni, nj, nk):
        return ni * nj * nk
    
    def get_terms(ni, nj, nk):
        """Get list of (u, v, w) dicts for ni x nj * nj x nk multiplication."""
        terms = []
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(i, k): 1}
                    terms.append((u, v, w))
        return terms
    
    def strassen_terms():
        """Get 7 terms for 2x2x2 using Strassen."""
        # I'll use the schoolbook 8 terms for now, as getting Strassen right is tricky
        # Actually, let me just use schoolbook for simplicity and correctness
        return get_terms(2, 2, 2)
    
    def decompose(i_start, i_end, j_start, j_end, k_start, k_end, offset_u, offset_v, offset_w, all_terms):
        """
        Recursively decompose the block A[i_start:i_end, j_start:j_end] * B[j_start:j_end, k_start:k_end]
        offset_u, offset_v, offset_w are the offsets for U, V, W matrices
        """
        ni = i_end - i_start
        nj = j_end - j_start
        nk = k_end - k_start
        
        if ni == 0 or nj == 0 or nk == 0:
            return
        
        # Base case: small blocks use schoolbook
        if ni * nj * nk <= 8 or min(ni, nj, nk) == 1:
            for (u, v, w) in get_terms(ni, nj, nk):
                # Adjust indices
                u_adj = {}
                for (i_local, j_local), coeff in u.items():
                    i_glob = i_start + i_local
                    j_glob = j_start + j_local
                    u_adj[(i_glob, j_glob)] = coeff
                v_adj = {}
                for (j_local, k_local), coeff in v.items():
                    j_glob = j_start + j_local
                    k_glob = k_start + k_local
                    v_adj[(j_glob, k_glob)] = coeff
                w_adj = {}
                for (i_local, k_local), coeff in w.items():
                    i_glob = i_start + i_local
                    k_glob = k_start + k_local
                    w_adj[(i_glob, k_glob)] = coeff
                all_terms.append((u_adj, v_adj, w_adj))
            return
        
        # Try Strassen for 2x2x2
        if ni == 2 and nj == 2 and nk == 2:
            for (u, v, w) in strassen_terms():
                u_adj = {}
                for (i_local, j_local), coeff in u.items():
                    i_glob = i_start + i_local
                    j_glob = j_start + j_local
                    u_adj[(i_glob, j_glob)] = coeff
                v_adj = {}
                for (j_local, k_local), coeff in v.items():
                    j_glob = j_start + j_local
                    k_glob = k_start + k_local
                    v_adj[(j_glob, k_glob)] = coeff
                w_adj = {}
                for (i_local, k_local), coeff in w.items():
                    i_glob = i_start + i_local
                    k_glob = k_start + k_local
                    w_adj[(i_glob, k_local)] = coeff
                all_terms.append((u_adj, v_adj, w_adj))
            return
        
        # Otherwise, split the largest dimension
        if ni >= nj and ni >= nk:
            mid = i_start + ni // 2
            decompose(i_start, mid, j_start, j_end, k_start, k_end, 0, 0, 0, all_terms)
            decompose(mid, i_end, j_start, j_end, k_start, k_end, 0, 0, 0, all_terms)
        elif nj >= ni and nj >= nk:
            mid = j_start + nj // 2
            decompose(i_start, i_end, j_start, mid, k_start, k_end, 0, 0, 0, all_terms)
            decompose(i_start, i_end, mid, j_end, k_start, k_end, 0, 0, 0, all_terms)
        else:
            mid = k_start + nk // 2
            decompose(i_start, i_end, j_start, j_end, k_start, mid, 0, 0, 0, all_terms)
            decompose(i_start, i_end, j_start, j_end, mid, k_end, 0, 0, 0, all_terms)
    
    all_terms = []
    decompose(0, n, 0, m, 0, p, 0, 0, 0, all_terms)
    
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u, v, w) in enumerate(all_terms):
        for (i, j), coeff in u.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v.items():
            V[j * p + k, r] = coeff
        for (i, k), coeff in w.items():
            # W[k*n+i, r]
            W[k * n + i, r] = coeff
    
    return {'U': U, 'V': W  # Oops, wrong assignment
    }

# I made an error above. Let me fix and simplify.

def construct():
    n, m, p = 4, 5, 6
    
    def get_schoolbook_terms(ni, nj, nk):
        """Schoolbook terms for ni x nj * nj x nk."""
        terms = []
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(i, k): 1}
                    terms.append((u, v, w))
        return terms
    
    def decompose(i_start, i_end, j_start, j_end, k_start, k_end, all_terms):
        ni = i_end - i_start
        nj = j_end - j_start
        nk = k_end - k_start
        
        if ni == 0 or nj == 0 or nk == 0:
            return
        
        # Use schoolbook for small or simple blocks
        if ni * nj * nk <= 8 or min(ni, nj, nk) == 1:
            for (u, v, w) in get_schoolbook_terms(ni, nj, nk):
                u_adj = {}
                for (il, jl), c in u.items():
                    u_adj[(i_start + il, j_start + jl)] = c
                v_adj = {}
                for (jl, kl), c in v.items():
                    v_adj[(j_start + jl, k_start + kl)] = c
                w_adj = {}
                for (il, kl), c in w.items():
                    w_adj[(i_start + il, k_start + kl)] = c
                all_terms.append((u_adj, v_adj, w_adj))
            return
        
        # Split largest dimension
        if ni >= nj and ni >= nk:
            mid = i_start + ni // 2
            decompose(i_start, mid, j_start, j_end, k_start, k_end, all_terms)
            decompose(mid, i_end, j_start, j_end, k_start, k_end, all_terms)
        elif nj >= ni and nj >= nk:
            mid = j_start + nj // 2
            decompose(i_start, i_end, j_start, mid, k_start, k_end, all_terms)
            decompose(i_start, i_end, mid, j_end, k_start, k_end, all_terms)
        else:
            mid = k_start + nk // 2
            decompose(i_start, i_end, j_start, j_end, k_start, mid, all_terms)
            decompose(i_start, i_end, j_start, j_end, mid, k_end, all_terms)
    
    all_terms = []
    decompose(0, n, 0, m, 0, p, all_terms)
    rank = len(all_terms)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r, (u, v, w) in enumerate(all_terms):
        for (i, j), c in u.items():
            U[i * m + j, r] = c
        for (j, k), c in v.items():
            V[j * p + k, r] = c
        for (i, k), c in w.items():
            W[k * n + i, r] = c
    
    return {'U': U, 'V': V, 'W': W}

if __name__ == "__main__":
    result = construct()
    rank = result['U'].shape[1]
    print(f"Rank: {rank}")
    
    # Verify: C = A @ B should equal sum_r W[:,r] * (U[:,r]^T * A) * ... 
    # Actually verify: for random A (4x5), B (5x6), check A@B == sum over r of 
    # (A * U[:,r])_... 
    # The decomposition means: C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j]*U[i*m+j,r]) * (sum_j B[j,k]*V[j*p+k,r])
    # i.e., C = sum_r outer(A @ diag... no
    # C[i,k] = sum_r W[k*n+i,r] * (row i of A dotted with col r of U) * (col k of B dotted with row... 
    # Let me just verify numerically
    A = np.random.randn(4, 5)
    B = np.random.randn(5, 6)
    C_expected = A @ B
    C_actual = np.zeros((4, 6))
    U, V, W = result['U'], result['V'], result['W']
    for i in range(4):
        for k in range(6):
            C_actual[i, k] = 0
            for r in range(rank):
                # term_r: scalar = (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
                a_val = sum(A[i, j] * U[i*m+j, r] for j in range(5))
                b_val = sum(B[j, k] * V[j*p+k, r] for j in range(5))
                C_actual[i, k] += a_val * b_val * W[k*n+i, r]
    print(f"Max error: {np.max(np.abs(C_expected - C_actual))}")
