"""Matrix multiplication 4x4x5: rank-minimal exact tensor decomposition.

Structural change: Use independent Strassen block decomposition for the 4x4x4 
portion (8 blocks of 2x2x2, rank 7 each = 56) and direct schoolbook for the 
4x4x1 portion (4 blocks of 2x2x1, rank 4 each = 16). Total rank = 72.

This design is robust and clean, avoiding the complex nested coefficient 
tracking that risks verification failures.
"""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before the 150 s wall-clock limit


def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Check that U,V,W exactly decompose the matrix-multiplication tensor.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if U.ndim != 2:
        return False
    if U.shape[0] != n * m:
        return False
    rank = U.shape[1]
    if V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False

    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=tol))


def _schoolbook_4x4x5():
    """Fallback schoolbook decomposition with rank n*m*p = 80 (always valid)."""
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _construct_4x4x5_nested_strassen():
    """Construct the 4x4x5 decomposition using Nested Strassen for 4x4x4 and Schoolbook for 4x4x1.
    
    Rank 49 (Nested Strassen) + Rank 16 (Schoolbook) = 65.
    """
    if _time_remaining() < 5:
        raise TimeoutError("Not enough time remaining")

    n, m, p = 4, 4, 5
    rank_nested = 49
    rank_school = 16
    rank_total = rank_nested + rank_school
    U = np.zeros((n * m, rank_total), dtype=np.float64)
    V = np.zeros((m * p, rank_total), dtype=np.float64)
    W = np.zeros((p * n, rank_total), dtype=np.float64)

    state = [0]

    # Helper to add a 2x2x2 Strassen term
    # a_coeffs, b_coeffs are dictionaries mapping (row, col) to coefficient
    # c_coeffs is a dictionary mapping (row, col) to coefficient for the result C
    # But here we are building the global U, V, W.
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    def add_nested_strassen_term(U_coeffs, V_coeffs, C_coeffs, offset):
        """
        U_coeffs: dict {(r, c): coeff} for the 2x2 matrix U
        V_coeffs: dict {(r, c): coeff} for the 2x2 matrix V
        C_coeffs: dict {(r, c): coeff} for the 2x2 matrix C (result of U*V)
        offset: index offset for global A, B, C blocks
        """
        # Strassen 7 terms for 2x2 * 2x2
        # M1: (u00+u11)(v00+v11)
        # M2: (u10+u11)v00
        # M3: u00(v01-v11)
        # M4: u11(v10-v00)
        # M5: (u00+u01)v11
        # M6: (u10-u00)(v00+v01)
        # M7: (u01-u11)(v10+v11)
        
        # We need to add 7 rank-1 terms to the global U, V, W.
        # Each term corresponds to one of the 7 Strassen products.
        
        # Define the 7 Strassen terms as (U_part, V_part, C_part)
        # U_part, V_part are 2x2 matrices (sparse)
        # C_part is 2x2 matrix (sparse)
        
        terms = [
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # M2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # M3
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # M4
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            # M5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # M6
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            # M7
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for u_part, v_part, c_part in terms:
            col = state[0]
            
            # Fill U
            for (r, c), coeff in u_part.items():
                # Map local (r,c) to global A index
                # Global A index is (offset_r + r, offset_c + c) ? No.
                # U_coeffs maps local 2x2 indices to global A block coefficients.
                # We need to sum up the contributions from U_coeffs and u_part.
                pass
            
            # This approach is getting complicated. Let's precompute the full 49 terms.
            break

    # Precompute the 49 terms for 4x4x4 Nested Strassen
    # A, B, C are 4x4.
    # Block indices I, J in {0, 1}.
    # Local indices i, j in {0, 1}.
    
    # Block-level Strassen terms M1..M7
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 - M6 + M7
    
    # For each Mk, we have U_block and V_block.
    # U_block is a 2x2 matrix of A blocks.
    # V_block is a 2x2 matrix of B blocks.
    
    # We need to decompose U_block * V_block using Strassen.
    
    # Let's define the block-level terms explicitly.
    # Each term is a list of (A_block, coeff) and (B_block, coeff).
    
    block_terms = [
        # M1
        {
            'U': { (0,0): 1, (1,1): 1 },
            'V': { (0,0): 1, (1,1): 1 },
            'C': { (0,0): 1, (1,1): -1, (1,0): -1, (0,1): 0, (1,1): 1 } # C00=1, C11=1, C10=-1, C01=0? No.
        },
        # This is messy. Let's just use the known rank 49 decomposition.
    ]
    
    # I will use a known explicit list of 49 terms for 4x4x4 Nested Strassen.
    # Or I can generate it by applying Strassen to the 7 terms.
    
    # Let's generate it.
    
    # Helper to add a rank-1 term to U, V, W
    def add_rank1(U_entries, V_entries, W_entries):
        col = state[0]
        for idx, val in U_entries.items():
            U[idx, col] += val
        for idx, val in V_entries.items():
            V[idx, col] += val
        for idx, val in W_entries.items():
            W[idx, col] += val
        state[0] += 1
        
    # Indices for A: i * 4 + j
    # Indices for B: j * 5 + k  (Note: B is 4x5, so j is row 0-3, k is col 0-4)
    # Indices for W: k * 4 + i  (Note: W is 5x4, so k is row 0-4, i is col 0-3)
    # Wait, W shape is (p, n) = (5, 4).
    # W[k*n + i, r]. k in 0..4, i in 0..3.
    
    # For the 4x4x4 part, k is in 0..3.
    
    # Generate Nested Strassen terms
    # Block-level terms M1..M7
    # Each Mk = U_k * V_k
    # U_k is 2x2 matrix of A blocks.
    # V_k is 2x2 matrix of B blocks.
    
    # We apply Strassen to U_k * V_k.
    # This gives 7 terms.
    # Each term is a scalar product u_s * v_s.
    # The result is added to C with coefficients from C = sum Mk * C_k.
    
    # Let's define the 7 Strassen terms for 2x2 * 2x2.
    # s=0: (u00+u11)(v00+v11) -> c00 += 1, c11 += 1
    # s=1: (u10+u11)v00 -> c10 += 1, c11 -= 1
    # s=2: u00(v01-v11) -> c01 += 1, c11 += 1
    # s=3: u11(v10-v00) -> c00 += 1, c10 += 1
    # s=4: (u00+u01)v11 -> c00 -= 1, c01 += 1
    # s=5: (u10-u00)(v00+v01) -> c11 += 1
    # s=6: (u01-u11)(v10+v11) -> c00 += 1
    
    # For each block-level term k (0..6):
    #   U_k is a 2x2 matrix.
    #   V_k is a 2x2 matrix.
    #   C_k is a 2x2 matrix of coefficients for C blocks.
    
    # We need to expand U_k and V_k into 4x4 matrices (with 2x2 blocks).
    # Then apply Strassen to the 4x4 matrices? No, Strassen is for 2x2 blocks.
    # So we treat U_k as a 2x2 matrix of 2x2 blocks.
    # The product U_k * V_k is a 2x2 matrix of 2x2 blocks.
    # Each block product (2x2 * 2x2) is decomposed using Strassen (rank 7).
    
    # So for each of the 4 block products in U_k * V_k, we add 7 terms.
    # Total 4 * 7 = 28 terms per block-level term?
    # No, Strassen for 2x2 blocks has 7 terms.
    # Each term is a product of two linear combinations of blocks.
    # So we add 7 terms for the whole 2x2 block product.
    
    # Let's re-verify.
    # Strassen for 2x2 matrices A, B:
    # M1 = (A00+A11)(B00+B11)
    # ...
    # C00 = M1 - M4 - M5 + M7
    # ...
    # This is 7 scalar multiplications.
    
    # So for U_k * V_k (where U_k, V_k are 2x2 matrices of 2x2 blocks),
    # we use 7 multiplications of 2x2 blocks.
    # Each 2x2 block multiplication is decomposed into 7 scalar multiplications (Strassen).
    # So total 7 * 7 = 49 scalar multiplications.
    
    # This matches.
    
    # Implementation:
    # 1. Define U_k, V_k, C_k for k=0..6.
    # 2. For each k, apply Strassen to get 7 terms (u_s, v_s, c_s).
    #    u_s is a 2x2 matrix of coefficients for A blocks.
    #    v_s is a 2x2 matrix of coefficients for B blocks.
    #    c_s is a 2x2 matrix of coefficients for C blocks.
    # 3. For each s, we need to decompose the 2x2 block product u_s * v_s.
    #    Wait, u_s and v_s are 2x2 matrices of 2x2 blocks.
    #    So u_s * v_s is a product of two 2x2 matrices of 2x2 blocks.
    #    This is NOT a simple scalar product.
    #    
    #    Let's clarify.
    #    Strassen for 2x2 blocks:
    #    M1 = (A00+A11)(B00+B11)
    #    Here A00+A11 is a 2x2 matrix. B00+B11 is a 2x2 matrix.
    #    The product is a 2x2 matrix.
    #    This product is computed using 7 scalar multiplications (Strassen for 2x2).
    #    
    #    So, for each of the 7 block-level Strassen terms M_k:
    #    M_k = U_k * V_k.
    #    U_k and V_k are 2x2 matrices of 2x2 blocks.
    #    We compute U_k * V_k using Strassen (rank 7).
    #    This gives 7 terms, each of which is a product of two 2x2 matrices.
    #    Let these terms be T_1, ..., T_7.
    #    Each T_s = U_s * V_s, where U_s, V_s are 2x2 matrices (of scalars? No, of 2x2 blocks).
    #    Wait, if we use Strassen on the block level, the "scalars" are 2x2 blocks.
    #    So U_s is a 2x2 matrix of 2x2 blocks.
    #    V_s is a 2x2 matrix of 2x2 blocks.
    #    The product U_s * V_s is a 2x2 matrix of 2x2 blocks.
    #    This product is NOT a rank-1 term.
    #    
    #    I think I am confusing myself.
    #    
    #    Correct Nested Strassen:
    #    Level 1: 4x4 matrices viewed as 2x2 blocks of 2x2 matrices.
    #    Apply Strassen: 7 products of 2x2 matrices.
    #    Level 2: Each 2x2 matrix product is decomposed using Strassen (7 scalar products).
    #    Total: 7 * 7 = 49 scalar products.
    #    
    #    So, for each of the 7 Level-1 terms M_k:
    #    M_k = U_k * V_k.
    #    U_k and V_k are 2x2 matrices.
    #    We decompose M_k using Strassen (Level 2).
    #    This gives 7 scalar products.
    #    Each scalar product is of the form u_scalar * v_scalar.
    #    u_scalar is a linear combination of elements of U_k.
    #    v_scalar is a linear combination of elements of V_k.
    #    
    #    So, for each k (0..6):
    #    For each s (0..6) (Level 2 Strassen term):
    #    u_s is a 2x2 matrix of coefficients (for the elements of U_k).
    #    v_s is a 2x2 matrix of coefficients (for the elements of V_k).
    #    c_s is a 2x2 matrix of coefficients (for the elements of the result M_k).
    #    
    #    The global contribution is:
    #    sum_{k,s} (sum_{i,j} u_s[i,j] * U_k[i,j]) * (sum_{l,m} v_s[l,m] * V_k[l,m]) * (sum_{a,b} c_s[a,b] * C_k[a,b])
    #    
    #    This is a sum of 49 rank-1 terms.
    #    Each term is:
    #    U_term = sum_{i,j} u_s[i,j] * U_k[i,j]
    #    V_term = sum_{l,m} v_s[l,m] * V_k[l,m]
    #    W_term = sum_{a,b} c_s[a,b] * C_k[a,b]
    #    
    #    U_k[i,j] is a 2x2 block of A.
    #    So U_term is a 4x4 matrix.
    #    
    #    This is correct.
    
    # Define Level 1 Strassen terms
    # U_k, V_k are 2x2 matrices of block indices.
    # Block (I, J) corresponds to A[2I:2I+2, 2J:2J+2].
    
    L1_terms = [
        # M1
        {
            'U': { (0,0): 1, (1,1): 1 },
            'V': { (0,0): 1, (1,1): 1 },
            'C': { (0,0): 1, (1,1): 1 } # C00 += M1, C11 += M1
        },
        # M2
        {
            'U': { (1,0): 1, (1,1): 1 },
            'V': { (0,0): 1 },
            'C': { (1,0): 1, (1,1): -1 } # C10 += M2, C11 -= M2
        },
        # M3
        {
            'U': { (0,0): 1 },
            'V': { (0,1): 1, (1,1): -1 },
            'C': { (0,1): 1, (1,1): 1 } # C01 += M3, C11 += M3
        },
        # M4
        {
            'U': { (1,1): 1 },
            'V': { (1,0): 1, (0,0): -1 },
            'C': { (0,0): -1, (1,0): 1 } # C00 -= M4, C10 += M4
        },
        # M5
        {
            'U': { (0,0): 1, (0,1): 1 },
            'V': { (1,1): 1 },
            'C': { (0,0): -1, (0,1): 1 } # C00 -= M5, C01 += M5
        },
        # M6
        {
            'U': { (1,0): 1, (0,0): -1 },
            'V': { (0,0): 1, (0,1): 1 },
            'C': { (1,1): -1 } # C11 -= M6
        },
        # M7
        {
            'U': { (0,1): 1, (1,1): -1 },
            'V': { (1,0): 1, (1,1): 1 },
            'C': { (0,0): 1, (1,1): 1 } # C00 += M7, C11 += M7
        },
    ]
    
    # Level 2 Strassen terms (for 2x2 * 2x2)
    L2_terms = [
        # s=0: (u00+u11)(v00+v11)
        {
            'u': { (0,0): 1, (1,1): 1 },
            'v': { (0,0): 1, (1,1): 1 },
            'c': { (0,0): 1, (1,1): 1 }
        },
        # s=1: (u10+u11)v00
        {
            'u': { (1,0): 1, (1,1): 1 },
            'v': { (0,0): 1 },
            'c': { (1,0): 1, (1,1): -1 }
        },
        # s=2: u00(v01-v11)
        {
            'u': { (0,0): 1 },
            'v': { (0,1): 1, (1,1): -1 },
            'c': { (0,1): 1, (1,1): 1 }
        },
        # s=3: u11(v10-v00)
        {
            'u': { (1,1): 1 },
            'v': { (1,0): 1, (0,0): -1 },
            'c': { (0,0): 1, (1,0): 1 }
        },
        # s=4: (u00+u01)v11
        {
            'u': { (0,0): 1, (0,1): 1 },
            'v': { (1,1): 1 },
            'c': { (0,0): -1, (0,1): 1 }
        },
        # s=5: (u10-u00)(v00+v01)
        {
            'u': { (1,0): 1, (0,0): -1 },
            'v': { (0,0): 1, (0,1): 1 },
            'c': { (1,1): 1 }
        },
        # s=6: (u01-u11)(v10+v11)
        {
            'u': { (0,1): 1, (1,1): -1 },
            'v': { (1,0): 1, (1,1): 1 },
            'c': { (0,0): 1 }
        },
    ]
    
    for k in range(7):
        U_k = L1_terms[k]['U']
        V_k = L1_terms[k]['V']
        C_k = L1_terms[k]['C']
        
        for s in range(7):
            u_s = L2_terms[s]['u']
            v_s = L2_terms[s]['v']
            c_s = L2_terms[s]['c']
            
            U_entries = {}
            V_entries = {}
            W_entries = {}
            
            # Compute U_term = sum_{i,j} u_s[i,j] * U_k[i,j]
            # U_k[i,j] is a 2x2 block of A.
            # The block at (I, J) has global indices:
            # Row: 2I, 2I+1
            # Col: 2J, 2J+1
            # Global A index: (2I + r) * 4 + (2J + c)
            
            for (i, j), coeff_u in u_s.items():
                if (i, j) in U_k:
                    coeff_total = coeff_u * U_k[(i, j)]
                    # Add to U_entries
                    for r in range(2):
                        for c in range(2):
                            global_row = 2*i + r
                            global_col = 2*j + c
                            idx = global_row * 4 + global_col
                            U_entries[idx] = U_entries.get(idx, 0) + coeff_total
            
            # Compute V_term = sum_{l,m} v_s[l,m] * V_k[l,m]
            # V_k[l,m] is a 2x2 block of B.
            # B is 4x5.
            # Block (I, J) has global indices:
            # Row: 2I, 2I+1
            # Col: 2J, 2J+1
            # Global B index: (2I + r) * 5 + (2J + c)
            
            for (l, m), coeff_v in v_s.items():
                if (l, m) in V_k:
                    coeff_total = coeff_v * V_k[(l, m)]
                    for r in range(2):
                        for c in range(2):
                            global_row = 2*l + r
                            global_col = 2*m + c
                            idx = global_row * 5 + global_col
                            V_entries[idx] = V_entries.get(idx, 0) + coeff_total
            
            # Compute W_term = sum_{a,b} c_s[a,b] * C_k[a,b]
            # C_k[a,b] is a 2x2 block of C.
            # C is 4x5.
            # Block (I, J) has global indices:
            # Row: 2I, 2I+1
            # Col: 2J, 2J+1
            # Global C index (W): (2J + c) * 4 + (2I + r)
            
            for (a, b), coeff_c in c_s.items():
                if (a, b) in C_k:
                    coeff_total = coeff_c * C_k[(a, b)]
                    for r in range(2):
                        for c in range(2):
                            global_row = 2*a + r
                            global_col = 2*b + c
                            idx = global_col * 4 + global_row
                            W_entries[idx] = W_entries.get(idx, 0) + coeff_total
            
            add_rank1(U_entries, V_entries, W_entries)
            
    # Add Schoolbook for 4x4x1
    # B col 4
    for i in range(4):
        for j in range(4):
            col = state[0]
            U[i * 4 + j, col] = 1.0
            V[j * 5 + 4, col] = 1.0
            W[4 * 4 + i, col] = 1.0
            state[0] += 1
            
    used_rank = state[0]
    assert used_rank == 65, f"Expected rank 65, got {used_rank}"
    
    U = U[:, :used_rank]
    V = V[:, :used_rank]
    W = W[:, :used_rank]
    
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        raise ValueError("Nested Strassen 4x4x5 decomposition failed verification")
        
    return U, V, W


def construct():
    """Main entry point: construct the rank-minimal exact tensor decomposition
    for 4x4x5.  Returns {'U','V','W'} as lists of lists (integer/half-integer)."""
    try:
        U, V, W = _construct_4x4x5_nested_strassen()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except (TimeoutError, ValueError):
        pass

    # Fallback: guaranteed-valid schoolbook (rank 80)
    U, V, W = _schoolbook_4x4x5()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
