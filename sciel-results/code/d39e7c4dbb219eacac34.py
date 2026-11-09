import time
import numpy as np

_DIM = (6, 6, 6)

def _verify_3x3_decomposition(terms):
    """Verify that the terms correctly decompose 3x3 matrix multiplication."""
    np.random.seed(42)
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    C_expected = A @ B
    
    C_computed = np.zeros((3, 3))
    for term in terms:
        a_val = 0.0
        b_val = 0.0
        for r, c, coeff in term['U']:
            a_val += coeff * A[r, c]
        for r, c, coeff in term['V']:
            b_val += coeff * B[r, c]
        prod = a_val * b_val
        for r, c, coeff in term['W']:
            C_computed[r, c] += coeff * prod
    
    return np.allclose(C_expected, C_computed, atol=1e-10)

def _strassen_3x3_terms():
    """Return 19 terms for 3x3 matrix multiplication."""
    # These terms are likely invalid, so this will fallback to 189.
    terms = []
    terms.append({'U': [(0,0,1),(1,1,1)], 'V': [(0,0,1),(1,1,1)], 'W': [(0,0,1),(1,1,1)]})
    terms.append({'U': [(0,0,1),(1,1,1)], 'V': [(0,1,1),(1,1,1)], 'W': [(0,1,1),(1,1,1)]})
    terms.append({'U': [(0,0,1),(2,2,1)], 'V': [(0,0,1),(2,2,1)], 'W': [(0,0,1),(2,2,1)]})
    terms.append({'U': [(0,0,1),(2,2,1)], 'V': [(0,2,1),(2,2,1)], 'W': [(0,2,1),(2,2,1)]})
    terms.append({'U': [(1,1,1),(2,2,1)], 'V': [(1,1,1),(2,2,1)], 'W': [(1,1,1),(2,2,1)]})
    terms.append({'U': [(1,1,1),(2,2,1)], 'V': [(1,2,1),(2,2,1)], 'W': [(1,2,1),(2,2,1)]})
    terms.append({'U': [(0,1,1),(1,2,1)], 'V': [(0,1,1),(1,2,1)], 'W': [(0,1,1),(1,2,1)]})
    terms.append({'U': [(0,1,1),(1,2,1)], 'V': [(1,2,1),(2,2,1)], 'W': [(0,2,1),(1,2,1)]})
    terms.append({'U': [(0,2,1),(2,0,1)], 'V': [(0,2,1),(2,0,1)], 'W': [(0,2,1),(2,0,1)]})
    terms.append({'U': [(0,2,1),(2,0,1)], 'V': [(2,0,1),(2,2,1)], 'W': [(0,0,1),(2,0,1)]})
    terms.append({'U': [(1,0,1),(2,1,1)], 'V': [(1,0,1),(2,1,1)], 'W': [(1,0,1),(2,1,1)]})
    terms.append({'U': [(1,0,1),(2,1,1)], 'V': [(2,1,1),(2,2,1)], 'W': [(1,1,1),(2,1,1)]})
    terms.append({'U': [(0,1,1),(1,0,1),(2,1,1)], 'V': [(0,1,1),(1,1,1),(2,1,1)], 'W': [(0,1,1),(1,1,1),(2,1,1)]})
    terms.append({'U': [(0,2,1),(1,2,1),(2,1,1)], 'V': [(0,2,1),(1,2,1),(2,1,1)], 'W': [(0,2,1),(1,2,1),(2,1,1)]})
    terms.append({'U': [(1,0,1),(2,0,1),(2,1,1)], 'V': [(1,0,1),(2,0,1),(2,1,1)], 'W': [(1,0,1),(2,0,1),(2,1,1)]})
    terms.append({'U': [(0,1,1),(1,2,1),(2,0,1)], 'V': [(0,1,1),(1,2,1),(2,0,1)], 'W': [(0,1,1),(1,2,1),(2,0,1)]})
    terms.append({'U': [(0,2,1),(1,0,1),(2,1,1)], 'V': [(0,2,1),(1,0,1),(2,1,1)], 'W': [(0,2,1),(1,0,1),(2,1,1)]})
    terms.append({'U': [(1,0,1),(2,0,1),(1,2,1)], 'V': [(1,0,1),(2,0,1),(1,2,1)], 'W': [(1,0,1),(2,0,1),(1,2,1)]})
    terms.append({'U': [(0,1,1),(2,0,1),(1,2,1)], 'V': [(0,1,1),(2,0,1),(1,2,1)], 'W': [(0,1,1),(2,0,1),(1,2,1)]})
    return terms

def construct():
    n, m, p = _DIM
    
    # Try 3x3 block approach first
    B = 3
    N_b = n // B  # = 2
    terms_3x3 = _strassen_3x3_terms()
    
    # Verify the 3x3 decomposition
    if _verify_3x3_decomposition(terms_3x3):
        rank_3x3 = len(terms_3x3)
        rank_total = N_b * N_b * N_b * rank_3x3
        
        U = [[0] * rank_total for _ in range(n * m)]
        V = [[0] * rank_total for _ in range(m * p)]
        W = [[0] * rank_total for _ in range(p * n)]
        
        for i in range(N_b):
            for k in range(N_b):
                for l in range(N_b):
                    for t in range(rank_3x3):
                        col_idx = (i * N_b + k) * (N_b * rank_3x3) + l * rank_3x3 + t
                        term = terms_3x3[t]
                        
                        for lr, lc, coeff in term['U']:
                            u_row = (i * B + lr) * m + (k * B + lc)
                            U[u_row][col_idx] += coeff
                            
                        for lr, lc, coeff in term['V']:
                            v_row = (k * B + lr) * p + (l * B + lc)
                            V[v_row][col_idx] += coeff
                            
                        for lr, lc, coeff in term['W']:
                            w_row = (l * B + lc) * n + (i * B + lr)
                            W[w_row][col_idx] += coeff
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to 2x2 Strassen
        B = 2
        N_b = n // B  # = 3
        rank = N_b * N_b * N_b * 7  # = 189
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_terms = [
            {'U': [(0, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
            {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1)], 'W': [(1, 0, 1), (1, 1, -1)]},
            {'U': [(0, 0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(0, 1, 1), (1, 1, 1)]},
            {'U': [(1, 1, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, 1), (1, 0, 1)]},
            {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(0, 0, -1), (0, 1, 1)]},
            {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(1, 1, 1)]},
            {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]}
        ]
        
        for i in range(N_b):
            for k in range(N_b):
                for l in range(N_b):
                    for t in range(7):
                        col_idx = (i * N_b + k) * (N_b * 7) + l * 7 + t
                        term = strassen_terms[t]
                        
                        for lr, lc, coeff in term['U']:
                            u_row = (i * B + lr) * m + (k * B + lc)
                            U[u_row][col_idx] += coeff
                            
                        for lr, lc, coeff in term['V']:
                            v_row = (k * B + lr) * p + (l * B + lc)
                            V[v_row][col_idx] += coeff
                            
                        for lr, lc, coeff in term['W']:
                            w_row = (l * B + lc) * n + (i * B + lr)
                            W[w_row][col_idx] += coeff
        
        return {"U": U, "V": V, "W": W}
