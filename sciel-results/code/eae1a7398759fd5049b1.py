import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 11
    
    # Initialize U, V, W as NumPy arrays for easier manipulation
    # U: (n*m, rank) = (4, 11)
    # V: (m*p, rank) = (6, 11)
    # W: (p*n, rank) = (6, 11)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to set values in U (index i, j for A[i,j])
    def set_u(i, j, r, val):
        U[i * m + j, r] = val
        
    # Helper to set values in V (index j, k for B[j,k])
    def set_v(j, k, r, val):
        V[j * p + k, r] = val
        
    # Helper to set values in W (index k, i for C[i,k] -> W[k*n+i])
    # Note: Task says W[k*n+i, r]. C is n x p. C[i,k].
    # So index in W is k*n + i.
    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # --- Strassen Part (for B columns 0 and 1) ---
    # This computes A * B[:, 0:2]
    
    # M1 = (A00 + A11) * (B00 + B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    # C00 += M1
    set_w(0, 0, 0, 1)
    # C11 += M1
    set_w(1, 1, 0, 1)

    # M2 = (A10 + A11) * B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    # C10 += M2
    set_w(0, 1, 1, 1)
    # C11 -= M2
    set_w(1, 1, 1, -1)

    # M3 = A00 * (B01 - B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    # C01 += M3
    set_w(1, 0, 2, 1)
    # C11 += M3
    set_w(1, 1, 2, 1)

    # M4 = A11 * (B10 - B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    # C00 += M4
    set_w(0, 0, 3, 1)
    # C10 += M4
    set_w(0, 1, 3, 1)

    # M5 = (A00 + A01) * B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    # C00 -= M5
    set_w(0, 0, 4, -1)
    # C01 += M5
    set_w(1, 0, 4, 1)

    # M6 = (A10 - A00) * (B00 + B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    # C11 += M6
    set_w(1, 1, 5, 1)

    # M7 = (A01 - A11) * (B10 + B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    # C00 += M7
    set_w(0, 0, 6, 1)

    # --- Schoolbook Part (for B column 2) ---
    # This computes A * B[:, 2]
    
    # M8 = A00 * B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    # C02 += M8
    set_w(2, 0, 7, 1)

    # M9 = A01 * B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    # C02 += M9
    set_w(2, 0, 8, 1)

    # M10 = A10 * B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    # C12 += M10
    set_w(2, 1, 9, 1)

    # M11 = A11 * B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    # C12 += M11
    set_w(2, 1, 10, 1)

    # Convert back to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
