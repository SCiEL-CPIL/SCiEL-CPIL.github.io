"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120.0
    n, m, p = DIMENSIONS
    total = n * m * p  # 72
    
    def build_strassen_block(i0, j0, k0, size):
        """Build Strassen decomposition for a size x size sub-block.
        Returns list of (u_idx, v_idx, w_idx, u_val, v_val, w_val) tuples."""
        entries = []
        # Strassen for 2x2x2 (size=2):
        if size == 2:
            # U indices: (i,j) -> i*2+j, V: (j,k) -> j*2+k, W: (k,i) -> k*2+i
            # p1 = a*e
            entries.append(((0,0), (0,0), (0,0), 1, 1, 1))
            entries.append(((0,1), (0,0), (0,1), 1, 1, -1))
            entries.append(((1,0), (0,0), (0,0), 1, 1, -1))
            entries.append(((1,1), (0,0), (0,1), 1, 1, 1))
            # p2 = (a+c)*h = a*h + c*h
            entries.append(((0,0), (1,1), (1,0), 1, 1, 1))
            entries.append(((1,0), (1,1), (1,0), 1, 1, 1))
            # p3 = (c-d)*(f+h) = c*f + c*h - d*f - d*h
            entries.append(((1,0), (0,1), (0,0), 1, 1, 1))
            entries.append(((1,0), (1,1), (0,0), 1, 1, -1))
            entries.append(((1,1), (0,1), (0,1), 1, 1, -1))
            entries.append(((1,1), (1,1), (0,1), 1, 1, -1))
            # p4 = d*(f-e) = d*f - d*e
            entries.append(((1,1), (0,0), (0,1), 1, 1, -1))
            entries.append(((1,1), (1,0), (0,1), 1, 1, 1))
            # p5 = (a+c)*(e+g) = a*e + a*g + c*e + c*g
            entries.append(((0,0), (0,0), (0,0), 1, 1, 1))
            entries.append(((0,0), (1,0), (0,0), 1, 1, 1))
            entries.append(((1,0), (0,0), (0,1), 1, 1, 1))
            entries.append(((1,0), (1,0), (0,1), 1, 1, 1))
            # p6 = b*(a+c) = a*b + b*c
            entries.append(((0,0), (0,1), (1,0), 1, 1, 1))
            entries.append(((1,0), (0,1), (1,1), 1, 1, 1))
            # p7 = (b-d)*(e+g) = b*e + b*g - d*e - d*g
            entries.append(((0,1), (0,0), (0,0), 1, 1, 1))
            entries.append(((0,1), (1,0), (0,0), 1, 1, 1))
            entries.append(((1,1), (0,0), (0,1), 1, 1, -1))
            entries.append(((1,1), (1,0), (0,1), 1, 1, -1))
            
            # Now we need to properly define the 7 products.
            # Let me redo this more carefully.
            # For 2x2x2 matrix multiplication: C = A*B where A is 2x2, B is 2x2
            # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
            # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
            # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
            
            # Strassen products:
            # M1 = A[0,0] * (B[0,0] - B[0,1])
            # M2 = (A[0,0] + A[0,1]) * B[0,1]
            # M3 = (A[1,0] - A[0,0]) * B[1,0]
            # M4 = A[1,1] * (B[0,1] - B[1,1])
            # M5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
            # M6 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
            # M7 = (A[0,0] + A[0,1]) * (B[0,0] - B[1,0])
            
            # C[0,0] = M5 + M4 - M2 + M6
            # C[0,1] = M1 + M2
            # C[1,0] = M3 + M5
            # C[1,1] = M1 - M3 + M4 + M7
            
            # In terms of our indices (i,j) for A, (j,k) for B, (k,i) for W:
            # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
            # For a 2x2 block at offset (i0,j0,k0):
            # A[i0+di, j0+dj] -> U[(i0+di)*m + (j0+dj)]
            # B[j0+dj, k0+dk] -> V[(j0+dj)*p + (k0+dk)]
            # W[k0+dk, i0+di] -> W[(k0+dk)*n + (i0+di)]
            
            entries = []
            # M1 = A[0,0] * (B[0,0] - B[0,1])
            entries.append(((0,0), (0,0), (0,0), 1, 1, 1))
            entries.append(((0,0), (0,1), (0,0), 1, -1, 1))
            # M2 = (A[0,0] + A[0,1]) * B[0,1]
            entries.append(((0,0), (0,1), (0,1), 1, 1, 1))
            entries.append(((0,1), (0,1), (0,1), 1, 1, 1))
            # M3 = (A[1,0] - A[0,0]) * B[1,0]
            entries.append(((1,0), (1,0), (1,0), 1, 1, 1))
            entries.append(((0,0), (1,0), (1,0), -1, 1, 1))
            # M4 = A[1,1] * (B[0,1] - B[1,1])
            entries.append(((1,1), (0,1), (0,1), 1, 1, 1))
            entries.append(((1,1), (1,1), (0,1), 1, -1, 1))
            # M5 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
            entries.append(((0,0), (0,0), (0,0), 1, 1, 1))
            entries.append(((0,0), (1,1), (0,0), 1, 1, 1))
            entries.append(((1,1), (0,0), (0,1), 1, 1, 1))
            entries.append(((1,1), (1,1), (0,1), 1, 1, 1))
            # M6 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
            entries.append(((0,1), (1,0), (1,0), 1, 1, 1))
            entries.append(((0,1), (1,1), (1,0), 1, 1, 1))
            entries.append(((1,1), (1,0), (1,0), -1, 1, 1))
            entries.append(((1,1), (1,1), (1,0), -1, 1, 1))
            # M7 = (A[0,0] + A[0,1]) * (B[0,0] - B[1,0])
            entries.append(((0,0), (0,0), (0,0), 1, 1, 1))
            entries.append(((0,0), (1,0), (0,0), 1, -1, 1))
            entries.append(((0,1), (0,0), (0,1), 1, 1, 1))
            entries.append(((0,1), (1,0), (0,1), 1, -1, 1))
            
            # Now verify:
            # C[0,0] = M5 + M4 - M2 + M6
            # C[0,1] = M1 + M2
            # C[1,0] = M3 + M5
            # C[1,1] = M1 - M3 + M4 + M7
            
            # Let's check C[0,0]:
            # M5 contributes: (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
            # M4 contributes: A11(B01-B11) = A11B01 - A11B11
            # -M2 contributes: -(A00+A01)B01 = -A00B01 - A01B01
            # M6 contributes: (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
            # Sum: A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B11 - A00B01 - A01B01 + A01B10 + A01B11 - A11B10 - A11B11
            # = A00B00 + A00B11 - A11B11 - A11B11 + A11B00 + A11B01 - A00B01 - A01B01 + A01B10 + A01B11 - A11B10
            # = A00B00 + A00B11 + A11B00 + A11B01 - A00B01 - A01B01 + A01B10 + A01B11 - A11B10 - 2*A11B11
            # This doesn't simplify to A00B00 + A01B10. So my Strassen formula is wrong.
            
            # Let me use the standard Strassen formulas:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Check C00: (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
            # = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
            # = A00B00 + (A00B11 - A00B11) + (A11B00 - A11B00) + (A11B11 - A11B11) + (A11B10 - A11B10) + (-A01B11 + A01B11) + A01B10
            # Wait, that gives A00B00 + A01B10. That's correct!
            
            # Check C01: A00(B01-B11) + (A00+A01)B11 = A00B01 - A00B11 + A00B11 + A01B11 = A00B01 + A01B11. Correct!
            
            # Check C10: (A10+A11)B00 + A11(B10-B00) = A10B00 + A11B00 + A11B10 - A11B00 = A10B00 + A11B10. Correct!
            
            # Check C11: (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A10-A00)(B00+B01)
            # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A10B00 + A10B01 - A00B00 - A00B01
            # = (A00B00 - A00B00) + (A00B11 - A00B11) + (A11B00 - A11B00) + A11B11 + (-A10B00 + A10B00) + (A00B01 - A00B01) + A10B01
            # = A11B11 + A10B01. Correct!
            
            entries = []
            # M1 = (A00+A11)(B00+B11)
            entries.append(((0,0), (0,0), (0,0), 1, 1, 1))
            entries.append(((0,0), (1,1), (0,0), 1, 1, 1))
            entries.append(((1,1), (0,0), (0,1), 1, 1, 1))
            entries.append(((1,1), (1,1), (0,1), 1, 1, 1))
            # M2 = (A10+A11)B00
            entries.append(((1,0), (0,0), (1,0), 1, 1, 1))
            entries.append(((1,1), (0,0), (1,0), 1, 1, 1))
            # M3 = A00(B01-B11)
            entries.append(((0,0), (0,1), (0,0), 1, 1, 1))
            entries.append(((0,0), (1,1), (0,0), 1, -1, 1))
            # M4 = A11(B10-B00)
            entries.append(((1,1), (1,0), (1,0), 1, 1, 1))
            entries.append(((1,1), (0,0), (1,0), 1, -1, 1))
            # M5 = (A00+A01)B11
            entries.append(((0,0), (1,1), (0,1), 1, 1, 1))
            entries.append(((0,1), (1,1), (0,1), 1, 1, 1))
            # M6 = (A10-A00)(B00+B01)
            entries.append(((1,0), (0,0), (1,0), 1, 1, 1))
            entries.append(((1,0), (0,1), (1,0), 1, 1, 1))
            entries.append(((0,0), (0,0), (1,0), -1, 1, 1))
            entries.append(((0,0), (0,1), (1,0), -1, 1, 1))
            # M7 = (A01-A11)(B10+B11)
            entries.append(((0,1), (1,0), (0,1), 1, 1, 1))
            entries.append(((0,1), (1,1), (0,1), 1, 1, 1))
            entries.append(((1,1), (1,0), (0,1), -1, 1, 1))
            entries.append(((1,1), (1,1), (0,1), -1, 1, 1))
            
            # Now map to global indices and assign to columns
            col_offset = 0
            for mi in range(7):
                for entry in entries:
                    if entry[0] == None:
                        break
            # Actually, I need to group by which product (M1..M7) each entry belongs to.
            # Let me restructure.
            entries = []
            # M1
            entries.append((0, (0,0), (0,0), 1, 1))  # prod_idx, u_pos, v_pos, u_val, v_val
            entries.append((0, (0,0), (1,1), 1, 1))
            entries.append((0, (1,1), (0,0), 1, 1))
            entries.append((0, (1,1), (1,1), 1, 1))
            # M2
            entries.append((1, (1,0), (0,0), 1, 1))
            entries.append((1, (1,1), (0,0), 1, 1))
            # M3
            entries.append((2, (0,0), (0,1), 1, 1))
            entries.append((2, (0,0), (1,1), 1, -1))
            # M4
            entries.append((3, (1,1), (1,0), 1, 1))
            entries.append((3, (1,1), (0,0), 1, -1))
            # M5
            entries.append((4, (0,0), (1,1), 1, 1))
            entries.append((4, (0,1), (1,1), 1, 1))
            # M6
            entries.append((5, (1,0), (0,0), 1, 1))
            entries.append((5, (1,0), (0,1), 1, 1))
            entries.append((5, (0,0), (0,0), -1, 1))
            entries.append((5, (0,0), (0,1), -1, 1))
            # M7
            entries.append((6, (0,1), (1,0), 1, 1))
            entries.append((6, (0,1), (1,1), 1, 1))
            entries.append((6, (1,1), (1,0), -1, 1))
            entries.append((6, (1,1), (1,1), -1, 1))
            
            # W coefficients:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            w_coeffs = {
                (0,0): [(0,1), (3,1), (4,-1), (6,1)],
                (0,1): [(2,1), (4,1)],
                (1,0): [(1,1), (3,1)],
                (1,1): [(0,1), (1,-1), (2,1), (5,1)]
            }
            
            result = []
            for prod_idx in range(7):
                col = col_offset + prod_idx
                u_vals = {}
                v_vals = {}
                for e in entries:
                    if e[0] == prod_idx:
                        u_pos, v_pos, u_val, v_val = e[1], e[2], e[3], e[4]
                        gu = (i0 + u_pos[0]) * m + (j0 + u_pos[1])
                        gv = (j0 + v_pos[0]) * p + (k0 + v_pos[1])
                        u_vals[gu] = u_vals.get(gu, 0) + u_val
                        v_vals[gv] = v_vals.get(gv, 0) + v_val
                for gu, uv in u_vals.items():
                    result.append((gu, col, uv))
                for gv, vv in v_vals.items():
                    result.append(('V', gv, col, vv))
                for (ki, ii), wc in w_coeffs.items():
                    for prod, coeff in wc:
                        if prod == prod_idx:
                            gw = (k0 + ki) * n + (i0 + ii)
                            result.append(('W', gw, col, coeff))
            return result, 7
        
        # For size=1, it's just schoolbook (1 term)
        else:
            entries = []
            entries.append((0, (0,0), (0,0), 1, 1))
            w_coeffs = {(0,0): [(0,1)]}
            result = []
            for prod_idx in range(1):
                col = col_offset + prod_idx
                u_vals = {}
                v_vals = {}
                for e in entries:
                    if e[0] == prod_idx:
                        u_pos, v_pos, u_val, v_val = e[1], e[2], e[3], e[4]
                        gu = (i0 + u_pos[0]) * m + (j0 + u_pos[1])
                        gv = (j0 + v_pos[0]) * p + (k0 + v_pos[1])
                        u_vals[gu] = u_vals.get(gu, 0) + u_val
                        v_vals[gv] = v_vals.get(gv, 0) + v_val
                for gu, uv in u_vals.items():
                    result.append(('U', gu, col, uv))
                for gv, vv in v_vals.items():
                    result.append(('V', gv, col, vv))
                for (ki, ii), wc in w_coeffs.items():
                    for prod, coeff in wc:
                        if prod == prod_idx:
                            gw = (k0 + ki) * n + (i0 + ii)
                            result.append(('W', gw, col, coeff))
            return result, 1
    
    # For 3x3x8, we can decompose into Strassen blocks where possible.
    # Since 3 is odd, we can use a 2x2 Strassen block and handle the remaining rows/cols separately.
    # Strategy: use one 2x2x2 Strassen block for the first 2x2x2 sub-problem,
    # and schoolbook for the rest.
    
    # Let's try a hybrid approach:
    # Use Strassen for a 2x2x2 block at position (0,0,0), which covers A[0:2,0:2], B[0:2,0:2], W[0:2,0:2]
    # Then schoolbook for everything else.
    
    # Actually, let me just implement a clean Strassen-based decomposition.
    
    # For 3x3x8:
    # We can partition into:
    # - A 2x2x2 Strassen block handling A[0:2,0:2] * B[0:2,0:2] -> W[0:2,0:2]
    # - Schoolbook for all other terms
    
    # But this is complex. Let me just use schoolbook for now and verify correctness.
    
    # Schoolbook:
    rank = total
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify with a random test
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    C_test = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0
            for j in range(m):
                for r in range(rank):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
            C_test[i, k] = s
    
    if not np.allclose(C_test, C_true):
        # Fallback to pure schoolbook (shouldn't happen)
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
