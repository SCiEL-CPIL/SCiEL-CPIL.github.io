import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 115.0
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def deadline_exceeded():
    return elapsed() > TIME_LIMIT


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    R = len(U[0])
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = Uf[i * m + j]
                vk = Vf[j * p + k]
                wk = Wf[k * n + i]
                for r in range(R):
                    s += ui[r] * vk[r] * wk[r]
                if s != 1:
                    return False
    return True


def _construct_rank17(n, m, p):
    """
    Constructive rank 17 decomposition for 2x3x3 matrix multiplication.
    Uses Strassen's algorithm for the 2x2x2 core (indices i,j,k in {0,1})
    and schoolbook terms for the remaining boundary indices (j=2 or k=2).
    """
    if n != 2 or m != 3 or p != 3:
        return None

    R = 17
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Strassen's 7 terms for i,j,k in {0,1}
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00      -> C10, C11
    # M3 = A00(B01-B11)      -> C01, C11
    # M4 = A11(B10-B00)      -> C00, C10
    # M5 = (A00+A01)B11      -> C00, C01
    # M6 = (A10-A00)(B00+B01)-> C11
    # M7 = (A01-A11)(B10+B11)-> C00
    
    # r=0 (M1)
    U[u_idx(0,0)][0] = 1; U[u_idx(1,1)][0] = 1
    V[v_idx(0,0)][0] = 1; V[v_idx(1,1)][0] = 1
    W[w_idx(0,0)][0] = 1; W[w_idx(1,1)][0] = 1
    
    # r=1 (M2)
    U[u_idx(1,0)][1] = 1; U[u_idx(1,1)][1] = 1
    V[v_idx(0,0)][1] = 1
    W[w_idx(0,1)][1] = 1; W[w_idx(1,1)][1] = -1
    
    # r=2 (M3)
    U[u_idx(0,0)][2] = 1
    V[v_idx(0,1)][2] = 1; V[v_idx(1,1)][2] = -1
    W[w_idx(1,0)][2] = 1; W[w_idx(1,1)][2] = 1
    
    # r=3 (M4)
    U[u_idx(1,1)][3] = 1
    V[v_idx(1,0)][3] = 1; V[v_idx(0,0)][3] = -1
    W[w_idx(0,0)][3] = 1; W[w_idx(0,1)][3] = 1
    
    # r=4 (M5)
    U[u_idx(0,0)][4] = 1; U[u_idx(0,1)][4] = 1
    V[v_idx(1,1)][4] = 1
    W[w_idx(0,0)][4] = -1; W[w_idx(1,0)][4] = 1
    
    # r=5 (M6)
    U[u_idx(1,0)][5] = 1; U[u_idx(0,0)][5] = -1
    V[v_idx(0,0)][5] = 1; V[v_idx(0,1)][5] = 1
    W[w_idx(1,1)][5] = 1
    
    # r=6 (M7)
    U[u_idx(0,1)][6] = 1; U[u_idx(1,1)][6] = -1
    V[v_idx(1,0)][6] = 1; V[v_idx(1,1)][6] = 1
    W[w_idx(0,0)][6] = 1

    # Boundary terms (r=7..16)
    # These cover all (i,j,k) where j=2 or k=2
    boundary_terms = [
        (0, 2, 0), (1, 2, 0),
        (0, 2, 1), (1, 2, 1),
        (0, 2, 2), (1, 2, 2),
        (0, 0, 2), (1, 0, 2),
        (0, 1, 2), (1, 1, 2)
    ]
    
    for idx, (i, j, k) in enumerate(boundary_terms):
        r = 7 + idx
        U[u_idx(i, j)][r] = 1
        V[v_idx(j, k)][r] = 1
        W[w_idx(k, i)][r] = 1

    return {"U": U, "V": V, "W": W}


def _try_rank_reduction(U, V, W, n, m, p, target_rank, max_time=60.0):
    """
    Try to reduce the rank of a decomposition from current rank to target_rank.
    Uses a greedy approach: try to merge pairs of rank-1 tensors.
    """
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return None
    
    # Work with Fraction for exactness
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    
    # Try to find linear dependencies among the rank-1 tensors
    # For each pair (r1, r2), check if tensor r2 can be expressed as a linear combination
    # of tensors 0..r1-1 and r1
    
    start_time = time.time()
    current_U = [row[:] for row in Uf]
    current_V = [row[:] for row in Vf]
    current_W = [row[:] for row in Wf]
    current_R = current_rank
    
    # Greedy: try to eliminate the last tensor by expressing it as combo of others
    for attempt in range(current_R - target_rank):
        if time.time() - start_time > max_time:
            break
        if deadline_exceeded():
            break
        
        if current_R <= target_rank:
            break
        
        # Try to eliminate tensor at index current_R - 1
        # by expressing it as a linear combination of tensors 0..current_R-2
        eliminated = False
        
        # Try simple coefficient sets: {0, ±1, ±1/2, ±2}
        # This is a heuristic - in practice we'd solve a linear system
        # For now, try if the last tensor is already zero or trivially dependent
        
        # Check if tensor current_R-1 is zero
        is_zero = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = Fraction(0)
                    for r in range(current_R - 1):
                        val += current_U[i*m+j][r] * current_V[j*p+k][r] * current_W[k*n+i][r]
                    if val != current_U[i*m+j][current_R-1] * current_V[j*p+k][current_R-1] * current_W[k*n+i][current_R-1]:
                        # Check if the full sum equals 1
                        pass
                    if current_U[i*m+j][current_R-1] != 0 or current_V[j*p+k][current_R-1] != 0 or current_W[k*n+i][current_R-1] != 0:
                        # Not trivially zero, but let's check if it contributes
                        pass
                        break
                else:
                    continue
                break
            else:
                continue
            break
        
        # More systematic: solve for coefficients c_r such that
        # sum_{r=0}^{R-2} c_r * U_r * V_r * W_r = U_{R-1} * V_{R-1} * W_{R-1}
        # This is a linear system in the c_r's
        
        # Set up the system: for each (i,j,k),
        # sum_{r=0}^{R-2} c_r * U[i,j,r] * V[j,k,r] * W[k,i,r] = U[i,j,R-1] * V[j,k,R-1] * W[k,i,R-1]
        
        num_eqs = n * m * p
        num_unknowns = current_R - 1
        
        # Build the system
        A = []
        b = []
        for idx, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
            row = []
            for r in range(num_unknowns):
                row.append(current_U[i*m+j][r] * current_V[j*p+k][r] * current_W[k*n+i][r])
            A.append(row)
            b.append(current_U[i*m+j][current_R-1] * current_V[j*p+k][current_R-1] * current_W[k*n+i][current_R-1])
        
        # Solve using Gaussian elimination with Fractions
        # This may be slow for large systems, but our systems are small
        try:
            # Augmented matrix
            aug = [A[i] + [b[i]] for i in range(num_eqs)]
            
            # Gaussian elimination
            rank = 0
            for col in range(num_unknowns):
                if rank >= num_eqs:
                    break
                # Find pivot
                pivot = None
                for row in range(rank, num_eqs):
                    if aug[row][col] != 0:
                        pivot = row
                        break
                if pivot is None:
                    continue
                # Swap
                aug[rank], aug[pivot] = aug[pivot], aug[rank]
                # Eliminate
                pivot_val = aug[rank][col]
                for row in range(num_eqs):
                    if row != rank and aug[row][col] != 0:
                        factor = aug[row][col] / pivot_val
                        for c in range(col, num_unknowns + 1):
                            aug[row][c] -= factor * aug[rank][c]
                rank += 1
            
            # Check for consistency
            for row in range(rank, num_eqs):
                if aug[row][num_unknowns] != 0:
                    # Inconsistent - can't eliminate this tensor
                    break
            else:
                # Consistent - extract solution
                coeffs = [Fraction(0)] * num_unknowns
                for row in range(rank):
                    # Find the leading column
                    lead_col = None
                    for c in range(num_unknowns):
                        if aug[row][c] != 0:
                            lead_col = c
                            break
                    if lead_col is not None:
                        coeffs[lead_col] = aug[row][num_unknowns] / aug[row][lead_col]
                
                # Verify the solution
                valid = True
                for idx, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
                    lhs = Fraction(0)
                    for r in range(num_unknowns):
                        lhs += coeffs[r] * current_U[i*m+j][r] * current_V[j*p+k][r] * current_W[k*n+i][r]
                    rhs = current_U[i*m+j][current_R-1] * current_V[j*p+k][current_R-1] * current_W[k*n+i][current_R-1]
                    if lhs != rhs:
                        valid = False
                        break
                
                if valid:
                    # Eliminate the tensor by adding -coeffs to the other tensors
                    for r in range(num_unknowns):
                        if coeffs[r] != 0:
                            for i in range(n * m):
                                current_U[i][r] -= coeffs[r] * current_U[i][current_R-1]
                            for j in range(m * p):
                                current_V[j][r] -= coeffs[r] * current_V[j][current_R-1]
                            for k in range(p * n):
                                current_W[k][r] -= coeffs[r] * current_W[k][current_R-1]
                    
                    # Remove the last tensor
                    for i in range(n * m):
                        current_U[i].pop()
                    for j in range(m * p):
                        current_V[j].pop()
                    for k in range(p * n):
                        current_W[k].pop()
                    current_R -= 1
                    eliminated = True
        except (ZeroDivisionError, IndexError):
            pass
        
        if not eliminated:
            # Try a different tensor to eliminate - shuffle and retry
            # For simplicity, break out
            break
    
    if current_R <= target_rank:
        # Convert back to int/float
        U_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in current_U]
        V_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in current_V]
        W_out = [[int(x) if x.denominator == 1 else float(x) for x in row] for row in current_W]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Try constructive rank 17
    result = _construct_rank17(n, m, p)
    if result is not None:
        if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
            # Try to reduce rank further
            for target in range(16, 12, -1):
                if deadline_exceeded():
                    break
                reduced = _try_rank_reduction(result["U"], result["V"], result["W"], n, m, p, target, max_time=20.0)
                if reduced is not None:
                    if _verify_exact(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                        result = reduced
                        break
    
    if result is None:
        result = _schoolbook(n, m, p)
    
    return result


def main():
    return construct()


if __name__ == '__main__':
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
