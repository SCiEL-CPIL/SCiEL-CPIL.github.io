import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin before the 150s kill

    def get_schoolbook():
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}

    def validate_fast(U, V, W):
        rank = len(U[0])
        U_3d = np.array(U, dtype=np.float64).reshape(n, m, rank)
        V_3d = np.array(V, dtype=np.float64).reshape(m, p, rank)
        W_3d = np.array(W, dtype=np.float64).reshape(p, n, rank)
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(T_recon, 1.0, atol=1e-9):
            return False
        return True

    def to_rational_list(mat):
        """Convert a numpy float array to a list of Fractions (or integers)."""
        result = []
        for row in mat:
            r = []
            for val in row:
                # Check if integer
                if abs(val - round(val)) < 1e-9:
                    r.append(int(round(val)))
                else:
                    # Check if half-integer
                    half = round(val * 2) / 2.0
                    if abs(val - half) < 1e-9:
                        if half == int(half):
                            r.append(int(half))
                        else:
                            r.append(Fraction(1, 2))
                    else:
                        # Fallback to fraction if possible, else float (though task requires int/half-int)
                        # For safety in validation, we assume the search produced valid candidates
                        r.append(Fraction(val).limit_denominator(1000000))
            result.append(r)
        return result

    def try_strassen_like_search(target_rank):
        """
        Attempt to find a decomposition by leveraging the structure of small matrix multiplication.
        For 2x3x5, the tensor is small. We use a random linear algebra approach:
        1. Fix U and V randomly with small integer entries.
        2. Solve the linear system for W.
        3. Check if W has integer/half-integer entries.
        
        To improve success rate, we use a "peeling" approach or simply brute force with better randomization.
        Since the space is huge, we try to fix the structure of U and V to be "sparse" or "structured".
        
        A more robust approach for small ranks:
        We know the rank of 2x3x5 is at most 30. 
        Let's try to find rank < 30.
        
        Strategy:
        Iterate over target_rank from 20 to 29.
        For each rank r:
          - Generate random U (6xr) and V (15xr) with entries in {-1, 0, 1}.
          - Form the linear system for W.
            The equation is: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
            This is not a standard linear system in W because W is multiplied by the product U*V.
            However, for a fixed r, the term is (U[i,j,r] * V[j,k,r]) * W[k,i,r].
            Let A_r[i,j,k] = U[i,j,r] * V[j,k,r].
            Then sum_r A_r[i,j,k] W_r[k,i] = 1.
            
            This is a linear system in the variables W_r[k,i].
            There are r * p * n = r * 10 variables.
            There are n * m * p = 30 equations.
            
            If r < 3, we have more equations than variables, so it's likely overdetermined.
            If r = 3, 30 vars, 30 eqs.
            If r > 3, underdetermined.
            
            So for r >= 3, we can solve the least squares or check consistency.
            Actually, we want an EXACT solution.
            We can use Gaussian elimination over the rationals.
            
            Let's build the matrix M where:
            M is 30 x (r*10).
            Each row corresponds to (i,j,k).
            Each column corresponds to (r', k', i').
            M[i,j,k; r',k',i'] = 1 if (k',i') == (k,i) else 0 ... wait.
            
            The equation is:
            sum_{r'} U[i,j,r'] V[j,k,r'] W[k,i,r'] = 1
            
            The variable is W[k,i,r'].
            The coefficient of W[k,i,r'] in equation (i,j,k) is:
            If the variable is W[k,i,r'], it only appears in equations where the indices (i,j,k) match?
            No, W[k,i,r'] appears in the sum for all j, for fixed i,k.
            So for a fixed (i,k,r'), the variable W[k,i,r'] appears in m equations (for j=0..m-1).
            The coefficient in equation (i,j,k) is U[i,j,r'] * V[j,k,r'].
            
            So the linear system is:
            For each (i,k), we have a sub-system for j=0..m-1:
            sum_{r'} (U[i,j,r'] * V[j,k,r']) * W[k,i,r'] = 1
            
            This is a set of linear equations.
            Let's flatten the variables W to a vector of length r*p*n.
            Let's flatten the equations to a vector of length n*m*p.
            
            Matrix A (size 30 x r*10):
            Row index: (i,j,k) -> i*m*p + j*p + k
            Col index: (r', k', i') -> r'*p*n + k'*n + i'
            A[row, col] = U[i, j, r'] * V[j, k, r'] if k==k' and i==i' else 0.
            
            Wait, W[k,i,r'] is the variable.
            In the equation for (i,j,k), the term is U[i,j,r'] V[j,k,r'] W[k,i,r'].
            So for a specific variable W[k', i', r'], it contributes to equation (i,j,k) only if k'=k and i'=i.
            In that case, the coefficient is U[i,j,r'] * V[j,k,r'].
            
            So A is sparse.
            We can solve A x = b where b is all 1s.
            
            If the system is consistent, we have a solution.
            Then we check if the solution x consists of integers or half-integers.
        """
        if time.time() > deadline:
            return None
        
        num_vars = target_rank * p * n
        num_eqs = n * m * p
        
        # We will try a few random U, V
        # To make it faster, we can use numpy for the linear solve if possible,
        # but we need exact rational arithmetic to ensure integer/half-integer.
        # Given the small size, we can use Fraction-based Gaussian elimination.
        
        # Optimization: Precompute the structure of A.
        # The non-zero pattern of A depends on U and V.
        
        max_attempts = 5000 if target_rank <= 25 else 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Generate random U, V with small integers
            # Using {-1, 0, 1} helps keep numbers small
            U_rand = np.random.randint(-1, 2, size=(n * m, target_rank))
            V_rand = np.random.randint(-1, 2, size=(m * p, target_rank))
            
            # Build the linear system matrix A (dense, but small)
            # A is num_eqs x num_vars
            # Using Fraction for exactness
            A = [[Fraction(0) for _ in range(num_vars)] for _ in range(num_eqs)]
            b = [Fraction(1) for _ in range(num_eqs)]
            
            # Fill A
            # Row (i,j,k), Col (r', k', i')
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row_idx = i * m * p + j * p + k
                        for r_prime in range(target_rank):
                            # The variable W[k, i, r_prime] is at col_idx
                            col_idx = r_prime * p * n + k * n + i
                            coeff = Fraction(U_rand[i * m + j, r_prime]) * Fraction(V_rand[j * p + k, r_prime])
                            A[row_idx][col_idx] += coeff
            
            # Solve A x = b using Gaussian Elimination
            # Since num_vars can be up to 29*10=290, and num_eqs=30.
            # This is an underdetermined system (usually).
            # We need to find IF there is a solution with integer/half-integer entries.
            # This is a hard problem (Diophantine).
            
            # Heuristic: 
            # 1. Find the general solution space.
            # 2. Check if there exists a vector in the affine space that has integer/half-integer coords.
            
            # Given the complexity, let's try a different tactic:
            # Fix W randomly too, and check if the residual is zero?
            # No, that's just random search.
            
            # Let's try to fix U and V, and solve for W using Least Squares, 
            # then check if the LS solution is close to integer/half-integer.
            # If the true rank is r, there exists an exact solution.
            # If we pick U,V such that an exact solution exists, the LS solution will be exact.
            
            # Convert A to numpy for LS
            A_np = np.array(A, dtype=np.float64)
            b_np = np.array(b, dtype=np.float64)
            
            # Solve least squares
            # If rank(A) < num_vars, the solution is not unique.
            # We want a solution with small entries.
            # Use pinv
            try:
                x_ls = np.linalg.lstsq(A_np, b_np, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            
            # Check if x_ls is close to integer or half-integer
            valid = True
            x_rational = []
            for val in x_ls:
                nearest_int = round(val)
                if abs(val - nearest_int) < 1e-9:
                    x_rational.append(Fraction(int(nearest_int)))
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < 1e-9:
                        if nearest_half == int(nearest_half):
                            x_rational.append(Fraction(int(nearest_half)))
                        else:
                            x_rational.append(Fraction(1, 2))
                    else:
                        valid = False
                        break
            
            if valid:
                # Verify exactness
                # Reconstruct W from x_rational
                W_full = [[0 for _ in range(p * n)] for _ in range(target_rank)]
                for r_prime in range(target_rank):
                    for k in range(p):
                        for i in range(n):
                            col_idx = r_prime * p * n + k * n + i
                            W_full[r_prime][k * n + i] = x_rational[col_idx]
                
                # Reshape U, V, W to (rows, rank)
                U_mat = [[0 for _ in range(target_rank)] for _ in range(n * m)]
                V_mat = [[0 for _ in range(target_rank)] for _ in range(m * p)]
                
                for i in range(n):
                    for j in range(m):
                        for r_prime in range(target_rank):
                            U_mat[i * m + j][r_prime] = int(U_rand[i * m + j, r_prime])
                
                for j in range(m):
                    for k in range(p):
                        for r_prime in range(target_rank):
                            V_mat[j * p + k][r_prime] = int(V_rand[j * p + k, r_prime])
                
                # Final validation
                if validate_fast(U_mat, V_mat, W_full):
                    return {
                        "U": U_mat,
                        "V": V_mat,
                        "W": W_full
                    }
        
        return None

    # Try ranks from 20 to 29
    # The schoolbook rank is 30.
    # We want to find rank < 30.
    for target_rank in range(20, 30):
        if time.time() > deadline:
            break
        
        result = try_strassen_like_search(target_rank)
        if result is not None:
            return result

    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
