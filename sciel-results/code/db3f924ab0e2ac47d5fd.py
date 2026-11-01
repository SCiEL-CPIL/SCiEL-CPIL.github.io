import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 2, 3
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit

    # Dimensions
    nm, mp, pn = n * m, m * p, p * n
    # nm=4, mp=6, pn=6

    # Target Tensor T[i*m+j, j*p+k, k*n+i] = 1
    # U index: a = i*m + j
    # V index: b = j*p + k
    # W index: c = k*n + i

    # 1. Construct the full tensor T as a vector of length nm*mp*pn
    T_flat = np.zeros(nm * mp * pn)
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = a * (mp * pn) + b * pn + c
                T_flat[idx] = 1.0
                indices.append((i, j, k, a, b, c))

    # 2. Generate a set of random rank-1 tensors
    # We need to find a set of R rank-1 tensors whose linear combination equals T.
    # Let R be the rank we are testing.
    # T = sum_{r=1}^R u_r v_r w_r
    # This is a system of linear equations in the coefficients if we fix the vectors,
    # but here we are searching for the vectors themselves.
    
    # Alternative approach: 
    # The space of tensors of size 4x6x6 has dimension 144.
    # The rank-1 tensors form a variety.
    # We can try to find a basis of the tensor space spanned by rank-1 tensors? No, that's just the whole space.
    
    # Let's use the "random linear combination" method.
    # We know the schoolbook rank is 12.
    # We suspect the rank is 10 (based on inspiration 3).
    # If rank is 10, there exist 10 rank-1 tensors u_r, v_r, w_r such that T = sum c_r (u_r v_r w_r).
    # Actually, the definition of rank is the minimal number of rank-1 tensors that sum to T.
    # So T = sum_{r=1}^R u_r v_r w_r.
    
    # Strategy: Randomized search for R rank-1 components.
    # This is generally hard (NP-hard). However, for small dimensions, we might get lucky or use structure.
    
    # Better Strategy: 
    # The current program uses Rank 11 (Strassen 7 + Schoolbook 4).
    # Inspiration 3 found Rank 10.
    # Let's try to find Rank 10 by searching for linear dependencies in a larger random set.
    
    # Method: 
    # 1. Generate many random rank-1 tensors (u, v, w).
    # 2. Check if the target tensor T is in the span of these random rank-1 tensors.
    # 3. If yes, find the coefficients.
    # 4. If the number of non-zero coefficients is <= R, we have a decomposition of rank <= R.
    
    # However, the rank-1 tensors must be *specific* to the problem structure? 
    # No, any rank-1 tensor in the space is valid. 
    # But wait, the entries must be "integer/half-integer".
    # The problem states: "factor matrices ... with integer/half-integer entries".
    # Random floats will not satisfy this.
    
    # CRITICAL CONSTRAINT: Entries must be integers or half-integers.
    # This means we are looking for a decomposition over Q (rationals) that can be scaled to Z or 1/2 Z.
    
    # Let's re-evaluate. 
    # The current solution uses integers.
    # If we find a rational solution, we can scale it.
    
    # Let's try to find a rank-10 decomposition using rational numbers.
    # We can set up a system of equations.
    # Let the rank be R.
    # We have variables:
    # U[a, r], V[b, r], W[c, r] for r in 0..R-1.
    # Equation: sum_{r=0}^{R-1} U[a,r] V[b,r] W[c,r] = T[a,b,c]
    
    # This is a system of bilinear/trilinear equations.
    # For R=10, variables = 10 * (4 + 6 + 6) = 160 variables.
    # Equations = 144.
    # Under-determined system.
    
    # We can fix some variables to break symmetry and scaling.
    # For example, set U[0, r] = 1 for all r? No, that might not be possible.
    # We can fix the first row of U to be standard basis?
    
    # Let's use a numerical solver (scipy.optimize.least_squares) to find a solution.
    # Then check if it can be rounded to integers/half-integers.
    
    def try_rank(R):
        if time.time() > deadline - 10:
            return None
        
        # Variables: U (4xR), V (6xR), W (6xR)
        # Total vars: R * (4+6+6) = 16R
        # For R=10, 160 vars.
        
        # Objective: minimize sum (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
        # Plus a penalty for non-integer/half-integer values?
        # Or just find *any* solution, then try to rationalize.
        
        # To make it tractable, we can fix some parameters.
        # The tensor is invariant under:
        # U -> D1 U, V -> D2 V, W -> D3 W where D1 D2 D3 = I (diagonal matrices)
        # And permutation of indices r.
        
        # Let's fix the first column of U to be 1s? 
        # No, let's fix U[:, 0] to be a specific vector, e.g., [1, 0, 0, 0]^T?
        # This might be too restrictive.
        
        # Let's try a different approach: 
        # Use the known Rank 11 solution and try to "compress" it.
        # The Rank 11 solution is:
        # 7 terms for k=0,1 (Strassen)
        # 4 terms for k=2 (Schoolbook)
        
        # Maybe we can mix the terms?
        # The Strassen part handles the 2x2x2 tensor.
        # The Schoolbook part handles the 2x2x1 tensor.
        # The full tensor is 2x2x3.
        
        # Is it possible to share terms between the k=0,1 block and k=2 block?
        # The k=2 block involves V indices 2, 5 (j=0,k=2; j=1,k=2).
        # The k=0,1 block involves V indices 0,1,3,4.
        # They are disjoint in V indices. So no direct sharing in V.
        # Similarly for W indices:
        # k=0,1 -> W indices 0,1,2,3.
        # k=2 -> W indices 4,5.
        # Disjoint in W.
        # U indices are shared (0,1,2,3).
        
        # So the decomposition is block-diagonal in the V and W spaces?
        # If so, the rank is Rank(T_{k=0,1}) + Rank(T_{k=2}).
        # Rank(T_{k=0,1}) = 7 (Strassen).
        # Rank(T_{k=2}) = 4 (Schoolbook for 2x2x1).
        # Total Rank = 11.
        
        # If this block-diagonal structure is inherent to the tensor, then Rank 10 is impossible?
        # Let's check if T_{k=0,1} and T_{k=2} are truly independent.
        # T[a,b,c] is non-zero only if b and c are consistent with the same k?
        # b = j*p + k. c = k*n + i.
        # If b corresponds to k=0, then c must correspond to k=0.
        # If b corresponds to k=2, then c must correspond to k=2.
        # There is no term where b is from k=0 and c is from k=2.
        # So the tensor is indeed a direct sum of two tensors:
        # T = T_01 + T_2
        # Where T_01 has support on V indices {0,1,3,4} and W indices {0,1,2,3}.
        # And T_2 has support on V indices {2,5} and W indices {4,5}.
        
        # The rank of a direct sum is the sum of the ranks.
        # Rank(T) = Rank(T_01) + Rank(T_2).
        
        # Rank(T_01) is the rank of the 2x2x2 matrix multiplication tensor.
        # It is known to be 7 (Strassen).
        
        # Rank(T_2) is the rank of the 2x2x1 matrix multiplication tensor (multiplying 2x2 by 2x1).
        # This is equivalent to multiplying a 2x2 matrix by a column vector.
        # C[i] = A[i,0]B[0] + A[i,1]B[1].
        # This requires 4 multiplications (A[0,0]B[0], A[0,1]B[1], A[1,0]B[0], A[1,1]B[1]).
        # Can we do it in 3?
        # If we could, we would have Rank(T_2) = 3.
        # Total Rank = 7 + 3 = 10.
        
        # So the question reduces to: What is the rank of 2x2x1 matrix multiplication?
        # Or more generally, what is the rank of multiplying a 2x2 matrix by a 2x1 matrix?
        
        # Let's analyze T_2.
        # U indices: 0,1,2,3 (i,j).
        # V indices: 2,5 (j,k=2). Let's call them v0 (j=0), v1 (j=1).
        # W indices: 4,5 (k=2,i). Let's call them w0 (i=0), w1 (i=1).
        
        # T_2[0, v0, w0] = 1 (A[0,0] B[0,2] -> C[0,2])
        # T_2[1, v1, w0] = 1 (A[0,1] B[1,2] -> C[0,2])
        # T_2[2, v0, w1] = 1 (A[1,0] B[0,2] -> C[1,2])
        # T_2[3, v1, w1] = 1 (A[1,1] B[1,2] -> C[1,2])
        
        # This is exactly the tensor for multiplying a 2x2 matrix by a 2x1 vector.
        # Is the rank 4? Or can it be 3?
        
        # Let's check if rank 3 is possible for 2x2x1.
        # Suppose C = sum_{r=1}^3 u_r v_r w_r.
        # This is a system of 4 equations (for the 4 non-zero entries).
        # Variables: 3 * (4 + 2 + 2) = 24 variables.
        # 4 equations.
        # Likely solvable.
        
        # Let's try to construct a rank-3 decomposition for T_2.
        # We need:
        # 1 = u0_r v0_r w0_r (sum over r)
        # 1 = u1_r v1_r w0_r
        # 1 = u2_r v0_r w1_r
        # 1 = u3_r v1_r w1_r
        # And all other entries must be 0.
        # Specifically:
        # u0_r v1_r w1_r = 0
        # u1_r v0_r w1_r = 0
        # u2_r v1_r w0_r = 0
        # u3_r v0_r w0_r = 0
        
        # Let's try to find such u, v, w.
        # Let's guess a structure.
        # Maybe v0_r and v1_r are related?
        
        # Let's use a random search for this specific sub-problem.
        # If we find a rank-3 decomposition for T_2 with rational entries, 
        # we can combine it with the rank-7 Strassen decomposition for T_01.
        # Total rank 10.
        
        # Let's write a solver for the 2x2x1 rank-3 problem.
        
        def find_rank3_2x2x1():
            # Variables:
            # U: 4x3
            # V: 2x3 (indices 0, 1 corresponding to j=0, j=1)
            # W: 2x3 (indices 0, 1 corresponding to i=0, i=1)
            
            # Equations:
            # E1: sum_r U[0,r] V[0,r] W[0,r] = 1
            # E2: sum_r U[1,r] V[1,r] W[0,r] = 1
            # E3: sum_r U[2,r] V[0,r] W[1,r] = 1
            # E4: sum_r U[3,r] V[1,r] W[1,r] = 1
            
            # Zeros:
            # Z1: sum_r U[0,r] V[1,r] W[1,r] = 0
            # Z2: sum_r U[1,r] V[0,r] W[1,r] = 0
            # Z3: sum_r U[2,r] V[1,r] W[0,r] = 0
            # Z4: sum_r U[3,r] V[0,r] W[0,r] = 0
            
            # We have 8 equations.
            # 24 variables.
            
            # Let's fix some variables to reduce symmetry.
            # Set V[0,0] = 1, V[1,0] = 0? No, V is 2x3.
            
            # Let's use scipy.optimize.least_squares.
            
            def residuals(x):
                U = x[0:12].reshape(4, 3)
                V = x[12:18].reshape(2, 3)
                W = x[18:24].reshape(2, 3)
                
                e1 = np.sum(U[0, :] * V[0, :] * W[0, :]) - 1
                e2 = np.sum(U[1, :] * V[1, :] * W[0, :]) - 1
                e3 = np.sum(U[2, :] * V[0, :] * W[1, :]) - 1
                e4 = np.sum(U[3, :] * V[1, :] * W[1, :]) - 1
                
                z1 = np.sum(U[0, :] * V[1, :] * W[1, :])
                z2 = np.sum(U[1, :] * V[0, :] * W[1, :])
                z3 = np.sum(U[2, :] * V[1, :] * W[0, :])
                z4 = np.sum(U[3, :] * V[0, :] * W[0, :])
                
                return np.array([e1, e2, e3, e4, z1, z2, z3, z4])
            
            # Initial guess
            x0 = np.random.randn(24)
            
            # Try multiple random starts
            for _ in range(100):
                if time.time() > deadline - 20:
                    return None
                x0 = np.random.randn(24)
                # Scale to reasonable values
                x0 *= 0.1
                
                try:
                    res = scipy.optimize.least_squares(residuals, x0, max_nfev=1000)
                    if res.success and np.allclose(res.fun, 0, atol=1e-6):
                        U = res.x[0:12].reshape(4, 3)
                        V = res.x[12:18].reshape(2, 3)
                        W = res.x[18:24].reshape(2, 3)
                        
                        # Check if we can rationalize
                        # The entries are floats. We need integers or half-integers.
                        # This is a hard constraint.
                        # If the solution is generic, it won't be rational.
                        # However, if a rational solution exists, the numerical solver might find a nearby one.
                        # We can try to round the values and check.
                        
                        # Let's check if the values are close to integers or half-integers.
                        # Multiply by 2 and check closeness to integers.
                        U2 = U * 2
                        V2 = V * 2
                        W2 = W * 2
                        
                        U_round = np.round(U2) / 2
                        V_round = np.round(V2) / 2
                        W_round = np.round(W2) / 2
                        
                        if np.allclose(U, U_round, atol=1e-3) and np.allclose(V, V_round, atol=1e-3) and np.allclose(W, W_round, atol=1e-3):
                            return U_round, V_round, W_round
                except Exception:
                    pass
            return None
        
        # We need to import scipy
        try:
            import scipy.optimize
        except ImportError:
            # Fallback if scipy is not available (though it is stated to be available)
            pass
        
        # Try to find rank 3 for T_2
        rank3_sol = find_rank3_2x2x1()
        
        if rank3_sol:
            U2, V2, W2 = rank3_sol
            # Construct the full rank-10 decomposition
            # Part 1: Strassen (Rank 7) for k=0,1
            # Part 2: Rank 3 for k=2
            
            # Re-use the Strassen construction from the current program
            # But we need to make sure the indices align.
            
            U_strassen = np.zeros((nm, 7))
            V_strassen = np.zeros((mp, 7))
            W_strassen = np.zeros((pn, 7))
            
            def add_strassen_term(idx, u_dict, v_dict, w_dict):
                for (i, j), coeff in u_dict.items():
                    a = i * 2 + j
                    U_strassen[a, idx] += coeff
                for (j, k), coeff in v_dict.items():
                    b = j * 3 + k
                    V_strassen[b, idx] += coeff
                for (k, i), coeff in w_dict.items():
                    c = k * 2 + i
                    W_strassen[c, idx] += coeff

            add_strassen_term(0, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
            add_strassen_term(1, {(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):1})
            add_strassen_term(2, {(0,0):1}, {(1,0):1, (1,1):-1}, {(1,0):1, (0,1):1})
            add_strassen_term(3, {(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (0,1):1})
            add_strassen_term(4, {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})
            add_strassen_term(5, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
            add_strassen_term(6, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
            
            # Part 2: Rank 3 for k=2
            # U2 is 4x3, V2 is 2x3, W2 is 2x3
            # We need to map V2 indices (0,1) to global V indices (2,5)
            # And W2 indices (0,1) to global W indices (4,5)
            
            U_part2 = np.zeros((nm, 3))
            V_part2 = np.zeros((mp, 3))
            W_part2 = np.zeros((pn, 3))
            
            # U2 is already in global U indices (0..3)
            U_part2[:, :] = U2
            
            # V2[0, r] -> V[2, r] (j=0, k=2)
            # V2[1, r] -> V[5, r] (j=1, k=2)
            V_part2[2, :] = V2[0, :]
            V_part2[5, :] = V2[1, :]
            
            # W2[0, r] -> W[4, r] (k=2, i=0)
            # W2[1, r] -> W[5, r] (k=2, i=1)
            W_part2[4, :] = W2[0, :]
            W_part2[5, :] = W2[1, :]
            
            U = np.hstack([U_strassen, U_part2])
            V = np.hstack([V_strassen, V_part2])
            W = np.hstack([W_strassen, W_part2])
            
            # Verify
            T_prime = np.einsum('ir,jr,kr->ijk', U, V, W)
            T_target = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        T_target[a, b, c] = 1.0
            
            if np.allclose(T_prime, T_target, atol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Fallback to Rank 11 (Current Program Logic)
        U_strassen = np.zeros((nm, 7))
        V_strassen = np.zeros((mp, 7))
        W_strassen = np.zeros((pn, 7))
        
        def add_strassen_term(idx, u_dict, v_dict, w_dict):
            for (i, j), coeff in u_dict.items():
                a = i * 2 + j
                U_strassen[a, idx] += coeff
            for (j, k), coeff in v_dict.items():
                b = j * 3 + k
                V_strassen[b, idx] += coeff
            for (k, i), coeff in w_dict.items():
                c = k * 2 + i
                W_strassen[c, idx] += coeff

        add_strassen_term(0, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
        add_strassen_term(1, {(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):1})
        add_strassen_term(2, {(0,0):1}, {(1,0):1, (1,1):-1}, {(1,0):1, (0,1):1})
        add_strassen_term(3, {(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (0,1):1})
        add_strassen_term(4, {(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1})
        add_strassen_term(5, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1})
        add_strassen_term(6, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        
        U_school = np.zeros((nm, 4))
        V_school = np.zeros((mp, 4))
        W_school = np.zeros((pn, 4))
        
        U_school[0, 0] = 1; V_school[2, 0] = 1; W_school[4, 0] = 1
        U_school[1, 1] = 1; V_school[5, 1] = 1; W_school[4, 1] = 1
        U_school[2, 2] = 1; V_school[2, 2] = 1; W_school[5, 2] = 1
        U_school[3, 3] = 1; V_school[5, 3] = 1; W_school[5, 3] = 1
        
        U = np.hstack([U_strassen, U_school])
        V = np.hstack([V_strassen, V_school])
        W = np.hstack([W_strassen, W_school])
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
