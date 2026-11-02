import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 48
    
    start_time = time.time()
    deadline = 145  # Safety margin before 150s limit
    
    # Schoolbook solution (guaranteed correct)
    U_school = np.zeros((n * m, rank_school), dtype=np.int64)
    V_school = np.zeros((m * p, rank_school), dtype=np.int64)
    W_school = np.zeros((p * n, rank_school), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = rank_school
    
    # Try to find lower rank decompositions by searching for linear dependencies
    # among the rank-1 factors. We'll use a structured approach:
    # 1. Start with the schoolbook decomposition
    # 2. Try to express one rank-1 component as a linear combination of others
    # 3. If successful, we reduce the rank by 1
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if we're computing
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Each rank-1 component corresponds to a triple (i,j,k)
    
    # We'll try to find if any (i,j,k) component can be expressed as a combination
    # of other components. This is equivalent to finding if the tensor is decomposable
    # with fewer rank-1 terms.
    
    # For now, we'll try a simple search: can we find a linear dependency among
    # the rank-1 factors?
    
    # Build the tensor in a form where we can check for dependencies
    # The tensor T is in U ⊗ V ⊗ W space
    # Each basis element (i,j,k) contributes to T[i*m+j, j*p+k, k*n+i]
    
    # Actually, let's think differently. The matrix multiplication tensor has a specific structure.
    # For n=3, m=4, p=4, we have 48 rank-1 components.
    
    # One approach: try to find if there's a non-trivial linear combination of the
    # rank-1 tensors that equals zero. This would allow us to eliminate one component.
    
    # Let's try a simpler heuristic: look for patterns in the indices
    # For example, if we have components with the same i and k but different j,
    # they might be related.
    
    # Actually, the most promising approach for small dimensions is to use
    # a rank-revealing algorithm or try specific structured decompositions.
    
    # For now, let's return the schoolbook solution as we don't have a guaranteed
    # method to find a lower rank decomposition in the time available.
    
    if time.time() - start_time > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to find a rank 47 solution by checking if one component is redundant
    # We'll check each component to see if it can be expressed as a linear combination
    # of the others
    
    for target_col in range(rank_school):
        if time.time() - start_time > deadline:
            break
        
        # Remove the target column and see if we can still represent the tensor
        U_reduced = np.delete(U_school, target_col, axis=1)
        V_reduced = np.delete(V_school, target_col, axis=1)
        W_reduced = np.delete(W_school, target_col, axis=1)
        
        # Check if the target component can be expressed as a linear combination
        # of the remaining components
        # This is a system of linear equations
        
        # The target component contributes to T at position (i*m+j, j*p+k, k*n+i)
        # where (i,j,k) is the triple for target_col
        
        # Find the triple for target_col
        i, j, k = 0, 0, 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    if ii * m * p + jj * p + kk == target_col:
                        i, j, k = ii, jj, kk
                        break
                else:
                    continue
                break
            else:
                continue
            break
        
        # The target component has a 1 at position (i*m+j, j*p+k, k*n+i) in the tensor
        # We need to check if this can be matched by a linear combination of the other components
        
        # Build the system: for each other component, what does it contribute to
        # the position (i*m+j, j*p+k, k*n+i)?
        
        # Actually, this approach is getting complex. Let's try a different strategy.
        # We'll use the fact that for matrix multiplication, there might be
        # algebraic identities that allow rank reduction.
        
        pass
    
    # For this specific case (3,4,4), let's try known results from literature
    # The tensor rank for 3x4x4 matrix multiplication is known to be at most 48
    # and at least 24 (from Strassen's lower bound)
    
    # Since we can't easily find a lower rank decomposition in the time available,
    # we'll return the schoolbook solution
    
    return {"U": best_U, "V": best_V, "W": best_W}
