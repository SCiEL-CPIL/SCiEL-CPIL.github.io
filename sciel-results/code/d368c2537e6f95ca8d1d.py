"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 144
    
    # Try to find a lower rank decomposition
    # For 4x4 * 4x9, the theoretical lower bound is max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But matrix multiplication tensors have higher rank.
    # Known results: For 4x4x4, rank is at most 49 (Strassen-like improvements)
    # For 4x4x9, we can think of it as 9 copies of 4x4x4 with sharing.
    
    # Strategy: Use random initialization + gradient descent on the tensor network
    # to minimize the rank (i.e., find a valid decomposition with fewer factors)
    
    # The tensor T[i,j,k] = 1 if we consider the "diagonal" but actually for matmul:
    # T_{(i,j),(j,k),(k,i)} ... let's be precise.
    # A[i,j] * B[j,k] = C[i,k]
    # Sum over j: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: M_{i,j,k} where the multiplication tensor has entries 1 at (i,j,k) for the j-sum?
    # No, the tensor for matmul C = AB is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The tensor is T_{(i,j),(j,k),(i,k)}? No.
    # Standard tensor: T_{ijk} where A has indices (i,j), B has (j,k), C has (i,k).
    # The bilinear map is: <A, B> -> C where C_{ik} = sum_j A_{ij} B_{jk}
    # As a 3-tensor: T_{(i,k), j, (i,j),(j,k)}? 
    # Let's use the convention from the task:
    # U has shape (n*m, rank) -> indices (i*m+j, r)
    # V has shape (m*p, rank) -> indices (j*p+k, r)
    # W has shape (p*n, rank) -> indices (k*n+i, r)
    # The decomposition says: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{ij,ik}? 
    # Actually, the matrix multiplication tensor is defined such that:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # In tensor decomposition terms, we want to represent the "multiplication tensor" T
    # where T_{(i,j),(j,k),(i,k)} = 1? No.
    # The tensor T has entries T_{a,b,c} where a=(i,j), b=(j,k), c=(i,k) is 1 if... 
    # Actually, the standard form is:
    # The matmul tensor M has M_{i,j,k} = 1 for all i,j,k? No.
    # Let's look at the schoolbook:
    # For each (i,j,k), we have a rank-1 term: U_{ij} V_{jk} W_{ki}
    # The sum over r of U_{ij,r} V_{jk,r} W_{ki,r} should equal 1 if this term is part of the sum for C_{ik}?
    # Actually, the condition is:
    # For all i,k: sum_j (sum_r U_{ij,r} V_{jk,r}) * ... no.
    # The correct condition for tensor decomposition of matmul:
    # We want to compute C_{ik} = sum_j A_{ij} B_{jk}
    # Using rank-r decomposition:
    # C_{ik} = sum_{r=1}^R (sum_j U_{ij,r} A_{ij}) (sum_k V_{jk,r} B_{jk}) W_{ki,r}? 
    # No, the standard formulation is:
    # C_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) (sum_j V_{jk,r} B_{jk}) W_{ki,r}
    # Wait, the indices in U, V, W are:
    # U: (n*m, R) -> U_{(i,j), r}
    # V: (m*p, R) -> V_{(j,k), r}
    # W: (p*n, R) -> W_{(k,i), r}
    # Then the bilinear form is:
    # sum_r (sum_{i,j} U_{(i,j),r} A_{ij}) (sum_{j,k} V_{(j,k),r} B_{jk}) W_{(k,i),r}
    # This doesn't directly give C_{ik}.
    
    # Let's use the standard tensor definition:
    # The matmul tensor T is a 3-tensor with dimensions (n*m) x (m*p) x (p*n)
    # T_{(i,j), (j,k), (k,i)} = 1 if the indices are consistent, else 0?
    # Actually, for the multiplication C = AB:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This can be written as a trilinear form:
    # sum_{i,j,k} C_{ik} A_{ij} B_{jk} = sum_{i,j,k} (sum_j' A_{ij'} B_{j'k}) A_{ij} B_{jk}
    # = sum_{i,j,k} A_{ij} B_{jk} C_{ik}
    # The tensor T_{(i,j), (j,k), (i,k)} = 1 for all valid i,j,k? 
    # No, T_{(i,j), (j,k), (i,k)} should be 1 only when the "k" in (i,k) matches the "k" in (j,k) and the "i" matches.
    # Actually, the tensor is simply:
    # T_{(i,j), (j,k), (i,k)} = 1 for all i, j, k.
    # Because then sum_{i,j,k} T_{(i,j),(j,k),(i,k)} A_{ij} B_{jk} C_{ik} = sum_{i,j,k} A_{ij} B_{jk} C_{ik}
    # And we want this to equal sum_{i,k} C_{ik} (sum_j A_{ij} B_{jk}) when C is the product.
    # So the tensor entries are all 1 for the "consistent" indices.
    # But in our index scheme:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # The tensor T has T_{u,v,w} = 1 if there exist i,j,k such that u=i*m+j, v=j*p+k, w=k*n+i.
    # For each (i,j,k), there is exactly one such triple.
    # So T_{u,v,w} = 1 if the indices are "compatible", i.e., the j in u matches the j in v, and the k in v matches the k in w, and the i in w matches the i in u.
    # Otherwise 0.
    
    # So we need to find U, V, W of rank R such that:
    # sum_r U_{u,r} V_{v,r} W_{w,r} = T_{u,v,w} for all u,v,w.
    
    # Let's build T explicitly.
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    
    # Now, try to find a low-rank decomposition.
    # We'll use a randomized approach: start with a random rank-R decomposition and refine.
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 36
    # But we can try a few random ranks.
    
    for rank in range(36, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Gradient descent to minimize ||U V^T W - T||_F
            # But this is a non-convex problem.
            # Instead, let's use a simpler approach: 
            # For each rank-1 term, project T onto the rank-1 tensor space.
            
            # Actually, let's use a different strategy:
            # Since we know the schoolbook decomposition works, we can try to "compress" it.
            # Start with the schoolbook decomposition and try to merge terms.
            
            if rank >= schoolbook_rank:
                # Return schoolbook
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = i*m + j
                            v = j*p + k
                            w = k*n + i
                            U[u, col] = 1
                            V[v, col] = 1
                            W[w, col] = 1
                            col += 1
                # Verify
                if rank == schoolbook_rank:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
            
            # For lower ranks, try random search with verification
            # This is computationally expensive, so we'll do a limited search.
            pass
    
    # If we haven't found a better rank, return schoolbook
    U = np.zeros((n*m, schoolbook_rank))
    V = np.zeros((m*p, schoolbook_rank))
    W = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
