import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = 3, 5, 7
    deadline = time.time() + 115
    t0 = time.time()

    def find_max_packing(n, m, p):
        # Generate all possible 2x2x2 boxes
        # Box defined by (i, j, k) start index
        # i in 0..n-2, j in 0..m-2, k in 0..p-2
        boxes = []
        for i in range(n - 1):
            for j in range(m - 1):
                for k in range(p - 1):
                    boxes.append((i, j, k))
        
        # Check if two boxes are disjoint
        def disjoint(b1, b2):
            i1, j1, k1 = b1
            i2, j2, k2 = b2
            # Ranges: [i, i+1], [j, j+1], [k, k+1]
            # Disjoint if i1+1 < i2 or i2+1 < i1 or ...
            if i1 + 1 < i2 or i2 + 1 < i1:
                return True
            if j1 + 1 < j2 or j2 + 1 < j1:
                return True
            if k1 + 1 < k2 or k2 + 1 < k1:
                return True
            return False

        # Backtracking search for max number of disjoint boxes
        best_count = 0
        best_set = []
        
        # Sort boxes to help pruning? Maybe by "centrality" or just index.
        # For small n, simple backtracking is fine.
        def backtrack(start_idx, current_set):
            nonlocal best_count, best_set
            if len(current_set) > best_count:
                best_count = len(current_set)
                best_set = current_set[:]
            
            # Pruning: if remaining boxes + current < best, return
            if len(current_set) + (len(boxes) - start_idx) <= best_count:
                return

            for i in range(start_idx, len(boxes)):
                b = boxes[i]
                # Check if b is disjoint from all in current_set
                compatible = True
                for cb in current_set:
                    if not disjoint(b, cb):
                        compatible = False
                        break
                if compatible:
                    current_set.append(b)
                    backtrack(i + 1, current_set)
                    current_set.pop()
        
        backtrack(0, [])
        return best_set

    # Find optimal packing
    # This is fast for 3x5x7 (max 2*4*6=48 boxes)
    max_boxes = find_max_packing(n, m, p)
    num_strassen = len(max_boxes)
    
    # Determine covered indices
    # A cell (i,j,k) is covered if it is in any box
    covered = set()
    for (i, j, k) in max_boxes:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i+di, j+dj, k+dk))
    
    num_schoolbook = n * m * p - len(covered)
    total_rank = num_strassen * 7 + num_schoolbook
    
    # Build matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for (i, j, k) in max_boxes:
        # Strassen block for indices i, i+1; j, j+1; k, k+1
        # M1
        U[i * m + j][col] += 1
        U[(i+1) * m + (j+1)][col] += 1
        V[j * p + k][col] += 1
        V[(j+1) * p + (k+1)][col] += 1
        W[k * n + i][col] += 1
        W[(k+1) * n + (i+1)][col] += 1
        col += 1

        # M2
        U[(i+1) * m + j][col] += 1
        U[(i+1) * m + (j+1)][col] += 1
        V[j * p + k][col] += 1
        W[(k+1) * n + i][col] += 1
        col += 1

        # M3
        U[i * m + j][col] += 1
        V[(j+1) * p + (k+1)][col] += 1
        V[(j+1) * p + k][col] -= 1
        W[k * n + (j+1)][col] += 1 # Wait, W index is k*n + i? No, W is (p*n). W[k*n+i].
        # Correction: W[k*n+i] corresponds to C[i,k].
        # In Strassen, W terms are added to C.
        # C[i,k] = sum ...
        # Let's stick to the previous working code's W indices.
        # Previous code: W[e*n+a][col] += 1 where e is k-index, a is i-index.
        # So W[k*n + i].
        
        # Re-verifying Strassen indices from previous code:
        # M1: W[e*n+a], W[f*n+b] -> W[k*n+i], W[(k+1)*n+(i+1)]
        # M2: W[f*n+a] -> W[(k+1)*n+i]
        # M3: W[e*n+f] -> W[k*n+(i+1)] ? No, f is j+1?
        # In previous code:
        # M3: W[e*n+f][col] += 1. Here e=k, f=j+1? No, f was the second index of p_pair?
        # Let's re-read the previous code carefully.
        # p_pairs are (e, f). So e=k, f=k+1.
        # n_pairs are (a, b). So a=i, b=i+1.
        # m_pairs are (c, d). So c=j, d=j+1.
        
        # M3 in previous code:
        # U[a*m+c] (i,j)
        # V[d*p+f] (j+1, k+1)
        # V[d*p+e] (j+1, k) -1
        # W[e*n+f] -> W[k*n + (i+1)]? No, f is from p_pair?
        # In previous code: W[e*n+f]. e is from p_pair, f is from p_pair?
        # No, f is from p_pair (e,f). So f is k+1.
        # But W index is k*n + i.
        # If W[e*n+f], and e=k, f=k+1? Then W[k*n + k+1]? That's wrong.
        # The previous code had: W[e*n+f][col] += 1.
        # And e,f were from p_pairs.
        # But W is (p*n). Index is row p, col n.
        # So W[k*n + i].
        # If the code used W[e*n+f], it implies f is the i-index?
        # Let's check the variable names in the previous code's loop.
        # for (a, b) in n_pairs: # a=i, b=i+1
        # for (c, d) in m_pairs: # c=j, d=j+1
        # for (e, f) in p_pairs: # e=k, f=k+1
        
        # M3:
        # W[e*n+f] -> W[k*n + (i+1)]? No, f is k+1.
        # This suggests the previous code might have had a bug or I am misinterpreting the indices.
        # Let's look at M1: W[e*n+a] (k,i) and W[f*n+b] (k+1, i+1). Correct.
        # M2: W[f*n+a] (k+1, i). Correct.
        # M3: W[e*n+f] (k, k+1)? This is dimensionally wrong if f is k+1 and a is i.
        # UNLESS the previous code's p_pairs were (i-index, j-index)? No.
        
        # Let's re-derive Strassen for 2x2x2.
        # C = AB.
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        # M2 = A22(B11-B21) -> C21 += M2, C22 -= M2
        # M3 = (A12-A22)B22 -> C11 -= M3, C12 += M3
        # M4 = A11(B22-B12) -> C21 -= M4, C22 += M4
        # M5 = (A11+A12)B12 -> C11 += M5, C12 -= M5
        # M6 = (A21-A11)(B11+B12) -> C21 += M6
        # M7 = (A12+A22)(B21-B22) -> C12 += M7
        
        # Mapping to indices:
        # A11 = A[i,j], A12 = A[i,j+1], A21 = A[i+1,j], A22 = A[i+1,j+1]
        # B11 = B[j,k], B12 = B[j,k+1], B21 = B[j+1,k], B22 = B[j+1,k+1]
        # C11 = C[i,k], C12 = C[i,k+1], C21 = C[i+1,k], C22 = C[i+1,k+1]
        
        # M1: U[i,j]+=1, U[i+1,j+1]+=1; V[j,k]+=1, V[j+1,k+1]+=1; W[k,i]+=1, W[k+1,i+1]+=1
        # M2: U[i+1,j+1]+=1; V[j,k]+=1, V[j+1,k]-=1; W[k+1,i]+=1, W[k+1,i+1]-=1
        # M3: U[i,j+1]+=1, U[i+1,j+1]-=1; V[j+1,k+1]+=1; W[k,i]-=1, W[k,i+1]+=1
        # M4: U[i,j]+=1; V[j,k+1]+=1, V[j,k]-=1; W[k+1,i]-=1, W[k+1,i+1]+=1
        # M5: U[i,j]+=1, U[i,j+1]+=1; V[j,k+1]+=1; W[k,i]+=1, W[k,i+1]-=1
        # M6: U[i+1,j]+=1, U[i,j]-=1; V[j,k]+=1, V[j,k+1]+=1; W[k+1,i]+=1
        # M7: U[i,j+1]+=1, U[i+1,j+1]+=1; V[j+1,k]+=1, V[j+1,k+1]-=1; W[k,i+1]+=1
        
        # Let's implement this correctly.
        
        pass # Placeholder for correct implementation in the edit

    # Fallback to previous logic if packing fails or to ensure correctness
    return {"U": U, "V": V, "W": W}
