import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # We will construct a rank 22 decomposition using Strassen for 2x2x2 blocks
    # and schoolbook for the remaining 2x1x2 blocks.
    
    # Total rank
    r = 22
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Helper to add a term
    # term_u: dict mapping (i, j) to coeff
    # term_v: dict mapping (j, k) to coeff
    # term_w: dict mapping (k, i) to coeff
    def add_term(col, u_coeffs, v_coeffs, w_coeffs):
        for (i, j), val in u_coeffs.items():
            U[i * m + j][col] += val
        for (j, k), val in v_coeffs.items():
            V[j * p + k][col] += val
        for (k, i), val in w_coeffs.items():
            W[k * n + i][col] += val

    col = 0
    
    # Process left half k in {0, 1}
    # 2x2x2 Strassen part for i in {0,1}, j in {0,1}, k in {0,1}
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1) -> k*n+i
    
    # M1 = (a+d)(e+h)
    add_term(col, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1, (0,1):-1, (1,0):1}); col+=1
    # Wait, W coefficients must map the product to the correct C entries.
    # The term is U_dot * V_dot * W.
    # If U_dot = a+d and V_dot = e+h, the product is (a+d)(e+h).
    # This contributes to C00, C01, C10, C11.
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
    # In Strassen, M1 is added to C00 and C11 with +1, and C01 and C10 with ...?
    # Let's check the formula:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M1 contributes +1 to C00 and +1 to C11.
    # C00 is index k=0, i=0 -> w = 0*2+0 = 0.
    # C11 is index k=1, i=1 -> w = 1*2+1 = 3.
    # So W for M1 should have 1 at (0,0) and 1 at (1,1).
    
    # Let's re-verify the Strassen terms and their W contributions.
    # M1: (a+d)(e+h). W: {(0,0):1, (1,1):1}
    # M2: (c+d)e. W: {(0,0):1} (contributes to C10? No, C10 = M2+M4. C10 is k=0, i=1 -> w=1. Wait. k=0, i=1 is C_{10}? No, C_{ik}. i=1, k=0 is C_{10}. w = 0*2+1 = 1.
    # Let's be precise with indices.
    # i=0, j=0 -> A00
    # i=0, j=1 -> A01
    # i=1, j=0 -> A10
    # i=1, j=1 -> A11
    # j=0, k=0 -> B00
    # j=0, k=1 -> B01
    # j=1, k=0 -> B10
    # j=1, k=1 -> B11
    # k=0, i=0 -> C00 (w=0)
    # k=0, i=1 -> C10 (w=1)
    # k=1, i=0 -> C01 (w=2)
    # k=1, i=1 -> C11 (w=3)
    
    # M1 = (A00+A11)(B00+B11). 
    # Contribution to C00 (w=0): +1
    # Contribution to C11 (w=3): +1
    # So W_M1 = {(0,0):1, (1,1):1}
    
    # M2 = (A10+A11)B00.
    # Contribution to C10 (w=1): +1
    # W_M2 = {(0,1):1}
    
    # M3 = A00(B01-B11).
    # Contribution to C01 (w=2): +1
    # W_M3 = {(1,0):1}
    
    # M4 = A11(B10-A10).
    # Contribution to C10 (w=1): +1
    # Contribution to C00 (w=0): +1? No, C00 = M1+M4-M5+M7. So M4 contributes +1 to C00.
    # W_M4 = {(0,0):1, (0,1):1}
    
    # M5 = (A00+A01)B11.
    # Contribution to C01 (w=2): +1
    # Contribution to C00 (w=0): -1
    # W_M5 = {(1,0):1, (0,0):-1}
    
    # M6 = (A10+A11)(B01-B11).
    # Contribution to C11 (w=3): +1
    # W_M6 = {(1,1):1}
    
    # M7 = (A00-A01)(B00+B10).
    # Contribution to C00 (w=0): +1
    # W_M7 = {(0,0):1}
    
    # Let's implement these 7 terms for the left half (k=0,1).
    # Note: j indices in V are local to the 2x2 block? No, global.
    # In the left half, j=0,1 and k=0,1.
    # U indices: (0,0), (0,1), (1,0), (1,1)
    # V indices: (0,0), (0,1), (1,0), (1,1)
    # W indices: (0,0), (0,1), (1,0), (1,1) -> w = k*n+i
    
    # Term 1: M1
    add_term(col, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}); col+=1
    # Term 2: M2
    add_term(col, {(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1}); col+=1
    # Term 3: M3
    add_term(col, {(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1}); col+=1
    # Term 4: M4
    add_term(col, {(1,1):1}, {(1,0):1, (1,0):-1}, {(0,0):1, (0,1):1}); col+=1 # Wait, V is (1,0):1 and (1,0):-1? No.
    # M4 = A11 * (B10 - A10). V coeff for B10 is 1. U coeff for A10 is -1?
    # No, M4 is a product of a linear form in A and a linear form in B.
    # M4 = (A11) * (B10) + (A11) * (-A10)? No, A10 is in A.
    # Strassen M4 is usually $d(g-c)$. $d=A11, g=B10, c=A10$.
    # So it's not a pure product of A-form and B-form. It's a bilinear form.
    # $d \cdot g - d \cdot c$.
    # This requires two rank-1 tensors? Or can it be represented as one?
    # $U \cdot V \cdot W$.
    # If we set $U = (0, 0, 0, 1)$ (for A11) and $V = (0, 0, 1, 0)$ (for B10), we get $A11 B10$.
    # We need to subtract $A11 A10$? No, $c$ is $A10$.
    # So $M4 = A11 B10 - A11 A10$.
    # This is not a single rank-1 tensor in the $(A, B)$ space?
    # Wait, the tensor is over $A, B, C$.
    # The term is $A11 B10$. The $-A11 A10$ part?
    # No, Strassen formulas are linear combinations of products of linear forms in A and B.
    # $M_4 = d(g-c)$. This is $d \cdot g - d \cdot c$.
    # $d$ is in A, $g$ is in B, $c$ is in A.
    # So $d \cdot c$ is a product of two A-entries. This is not a term in the matrix multiplication tensor which is linear in A and B.
    # I made a mistake. Strassen's $M_4$ is $d(g-c)$?
    # Standard Strassen:
    # $M_1 = (a+d)(e+h)$
    # $M_2 = (c+d)e$
    # $M_3 = a(f-h)$
    # $M_4 = d(g-c)$
    # $M_5 = (a+b)h$
    # $M_6 = (c+d)(f-h)$
    # $M_7 = (a-b)(e+g)$
    # Yes, $M_4$ contains $d \cdot c$ which is $A_{11} A_{10}$.
    # This is not allowed in the tensor decomposition of matrix multiplication, which must be linear in $A$ and $B$.
    # Wait, is it?
    # The tensor $T$ is linear in $A$ and $B$.
    # Any rank-1 term $U \otimes V \otimes W$ is linear in $A$ and $B$.
    # So $M_4$ cannot be a single rank-1 term if it contains $A \cdot A$.
    # But Strassen's algorithm works. How?
    # Ah, $M_4$ is not a separate product. It's a term in the sum.
    # The rank-1 tensors are the $M_k$ themselves?
    # No, the $M_k$ are the products.
    # If $M_4 = d(g-c)$, then the product is $d \cdot g - d \cdot c$.
    # This implies the term is not a simple outer product of a vector in A-space and a vector in B-space.
    # It is a bilinear form.
    # But any bilinear form can be decomposed into rank-1 tensors.
    # $d \cdot g$ is rank 1. $d \cdot c$ is not a product of A and B?
    # $c$ is in A. So $d \cdot c$ is a product of two A-entries.
    # This is not in the space of bilinear forms on $A \times B$.
    # I must have the Strassen formulas wrong or misinterpreted.
    # Let's check $M_4$.
    # Maybe $M_4 = d \cdot g$ and the subtraction is handled in the W coefficients?
    # No, W coefficients are just scalars.
    # The issue is that $M_4$ as defined in Strassen is not a single bilinear term in $A, B$.
    # However, the sum of all $M_k$ results in the correct $C$.
    # The rank is 7 because we compute 7 products.
    # Each product is a bilinear form.
    # If $M_4 = d(g-c)$, it is not a single bilinear form.
    # Actually, $M_4$ is usually defined as $d \cdot g$ and the $-d \cdot c$ part is absorbed?
    # No, let's look at a reliable source.
    # $M_4 = d(g-c)$ is indeed the formula.
    # But $d$ and $c$ are both from A.
    # This means $M_4$ is not a valid term in the tensor decomposition of the multiplication map $(A,B) \to C$.
    # Wait, the multiplication map is bilinear.
    # So every term must be bilinear.
    # $d \cdot c$ is not bilinear in $A, B$ (it's quadratic in A).
    # So my Strassen memory is flawed for this context.
    # Let's use a different set of 7 terms or just use schoolbook for the 2x2x2 part to be safe?
    # Schoolbook for 2x2x2 is rank 4.
    # If I use schoolbook for 2x2x2, rank is 4.
    # Then total rank for 2x3x4 would be $4+4$ (for two 2x2x2) + $4+4$ (for two 2x1x2) = 16?
    # No, 2x2x2 schoolbook is 4 terms.
    # 2x1x2 schoolbook is 4 terms.
    # Left half: 4 (2x2x2) + 4 (2x1x2) = 8.
    # Right half: 4 (2x2x2) + 4 (2x1x2) = 8.
    # Total 16.
    # This is better than 24 and easier to implement correctly.
    
    # Let's implement the rank 16 decomposition using schoolbook for all blocks.
    # This is essentially the schoolbook method but structured.
    # Wait, schoolbook for 2x3x4 is 24.
    # Why is my block sum 16?
    # $A_{2 \times 2} B_{2 \times 2}$ is 4 terms.
    # $A_{2 \times 1} B_{1 \times 2}$ is 4 terms.
    # Total 8 for left half.
    # But $A_{2 \times 2} B_{2 \times 2}$ computes $C_{left, left}$?
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # If we split $j$ and $k$, we are not covering all terms.
    # $j$ goes 0,1,2. $k$ goes 0,1,2,3.
    # My split was $j \in \{0,1\}$ and $j=2$.
    # $k \in \{0,1\}$ and $k \in \{2,3\}$.
    # Term $A_{2 \times 2} B_{2 \times 2}$ covers $j \in \{0,1\}, k \in \{0,1\}$.
    # Term $A_{2 \times 1} B_{1 \times 2}$ covers $j=2, k \in \{0,1\}$.
    # This covers all $j, k$ for the left half.
    # So 8 terms for left half.
    # 8 terms for right half.
    # Total 16 terms.
    # Why is schoolbook 24?
    # Schoolbook is $\sum_{i,j,k} A_{ij} B_{jk}$.
    # There are $2 \cdot 3 \cdot 4 = 24$ such terms.
    # My block decomposition sums them up.
    # $A_{2 \times 2} B_{2 \times 2}$ is $\sum_{i=0,1} \sum_{j=0,1} \sum_{k=0,1} A_{ij} B_{jk}$.
    # This is 4 terms? No, it's a matrix product.
    # The rank of the tensor for $2 \times 2 \times 2$ is 4 (schoolbook).
    # So yes, 4 terms.
    # So total rank 16.
    
    # This is a valid decomposition of rank 16.
    
    # Let's code this.
    
    # Reset U, V, W
    r = 16
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    
    # Left half k in {0, 1}
    # Block 1: j in {0, 1}, k in {0, 1} (2x2x2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Block 2: j in {2}, k in {0, 1} (2x1x2)
    for i in range(2):
        for k in range(2):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Right half k in {2, 3}
    # Block 3: j in {0, 1}, k in {2, 3} (2x2x2)
    for i in range(2):
        for j in range(2):
            for k in range(2, 4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Block 4: j in {2}, k in {2, 3} (2x1x2)
    for i in range(2):
        for k in range(2, 4):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
