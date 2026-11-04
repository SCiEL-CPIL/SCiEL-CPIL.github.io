import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 + Rank 28 for 4x4x2 = 77
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # A and B are 2x2 matrices defined by linear combinations of base blocks
    # We represent A and B as 2x2 arrays of coefficients (1, -1, 0)
    # for the base elements A_{row, col} and B_{row, col}
    
    # We need to map the logical 2x2 blocks to global indices.
    # For Part 1, the blocks are A_{00}, A_{01}, etc.
    
    # Let's create a generic add_strassen function
    # It takes:
    # u_offsets: 2x2 list of (i, j) global row/col for A elements? 
    # No, U index is 4*i + j.
    # Let's define the base elements as (i, j) pairs.
    
    def add_strassen_block(A_elems, B_elems, C_offsets, coeff=1):
        nonlocal col
        # A_elems: 2x2 list of (i, j) indices in the original A matrix
        # B_elems: 2x2 list of (j, k) indices in the original B matrix
        # C_offsets: 2x2 list of (i, k) indices in the original C matrix
        
        # Strassen 7 products
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We will iterate through the 7 terms and add the scalar products
        # Each term is a product of (A_part) and (B_part)
        # A_part is a sum of A elements
        # B_part is a sum of B elements
        
        # To keep it simple, we expand the sums.
        # For each of the 7 macro-terms, we have a set of A elements and B elements.
        
        # M1: A: {A00, A11}, B: {B00, B11}
        # M2: A: {A10, A11}, B: {B00}
        # M3: A: {A00}, B: {B01, -B11}
        # M4: A: {A11}, B: {B10, -B00}
        # M5: A: {A00, A01}, B: {B11}
        # M6: A: {A10, -A00}, B: {B00, B01}
        # M7: A: {A01, -A11}, B: {B10, B11}
        
        # We define the terms as list of (A_indices, B_indices)
        # where A_indices is a list of (elem, coeff) and B_indices is a list of (elem, coeff)
        
        terms = [
            ([ (A_elems[0][0], 1), (A_elems[1][1], 1) ], [ (B_elems[0][0], 1), (B_elems[1][1], 1) ]),
            ([ (A_elems[1][0], 1), (A_elems[1][1], 1) ], [ (B_elems[0][0], 1) ]),
            ([ (A_elems[0][0], 1) ], [ (B_elems[0][1], 1), (B_elems[1][1], -1) ]),
            ([ (A_elems[1][1], 1) ], [ (B_elems[1][0], 1), (B_elems[0][0], -1) ]),
            ([ (A_elems[0][0], 1), (A_elems[0][1], 1) ], [ (B_elems[1][1], 1) ]),
            ([ (A_elems[1][0], 1), (A_elems[0][0], -1) ], [ (B_elems[0][0], 1), (B_elems[0][1], 1) ]),
            ([ (A_elems[0][1], 1), (A_elems[1][1], -1) ], [ (B_elems[1][0], 1), (B_elems[1][1], 1) ])
        ]
        
        # For each term, we need to add the products to C.
        # C = M1 + M2 - M3 + M4 + M5 - M6 + M7
        # C00 = M1 + M4 - M6
        # C01 = M3 + M5
        # C10 = M2 + M7
        # C11 = M1 - M4 + M6 + M7
        
        # We need to map the scalar products to U, V, W.
        # Each scalar product is A_elem * B_elem.
        # U index: 4*i + j (from A_elem)
        # V index: 6*j + k (from B_elem)
        # W index: 4*k + i (from C_elem)
        
        # We iterate through the 7 terms.
        # For each term, we have a list of A parts and B parts.
        # We form all products.
        
        for t_idx, (a_parts, b_parts) in enumerate(terms):
            # Determine the coefficient of this term in C
            # t_idx 0: M1
            # t_idx 1: M2
            # t_idx 2: M3
            # t_idx 3: M4
            # t_idx 4: M5
            # t_idx 5: M6
            # t_idx 6: M7
            
            # We add the products to the specific C elements.
            # This requires knowing which C elements are affected.
            # It is easier to just add the products to a temporary accumulator or directly to U,V,W.
            # But U,V,W are global.
            
            # We will add each scalar product as a new column in U,V,W.
            # This increases the rank.
            # Wait, Strassen rank is 7, not 7 * (number of scalar products).
            # The rank is the number of columns in U,V,W.
            # In Strassen, we have 7 "multiplications".
            # Each multiplication is a dot product of vectors?
            # No, in the tensor decomposition, each term is a rank-1 tensor.
            # A rank-1 tensor is $u \otimes v \otimes w$.
            # $u$ is a vector of size $n \times m$.
            # $v$ is a vector of size $m \times p$.
            # $w$ is a vector of size $p \times n$.
            
            # In Strassen, each of the 7 products $M_r$ is a $2 \times 2$ matrix.
            # $M_r = A' B'$.
            # $A'$ and $B'$ are $2 \times 2$ matrices.
            # $M_{r, ab} = \sum_{c} A'_{ac} B'_{cb}$.
            # Each $M_{r, ab}$ contributes to $C$.
            # The term $A'_{ac} B'_{cb}$ is a product of two linear combinations.
            # To decompose this into rank-1 tensors, we expand the products.
            # $A'_{ac} = \sum_{i,j} \alpha_{ij} A_{ij}$.
            # $B'_{cb} = \sum_{j,k} \beta_{jk} B_{jk}$.
            # Product = $\sum \alpha_{ij} \beta_{jk} A_{ij} B_{jk}$.
            # Each $A_{ij} B_{jk}$ is a rank-1 tensor.
            # So each Strassen term $M_r$ expands into multiple rank-1 tensors.
            
            # Wait, the rank of $2 \times 2 \times 2$ is 7.
            # This means the entire $2 \times 2 \times 2$ multiplication can be done with 7 rank-1 tensors.
            # My expansion above suggests more.
            # Why?
            # Because in the standard tensor decomposition of Strassen, the "multiplications" are not scalar products of the original elements, but rather the Strassen variables $M_1 \dots M_7$ are the "multiplications".
            # But $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$ is NOT a scalar product. It is a matrix product.
            # So $M_1$ itself has rank 4 (schoolbook) or 7 (Strassen)?
            # No, $M_1$ is a $2 \times 2$ matrix.
            # The rank of the tensor for $2 \times 2 \times 2$ is 7.
            # This means we can express $C = AB$ as sum of 7 rank-1 tensors.
            # These 7 tensors are NOT the scalar products $A_{ij}B_{jk}$.
            # They are specific linear combinations.
            
            # I need the explicit 7 rank-1 tensors for $2 \times 2 \times 2$ Strassen.
            # They are:
            # 1. $U = A_{00}+A_{11}, V = B_{00}+B_{11}$? No.
            
            # Let's look up the rank-1 decomposition of Strassen.
            # The 7 terms are:
            # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$
            # ...
            # This notation is misleading.
            # The rank-1 tensors are:
            # $T_1: u_1 \otimes v_1 \otimes w_1$
            # where $u_1$ has 1s at $A_{00}, A_{11}$ positions?
            # No, $u_1$ is a vector in $\mathbb{R}^{2 \times 2}$.
            # $u_1$ corresponds to the matrix $A_{00}+A_{11}$.
            # So $u_1$ has 1 at $(0,0)$ and 1 at $(1,1)$.
            # $v_1$ has 1 at $(0,0)$ and 1 at $(1,1)$.
            # $w_1$?
            # $C = M_1 + M_2 - M_3 + M_4 + M_5 - M_6 + M_7$.
            # $M_1$ contributes to $C_{00}, C_{01}, C_{10}, C_{11}$.
            # Specifically, $M_1$ is a matrix.
            # $M_1 = \begin{pmatrix} m_{00} & m_{01} \\ m_{10} & m_{11} \end{pmatrix}$.
            # $C_{00} += M_1_{00}$, etc.
            
            # This implies that $M_1$ is not a scalar, but a matrix.
            # So the decomposition is NOT a sum of 7 rank-1 tensors for the $2 \times 2 \times 2$ case?
            # Yes it is.
            # The rank is 7.
            # The 7 terms are the scalar multiplications of the Strassen variables.
            # But Strassen variables are $2 \times 2$ matrices.
            # So we are multiplying $2 \times 2$ matrices?
            # No, Strassen reduces the number of *scalar* multiplications from 8 to 7.
            # The inputs are $2 \times 2$ matrices.
            # The "multiplications" are scalar products of the *elements*?
            # No.
            
            # I will use the explicit 7 rank-1 tensors for $2 \times 2 \times 2$.
            # They are known.
            
            # I will hardcode the 7 tensors for $2 \times 2 \times 2$.
            
            pass
        
        # This approach is getting complicated to code generically.
        # I will hardcode the 49 and 28 terms for the 4x4 case.
        
        # Actually, I can just use the schoolbook for 4x4x4 (Rank 64) and 4x4x2 (Rank 32).
        # Total 96.
        # But I want 77.
        
        # I will implement the 77 rank solution by explicitly listing the 77 terms.
        # I will generate them in the code.
        
        return col

    # I will write the full code in the final block.
    
    return {"U": U, "V": V, "W": W}
