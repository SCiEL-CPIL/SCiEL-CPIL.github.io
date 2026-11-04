"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix.
    
    We use a block decomposition strategy:
    1. Split A (2x3) into A1 (2x2) and A2 (2x1).
    2. Split B (3x6) into B1 (2x6) and B2 (1x6).
    3. C = A1*B1 + A2*B2.
    
    For the term A2*B2 (2x1 * 1x6):
    This is essentially an outer product of a 2x1 matrix and a 1x6 matrix.
    The tensor T[i, 0, k] = A[i,0] * B[0,k].
    This can be decomposed into 6 rank-1 terms? No.
    Let's look at the tensor T_{i, j, k} where j is the index in the middle dimension (size 1).
    T_{i, 0, k} = 1 for all i in {0,1}, k in {0..5}.
    This is a matrix of 1s of size 2x6.
    The rank of the tensor is the matrix rank? No, it's the tensor rank.
    The tensor is a slice of the multiplication tensor.
    Actually, $R(2, 1, 6)$:
    $C_{i,k} = A_{i,1} B_{1,k}$.
    The tensor has entries $T_{i, 1, k} = 1$.
    We can write $T_{i, 1, k} = \sum_{r=1}^{6} U_{i,r} V_{1,r} W_{k,r}$?
    If we set $V_{1,r} = 1$, $U_{i,r} = 1$ for all i, and $W_{k,r}$ to be the identity-like basis?
    No.
    Let's check $R(2, 1, 6)$.
    The tensor is $T \in \mathbb{R}^{2 \times 1 \times 6}$.
    $T_{i, 0, k} = 1$.
    This is equivalent to multiplying a 2x1 matrix by a 1x6 matrix.
    The result is a 2x6 matrix.
    The multiplication tensor for $n \times 1$ by $1 \times p$ is simply the tensor with all 1s.
    The rank of the all-1s tensor of size $n \times 1 \times p$ is $\max(n, p)$? No.
    The rank of the all-1s matrix $n \times p$ is 1.
    The tensor rank of the all-1s tensor $n \times 1 \times p$?
    $T_{i, 0, k} = 1$.
    We can write $1 = \sum_{r=1}^{R} u_{i,r} v_{0,r} w_{k,r}$.
    If we choose $v_{0,r} = 1$ for all r, then we need $1 = \sum_{r=1}^{R} u_{i,r} w_{k,r}$.
    This means the matrix $U W^T$ (size $n \times p$) must be all 1s.
    The minimum rank of a matrix that is all 1s is 1.
    So $R(2, 1, 6) = 1$.
    Wait, is that right?
    If $R=1$, $u_{i,1} w_{k,1} = 1$.
    Let $u_{i,1} = 1, w_{k,1} = 1$.
    Then $T_{i, 0, k} = 1 \cdot 1 \cdot 1 = 1$.
    Yes, rank is 1.
    
    So the term A2*B2 contributes rank 1.
    
    Now for A1*B1 (2x2 * 2x6):
    Split B1 (2x6) into B11 (2x3) and B12 (2x3).
    $A1 B1 = A1 B11 + A1 B12$.
    Each term is $2 \times 2 \times 3$ multiplication.
    What is $R(2, 2, 3)$?
    Schoolbook is $2 \times 2 \times 3 = 12$.
    Can we do better?
    $C_{i,k} = A_{i,1} B_{1,k} + A_{i,2} B_{2,k}$.
    This is a linear combination of two $2 \times 1 \times 3$ multiplications.
    $R(2, 1, 3) = 1$ (by same logic as above, all-1s tensor $2 \times 1 \times 3$ has rank 1).
    So $R(2, 2, 3) \le R(2, 1, 3) + R(2, 1, 3) = 1 + 1 = 2$?
    Let's check.
    Term 1: $A_{i,1} B_{1,k}$. Tensor $T^{(1)}_{i, 1, k} = 1$. Rank 1.
    Term 2: $A_{i,2} B_{2,k}$. Tensor $T^{(2)}_{i, 2, k} = 1$. Rank 1.
    Total tensor $T_{i, j, k} = \delta_{j,1} + \delta_{j,2}$? No.
    The multiplication tensor for $2 \times 2 \times 3$ has indices $i \in \{0,1\}, j \in \{0,1\}, k \in \{0,1,2\}$.
    $T_{i, j, k} = 1$ if the entry is part of the multiplication?
    No, the tensor for matrix multiplication $A B$ where $A$ is $n \times m$ and $B$ is $m \times p$ is defined such that
    $(AB)_{i,k} = \sum_{j=0}^{m-1} A_{i,j} B_{j,k}$.
    The tensor $T$ has components $T_{i,j,k}$ such that $T_{i,j,k} = 1$ for all valid $i,j,k$.
    Wait, this is the "structure tensor".
    The rank of the structure tensor for $n \times m \times p$ multiplication is the bilinear complexity.
    
    For $2 \times 1 \times 3$:
    $C_{i,k} = A_{i,0} B_{0,k}$.
    $T_{i, 0, k} = 1$.
    As established, rank is 1.
    
    For $2 \times 2 \times 3$:
    $C_{i,k} = A_{i,0} B_{0,k} + A_{i,1} B_{1,k}$.
    This is the sum of two $2 \times 1 \times 3$ tensors.
    The first tensor is non-zero only when $j=0$.
    The second tensor is non-zero only when $j=1$.
    Since the supports in the $j$-dimension are disjoint, the rank of the sum is the sum of the ranks.
    Rank($T^{(1)}$) = 1.
    Rank($T^{(2)}$) = 1.
    Total rank = 2.
    
    So $R(2, 2, 3) = 2$.
    
    Then $R(2, 2, 6) = R(2, 2, 3) + R(2, 2, 3) = 2 + 2 = 4$.
    
    So the total rank for $2 \times 3 \times 6$ is:
    $R(2, 2, 6) + R(2, 1, 6) = 4 + 1 = 5$.
    
    Let's verify this logic.
    Is $R(2, 1, 3) = 1$?
    $T_{i, 0, k} = 1$.
    Decomposition: $T = u \otimes v \otimes w$.
    $u_i = 1, v_0 = 1, w_k = 1$.
    $T_{i, 0, k} = 1 \cdot 1 \cdot 1 = 1$.
    Yes.
    
    Is $R(2, 2, 3) = 2$?
    $T_{i, j, k} = 1$ for all $i, j, k$.
    Wait, for matrix multiplication $A (2 \times 2) B (2 \times 3)$,
    $C_{i,k} = A_{i,0} B_{0,k} + A_{i,1} B_{1,k}$.
    The tensor is $T = T^{(0)} + T^{(1)}$.
    $T^{(0)}_{i, 0, k} = 1$, $T^{(0)}_{i, 1, k} = 0$.
    $T^{(1)}_{i, 0, k} = 0$, $T^{(1)}_{i, 1, k} = 1$.
    Rank($T^{(0)}$) = 1.
    Rank($T^{(1)}$) = 1.
    Since they have disjoint support in the $j$-index, Rank($T$) = Rank($T^{(0)}$) + Rank($T^{(1)}$) = 2.
    
    So $R(2, 2, 6)$:
    $B$ is $2 \times 6$. Split into $B_L (2 \times 3)$ and $B_R (2 \times 3)$.
    $C = A B_L + A B_R$.
    Rank = $R(2, 2, 3) + R(2, 2, 3) = 2 + 2 = 4$.
    
    Total Rank = $4 + 1 = 5$.
    
    This seems extremely low. Let's double check if there are any dependencies.
    The decomposition is a sum of rank-1 tensors.
    We are constructing $U, V, W$ matrices.
    
    Let's explicitly construct the rank-1 terms.
    
    Term 1: $A_{i,0} B_{0,k}$ for $k \in \{0,1,2\}$.
    $T^{(1)}_{i, 0, k} = 1$.
    $U^{(1)}_{i, 0} = 1, V^{(1)}_{0, 0} = 1, W^{(1)}_{k, 0} = 1$.
    Indices:
    $U$ row $i$ (0..1), col 0.
    $V$ row 0, col 0.
    $W$ row $k$ (0..2), col 0.
    
    Term 2: $A_{i,1} B_{1,k}$ for $k \in \{0,1,2\}$.
    $T^{(2)}_{i, 1, k} = 1$.
    $U^{(2)}_{i, 0} = 1, V^{(2)}_{1, 0} = 1, W^{(2)}_{k, 0} = 1$.
    Note: We can reuse the same rank slot if the tensors are identical?
    No, the tensors are different because the $j$ index is different.
    However, in the final decomposition, we just sum over all rank slots.
    We can combine them if they share structure?
    Actually, $T^{(1)}$ and $T^{(2)}$ are linearly independent.
    So we need 2 slots for the $2 \times 2 \times 3$ block.
    
    So for $A_{11} B_{11}$ (where $A_{11}$ is $2 \times 2$ and $B_{11}$ is $2 \times 3$):
    We use 2 rank slots.
    Slot 1: $U[i, 0]=1, V[0, 0]=1, W[k, 0]=1$ for $k \in 0..2$.
    Slot 2: $U[i, 1]=1, V[1, 1]=1, W[k, 1]=1$ for $k \in 0..2$.
    
    For $A_{11} B_{12}$ (where $B_{12}$ is $2 \times 3$ corresponding to $k \in 3..5$):
    We use 2 more rank slots.
    Slot 3: $U[i, 2]=1, V[0, 2]=1, W[k, 2]=1$ for $k \in 3..5$.
    Slot 4: $U[i, 3]=1, V[1, 3]=1, W[k, 3]=1$ for $k \in 3..5$.
    
    For $A_{12} B_{12}$ (where $A_{12}$ is $2 \times 1$ and $B_{12}$ is $1 \times 6$):
    Wait, $A_{12}$ corresponds to $j=2$ (the 3rd column of A).
    $B_{12}$ corresponds to the 3rd row of B (which is $1 \times 6$).
    $C_{i,k} += A_{i,2} B_{2,k}$.
    This is a $2 \times 1 \times 6$ multiplication.
    Rank is 1.
    Slot 5: $U[i, 4]=1, V[2, 4]=1, W[k, 4]=1$ for $k \in 0..5$.
    
    Total rank 5.
    
    Let's verify the indices.
    $A$ is $2 \times 3$. Indices $i \in \{0,1\}, j \in \{0,1,2\}$.
    $B$ is $3 \times 6$. Indices $j \in \{0,1,2\}, k \in \{0..5\}$.
    $C$ is $2 \times 6$. Indices $i \in \{0,1\}, k \in \{0..5\}$.
    
    The tensor $T_{i, j, k}$ is 1 for all valid $i, j, k$.
    My decomposition:
    $T = T_1 + T_2 + T_3 + T_4 + T_5$.
    
    $T_1$: $i \in \{0,1\}, j=0, k \in \{0,1,2\}$.
    $T_1_{i, 0, k} = 1$.
    $U_1[i, 0]=1, V_1[0, 0]=1, W_1[k, 0]=1$ (for $k<3$).
    
    $T_2$: $i \in \{0,1\}, j=1, k \in \{0,1,2\}$.
    $T_2_{i, 1, k} = 1$.
    $U_2[i, 1]=1, V_2[1, 1]=1, W_2[k, 1]=1$ (for $k<3$).
    
    $T_3$: $i \in \{0,1\}, j=0, k \in \{3,4,5\}$.
    $T_3_{i, 0, k} = 1$.
    $U_3[i, 2]=1, V_3[0, 2]=1, W_3[k, 2]=1$ (for $k \ge 3$).
    
    $T_4$: $i \in \{0,1\}, j=1, k \in \{3,4,5\}$.
    $T_4_{i, 1, k} = 1$.
    $U_4[i, 3]=1, V_4[1, 3]=1, W_4[k, 3]=1$ (for $k \ge 3$).
    
    $T_5$: $i \in \{0,1\}, j=2, k \in \{0..5\}$.
    $T_5_{i, 2, k} = 1$.
    $U_5[i, 4]=1, V_5[2, 4]=1, W_5[k, 4]=1$ (for all $k$).
    
    Let's check if $T = \sum T_r$.
    For any $(i, j, k)$:
    If $j=0$:
      If $k < 3$: $T_1$ contributes 1. $T_2, T_3, T_4, T_5$ are 0 (wrong j or k).
      Sum = 1. Correct.
      If $k \ge 3$: $T_3$ contributes 1. Sum = 1. Correct.
    If $j=1$:
      If $k < 3$: $T_2$ contributes 1. Sum = 1. Correct.
      If $k \ge 3$: $T_4$ contributes 1. Sum = 1. Correct.
    If $j=2$:
      $T_5$ contributes 1 for all $k$. Sum = 1. Correct.
    
    This works! Rank is 5.
    
    The matrices:
    $U$ shape $(2 \times 3, 5) = (6, 5)$.
    $V$ shape $(3 \times 6, 5) = (18, 5)$.
    $W$ shape $(6 \times 2, 5) = (12, 5)$.
    
    Wait, the problem statement says:
    "output factor follows the official trace convention W[k*n+i,r]"
    So $W$ is indexed by $k$ and $i$.
    $W_{k, i, r}$.
    Shape $(p \times n, \text{rank}) = (6 \times 2, 5) = (12, 5)$.
    
    Let's map the indices.
    $U$ index: $i \times m + j$. ($i \in 0..1, j \in 0..2$).
    $V$ index: $j \times p + k$. ($j \in 0..2, k \in 0..5$).
    $W$ index: $k \times n + i$. ($k \in 0..5, i \in 0..1$).
    
    Let's fill the arrays.
    
    Rank 5.
    
    Slot 0 (T1):
    $U[i, 0] = 1$ for $i=0,1$.
    $V[0 \times 6 + 0, 0] = 1$? No, $V$ index is $j \times p + k$.
    For $T_1$, $j=0, k \in \{0,1,2\}$.
    So $V[0, 0] = 1, V[1, 0] = 1, V[2, 0] = 1$.
    $W[k \times 2 + i, 0] = 1$ for $k \in \{0,1,2\}, i \in \{0,1\}$.
    So $W[0, 0]=1, W[1, 0]=1, W[2, 0]=1, W[3, 0]=1, W[4, 0]=1, W[5, 0]=1$.
    (Indices: $k=0 \to 0,1; k=1 \to 2,3; k=2 \to 4,5$).
    
    Slot 1 (T2):
    $U[i, 1] = 1$ for $i=0,1$.
    $V[1 \times 6 + k, 1] = 1$ for $k \in \{0,1,2\}$.
    So $V[6, 1]=1, V[7, 1]=1, V[8, 1]=1$.
    $W[k \times 2 + i, 1] = 1$ for $k \in \{0,1,2\}, i \in \{0,1\}$.
    So $W[0, 1]=1, W[1, 1]=1, W[2, 1]=1, W[3, 1]=1, W[4, 1]=1, W[5, 1]=1$.
    
    Slot 2 (T3):
    $U[i, 2] = 1$ for $i=0,1$.
    $V[0 \times 6 + k, 2] = 1$ for $k \in \{3,4,5\}$.
    So $V[3, 2]=1, V[4, 2]=1, V[5, 2]=1$.
    $W[k \times 2 + i, 2] = 1$ for $k \in \{3,4,5\}, i \in \{0,1\}$.
    So $W[6, 2]=1, W[7, 2]=1, W[8, 2]=1, W[9, 2]=1, W[10, 2]=1, W[11, 2]=1$.
    
    Slot 3 (T4):
    $U[i, 3] = 1$ for $i=0,1$.
    $V[1 \times 6 + k, 3] = 1$ for $k \in \{3,4,5\}$.
    So $V[9, 3]=1, V[10, 3]=1, V[11, 3]=1$.
    $W[k \times 2 + i, 3] = 1$ for $k \in \{3,4,5\}, i \in \{0,1\}$.
    So $W[6, 3]=1, W[7, 3]=1, W[8, 3]=1, W[9, 3]=1, W[10, 3]=1, W[11, 3]=1$.
    
    Slot 4 (T5):
    $U[i, 4] = 1$ for $i=0,1$.
    $V[2 \times 6 + k, 4] = 1$ for $k \in \{0..5\}$.
    So $V[12, 4]=1, ..., V[17, 4]=1$.
    $W[k \times 2 + i, 4] = 1$ for $k \in \{0..5\}, i \in \{0,1\}$.
    So $W[0, 4]=1, ..., W[11, 4]=1$.
    
    This gives a rank of 5.
    
    Can we do better?
    The rank of $2 \times 1 \times 6$ is 1.
    The rank of $2 \times 2 \times 3$ is 2.
    The rank of $2 \times 2 \times 6$ is 4.
    Sum is 5.
    
    Is it possible that $R(2, 2, 3) < 2$?
    $R(2, 2, 3) \ge R(2, 1, 3) = 1$.
    But we have two disjoint supports.
    If we try to combine them, we would need a rank-1 tensor that covers both $j=0$ and $j=1$.
    But a rank-1 tensor $u \otimes v \otimes w$ has $v_j$ fixed.
    If $v_0 \neq 0$ and $v_1 \neq 0$, then $T_{i, 0, k}$ and $T_{i, 1, k}$ are proportional.
    Specifically $T_{i, 1, k} / T_{i, 0, k} = v_1 / v_0$.
    In our case, $T_{i, 0, k} = 1$ and $T_{i, 1, k} = 1$.
    So ratio is 1.
    So we could potentially use one rank-1 tensor for both if $v_0 = v_1$.
    Let's check.
    $T_{i, j, k} = u_i v_j w_k$.
    If $v_0 = 1, v_1 = 1$, then $T_{i, 0, k} = u_i w_k$ and $T_{i, 1, k} = u_i w_k$.
    We need $u_i w_k = 1$ for all $i, k$.
    This implies $u_i = c, w_k = 1/c$.
    So yes, we can cover $j=0$ and $j=1$ with a single rank-1 tensor!
    
    So $R(2, 2, 3) = 1$?
    Let's verify.
    Tensor $T_{i, j, k} = 1$ for $i \in \{0,1\}, j \in \{0,1\}, k \in \{0,1,2\}$.
    Decomposition: $u_i = 1, v_j = 1, w_k = 1$.
    $T_{i, j, k} = 1 \cdot 1 \cdot 1 = 1$.
    Yes!
    
    So $R(2, 2, 3) = 1$.
    Then $R(2, 2, 6) = 1 + 1 = 2$.
    Total Rank = $2 + 1 = 3$.
    
    Let's re-verify $R(2, 2, 6)$.
    $B$ is $2 \times 6$. Split into $B_L (2 \times 3)$ and $B_R (2 \times 3)$.
    $C = A B_L + A B_R$.
    $T = T_L + T_R$.
    $T_L$ is non-zero for $k \in \{0,1,2\}$.
    $T_R$ is non-zero for $k \in \{3,4,5\}$.
    Disjoint supports in $k$.
    So Rank($T$) = Rank($T_L$) + Rank($T_R$).
    Rank($T_L$) = $R(2, 2, 3) = 1$.
    Rank($T_R$) = $R(2, 2, 3) = 1$.
    Total Rank = 2.
    
    So total rank is 3.
    
    Let's construct the rank-3 decomposition.
    
    Slot 0: Covers $j \in \{0,1\}, k \in \{0,1,2\}$.
    $U[i, 0] = 1$ for $i=0,1$.
    $V[j, 0] = 1$ for $j=0,1$ and $k \in \{0,1,2\}$?
    Wait, $V$ index is $j \times p + k$.
    So $V[0 \times 6 + 0, 0] = 1, V[0 \times 6 + 1, 0] = 1, V[0 \times 6 + 2, 0] = 1$.
    $V[1 \times 6 + 0, 0] = 1, V[1 \times 6 + 1, 0] = 1, V[1 \times 6 + 2, 0] = 1$.
    $W[k \times 2 + i, 0] = 1$ for $k \in \{0,1,2\}, i \in \{0,1\}$.
    
    Slot 1: Covers $j \in \{0,1\}, k \in \{3,4,5\}$.
    $U[i, 1] = 1$ for $i=0,1$.
    $V[0 \times 6 + 3, 1] = 1, V[0 \times 6 + 4, 1] = 1, V[0 \times 6 + 5, 1] = 1$.
    $V[1 \times 6 + 3, 1] = 1, V[1 \times 6 + 4, 1] = 1, V[1 \times 6 + 5, 1] = 1$.
    $W[k \times 2 + i, 1] = 1$ for $k \in \{3,4,5\}, i \in \{0,1\}$.
    
    Slot 2: Covers $j=2, k \in \{0..5\}$.
    $U[i, 2] = 1$ for $i=0,1$.
    $V[2 \times 6 + k, 2] = 1$ for $k \in \{0..5\}$.
    $W[k \times 2 + i, 2] = 1$ for $k \in \{0..5\}, i \in \{0,1\}$.
    
    Total rank 3.
    
    Can we do rank 2?
    $R(2, 2, 6) = 2$.
    $R(2, 1, 6) = 1$.
    Sum = 3.
    Is it possible to merge the $j=2$ part with the others?
    The $j=2$ part has $V[2, k] \neq 0$.
    The other parts have $V[2, k] = 0$.
    So they are linearly independent in the $V$ dimension?
    Not necessarily, but the supports in $j$ are disjoint.
    $T^{(0,1)}$ has support $j \in \{0,1\}$.
    $T^{(2)}$ has support $j=2$.
    So Rank($T$) = Rank($T^{(0,1)}$) + Rank($T^{(2)}$) = 2 + 1 = 3.
    
    So the minimal rank is 3.
    
    Let's implement this.
    
    """
    n, m, p = DIMENSIONS
    
    U = np.zeros((n * m, 3))
    V = np.zeros((m * p, 3))
    W = np.zeros((p * n, 3))
    
    # Slot 0: j in {0,1}, k in {0,1,2}
    for i in range(n):
        U[i * m + 0, 0] = 1.0
        U[i * m + 1, 0] = 1.0
        U[i * m + 2, 0] = 0.0
        
    for j in [0, 1]:
        for k in range(3):
            V[j * p + k, 0] = 1.0
            
    for k in range(3):
        for i in range(n):
            W[k * n + i, 0] = 1.0
            
    # Slot 1: j in {0,1}, k in {3,4,5}
    for i in range(n):
        U[i * m + 0, 1] = 1.0
        U[i * m + 1, 1] = 1.0
        U[i * m + 2, 1] = 0.0
        
    for j in [0, 1]:
        for k in range(3, 6):
            V[j * p + k, 1] = 1.0
            
    for k in range(3, 6):
        for i in range(n):
            W[k * n + i, 1] = 1.0
            
    # Slot 2: j in {2}, k in {0..5}
    for i in range(n):
        U[i * m + 2, 2] = 1.0
        U[i * m + 0, 2] = 0.0
        U[i * m + 1, 2] = 0.0
        
    for k in range(6):
        V[2 * p + k, 2] = 1.0
        
    for k in range(6):
        for i in range(n):
            W[k * n + i, 2] = 1.0
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
