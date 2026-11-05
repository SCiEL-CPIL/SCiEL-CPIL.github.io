import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We'll use a partition approach:
    # Split into blocks and apply Strassen or Winograd where beneficial
    # 
    # Strategy: Partition n=3 as [1,2], m=5 as [1,2,2], p=5 as [1,2,2]
    # Blocks: (1,1,1) schoolbook=1, (1,1,2) schoolbook=2, (1,2,1) schoolbook=2,
    #         (1,2,2) schoolbook=4, (2,1,1) schoolbook=2, (2,1,2) schoolbook=4,
    #         (2,2,1) schoolbook=4, (2,2,2) Strassen=7
    # Total: 1+2+2+4+2+4+4+7 = 26... wait that's too optimistic
    # 
    # Actually for 3x5x5, the schoolbook rank is 3*5*5=75
    # With one 2x2x2 Strassen block we save 8-7=1, so rank=74? No...
    # 
    # Let me think more carefully. The current program achieves rank 71.
    # It uses partition n=[1,2], m=[2,1,2], p=[2,1,2]
    # 
    # n=3: [1,2]
    # m=5: [2,1,2] 
    # p=5: [2,1,2]
    #
    # Blocks and their ranks:
    # (n_i, m_j, p_k) where:
    # i=0: n=1, i=1: n=2
    # j=0: m=2, j=1: m=1, j=2: m=2
    # k=0: p=2, k=1: p=1, k=2: p=2
    #
    # (1,2,2): schoolbook=4
    # (1,2,1): schoolbook=2
    # (1,2,2): schoolbook=4
    # (2,2,2): Strassen=7
    # (2,2,1): schoolbook=4
    # (2,2,2): schoolbook=8
    # (2,1,2): schoolbook=4
    # (2,1,1): schoolbook=2
    # (2,1,2): schoolbook=4
    #
    # Total: 4+2+4+7+4+8+4+2+4 = 39? That doesn't match 71.
    #
    # Wait, I need to recount. There are 2*3*3=18 blocks.
    # 
    # Let me just trust the current program's approach and try to find a better partition.
    # 
    # Actually, the key insight is to try different partitions that maximize the number
    # of 2x2x2 Strassen blocks.
    #
    # For n=3, m=5, p=5:
    # Option A: n=[1,2], m=[2,3], p=[2,3]
    #   (1,2,2): 4
    #   (1,2,3): 6
    #   (1,3,2): 6
    #   (1,3,3): 9
    #   (2,2,2): 7 (Strassen)
    #   (2,2,3): 12
    #   (2,3,2): 12
    #   (2,3,3): 18
    #   Total: 4+6+6+9+7+12+12+18 = 74
    #
    # Option B: n=[2,1], m=[2,3], p=[2,3]
    #   Same as A just reordered: 74
    #
    # Option C: n=[1,2], m=[3,2], p=[3,2]
    #   (1,3,3): 9
    #   (1,3,2): 6
    #   (1,2,3): 6
    #   (1,2,2): 4
    #   (2,3,3): 18
    #   (2,3,2): 12
    #   (2,2,3): 12
    #   (2,2,2): 7
    #   Total: 9+6+6+4+18+12+12+7 = 74
    #
    # Option D: Current n=[1,2], m=[2,1,2], p=[2,1,2]
    #   Let me compute this properly.
    #   n parts: [1, 2]
    #   m parts: [2, 1, 2]
    #   p parts: [2, 1, 2]
    #   
    #   i=0 (n=1), j=0 (m=2), k=0 (p=2): 1*2*2=4
    #   i=0 (n=1), j=0 (m=2), k=1 (p=1): 1*2*1=2
    #   i=0 (n=1), j=0 (m=2), k=2 (p=2): 1*2*2=4
    #   i=0 (n=1), j=1 (m=1), k=0 (p=2): 1*1*2=2
    #   i=0 (n=1), j=1 (m=1), k=1 (p=1): 1*1*1=1
    #   i=0 (n=1), j=1 (m=1), k=2 (p=2): 1*1*2=2
    #   i=0 (n=1), j=2 (m=2), k=0 (p=2): 1*2*2=4
    #   i=0 (n=1), j=2 (m=2), k=1 (p=1): 1*2*1=2
    #   i=0 (n=1), j=2 (m=2), k=2 (p=2): 1*2*2=4
    #   i=1 (n=2), j=0 (m=2), k=0 (p=2): Strassen 2x2x2=7
    #   i=1 (n=2), j=0 (m=2), k=1 (p=1): 2*2*1=4
    #   i=1 (n=2), j=0 (m=2), k=2 (p=2): 2*2*2=8
    #   i=1 (n=2), j=1 (m=1), k=0 (p=2): 2*1*2=4
    #   i=1 (n=2), j=1 (m=1), k=1 (p=1): 2*1*1=2
    #   i=1 (n=2), j=1 (m=1), k=2 (p=2): 2*1*2=4
    #   i=1 (n=2), j=2 (m=2), k=0 (p=2): 2*2*2=8
    #   i=1 (n=2), j=2 (m=2), k=1 (p=1): 2*2*1=4
    #   i=1 (n=2), j=2 (m=2), k=2 (p=2): 2*2*2=8
    #   
    #   Total: 4+2+4+2+1+2+4+2+4+7+4+8+4+2+4+8+4+8 = 71
    #
    # So the current program gets 71. Let me try to do better.
    #
    # Option E: n=[1,2], m=[2,2,1], p=[2,2,1]
    #   i=0 (n=1):
    #     j=0 (m=2): 1*2*2=4, 1*2*2=4, 1*2*1=2
    #     j=1 (m=2): 1*2*2=4, 1*2*2=4, 1*2*1=2
    #     j=2 (m=1): 1*1*2=2, 1*1*2=2, 1*1*1=1
    #   i=1 (n=2):
    #     j=0 (m=2): 2*2*2=7(Strassen), 2*2*2=8, 2*2*1=4
    #     j=1 (m=2): 2*2*2=8, 2*2*2=8, 2*2*1=4
    #     j=2 (m=1): 2*1*2=4, 2*1*2=4, 2*1*1=2
    #   Total: 4+4+2+4+4+2+2+2+1+7+8+4+8+8+4+4+4+2 = 74
    #
    # Option F: n=[2,1], m=[2,2,1], p=[2,2,1]
    #   Same structure, same total: 74
    #
    # Option G: n=[1,2], m=[1,2,2], p=[1,2,2]
    #   i=0 (n=1):
    #     j=0 (m=1): 1*1*1=1, 1*1*2=2, 1*1*2=2
    #     j=1 (m=2): 1*2*1=2, 1*2*2=4, 1*2*2=4
    #     j=2 (m=2): 1*2*1=2, 1*2*2=4, 1*2*2=4
    #   i=1 (n=2):
    #     j=0 (m=1): 2*1*1=2, 2*1*2=4, 2*1*2=4
    #     j=1 (m=2): 2*2*1=4, 2*2*2=7(Strassen), 2*2*2=8
    #     j=2 (m=2): 2*2*1=4, 2*2*2=8, 2*2*2=8
    #   Total: 1+2+2+2+4+4+2+4+4+2+4+4+4+7+8+4+8+8 = 69? Let me recount.
    #   i=0: 1+2+2+2+4+4+2+4+4 = 25
    #   i=1: 2+4+4+4+7+8+4+8+8 = 49
    #   Total: 25+49 = 74
    #
    # Hmm, all giving 74 except the current 71. Let me re-examine.
    #
    # Actually wait, in option G, only one block is 2x2x2 (at i=1,j=1,k=1). 
    # Let me try to get TWO 2x2x2 blocks.
    #
    # For two 2x2x2 Strassen blocks, I need two blocks with ns=2,ms=2,ps=2.
    # n=3 can only have one part of size 2 (since 3=1+2).
    # So I can have at most one n-part that is 2, and it can combine with multiple m,p parts.
    # But each (i,j,k) block is independent. So if n has a part of size 2, and m has parts
    # of size 2, and p has parts of size 2, then I can have multiple (2,2,2) blocks.
    #
    # Current: n=[1,2], m=[2,1,2], p=[2,1,2]
    # The (2,2,2) block occurs at (i=1,j=0,k=0). Only one such block.
    #
    # What if n=[1,2], m=[2,2,1], p=[2,2,1]?
    # Then (i=1,j=0,k=0) is (2,2,2) and (i=1,j=1,k=1) is (2,2,2). TWO Strassen blocks!
    #
    # Let me compute:
    # i=0 (n=1):
    #   j=0 (m=2): k=0(p=2):4, k=1(p=2):4, k=2(p=1):2
    #   j=1 (m=2): k=0(p=2):4, k=1(p=2):4, k=2(p=1):2
    #   j=2 (m=1): k=0(p=2):2, k=1(p=2):2, k=2(p=1):1
    # i=1 (n=2):
    #   j=0 (m=2): k=0(p=2):7(Strassen), k=1(p=2):8, k=2(p=1):4
    #   j=1 (m=2): k=0(p=2):8, k=1(p=2):7(Strassen), k=2(p=1):4
    #   j=2 (m=1): k=0(p=2):4, k=1(p=2):4, k=2(p=1):2
    # Total: (4+4+2+4+4+2+2+2+1) + (7+8+4+8+7+4+4+4+2) = 25 + 48 = 73
    #
    # Hmm, 73 > 71. The issue is that we gain 1 from the second Strassen (8->7) but lose
    # because the other blocks change.
    #
    # Let me try n=[1,2], m=[2,2,1], p=[1,2,2]:
    # i=0 (n=1):
    #   j=0 (m=2): k=0(p=1):2, k=1(p=2):4, k=2(p=2):4
    #   j=1 (m=2): k=0(p=1):2, k=1(p=2):4, k=2(p=2):4
    #   j=2 (m=1): k=0(p=1):1, k=1(p=2):2, k=2(p=2):2
    # i=1 (n=2):
    #   j=0 (m=2): k=0(p=1):4, k=1(p=2):8, k=2(p=2):7(Strassen)
    #   j=1 (m=2): k=0(p=1):4, k=1(p=2):7(Strassen), k=2(p=2):8
    #   j=2 (m=1): k=0(p=1):2, k=1(p=2):4, k=2(p=2):4
    # Total: (2+4+4+2+4+4+1+2+2) + (4+8+7+4+7+8+2+4+4) = 25 + 48 = 73
    #
    # Still 73. Let me try to think about what makes the current 71 work.
    #
    # Current partition gives rank 71. Schoolbook is 75. Saving is 4.
    # One Strassen block saves 1 (8->7). So where does the other 3 savings come from?
    #
    # Wait, I think I'm misunderstanding. The partition approach with Strassen is:
    # For blocks that are exactly 2x2x2, use rank 7 instead of 8.
    # All other blocks use schoolbook.
    #
    # Current: 75 - 1 = 74? But the program says 71. Let me re-examine.
    #
    # Oh wait, I think the issue is that my block counting might be wrong.
    # Let me just trust the current program and try variations.
    #
    # Let me try a completely different approach: use the known rank 71 result
    # from the literature and try to verify it.
    #
    # Actually, for 3x5x5, the known optimal rank is 71 (or possibly lower).
    # Let me try to construct it more carefully.
    
    # Let me try the partition: n=[1,2], m=[1,2,2], p=[1,2,2]
    # This gives:
    # i=0(n=1), j=0(m=1): k=0(p=1):1, k=1(p=2):2, k=2(p=2):2 => 5
    # i=0(n=1), j=1(m=2): k=0(p=1):2, k=1(p=2):4, k=2(p=2):4 => 10
    # i=0(n=1), j=2(m=2): k=0(p=1):2, k=1(p=2):4, k=2(p=2):4 => 10
    # i=1(n=2), j=0(m=1): k=0(p=1):2, k=1(p=2):4, k=2(p=2):4 => 10
    # i=1(n=2), j=1(m=2): k=0(p=1):4, k=1(p=2):7(Str), k=2(p=2):8 => 19
    # i=1(n=2), j=2(m=2): k=0(p=1):4, k=1(p=2):8, k=2(p=2):7(Str) => 19
    # Total: 5+10+10+10+19+19 = 73
    
    # Hmm. Let me try to just use the same approach as the current program
    # but with a different partition.
    
    # Actually, I realize I should just carefully implement the current known-good
    # approach and then try small variations.
    
    rank = 71
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def schoolbook_block(i0, j0, k0, n_s, m_s, p_s):
        nonlocal col
        for i_s in range(n_s):
            for j_s in range(m_s):
                for k_s in range(p_s):
                    i = i0 + i_s
                    j = j0 + j_s
                    k = k0 + k_s
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    def strassen_block(i0, j0, k0):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[i0*m+j0][col]=1; U[(i0+1)*m+(j0+1)][col]=1
        V[j0*p+k0][col]=1; V[(j0+1)*p+(k0+1)][col]=1
        W[k0*n+i0][col]=1; W[(k0+1)*n+(i0+1)][col]=1
        col+=1
        # M2 = (A10+A11)B00
        U[(i0+1)*m+j0][col]=1; U[(i0+1)*m+(j0+1)][col]=1
        V[j0*p+k0][col]=1
        W[k0*n+(i0+1)][col]=1; W[(k0+1)*n+(i0+1)][col]=-1
        col+=1
        # M3 = A00(B01-B11)
        U[i0*m+j0][col]=1
        V[j0*p+(k0+1)][col]=1; V[(j0+1)*p+(k0+1)][col]=-1
        W[(k0+1)*n+i0][col]=1; W[(k0+1)*n+(i0+1)][col]=1
        col+=1
        # M4 = A11(B10-B00)
        U[(i0+1)*m+(j0+1)][col]=1
        V[(j0+1)*p+k0][col]=1; V[j0*p+k0][col]=-1
        W[k0*n+i0][col]=1; W[k0*n+(i0+1)][col]=1
        col+=1
        # M5 = (A00+A01)B11
        U[i0*m+j0][col]=1; U[i0*m+(j0+1)][col]=1
        V[(j0+1)*p+(k0+1)][col]=1
        W[k0*n+i0][col]=-1; W[(k0+1)*n+i0][col]=1
        col+=1
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*m+j0][col]=1; U[i0*m+j0][col]=-1
        V[j0*p+k0][col]=1; V[j0*p+(k0+1)][col]=1
        W[(k0+1)*n+(i0+1)][col]=1
        col+=1
        # M7 = (A01-A11)(B10+B11)
        U[i0*m+(j0+1)][col]=1; U[(i0+1)*m+(j0+1)][col]=-1
        V[(j0+1)*p+k0][col]=1; V[(j0+1)*p+(k0+1)][col]=1
        W[k0*n+i0][col]=1
        col+=1
    
    # Use the same partition as the current program:
    # n=[1,2], m=[2,1,2], p=[2,1,2]
    n_parts = [1, 2]
    m_parts = [2, 1, 2]
    p_parts = [2, 1, 2]
    n_off = [0, 1]
    m_off = [0, 2, 3]
    p_off = [0, 2, 3]
    
    for bi in range(2):
        for bj in range(3):
            for bk in range(3):
                ns = n_parts[bi]
                ms = m_parts[bj]
                ps = p_parts[bk]
                i0 = n_off[bi]
                j0 = m_off[bj]
                k0 = p_off[bk]
                
                if ns == 2 and ms == 2 and ps == 2:
                    strassen_block(i0, j0, k0)
                else:
                    schoolbook_block(i0, j0, k0, ns, ms, ps)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Time check
    elapsed = time.time() - start
    if elapsed > 145:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
