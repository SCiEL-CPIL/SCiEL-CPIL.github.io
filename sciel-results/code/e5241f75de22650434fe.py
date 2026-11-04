import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s wall-clock limit

    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72

    # Initialize with schoolbook decomposition as the fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": U_school[:n*m] if False else W_school}
    best_rank = schoolbook_rank

    # Actually fix the best_result initialization
    best_result = {"U": U_school, "V": V_school, "W": W_school}

    def check_full(U, V, W, rank):
        """Check if the decomposition exactly represents 3x3x8 matrix multiplication."""
        for i in range(3):
            for j in range(3):
                for k in range(8):
                    s = 0
                    for r in range(rank):
                        s = s + U[i * 3 + j][r] * V[j * 8 + k][r] * W[k * 3 + i][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    def check_333(U, V, W, rank):
        """Check 3x3x3 decomposition."""
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for r in range(rank):
                        s = s + U[i * 3 + j][r] * V[j][r] * W[k][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    def check_332(U, V, W, rank):
        """Check 3x3x2 decomposition."""
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    s = 0
                    for r in range(rank):
                        s = s + U[i * 3 + j][r] * V[j * 2 + k][r] * W[k * 3 + i][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    def strassen_333():
        """
        Strassen's algorithm for 3x3 matrix multiplication.
        Returns U (9x21), V (3x21), W (3x21) for rank 21.
        
        The 7 Strassen products are combined for 3x3.
        Actually, standard Strassen is for 2x2. For 3x3, we can use
        a direct construction. Let's use the known rank-21 decomposition.
        
        Actually, the minimal rank for 3x3x3 is known to be 23 (Landsberg bound).
        Let's just do a direct search for small ranks.
        """
        return None

    def search_333(target_rank, time_limit=10, seed=42):
        """Search for a rank decomposition of 3x3x3 using structured random search."""
        start = time.time()
        random.seed(seed)
        
        # Use values from {-1, 0, 1} primarily
        vals = [-1, 0, 1]
        vals2 = [-2, -1, -0.5, 0, 0.5, 1, 2]
        
        best_found = None
        
        for attempt in range(50000):
            if time.time() > start + time_limit:
                break
            
            # Generate random matrices with mostly small integers
            U = [[random.choice(vals) for _ in range(target_rank)] for _ in range(9)]
            V = [[random.choice(vals) for _ in range(target_rank)] for _ in range(3)]
            W = [[random.choice(vals) for _ in range(target_rank)] for _ in range(3)]
            
            # Occasionally perturb with larger values
            if attempt % 5 == 0:
                for i in range(9):
                    for r in range(target_rank):
                        if random.random() < 0.05:
                            U[i][r] = random.choice(vals2)
                for i in range(3):
                    for r in range(target_rank):
                        if random.random() < 0.05:
                            V[i][r] = random.choice(vals2)
                        if random.random() < 0.05:
                            W[i][r] = random.choice(vals2)
            
            if check_333(U, V, W, target_rank):
                return U, V, W, target_rank
        
        return None

    def search_332(target_rank, time_limit=5, seed=43):
        """Search for a rank decomposition of 3x3x2."""
        start = time.time()
        random.seed(seed)
        vals = [-1, 0, 1]
        
        for attempt in range(50000):
            if time.time() > start + time_limit:
                break
            
            U = [[random.choice(vals) for _ in range(target_rank)] for _ in range(9)]
            V = [[random.choice(vals) for _ in range(target_rank)] for _ in range(6)]
            W = [[random.choice(vals) for _ in range(target_rank)] for _ in range(6)]
            
            if check_332(U, V, W, target_rank):
                return U, V, W, target_rank
        
        return None

    # Strategy: Try to find better decompositions
    # Known bounds: 3x3x3 has rank at least 23, 3x3x2 has rank at least 15
    
    # Try various target ranks for 3x3x3
    found_333 = []
    target_ranks_333 = [23, 24, 25]
    
    for tr in target_ranks_333:
        if time.time() > deadline - 30:
            break
        res = search_333(tr, time_limit=min(15, deadline - 30 - (time.time() - start_time)), seed=42 + tr)
        if res:
            found_333.append(res)
            print(f"Found 3x3x3 rank {tr}")
            break  # Use the best (lowest rank) found
    
    # Try 3x3x2
    found_332 = None
    target_ranks_332 = [15, 16, 17, 18]
    
    for tr in target_ranks_332:
        if time.time() > deadline - 15:
            break
        res = search_332(tr, time_limit=min(10, deadline - 15 - (time.time() - start_time)), seed=43 + tr)
        if res:
            found_332 = res
            print(f"Found 3x3x2 rank {tr}")
            break
    
    # Construct full decomposition if we found components
    if found_333 and found_332:
        U1, V1, W1, r1 = found_333[0]
        U2, V2, W2, r2 = found_333[0]  # Use same for second block
        U3, V3, W3, r3 = found_332
        
        # Actually we need two independent 3x3x3 decompositions
        # Let's search for a second one with different seed
        if time.time() < deadline - 20:
            res2 = search_333(r1, time_limit=min(10, deadline - 20 - (time.time() - start_time)), seed=100 + r1)
            if res2:
                U2, V2, W2, r2 = res2
            else:
                U2, V2, W2, r2 = U1, V1, W1, r1
        
        total_rank = r1 + r2 + r3
        
        U_full = [[0] * total_rank for _ in range(9)]
        V_full = [[0] * total_rank for _ in range(24)]
        W_full = [[0] * total_rank for _ in range(24)]
        
        # Block 1: k=0,1,2
        for i in range(9):
            for r in range(r1):
                U_full[i][r] = U1[i][r]
        for j in range(3):
            for k in range(3):
                for r in range(r1):
                    V_full[j * 8 + k][r] = V1[j][r]
        for k in range(3):
            for i in range(3):
                for r in range(r1):
                    W_full[k * 3 + i][r] = W1[k][r]
        
        # Block 2: k=3,4,5
        offset1 = r1
        for i in range(9):
            for r in range(r2):
                U_full[i][offset1 + r] = U2[i][r]
        for j in range(3):
            for k in range(3):
                for r in range(r2):
                    V_full[j * 8 + (3 + k)][offset1 + r] = V2[j][r]
        for k in range(3):
            for i in range(3):
                for r in range(r2):
                    W_full[(3 + k) * 3 + i][offset1 + r] = W2[k][r]
        
        # Block 3: k=6,7
        offset2 = r1 + r2
        for i in range(9):
            for r in range(r3):
                U_full[i][offset2 + r] = U3[i][r]
        for j in range(3):
            for k in range(2):
                for r in range(r3):
                    V_full[j * 8 + (6 + k)][offset2 + r] = V3[j * 2 + k][r]
        for k in range(2):
            for i in range(3):
                for r in range(r3):
                    W_full[(6 + k) * 3 + i][offset2 + r] = W3[k * 3 + i][r]
        
        if check_full(U_full, V_full, W_full, total_rank):
            best_result = {"U": U_full, "V": V_full, "W": W_full}
            best_rank = total_rank
            print(f"Found full decomposition with rank {total_rank}")
        else:
            print("Verification failed for combined blocks")
    
    return best_result
