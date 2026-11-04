import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    DEADLINE = 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Schoolbook baseline for 2x3x9 (rank 54)
    rank_54 = n * m * p
    U_54 = [[0] * rank_54 for _ in range(nm)]
    V_54 = [[0] * rank_54 for _ in range(mp)]
    W_54 = [[0] * rank_54 for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_54[i * m + j][col] = 1
                V_54[j * p + k][col] = 1
                W_54[k * n + i][col] = 1
                col += 1
    best = {"U": [row[:] for row in U_54], "V": [row[:] for row in V_54], "W": [row[:] for row in W_54]}
    best_rank = rank_54

    # --- Search for rank-17 decomposition of 2x3x3 ---
    n2, m2, p2 = 2, 3, 3
    nm2 = n2 * m2
    mp2 = m2 * p2
    pn2 = p2 * n2
    target_rank = 17

    def error_233(U, V, W, rank):
        err = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    s = 0
                    for r in range(rank):
                        s += U[i*m2+j][r] * V[j*p2+k][r] * W[k*n2+i][r]
                    err += (s - 1) ** 2
        return err

    def verify_233(U, V, W, rank):
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    s = 0
                    for r in range(rank):
                        s += U[i*m2+j][r] * V[j*p2+k][r] * W[k*n2+i][r]
                    if s != 1:
                        return False
        return True

    found_233 = None

    def local_search_233(U, V, W, max_steps, deadline):
        rank = target_rank
        current_err = error_233(U, V, W, rank)
        if current_err == 0:
            if verify_233(U, V, W, rank):
                return (U, V, W)
            return None

        for step in range(max_steps):
            if time.time() > deadline:
                break
            if current_err == 0:
                if verify_233(U, V, W, rank):
                    return (U, V, W)
                break

            # Try a random perturbation
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, nm2 - 1)
                col = random.randint(0, rank - 1)
                old_val = U[row][col]
                U[row][col] += random.choice([-1, 1])
            elif mat == 'V':
                row = random.randint(0, mp2 - 1)
                col = random.randint(0, rank - 1)
                old_val = V[row][col]
                V[row][col] += random.choice([-1, 1])
            else:
                row = random.randint(0, pn2 - 1)
                col = random.randint(0, rank - 1)
                old_val = W[row][col]
                W[row][col] += random.choice([-1, 1])

            new_err = error_233(U, V, W, rank)
            if new_err < current_err:
                current_err = new_err
            else:
                # Revert
                if mat == 'U':
                    U[row][col] = old_val
                elif mat == 'V':
                    V[row][col] = old_val
                else:
                    W[row][col] = old_val

        if current_err == 0:
            if verify_233(U, V, W, rank):
                return (U, V, W)
        return None

    # Strategy 1: Start from a structured initialization
    # Initialize with a pattern that has some structure
    for attempt in range(30):
        if time.time() > DEADLINE - 30:
            break

        # Different initialization strategies
        if attempt % 3 == 0:
            # Sparse random
            U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(nm2)]
            V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(mp2)]
            W = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(pn2)]
        elif attempt % 3 == 1:
            # Denser random
            U = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(nm2)]
            V = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(mp2)]
            W = [[random.choice([-2, -1, 0, 1, 2]) for _ in range(target_rank)] for _ in range(pn2)]
        else:
            # Very sparse
            U = [[0] * target_rank for _ in range(nm2)]
            V = [[0] * target_rank for _ in range(mp2)]
            W = [[0] * target_rank for _ in range(pn2)]
            # Add some 1s
            for _ in range(15):
                U[random.randint(0, nm2-1)][random.randint(0, target_rank-1)] = random.choice([-1, 1])
            for _ in range(15):
                V[random.randint(0, mp2-1)][random.randint(0, target_rank-1)] = random.choice([-1, 1])
            for _ in range(15):
                W[random.randint(0, pn2-1)][random.randint(0, target_rank-1)] = random.choice([-1, 1])

        result = local_search_233(U, V, W, 200, DEADLINE - 20)
        if result:
            found_233 = result
            break

    # If found 2x3x3 rank 17, construct 2x3x9
    if found_233:
        U_233, V_233, W_233 = found_233
        total_rank = 3 * target_rank

        U_9 = [[0] * total_rank for _ in range(nm)]
        V_9 = [[0] * total_rank for _ in range(mp)]
        W_9 = [[0] * total_rank for _ in range(pn)]

        for block in range(3):
            offset = block * target_rank
            for i in range(n2):
                for j in range(m2):
                    for r in range(target_rank):
                        U_9[i*m2+j][offset+r] = U_233[i*m2+j][r]

            for j in range(m2):
                for k in range(p2):
                    for r in range(target_rank):
                        V_9[j*p + (block*p2 + k)][offset+r] = V_233[j*p2+k][r]

            for k in range(p2):
                for i in range(n2):
                    for r in range(target_rank):
                        W_9[(block*p2 + k)*n + i][offset+r] = W_233[k*n2+i][r]

        def verify_full(U, V, W, rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if s != 1:
                            return False
            return True

        if verify_full(U_9, V_9, W_9, total_rank):
            best = {"U": U_9, "V": V_9, "W": W_9}
            best_rank = total_rank

    return best
