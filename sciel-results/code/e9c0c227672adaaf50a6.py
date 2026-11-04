import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def is_valid(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r]
            v = V[:, r]
            w = W[:, r]
            u_reshaped = u.reshape(n, m)
            v_reshaped = v.reshape(m, p)
            w_reshaped = w.reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped, optimize=True)
        return np.allclose(Rec, 1.0, atol=1e-9)

    # For 3x3x5, the Strassen-like algorithm is not directly applicable.
    # We try a systematic search over small integer/half-integer solutions.
    # The rank is likely close to the schoolbook rank of 45, or possibly
    # reducible to around 30-40.
    
    # Let's try a more structured approach: 
    # We know that for matrix multiplication, there are known lower bounds.
    # For 3x3x3, Strassen gives 7, but 3x3x5 is less studied.
    # We'll attempt a simple local search with integer constraints.
    
    def search_integer_solutions(max_rank, max_search_time=30.0):
        nonlocal best_solution, best_rank
        search_start = time.time()
        np.random.seed(42)
        
        # We'll try to find solutions by randomly generating sparse matrices
        # and then checking if they can be adjusted to form a valid decomposition.
        # This is a heuristic approach.
        
        for rank in range(max_rank, 20, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
                
            for _ in range(10):  # 10 attempts per rank
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                    
                # Randomly initialize with small integers
                U = np.random.randint(-2, 3, (n * m, rank))
                V = np.random.randint(-2, 3, (m * p, rank))
                W = np.random.randint(-2, 3, (p * n, rank))
                
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
        
        return False

    # Try to find a better solution
    search_integer_solutions(best_rank - 1)

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
