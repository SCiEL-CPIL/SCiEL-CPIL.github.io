<<<<<<< SEARCH
    # Verify the decomposition is correct
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for all i,j,k, sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] should equal 1
    # This is the matrix multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j  # i*4+j
                v_idx = j * p + k  # j*4+k
                w_idx = k * n + i  # k*4+i
                tensor_val = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                if abs(tensor_val - 1.0) > 1e-10:
                    # If verification fails, we still return the result
                    # but this indicates a bug
                    pass
    
    # Check deadline
    if time.time() > deadline - 5:
        # Return immediately to stay within time limit
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
=======
    # Convert to numpy arrays for the final return
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check deadline
    if time.time() > deadline - 5:
        return {"U": U_np, "V": V_np, "W": W_np}
    
    return {"U": U_np, "V": V_np, "W": W_np}
>>>>>>> REPLACE
