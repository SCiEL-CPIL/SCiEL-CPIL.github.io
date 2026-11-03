<<<<<<< SEARCH
    # Convert to lists with integer/half-integer entries if possible
    # The problem says "integer/half-integer entries"
    # Let's try to round to nearest half-integer
    def to_half_int(mat):
        mat = mat * 2
        mat_rounded = np.round(mat)
        mat_rounded = mat_rounded / 2
        # Check if rounding is valid
        if not np.allclose(mat, mat_rounded, atol=1e-3):
            # Try integer
            mat_int = np.round(mat)
            if np.allclose(mat, mat_int, atol=1e-3):
                return mat_int.tolist()
            # Return as is (floats are acceptable if they're half-integers)
            return mat_rounded.tolist()
        return mat_rounded.tolist()
    
    U_list = to_half_int(best_U)
    V_list = to_half_int(best_V)
    W_list = to_half_int(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
=======
    def is_half_integer(mat, tol=1e-5):
        scaled = mat * 2
        rounded = np.round(scaled)
        return np.allclose(scaled, rounded, atol=tol)

    # Check if best solution is valid and close to half-integers
    use_best = False
    if best_rank < rank_sb:
        if (is_half_integer(best_U) and is_half_integer(best_V) and is_half_integer(best_W)):
            # Verify reconstruction with rounded values to be safe
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            T_recon = np.zeros((n*m, m*p, p*n))
            for c in range(best_rank):
                T_recon += U_rounded[:, c] * V_rounded[:, c] * W_rounded[:, c]
            
            if np.allclose(T_recon, T, atol=1e-5):
                use_best = True
                best_U = U_rounded
                best_V = V_rounded
                best_W = W_rounded
    
    if use_best:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
    else:
        U_list = U_sb.tolist()
        V_list = V_sb.tolist()
        W_list = W_sb.tolist()
        
    return {"U": U_list, "V": V_list, "W": W_list}
>>>>>>> REPLACE
