<<<<<<< SEARCH
    return {"U": np.array(U, dtype=int), "V": np.array(V, dtype=int), "W": np.array(W, dtype=int)}
=======
    U_out = np.ascontiguousarray(np.array(U, dtype=np.float64))
    V_out = np.ascontiguousarray(np.array(V, dtype=np.float64))
    W_out = np.ascontiguousarray(np.array(W, dtype=np.float64))
    
    # Final shape sanity check
    if U_out.shape != (n*m, rank) or V_out.shape != (m*p, rank) or W_out.shape != (p*n, rank):
        # This should never happen, but if it does, we have a bug in construction
        pass
        
    return {"U": U_out, "V": V_out, "W": W_out}
>>>>>>> REPLACE
