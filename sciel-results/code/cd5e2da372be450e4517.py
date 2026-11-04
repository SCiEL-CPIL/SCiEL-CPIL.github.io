<<<<<<< SEARCH
    # Final: snap to half-integers and verify
    U_final = _snap_to_half_integer(best_U.tolist())
    V_final = _snap_to_half_integer(best_V.tolist())
    W_final = _snap_to_half_integer(best_W.tolist())
    
    if not _verify_fast(U_final, V_final, W_final, n, m, p):
        # Fallback to the unsnapped version if it's better
        if _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            U_final = best_U.tolist()
            V_final = best_V.tolist()
            W_final = best_W.tolist()
        else:
            # Ultimate fallback
            result = _schoolbook(n, m, p)
            return result
    
    return {"U": U_final, "V": V_final, "W": W_final}
=======
    # Final: snap to half-integers and verify
    U_final = _snap_to_half_integer(best_U.tolist())
    V_final = _snap_to_half_integer(best_V.tolist())
    W_final = _snap_to_half_integer(best_W.tolist())
    
    if not _verify_fast(U_final, V_final, W_final, n, m, p):
        # Fallback to the unsnapped version if it's better
        if _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            U_final = best_U.tolist()
            V_final = best_V.tolist()
            W_final = best_W.tolist()
        else:
            # Ultimate fallback
            result = _schoolbook(n, m, p)
            return result
    
    return {"U": U_final, "V": V_final, "W": W_final}
>>>>>>> REPLACE
