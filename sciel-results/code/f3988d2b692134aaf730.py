def _discrete_polish(U, V, W, n, m, p, deadline):
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    current_err = _compute_error(Ua, Va, Wa, n, m, p)
    
    for it in range(20):
        if time.time() > deadline:
            break
        improved = False
        
        # Try flipping U entries
        for idx in range(Ua.size):
            if time.time() > deadline: break
            orig = Ua.flat[idx]
            cand1 = np.floor(orig * 2) / 2.0
            cand2 = np.ceil(orig * 2) / 2.0
            
            if abs(orig - cand1) < 1e-9 or abs(orig - cand2) < 1e-9:
                continue
            
            Ua.flat[idx] = cand1
            err1 = _compute_error(Ua, Va, Wa, n, m, p)
            if err1 < current_err - 1e-9:
                current_err = err1
                improved = True
                continue
            
            Ua.flat[idx] = cand2
            err2 = _compute_error(Ua, Va, Wa, n, m, p)
            if err2 < current_err - 1e-9:
                current_err = err2
                improved = True
                continue
            
            Ua.flat[idx] = orig
        
        # ... similar for V, W
        
        if not improved:
            break
            
    return Ua.tolist(), Va.tolist(), Wa.tolist()
