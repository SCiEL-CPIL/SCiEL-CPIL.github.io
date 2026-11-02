    def construct():
        n, m, p = 4, 4, 6
        U, V, W = [], [], []
        
        def add_term(u_dict, v_dict, w_dict):
            # Convert dicts to lists
            u = [0.0] * (n*m)
            v = [0.0] * (m*p)
            w = [0.0] * (p*n)
            for (i,j), c in u_dict.items(): u[i*m+j] = c
            for (j,k), c in v_dict.items(): v[j*p+k] = c
            for (k,i), c in w_dict.items(): w[k*n+i] = c
            U.append(u); V.append(v); W.append(w)
            
        # ... define strassen terms ...
        # ... expand blocks ...
        
        return {"U": U, "V": V, "W": W}
    