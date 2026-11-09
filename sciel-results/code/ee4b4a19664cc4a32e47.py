    def refine_sup_norm(coeffs):
        # coeffs: array of shape (N,)
        # 1. Coarse grid
        vals = np.sum(coeffs[None, :] * Z, axis=1)
        abs_vals = np.abs(vals)
        max_idx = np.argmax(abs_vals)
        
        # 2. Refine around max_idx
        # The peak is between max_idx-1 and max_idx+1 (circular)
        # Use a small local grid or golden section
        # Golden section is better for accuracy
        
        theta_max = theta[max_idx]
        # Search interval [theta_max - step, theta_max + step]
        # step is half the grid spacing
        step = theta[1] - theta[0]
        
        def f(t):
            z = np.exp(1j * t)
            # Evaluate P(z)
            # coeffs is 1D
            # z^k
            # Use Horner or dot product
            # z_powers = np.array([z**k for k in range(N)])
            # return np.abs(np.sum(coeffs * z_powers))
            
            # Vectorized:
            z_k = np.exp(1j * t * np.arange(N))
            return np.abs(np.dot(coeffs, z_k))
            
        # Golden Section Search
        a = theta_max - step
        b = theta_max + step
        gr = (np.sqrt(5) - 1) / 2 # Golden ratio conjugate ~ 0.618
        
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = f(c)
        fd = f(d)
        
        # Maximize f
        for _ in range(50): # 50 iterations is plenty for 1e-9 accuracy
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
                
        return max(fc, fd)
    