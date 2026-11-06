import numpy as np
import time

def construct():
    N = 24
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Pre-compute unit circle points for fast evaluation
    # Dense sampling for initial peak detection
    M = 20000
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * theta)
    
    def get_sup_norm(coeffs):
        """
        Compute the sup-norm of P(z) = sum c_k z^k on |z|=1.
        Uses dense sampling and refinement of the top peaks.
        coeffs: array of length N with values +1 or -1
        """
        # Evaluate on dense grid
        # P(z) = sum c_k z^k
        # We can use FFT if we padded, but for N=24, direct sum is fast enough with vectorization
        # P(z_j) = sum_k c_k z_j^k
        # Let Z be a matrix where Z[j, k] = z_j^k
        # Then P = Z @ coeffs
        # To save memory/time, we can compute powers iteratively or use einsum
        # For N=24, Z is 20000x24, which is small.
        
        # Efficient evaluation:
        # p_vals = np.zeros(M, dtype=complex)
        # for k in range(N):
        #     p_vals += coeffs[k] * (z ** k)
        
        # Even faster:
        # Powers of z
        # z_powers = np.vstack([z ** k for k in range(N)]).T # 20000x24
        # p_vals = z_powers @ coeffs
        
        # Let's use the iterative method to avoid creating the large power matrix repeatedly
        # Or just use the matrix multiplication as it's C-optimized in numpy
        # We can precompute the power matrix once!
        
        # We'll pass the precomputed power matrix to make this fast
        p_vals = power_matrix @ coeffs
        mag = np.abs(p_vals)
        
        # Find the maximum
        max_val = np.max(mag)
        
        # Refine the peak
        # Find the index of the max
        idx = np.argmax(mag)
        
        # Golden section search around the peak
        # The peak is between theta[idx-1] and theta[idx+1] (circular)
        # Step size is 2*pi/M
        step = 2 * np.pi / M
        t0 = theta[(idx - 1) % M]
        t1 = theta[(idx + 1) % M]
        
        # Ensure t0 < t1 for golden section (handle wrap-around)
        if t0 > t1:
            t1 += 2 * np.pi
            
        def eval_at(t):
            z_t = np.exp(1j * t)
            # P(z_t) = sum c_k z_t^k
            # We can compute this quickly
            # z_powers_t = z_t ** np.arange(N)
            # return np.abs(np.sum(coeffs * z_powers_t))
            
            # Reusing logic:
            # Actually, for a single point, it's very fast.
            z_t_powers = np.power(z_t, np.arange(N))
            return np.abs(np.dot(coeffs, z_t_powers))
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        a, b = t0, t1
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = eval_at(c)
        fd = eval_at(d)
        
        for _ in range(50): # 50 iterations is plenty for 1e-9 accuracy
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = eval_at(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = eval_at(d)
                
        refined_max = max(fc, fd)
        
        # Check a few other local peaks just in case
        # Get top 5 indices
        top_indices = np.argsort(mag)[-5:][::-1]
        best_refined = refined_max
        
        for idx2 in top_indices:
            if idx2 == idx:
                continue
            t0_2 = theta[(idx2 - 1) % M]
            t1_2 = theta[(idx2 + 1) % M]
            if t0_2 > t1_2:
                t1_2 += 2 * np.pi
            a2, b2 = t0_2, t1_2
            c2 = b2 - gr * (b2 - a2)
            d2 = a2 + gr * (b2 - a2)
            fc2 = eval_at(c2)
            fd2 = eval_at(d2)
            for _ in range(50):
                if fc2 < fd2:
                    b2 = d2
                    d2 = c2
                    fd2 = fc2
                    c2 = b2 - gr * (b2 - a2)
                    fc2 = eval_at(c2)
                else:
                    a2 = c2
                    c2 = d2
                    fc2 = fd2
                    d2 = a2 + gr * (b2 - a2)
                    fd2 = eval_at(d2)
            local_best = max(fc2, fd2)
            if local_best > best_refined:
                best_refined = local_best
                
        return best_refined

    # Precompute power matrix
    # power_matrix[j, k] = z_j^k
    # We can build this efficiently
    # z is shape (M,)
    # We want shape (M, N)
    # Using broadcasting: z[:, None] ** np.arange(N)[None, :]
    power_matrix = z[:, None] ** np.arange(N)[None, :]
    
    def local_search(coeffs):
        """
        Perform local search by flipping single coefficients.
        Returns the best coeffs and norm found.
        """
        coeffs = np.array(coeffs, dtype=float)
        current_norm = get_sup_norm(coeffs)
        improved = True
        while improved:
            improved = False
            for i in range(N):
                # Flip coefficient i
                coeffs[i] *= -1
                new_norm = get_sup_norm(coeffs)
                if new_norm < current_norm:
                    current_norm = new_norm
                    improved = True
                else:
                    # Revert
                    coeffs[i] *= -1
        return coeffs, current_norm

    best_coeffs = None
    best_norm = float('inf')
    
    # Generate initial candidates
    # 1. Rudin-Shapiro sequence
    # RS sequence: 1, 1, 1, -1, 1, 1, -1, -1 ...
    # Recurrence: a_{2n} = a_n, a_{2n+1} = (-1)^n a_n
    # Let's generate a long RS sequence and take first 24
    rs = [1]
    for i in range(1, 100):
        if i % 2 == 0:
            n = i // 2
            rs.append(rs[n])
        else:
            n = (i - 1) // 2
            rs.append((-1) ** n * rs[n])
    rs_init = rs[:24]
    
    # 2. All ones
    ones_init = [1] * 24
    
    # 3. Random inits
    np.random.seed(42)
    inits = [rs_init, ones_init]
    for _ in range(20):
        inits.append(np.random.choice([-1, 1], size=N).tolist())
        
    # Also try some structured randoms
    # E.g. pairs of same sign, etc.
    
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        coeffs, norm = local_search(init)
        
        if norm < best_norm:
            best_norm = norm
            best_coeffs = coeffs.copy()
            
    # If we have time, do some simulated annealing or more random restarts
    # The local search from RS is very strong for this problem.
    # Let's add a few more random restarts if time permits.
    
    attempts = 0
    while time.time() - start_time < timeout - 5:
        init = np.random.choice([-1, 1], size=N).tolist()
        coeffs, norm = local_search(init)
        if norm < best_norm:
            best_norm = norm
            best_coeffs = coeffs.copy()
        attempts += 1
        
    # Return the best coefficients found
    # The problem asks for coefficients in {+1, -1}
    # Ensure they are integers
    return [int(c) for c in best_coeffs]

if __name__ == "__main__":
    coeffs = construct()
    print(coeffs)
