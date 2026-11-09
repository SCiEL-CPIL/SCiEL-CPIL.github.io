<<<<<<< SEARCH
def coefficients(stages):
    """Optimize 12-stage product formula coefficients via direct numerical optimization."""
    if stages != 12:
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    # Initialize cache
    cache = _get_eig()
    
    # Start with equal weights
    best_a = [1.0/12] * 12
    best_b = [1.0/12] * 12
    best_err = _compute_error_fast(best_a, best_b)
    
    # --- Phase 1: Multi-start with diverse initial guesses ---
    initial_guesses = []
    
    # Equal weights
    initial_guesses.append(([1.0/12]*12, [1.0/12]*12))
    
    # Strang-like: a and b equal, with slight edge emphasis
    for edge_w in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        a_g = [1.0/12] * 12
        b_g = [1.0/12] * 12
        for k in range(12):
            dist = min(k, 11-k)
            a_g[k] = 1.0/12 * (1.0 + 0.3*(6-dist)/6.0)
            b_g[k] = 1.0/12 * (1.0 + 0.3*(6-dist)/6.0)
        sa = sum(a_g); sb = sum(b_g)
        a_g = [x/sa for x in a_g]
        b_g = [x/sb for x in b_g]
        initial_guesses.append((a_g, b_g))
    
    # 4th-order Suzuki-like: negative coefficients at edges
    c = 6**(1/3)
    alpha1 = 1.0/(6.0 - 6.0*c)
    alpha2 = -1.0/(6.0 - 6.0*c)
    beta1 = 1.0/(2.0 - 6.0*c)
    a_g = [alpha1, alpha2, alpha2, alpha1] * 3
    b_g = [beta1, beta1, beta1, beta1] * 3
    sa = sum(a_g); sb = sum(b_g)
    if abs(sa) > 1e-10:
        a_g = [x/sa for x in a_g]
    if abs(sb) > 1e-10:
        b_g = [x/sb for x in b_g]
    initial_guesses.append((a_g, b_g))
    
    # Asymmetric patterns
    for asym in [0.2, 0.5, 1.0]:
        a_g = [1.0/12] * 12
        b_g = [1.0/12] * 12
        for k in range(12):
            a_g[k] = 1.0/12 * (1.0 + asym * k / 11.0)
            b_g[k] = 1.0/12 * (1.0 - asym * k / 11.0)
        sa = sum(a_g); sb = sum(b_g)
        if abs(sa) > 1e-10:
            a_g = [x/sa for x in a_g]
        if abs(sb) > 1e-10:
            b_g = [x/sb for x in b_g]
        initial_guesses.append((a_g, b_g))
    
    # Try each initial guess
    for a_g, b_g in initial_guesses:
        if _remaining() < 5:
            break
        err = _compute_error_fast(a_g, b_g)
        if err < best_err:
            best_err = err
            best_a = a_g[:]
            best_b = b_g[:]
    
    # --- Phase 2: Nelder-Mead optimization on all 24 parameters ---
    def objective(x):
        a = list(x[:12])
        b = list(x[12:24])
        sa = sum(a)
        sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        a = [max(-50.0, min(50.0, x)) for x in a]
        b = [max(-50.0, min(50.0, x)) for x in b]
        sa = sum(a); sb = sum(b)
        if abs(sa) < 1e-15 or abs(sb) < 1e-15:
            return 1e10
        a = [x/sa for x in a]
        b = [x/sb for x in b]
        return _compute_error_fast(a, b)
    
    # Run Nelder-Mead from best initial guess
    x0 = best_a + best_b
    x_opt, f_opt = _nelder_mead(objective, x0, max_iter=400, step_size=0.02)
    
    if f_opt < best_err:
        best_err = f_opt
        best_a = list(x_opt[:12])
        best_b = list(x_opt[12:24])
    
    # --- Phase 3: Second Nelder-Mead with smaller steps from best ---
    if _remaining() > 25:
        x0 = best_a + best_b
        x_opt, f_opt = _nelder_mead(objective, x0, max_iter=300, step_size=0.005)
        
        if f_opt < best_err:
            best_err = f_opt
            best_a = list(x_opt[:12])
            best_b = list(x_opt[12:24])
    
    # --- Phase 4: Final polish with very small steps ---
    if _remaining() > 15:
        x0 = best_a + best_b
        x_opt, f_opt = _nelder_mead(objective, x0, max_iter=200, step_size=0.001)
        
        if f_opt < best_err:
            best_err = f_opt
            best_a = list(x_opt[:12])
            best_b = list(x_opt[12:24])
    
    # Final normalization
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    # Clip
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    
    # Re-normalize after clipping
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa) > 1e-15:
        best_a = [x/sa for x in best_a]
    if abs(sb) > 1e-15:
        best_b = [x/sb for x in best_b]
    
    return best_a, best_b
=======
def coefficients(stages):
    """Optimize 12-stage product formula using symmetric reduced-space L-BFGS-B."""
    if stages != 12:
        a = [1.0/stages] * stages
        b = [1.0/stages] * stages
        return a, b
    
    # Initialize cache
    cache = _get_eig()
    
    # --- Generate diverse symmetric initial guesses in reduced 8D space ---
    # Each guess is [a0,a1,a2,a3,a4,b0,b1,b2,b3,b4]
    # where a5 = (1-sum(a0..a4))/2, etc.
    
    initial_guesses = []
    
    # 1. Equal weights: a[k]=1/12 for all k -> symmetric
    # a0..a4 = 1/12, a5 = (1 - 5/12)/2 = (7/12)/2 = 7/24
    # But we need sum of all 12 = 1. With symmetry: 2*(a0+a1+a2+a3+a4+a5) = 1
    # So a0..a5 sum to 1/2. Equal: each = 1/12.
    eq = 1.0/12.0
    initial_guesses.append([eq]*5 + [eq]*5)
    
    # 2. Edge-emphasized: larger weights at k=0 and k=11 (which are a0 and a5 mirrored)
    for edge_factor in [1.2, 1.5, 2.0]:
        a_half = [1.0/12.0] * 6
        a_half[0] *= edge_factor
        a_half[5] *= edge_factor
        s = sum(a_half)
        a_half = [x/s * 0.5 for x in a_half]  # scale so sum = 0.5
        b_half = a_half[:]
        initial_guesses.append(a_half[:5] + b_half[:5])
    
    # 3. Center-emphasized: larger weights at k=5,6 (which are a5 mirrored)
    for center_factor in [1.2, 1.5, 2.0]:
        a_half = [1.0/12.0] * 6
        a_half[5] *= center_factor
        s = sum(a_half)
        a_half = [x/s * 0.5 for x in a_half]
        b_half = a_half[:]
        initial_guesses.append(a_half[:5] + b_half[:5])
    
    # 4. Suzuki-like 4th order pattern, made symmetric
    # Standard 4th order: [c, -c, -c, c] repeated, with c = 1/(2-2*2^(1/3))
    # For 12 stages, use 3 repetitions of 4-stage pattern
    c4 = 1.0 / (2.0 - 2.0 * (2.0 ** (1.0/3.0)))
    pattern4 = [c4, -c4, -c4, c4]
    a12 = pattern4 * 3
    b12 = [1.0/12.0] * 12
    # Make symmetric: a[k] = a[11-k]
    a_sym = [(a12[k] + a12[11-k]) / 2.0 for k in range(12)]
    b_sym = [(b12[k] + b12[11-k]) / 2.0 for k in range(12)]
    # Extract half
    a_half_s = a_sym[:6]
    b_half_s = b_sym[:6]
    initial_guesses.append(a_half_s[:5] + b_half_s[:5])
    
    # 5. Random symmetric guesses
    rng = np.random.default_rng(42)
    for _ in range(5):
        a_half_r = rng.uniform(0.0, 0.2, size=6)
        b_half_r = rng.uniform(0.0, 0.2, size=6)
        # Scale to sum to 0.5
        sa = sum(a_half_r)
        sb = sum(b_half_r)
        if sa > 1e-10:
            a_half_r = [x/sa * 0.5 for x in a_half_r]
        if sb > 1e-10:
            b_half_r = [x/sb * 0.5 for x in b_half_r]
        initial_guesses.append(a_half_r[:5] + b_half_r[:5])
    
    # --- Multi-start L-BFGS-B optimization ---
    best_params = None
    best_err = float('inf')
    
    for guess in initial_guesses:
        if _remaining() < 10:
            break
        
        params_opt, err_opt = _optimize_from_start(np.array(guess))
        
        if err_opt < best_err:
            best_err = err_opt
            best_params = params_opt
    
    if best_params is None:
        # Fallback to equal weights
        best_params = np.array([1.0/12.0] * 10)
        best_a, best_b = _expand_symmetric(best_params)
        best_err = _compute_error_fast(best_a, best_b)
    
    # --- Final polish with tighter tolerance ---
    if _remaining() > 20:
        params_polish, err_polish = _optimize_from_start(best_params)
        if err_polish < best_err:
            best_err = err_polish
            best_params = params_polish
    
    # Expand to full coefficients
    best_a, best_b = _expand_symmetric(best_params)
    
    # Clip to valid range
    best_a = [max(-50.0, min(50.0, x)) for x in best_a]
    best_b = [max(-50.0, min(50.0, x)) for x in best_b]
    
    # Verify sums and renormalize if needed
    sa = sum(best_a)
    sb = sum(best_b)
    if abs(sa - 1.0) > 1e-8 and abs(sa) > 1e-10:
        best_a = [x/sa for x in best_a]
    if abs(sb - 1.0) > 1e-8 and abs(sb) > 1e-10:
        best_b = [x/sb for x in best_b]
    
    return best_a, best_b
>>>>>>> REPLACE
