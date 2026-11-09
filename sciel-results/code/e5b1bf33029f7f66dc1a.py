"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _sup_norm(coeffs, n_samples=4096):
    """Compute max |P(z)| over |z|=1 via dense sampling + golden-section refinement."""
    c = np.array(coeffs, dtype=np.complex128)
    # Dense sampling
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    z = np.exp(1j * thetas)
    # Evaluate polynomial at all points
    # P(z) = sum c_k z^k
    # Use np.polynomial or direct
    vals = np.zeros(n_samples, dtype=np.complex128)
    for k, ck in enumerate(c):
        vals += ck * (z ** k)
    mag = np.abs(vals)
    idx = np.argmax(mag)
    theta_best = thetas[idx]
    
    # Golden-section refinement around the peak
    # Find a small interval around theta_best
    step = 2 * np.pi / n_samples
    a = theta_best - step
    b = theta_best + step
    # Make sure a,b are within [0, 2*pi) - handle wrap-around
    # For simplicity, work in unwrapped coordinates
    gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate ~0.618
    
    def f(theta):
        z = np.exp(1j * theta)
        v = 0
        for k, ck in enumerate(c):
            v += ck * (z ** k)
        return np.abs(v)
    
    # Golden section search
    c_pt = b - gr * (b - a)
    d_pt = a + gr * (b - a)
    fc = f(c_pt)
    fd = f(d_pt)
    
    for _ in range(100):
        if fc < fd:
            b = d_pt
            d_pt = c_pt
            fd = fc
            c_pt = b - gr * (b - a)
            fc = f(c_pt)
        else:
            a = c_pt
            c_pt = d_pt
            fc = fd
            d_pt = a + gr * (b - a)
            fd = f(d_pt)
    
    theta_opt = (a + b) / 2
    return f(theta_opt)

def _score(coeffs):
    """Score = sup_norm / sqrt(24)"""
    return _sup_norm(coeffs) / np.sqrt(24)

def construct():
    start_time = time.time()
    deadline = 145.0  # seconds, safety margin before 150s kill
    
    # Start with the known optimal or near-optimal polynomials
    # The exhaustive optimum for n=24 is 1.2583758
    # Let's try a few known good candidates and then do local search
    
    # Golay-Rudin-Shapiro truncated to 24
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    grs = p[:24]
    
    # Try a set of initial candidates
    candidates = [grs]
    
    # Also try some random starts and known patterns
    np.random.seed(42)
    for _ in range(20):
        candidates.append(list(np.random.choice([-1, 1], size=24)))
    
    # Evaluate all candidates
    best_coeffs = None
    best_score = float('inf')
    
    for cand in candidates:
        s = _score(cand)
        if s < best_score:
            best_score = s
            best_coeffs = cand[:]
    
    # Local search: flip single coefficients, then pairs
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Try flipping each coefficient
        for i in range(24):
            if time.time() - start_time > deadline:
                break
            old = best_coeffs[i]
            best_coeffs[i] = -old
            s = _score(best_coeffs)
            if s < best_score:
                best_score = s
                improved = True
            else:
                best_coeffs[i] = old
        
        if time.time() - start_time > deadline:
            break
        
        # Try flipping pairs of coefficients
        for i in range(24):
            for j in range(i + 1, 24):
                if time.time() - start_time > deadline:
                    break
                old_i, old_j = best_coeffs[i], best_coeffs[j]
                best_coeffs[i] = -old_i
                best_coeffs[j] = -old_j
                s = _score(best_coeffs)
                if s < best_score:
                    best_score = s
                    improved = True
                else:
                    best_coeffs[i] = old_i
                    best_coeffs[j] = old_j
    
    # If we have time, try more random restarts with local search
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        np.random.seed(123)
        for _ in range(50):
            if time.time() - start_time > deadline - 2:
                break
            cand = list(np.random.choice([-1, 1], size=24))
            s = _score(cand)
            if s < best_score:
                best_coeffs = cand[:]
                best_score = s
                # Local search from here
                for _iter in range(3):
                    if time.time() - start_time > deadline - 2:
                        break
                    for i in range(24):
                        if time.time() - start_time > deadline - 2:
                            break
                        old = best_coeffs[i]
                        best_coeffs[i] = -old
                        s2 = _score(best_coeffs)
                        if s2 < best_score:
                            best_score = s2
                        else:
                            best_coeffs[i] = old
    
    return best_coeffs
# EVOLVE-BLOCK-END
