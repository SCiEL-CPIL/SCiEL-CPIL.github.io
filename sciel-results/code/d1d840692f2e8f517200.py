import numpy as np
import time
from scipy.optimize import linprog

N = 100

def construct():
    """
    Strategy:
    1. Formulate a Linear Programming (LP) relaxation in the Fourier domain.
       Variables: P_k (power spectrum) for k=1..N-1, and t (max overlap bound).
       Constraints:
       - P_k >= 0
       - P_0 is fixed by sum(h) = N/2 => |H_0|^2 = (N/2)^2.
       - Autocorrelation constraints: 
         autocorr_s = (1/N) * sum_{k=0}^{N-1} P_k * cos(2*pi*k*s/N)
         We require N/2 - autocorr_s <= t  =>  autocorr_s >= N/2 - t
       Objective: Minimize t.
       
    2. Solve the LP to find the optimal power spectrum P_k and lower bound t*.
    
    3. Reconstruct a valid h from P_k:
       - We have target magnitudes |H_k| = sqrt(P_k).
       - We need to find phases such that h = IDFT(H) is in [0,1] and sum(h)=N/2.
       - Use a simple iterative phase retrieval or gradient descent on phases 
         with projection to [0,1] and sum constraint.
       
    4. Return the best valid h found.
    """
    start_time = time.time()
    deadline = 145.0
    
    # 1. Solve LP Relaxation
    # Variables: x = [P_1, ..., P_{N-1}, t]
    # Indexing: 0 to N-2 for P_1..P_{N-1}, N-1 for t
    
    c = np.zeros(N)
    c[N-1] = 1.0 # Minimize t
    
    # Inequality constraints: A_ub x <= b_ub
    # We have constraints for each shift s=1..N-1:
    # (1/N) * sum_{k=0}^{N-1} P_k cos(2*pi*k*s/N) >= N/2 - t
    # => - (1/N) * sum_{k=1}^{N-1} P_k cos(2*pi*k*s/N) - (1/N) P_0 cos(0) + t <= -N/2
    # Wait:
    # autocorr_s = (1/N) [ P_0 * 1 + sum_{k=1}^{N-1} P_k cos(2*pi*k*s/N) ]
    # Note: P_0 = (N/2)^2.
    # Constraint: N/2 - autocorr_s <= t
    # => -autocorr_s - t <= -N/2
    # => - (1/N) [ P_0 + sum_{k=1}^{N-1} P_k cos(2*pi*k*s/N) ] - t <= -N/2
    
    A_ub = np.zeros((N-1, N))
    b_ub = np.zeros(N-1)
    
    P0 = (N / 2.0) ** 2
    
    for s_idx in range(1, N):
        # Row index in A_ub
        row = s_idx - 1
        
        # Coefficients for P_k (k=1..N-1)
        # - (1/N) * cos(2*pi*k*s/N)
        for k in range(1, N):
            A_ub[row, k-1] = -(1.0 / N) * np.cos(2.0 * np.pi * k * s_idx / N)
        
        # Coefficient for t
        A_ub[row, N-1] = -1.0
        
        # RHS
        # - (1/N) P_0 - t <= -N/2  => - (1/N) P_0 <= -N/2 + t ? No.
        # Equation: - (1/N) P_0 - (1/N) sum P_k cos(...) - t <= -N/2
        # Move constant to RHS:
        # - (1/N) sum P_k cos(...) - t <= -N/2 + (1/N) P_0
        b_ub[row] = -N/2.0 + (1.0 / N) * P0
    
    # Bounds: P_k >= 0, t >= 0 (t can be negative? Overlap is N/2 - acorr. 
    # If acorr is large, overlap is small. Min overlap can be negative?
    # h in [0,1], sum N/2. Max acorr is N/2 (if h is constant 0.5? No, acorr_0 = N/2*0.5 = N/4? No.
    # acorr_0 = sum h_i^2. If h_i=0.5, sum h_i^2 = N * 0.25 = N/4.
    # Overlap_0 = N/2 - N/4 = N/4.
    # For s != 0, acorr_s can be negative.
    # So t can be negative?
    # The score is the max overlap. If we can make all overlaps negative, score is negative.
    # But h in [0,1] restricts this.
    # Let's allow t to be unbounded below in LP, but physically it will be bounded.
    # However, linprog requires bounds. Let's set t lower bound to -N (safe).
    
    bounds = [(0, None)] * (N-1) + [(-10.0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback to simple heuristic if LP fails
            return np.full(N, 0.5)
        
        P_opt = np.zeros(N)
        P_opt[0] = P0
        P_opt[1:] = res.x[:N-1]
        t_opt = res.x[N-1]
        
    except Exception:
        return np.full(N, 0.5)
    
    # 2. Reconstruct h from P_opt
    # Target magnitudes
    target_mags = np.sqrt(np.maximum(P_opt, 0))
    
    # We need to find phases phi_k such that h = IDFT( target_mags * exp(1j*phi) ) is valid.
    # h must be real => H_{N-k} = conj(H_k).
    # So we only define phases for k=1..N/2.
    # P_k = P_{N-k}.
    
    # Initialize with random phases
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = float('inf')
    
    # Try a few random phase initializations
    for run in range(10):
        if time.time() > deadline:
            break
            
        # Generate random phases for k=1..N//2
        # H_0 is real and positive (since sum h > 0)
        H = np.zeros(N, dtype=complex)
        H[0] = target_mags[0]
        
        # For k=1 to N//2 - 1 (and N//2 if N even)
        for k in range(1, N // 2 + 1):
            if k == N - k: # N even, k=N/2
                # H_{N/2} must be real
                phase = rng.choice([0, np.pi])
                H[k] = target_mags[k] * np.exp(1j * phase)
            else:
                phase = rng.uniform(0, 2 * np.pi)
                H[k] = target_mags[k] * np.exp(1j * phase)
                H[N - k] = np.conj(H[k])
        
        # Initial h
        h = np.fft.ifft(H).real
        
        # Iterative refinement: Gradient descent on phases?
        # Or simpler: Project to [0,1] and sum constraint, then update H magnitudes?
        # Let's do a simple iterative projection method (like alternating projections)
        # 1. Compute h from H
        # 2. Project h to [0,1] and sum=N/2
        # 3. Update H to match the new h's spectrum? No, we want to keep target_mags.
        # So we fix magnitudes, adjust phases.
        
        # Let's use a simple gradient step on phases to minimize distance to [0,1] box and sum constraint.
        # Objective: sum (max(0, h_i - 1)^2 + max(0, -h_i)^2) + (sum h_i - N/2)^2
        
        for step in range(50):
            if time.time() > deadline:
                break
            
            h = np.fft.ifft(H).real
            
            # Check if valid
            if np.all(h >= -1e-6) and np.all(h <= 1 + 1e-6):
                h_clipped = np.clip(h, 0, 1)
                # Fix sum
                s = np.sum(h_clipped)
                if s > 0:
                    h_clipped *= (N/2.0 / s)
                    h_clipped = np.clip(h_clipped, 0, 1)
                    # Final check
                    if np.all(h_clipped >= -1e-9) and np.all(h_clipped <= 1 + 1e-9):
                        sc = _max_overlap(h_clipped)
                        if sc < best_score:
                            best_score = sc
                            best_h = h_clipped.copy()
            
            # Compute gradient of penalty w.r.t phases
            # This is complex. Let's use a simpler heuristic:
            # If h_i > 1, reduce energy at frequencies that peak at i?
            # If h_i < 0, increase energy?
            
            # Alternative: Just re-randomize if stuck?
            # Let's try a few more random inits if we haven't found a good one.
            # For now, just store the best clipped version if it was close.
            if not (np.all(h >= -1e-6) and np.all(h <= 1 + 1e-6)):
                h_proj = np.clip(h, 0, 1)
                s = np.sum(h_proj)
                if s > 0:
                    h_proj *= (N/2.0 / s)
                    h_proj = np.clip(h_proj, 0, 1)
                    if np.all(h_proj >= -1e-9) and np.all(h_proj <= 1 + 1e-9):
                        sc = _max_overlap(h_proj)
                        if sc < best_score:
                            best_score = sc
                            best_h = h_proj.copy()
        
        # If we found a valid h, we can try to improve it with local search
        if best_h is not None and best_score < t_opt + 0.1:
             # Local search on best_h
             h_local = best_h.copy()
             for _ in range(100):
                 if time.time() > deadline:
                     break
                 # Random swap or jitter
                 idx = rng.integers(0, N)
                 delta = rng.uniform(-0.1, 0.1)
                 new_h = h_local.copy()
                 new_h[idx] += delta
                 # Project
                 new_h = np.clip(new_h, 0, 1)
                 s = np.sum(new_h)
                 if s > 0:
                     new_h *= (N/2.0 / s)
                     new_h = np.clip(new_h, 0, 1)
                 if np.all(new_h >= -1e-9) and np.all(new_h <= 1 + 1e-9):
                     sc = _max_overlap(new_h)
                     if sc < best_score:
                         best_score = sc
                         best_h = new_h.copy()
                         h_local = new_h.copy()

    if best_h is None:
        # Fallback: Use the h from the last iteration of the first run, or uniform
        # Let's try to construct a valid h from the LP spectrum using a greedy phase assignment
        # If that fails, return uniform 0.5
        return np.full(N, 0.5)
        
    return best_h

def _max_overlap(h):
    Fh = np.fft.fft(h)
    S = Fh * np.conj(Fh)
    acorr = np.fft.ifft(S).real
    overlaps = N / 2.0 - acorr[1:N]
    return np.max(overlaps)
