"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time

N = 400
DEADLINE = 145  # seconds, safety margin before 150s kill


def _compute_c3(h):
    """Compute the upper bound C3 for the given step heights.
    
    C3 = max |autoconvolution| where autoconvolution is the linear
    autoconvolution of the step height sequence.
    """
    n = len(h)
    nfft = 2 * n
    H = np.fft.rfft(h, n=nfft)
    conv = np.fft.irfft(H * H, n=nfft)
    # Linear autoconvolution has 2n-1 points
    return np.max(np.abs(conv[:2 * n - 1]))


def _local_search(h, max_iter=1000, deadline=None):
    """Greedy local search: flip bits to reduce C3."""
    best_h = h.copy()
    best_c3 = _compute_c3(best_h)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        # Try flipping each position
        for i in range(N):
            h_new = best_h.copy()
            h_new[i] *= -1
            c3_new = _compute_c3(h_new)
            if c3_new < best_c3 - 1e-12:
                best_h = h_new
                best_c3 = c3_new
                improved = True
        
        if not improved:
            break
    
    return best_h, best_c3


def _simulated_annealing(h0, n_iterations=50000, deadline=None):
    """Simulated annealing to minimize C3."""
    h = h0.copy()
    current_c3 = _compute_c3(h)
    best_h = h.copy()
    best_c3 = current_c3
    
    T_start = 1.0
    T_end = 0.01
    n = len(h)
    
    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Cool down
        T = T_start * (T_end / T_start) ** (i / n_iterations)
        
        # Random flip
        idx = np.random.randint(n)
        h[idx] *= -1
        new_c3 = _compute_c3(h)
        
        delta = new_c3 - current_c3
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_c3 = new_c3
            if new_c3 < best_c3:
                best_c3 = new_c3
                best_h = h.copy()
        else:
            h[idx] *= -1  # revert
    
    return best_h, best_c3


def _initialize_structured():
    """Initialize with a structured sequence (e.g., based on quadratic residues or m-sequence)."""
    # Use a pseudo-random initialization with some structure
    np.random.seed(42)
    h = np.random.choice([-1.0, 1.0], size=N)
    return h


def _initialize_msequence_like():
    """Initialize with an m-sequence-like pattern for good autocorrelation properties."""
    # For N=400, we can't use a perfect m-sequence, but we can use a
    # pattern based on a linear feedback shift register concept
    h = np.zeros(N)
    # Use a simple LFSR-like pattern
    state = 1
    for i in range(N):
        h[i] = 1.0 if (state & 1) else -1.0
        # Update state (simple polynomial)
        state = (state << 1) | ((state >> 9) ^ (state >> 7) ^ (state >> 4) ^ 1)
        state &= 0xFFFF  # Keep 16 bits
    return h


def construct():
    """Construct the optimal signed step function. Returns sequence of signed step heights."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    np.random.seed(42)
    
    best_h = None
    best_c3 = np.inf
    
    # Phase 1: Try multiple initializations with local search
    initializations = []
    
    # Random initialization
    initializations.append(np.random.choice([-1.0, 1.0], size=N))
    
    # Structured initialization
    initializations.append(_initialize_msequence_like())
    
    # Alternating blocks
    h_alt = np.ones(N)
    h_alt[N//4:N//2] = -1
    h_alt[3*N//4:] = -1
    initializations.append(h_alt)
    
    # Sine wave sign
    t = np.linspace(0, 4*np.pi, N)
    initializations.append(np.sign(np.sin(t)))
    initializations[-1][initializations[-1] == 0] = 1.0
    
    for h0 in initializations:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        # Local search from this initialization
        h_ls, c3_ls = _local_search(h0, max_iter=200, deadline=deadline - 5)
        if c3_ls < best_c3:
            best_c3 = c3_ls
            best_h = h_ls
    
    # Phase 2: Simulated annealing from best solution
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        h_sa, c3_sa = _simulated_annealing(best_h, n_iterations=20000, deadline=deadline - 5)
        if c3_sa < best_c3:
            best_c3 = c3_sa
            best_h = h_sa
    
    # Phase 3: Final local search polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        h_final, c3_final = _local_search(best_h, max_iter=500, deadline=deadline - 2)
        if c3_final < best_c3:
            best_c3 = c3_final
            best_h = h_final
    
    return best_h
