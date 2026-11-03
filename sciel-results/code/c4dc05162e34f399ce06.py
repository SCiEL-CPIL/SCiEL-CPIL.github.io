"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _autocorr(signed):
    """Compute autocorrelation of signed sequence."""
    n = len(signed)
    acf = np.correlate(signed, signed, mode='full')
    # Normalize by N
    acf = acf / n
    # Take the positive lags (center to right)
    return acf[n - 1:]


def _peak_sidelobe(signed):
    """Peak sidelobe: max |autocorr| for non-zero lags."""
    acf = _autocorr(signed)
    # Exclude zero lag (acf[0])
    return np.max(np.abs(acf[1:])) if len(acf) > 1 else 0.0


def _harmonic_boundary_init(n, rng):
    """
    Initialize signs from zero-crossings of a 2D discrete harmonic function.
    
    We create a 2D grid where the bottom row is our 1D sequence.
    The harmonic extension solves the discrete Laplace equation with
    Dirichlet boundary conditions. The zero-crossing pattern of the
    harmonic extension gives us a smooth 1D sign sequence.
    """
    # Create a 2D grid: height H, width W
    H = 32  # Height of harmonic extension
    W = n
    
    # Initialize bottom boundary with random signs
    bottom = rng.choice([-1.0, 1.0], size=W)
    
    # Set up the 2D grid with bottom boundary fixed
    grid = np.zeros((H, W))
    grid[-1, :] = bottom
    grid[0, :] = 0.0  # Top boundary: zero
    
    # Solve discrete Laplace equation iteratively (Gauss-Seidel)
    # This enforces global smoothness
    for _ in range(100):  # Fixed iterations for determinism
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                grid[i, j] = 0.25 * (grid[i-1, j] + grid[i+1, j] + 
                                       grid[i, j-1] + grid[i, j+1])
    
    # Derive signs from zero-crossings of the harmonic extension
    # Use the row closest to the boundary
    row = grid[-2, :]
    signs = np.where(row >= 0, 1.0, -1.0)
    
    return signs


def _anneal_from_harmonic(signed, rng, max_iters=5000):
    """Simulated annealing starting from harmonic boundary initialization."""
    n = len(signed)
    current = signed.copy()
    current_score = _peak_sidelobe(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.1  # Initial temperature
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Random flip
        idx = rng.randint(n)
        new_signs = current.copy()
        new_signs[idx] *= -1
        
        new_score = _peak_sidelobe(new_signs)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = new_signs
            current_score = new_score
            if current_score < best_score:
                best = current.copy()
                best_score = current_score
        
        T *= cooling
        if T < 1e-6:
            break
    
    return best, best_score


def _greedy_polish(signed, max_passes=20):
    """Greedy local search: flip each position if it improves peak sidelobe."""
    n = len(signed)
    current = signed.copy()
    current_score = _peak_sidelobe(current)
    best = current.copy()
    best_score = current_score
    
    for _ in range(max_passes):
        improved = False
        for idx in range(n):
            if time.time() > deadline:
                break
            
            new_signs = current.copy()
            new_signs[idx] *= -1
            new_score = _peak_sidelobe(new_signs)
            
            if new_score < current_score:
                current = new_signs
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best = current.copy()
                    best_score = current_score
        
        if not improved:
            break
    
    return best, best_score


def construct():
    global deadline
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = np.random.RandomState(42)
    
    best_overall = None
    best_overall_score = np.inf
    
    # Multiple restarts with different harmonic boundary initializations
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Get harmonic boundary initialization
        signed = _harmonic_boundary_init(N, rng)
        
        # Anneal from this initialization
        annealed, anneal_score = _anneal_from_harmonic(signed, rng, max_iters=3000)
        
        # Greedy polish
        polished, polish_score = _greedy_polish(annealed, max_passes=15)
        
        if polish_score < best_overall_score:
            best_overall = polished
            best_overall_score = polish_score
        
        # If we're doing well, continue
        if time.time() > deadline - 10:
            break
    
    # Final greedy polish on the best result
    if best_overall is not None:
        final, final_score = _greedy_polish(best_overall, max_passes=30)
        if final_score < best_overall_score:
            best_overall = final
    
    if best_overall is None:
        # Fallback: all ones
        best_overall = np.ones(N)
    
    return best_overall
# EVOLVE-BLOCK-END
