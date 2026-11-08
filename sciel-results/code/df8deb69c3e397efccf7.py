"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Design a balanced binary sequence h of length N (N/2 ones) to minimize
    max_{k=1..N-1} overlap[k] where overlap[k] = sum_i h[i]*(1-h[(i+k)%N]).
    
    For binary h with sum=N/2: overlap[k] = N/2 - autocorr[k]
    So minimizing max overlap = maximizing min_{k!=0} autocorr[k].
    
    Uses efficient local search with O(N) delta updates per swap.
    """
    np.random.seed(42)
    
    def compute_autocorr(h):
        """Compute circular autocorrelation for all shifts using FFT."""
        n = len(h)
        f = np.fft.fft(h, n=2*n)
        ac = np.real(np.fft.ifft(f * np.conj(f)))[:n]
        return ac
    
    def get_objective(autocorr):
        """Return max overlap = N/2 - min(autocorr[1:])"""
        return N / 2.0 - np.min(autocorr[1:])
    
    def try_move(h, i, j, autocorr):
        """
        Try moving a 1 from position i to position j (h[i]=1, h[j]=0).
        Returns (new_h, new_autocorr, new_obj) if improved, else None.
        
        When we flip h[i]: 1->0 and h[j]: 0->1, the change in autocorr[k] is:
        delta[k] = -h[(i+k)%N] + h[(j+k)%N] for the 1->0 at i
                  + h[(i+k)%N] - h[(j+k)%N] for the 0->1 at j
        Wait, let me think more carefully.
        
        autocorr[k] = sum_t h[t]*h[(t+k)%N]
        
        Change at position i: h[i] goes from 1 to 0.
        This affects terms where t=i: h[i]*h[(i+k)%N] changes from h[(i+k)%N] to 0, delta = -h[(i+k)%N]
        And terms where (t+k)%N = i, i.e., t = (i-k)%N: h[(i-k)%N]*h[i] changes from h[(i-k)%N] to 0, delta = -h[(i-k)%N]
        
        Change at position j: h[j] goes from 0 to 1.
        This affects terms where t=j: h[j]*h[(j+k)%N] changes from 0 to h[(j+k)%N], delta = +h[(j+k)%N]
        And terms where (t+k)%N = j, i.e., t = (j-k)%N: h[(j-k)%N]*h[j] changes from 0 to h[(j-k)%N], delta = +h[(j-k)%N]
        
        So total delta[k] = -h[(i+k)%N] - h[(i-k)%N] + h[(j+k)%N] + h[(j-k)%N]
        """
        if h[i] == h[j]:
            return None
        
        # i has 1, j has 0 (or vice versa); we move 1 from i to j
        # Ensure i is the 1-position and j is the 0-position
        if h[i] == 0 and h[j] == 1:
            i, j = j, i
        
        # h[i] = 1, h[j] = 0
        # Compute delta for each shift k
        # delta[k] = -h[(i+k)%N] - h[(i-k)%N] + h[(j+k)%N] + h[(j-k)%N]
        
        # Vectorized computation
        k = np.arange(N)
        delta = -h[(i + k) % N] - h[(i - k) % N] + h[(j + k) % N] + h[(j - k) % N]
        
        new_autocorr = autocorr + delta
        
        # Verify balance is maintained (it is, since we're swapping)
        new_obj = get_objective(new_autocorr)
        old_obj = get_objective(autocorr)
        
        if new_obj < old_obj - 1e-12:
            new_h = h.copy()
            new_h[i] = 0
            new_h[j] = 1
            return (new_h, new_autocorr, new_obj)
        
        return None
    
    # Start with alternating pattern: 1,0,1,0,...
    h = np.zeros(N)
    h[0::2] = 1.0
    
    # Verify balance
    assert np.sum(h) == N / 2, f"Sum is {np.sum(h)}, expected {N/2}"
    
    autocorr = compute_autocorr(h)
    best_h = h.copy()
    best_autocorr = autocorr.copy()
    best_obj = get_objective(autocorr)
    
    # Local search with multiple restarts
    max_iterations = 50000
    
    for restart in range(10):
        if restart == 0:
            h = best_h.copy()
            autocorr = best_autocorr.copy()
        else:
            # Perturb the best solution
            h = best_h.copy()
            # Randomly flip a few pairs
            for _ in range(5):
                ones_pos = np.where(h == 1)[0]
                zeros_pos = np.where(h == 0)[0]
                if len(ones_pos) > 0 and len(zeros_pos) > 0:
                    i = np.random.choice(ones_pos)
                    j = np.random.choice(zeros_pos)
                    h[i], h[j] = h[j], h[i]
            autocorr = compute_autocorr(h)
        
        current_obj = get_objective(autocorr)
        
        for iteration in range(max_iterations):
            improved = False
            # Try all moves
            ones_pos = np.where(h == 1)[0]
            zeros_pos = np.where(h == 0)[0]
            
            # Shuffle for diversity
            np.random.shuffle(ones_pos)
            np.random.shuffle(zeros_pos)
            
            for i in ones_pos:
                for j in zeros_pos:
                    result = try_move(h, i, j, autocorr)
                    if result is not None:
                        h, autocorr, current_obj = result
                        improved = True
                        if current_obj < best_obj:
                            best_obj = current_obj
                            best_h = h.copy()
                            best_autocorr = autocorr.copy()
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    return best_h
# EVOLVE-BLOCK-END
