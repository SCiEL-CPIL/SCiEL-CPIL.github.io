"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _calc_psl(s):
    """Calculate Peak Sidelobe Level (max abs autocorrelation for lags 1..N-1)."""
    # Linear autocorrelation
    # np.correlate(s, s, 'full') returns 2N-1 values
    # Index N-1 is lag 0
    # Indices N to 2N-2 are lags 1 to N-1
    r = np.correlate(s, s, mode='full')
    sidelobes = r[N:2*N-1]
    return np.max(np.abs(sidelobes))

def _generate_chirp(n):
    """Generate a binary chirp sequence."""
    t = np.arange(n)
    # Quadratic phase: sin(pi * t^2 / n)
    # This creates a frequency sweep
    return np.sign(np.sin(np.pi * t**2 / n)).astype(int)

def _generate_mseq_like(n):
    """
    Generate a sequence resembling an m-sequence or just a good pseudo-random.
    For n=400, we can't use a perfect m-seq (len 511 or 255).
    We can truncate a 511 m-seq.
    """
    # Generate 511 length m-sequence
    # Polynomial x^9 + x^4 + 1 (0x011 in binary 10001, but taps at 9,4,0)
    # Standard LFSR implementation
    def gen_mseq_len(target_len):
        # Find a primitive polynomial for degree 9
        # x^9 + x^4 + 1 is primitive
        # Taps: 9, 4, 0 (1-based indexing for feedback)
        # Register size 9
        reg = [1] + [0]*8 
        seq = []
        for _ in range(target_len + 10):
            # Feedback
            # Bit to shift out is reg[0]
            # New bit is XOR of taps
            # Taps at 9 (reg[8]) and 4 (reg[3])? 
            # Convention varies. Let's use a simple LFSR.
            # Polynomial 0x11 (17) for degree 4? No, we need degree 9.
            # Let's just use a high-quality random sequence as a fallback if m-seq is complex.
            # Actually, a truncated m-sequence is known to be good.
            # Let's use a simpler approach: 
            # The "Gold Code" or just random is often sufficient for this score gap.
            # Given the risk of LFSR implementation errors, I will rely on Chirp and Random + SA.
            pass
        return np.random.choice([-1, 1], target_len)

    # Fallback to random if m-seq logic is too risky
    return np.random.choice([-1, 1], n)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin for safety

    # 1. Generate Initial Candidates
    candidates = []
    
    # Seed (Baseline)
    candidates.append(np.ones(N, dtype=int))
    
    # Random Candidates
    for _ in range(10):
        candidates.append(np.random.choice([-1, 1], N).astype(int))
        
    # Chirp Candidate
    candidates.append(_generate_chirp(N))
    
    # Evaluate Initial Candidates
    best_s = None
    best_psl = np.inf
    
    for s in candidates:
        psl = _calc_psl(s)
        if psl < best_psl:
            best_psl = psl
            best_s = s.copy()
            
    # 2. Simulated Annealing / Local Search
    # We will use a hybrid: Greedy with occasional random restarts or SA.
    # Given N=400, _calc_psl is fast.
    
    current_s = best_s
    current_psl = best_psl
    
    # Temperature for SA
    # Initial temp should be comparable to typical delta PSL
    # PSL range is roughly 10 to 400.
    T = 10.0
    cooling_rate = 0.99
    
    # Pre-allocate
    s_copy = current_s.copy()
    
    while time.time() < deadline:
        # Pick a random bit to flip
        i = np.random.randint(N)
        
        # Flip
        s_copy[i] = -s_copy[i]
        
        # Calculate new PSL
        new_psl = _calc_psl(s_copy)
        
        delta = new_psl - current_psl
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_s = s_copy.copy()
            current_psl = new_psl
        else:
            # Revert
            s_copy[i] = -s_copy[i]
            
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1 # Floor temperature to allow exploration
            
        # Optional: Reset temperature if stuck?
        # Keep it simple for now.
        
    return current_s.astype(float)

# EVOLVE-BLOCK-END
