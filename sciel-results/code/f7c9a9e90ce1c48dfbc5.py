"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Initialize with random signs
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Local search to minimize peak sidelobe
    # We minimize max_{k != 0} |c_k|
    best_x = x.copy()
    best_val = np.inf
    
    # Calculate initial metric
    c = np.convolve(best_x, best_x, mode='full')
    # c is length 2N-1. Center is N-1.
    center = N - 1
    # Sidelobes are c[0:center] and c[center+1:]
    # We want to minimize the max absolute value of these
    sid = np.concatenate([c[:center], c[center+1:]])
    best_val = np.max(np.abs(sid))
    
    for _ in range(50): # Iterations
        improved = False
        for i in range(N):
            # Flip x[i]
            x[i] *= -1
            
            # Update autocorrelation efficiently? 
            # For N=400, full recompute is 400*800 ops = 320k ops. 
            # 50 * 400 * 320k = 6.4 billion ops. Too slow in Python.
            # Need faster update or fewer iterations.
            # Let's just do a few random flips or use FFT.
            pass
        
        # Let's use a simpler approach: Random restarts + few steps
        break

    # Better approach: Use FFT based gradient or just return a known good pattern?
    # Since N=400 is not a standard Barker length (max 13), we need to generate.
    # Let's do a simple random search with local improvement using efficient correlation update?
    # Or just return a sequence that is "good enough".
    
    # A simpler heuristic: 
    # Start with all 1s? No, high sidelobes.
    # Start with random.
    
    # Let's try 100 random restarts, pick the one with lowest peak sidelobe.
    # Computing correlation for N=400 is fast with np.correlate or FFT.
    # np.correlate is O(N^2). 400^2 = 160,000.
    # 100 restarts * 160,000 = 16,000,000 ops. Very fast.
    
    best_x = None
    best_score = np.inf
    
    for _ in range(100):
        x = np.random.choice([-1.0, 1.0], size=N)
        c = np.correlate(x, x, mode='full')
        center = N - 1
        sid = np.concatenate([c[:center], c[center+1:]])
        score = np.max(np.abs(sid))
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
            
    return best_x
# EVOLVE-BLOCK-END
