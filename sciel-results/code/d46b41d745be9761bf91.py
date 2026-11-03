"""Combined program: Simulated Annealing with Block Moves and Precomputed FFT Basis."""
import numpy as np
import time
import random

N = 400
# FFT size must be at least 2*N - 1
_M = 1
while _M < 2 * N - 1:
    _M *= 2

# Precompute FFT basis vectors for fast spectrum updates
# basis[i] is the vector [exp(-2j*pi*k*i/_M) for k in 0.._M//2]
_FREQS = np.arange(_M // 2 + 1)
_BASIS = np.exp(-2j * np.pi * np.outer(_FREQS, np.arange(N)) / _M)
# _BASIS shape: (_M//2+1, N)

def _compute_autoconv_from_spectrum(H):
    """
    Compute linear autoconvolution from a pre-computed rfft spectrum H.
    H should be of length _M//2 + 1.
    """
    # H * H is the power spectrum
    # irfft returns the full convolution
    ac = np.fft.irfft(H * H, _M)
    return ac[:2 * N - 1]

def _score_from_spectrum(H, s):
    """
    Compute score from spectrum H and sum s.
    """
    if abs(s) < 1e-9:
        return 1e10
    ac = _compute_autoconv_from_spectrum(H)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

def _get_spectrum(h):
    """Compute rfft of h."""
    return np.fft.rfft(h, _M)

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # 5s margin for safety
    
    best_h = np.ones(N)
    best_score = np.inf
    best_s = np.sum(best_h)
    best_H = _get_spectrum(best_h)
    best_score = _score_from_spectrum(best_H, best_s)
    
    # Helper to update spectrum and sum when flipping a block
    def flip_block(h, H, s, start, length):
        """
        Flip signs of h[start:start+length].
        Returns new h, H, s.
        Note: This modifies h in place for speed, but we need to be careful with backtracking.
        For SA, we usually copy or track changes. Given N=400, copying h is cheap.
        """
        # Indices to flip
        idxs = np.arange(start, start + length)
        # The change in h is -2 * h[idxs]
        # The change in H is sum( delta_h * basis[:, idxs] )
        # delta_h = -2 * h[idxs]
        
        # Vectorized update for H
        # _BASIS[:, idxs] has shape (_M//2+1, length)
        # h[idxs] has shape (length,)
        # We need sum over length: _BASIS[:, idxs] @ (-2 * h[idxs])
        delta_H = _BASIS[:, idxs] @ (-2.0 * h[idxs])
        
        H_new = H + delta_H
        s_new = s - 4.0 * np.sum(h[idxs])
        
        # Update h
        h[idxs] = -h[idxs]
        
        return h, H_new, s_new

    # Simulated Annealing Parameters
    # We use multiple restarts with different initializations
    inits = []
    x = np.arange(N)
    
    # 1. Random signs
    rng = np.random.default_rng(42)
    for _ in range(10):
        inits.append(rng.choice([1.0, -1.0], N))
        
    # 2. Structured inits (to break symmetry)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(2 * np.pi * x / N))
    inits.append(np.sign(np.sin(5 * np.pi * x / N)))
    
    # 3. Gaussian-ish (though we flip signs, the magnitude doesn't matter for the final binary output, 
    # but starting from smooth functions might lead to better local minima structure)
    # Actually, since we are flipping signs, the initial magnitudes don't matter, only the signs.
    # So we just need diverse sign patterns.
    
    # Add some random block patterns
    for _ in range(5):
        h_pat = np.ones(N)
        for _ in range(N // 10):
            start = rng.integers(0, N)
            length = rng.integers(1, 20)
            h_pat[start:start+length] = -1
        inits.append(h_pat)

    # Annealing Schedule
    # We want to spend most of the time here.
    # Total time ~ 140s.
    # Let's do 5 restarts, each getting ~25s.
    
    T_start = 1.0
    T_end = 1e-6
    # Cooling rate alpha such that T_end = T_start * alpha^steps
    # We don't know steps exactly, so we use time-based cooling or fixed steps.
    # Fixed steps is easier to control.
    # Let's estimate steps: 1 step ~ 0.1ms? 
    # _BASIS @ h is (256, 400) dot (400,) -> 256*400 mults ~ 100k ops. Very fast.
    # IFFT is 512 log 512 ~ 4600 ops.
    # So one step is < 0.1ms. 
    # 1000 steps = 0.1s.
    # We can do ~100,000 steps in 10s.
    
    steps_per_restart = 20000
    alpha = (T_end / T_start) ** (1.0 / steps_per_restart)
    
    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        h = h_init.copy()
        h = np.sign(h)
        h[h == 0] = 1.0
        s = np.sum(h)
        H = _get_spectrum(h)
        current_score = _score_from_spectrum(H, s)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            best_s = s
            best_H = H.copy()

        T = T_start
        no_improve_steps = 0
        max_no_improve = 2000 # Restart local search if stuck for too long? Or just keep going.
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            # Choose move: Flip a random block
            # Block size distribution: mostly small, some large
            if random.random() < 0.8:
                length = random.choice([1, 2, 3, 5, 7])
            else:
                length = random.randint(10, 50)
            
            start = random.randint(0, N - length)
            
            # Perform flip
            h_new, H_new, s_new = flip_block(h, H, s, start, length)
            
            # Calculate new score
            new_score = _score_from_spectrum(H_new, s_new)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                h = h_new
                H = H_new
                s = s_new
                current_score = new_score
                no_improve_steps = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
                    best_s = s
                    best_H = H.copy()
            else:
                # Reject: revert
                # We modified h in place in flip_block, so we must revert
                idxs = np.arange(start, start + length)
                h[idxs] = -h[idxs]
                # H and s are not modified in place in flip_block (it returns new arrays/vars), 
                # but wait, flip_block returns new H and s. 
                # However, it DID modify h in place.
                # So we just reverted h. H and s are still the old ones because we didn't assign them back.
                # Correct.
            
            # Cool down
            T *= alpha
            
            # Optional: Restart temperature if stuck?
            # If no improvement for a long time, we might be stuck in a local min.
            # But SA is designed to escape. Let's keep it simple.

    # Final polish: Greedy local search with single bit flips (fast)
    if time.time() < deadline - 2:
        h = best_h.copy()
        s = best_s
        H = best_H.copy()
        current_score = best_score
        
        # Greedy: try flipping each bit once
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                # Flip bit i
                h[i] = -h[i]
                # Update H
                delta_H = _BASIS[:, i] * (-2.0 * h[i]) # Note: h[i] is already flipped, so -2 * new_val is -2 * (-old_val) = 2*old_val. 
                # Wait. 
                # old_h[i] = v. new_h[i] = -v.
                # delta_h = -2v.
                # In the code above: h[i] is -v. 
                # delta_H should be -2 * v * basis.
                # My flip_block logic: delta_H = basis @ (-2 * h[idxs]) where h[idxs] is the OLD value?
                # Let's re-verify flip_block.
                # In flip_block:
                #   idxs = ...
                #   delta_H = _BASIS[:, idxs] @ (-2.0 * h[idxs])
                #   h[idxs] = -h[idxs]
                # So h[idxs] inside the dot product is the OLD value. Correct.
                
                # In this manual loop:
                # We already did h[i] = -h[i]. So h[i] is NEW value.
                # OLD value was -h[i].
                # delta_H = _BASIS[:, i] * (-2.0 * (-h[i])) = _BASIS[:, i] * (2.0 * h[i])
                
                H_trial = H + _BASIS[:, i] * (2.0 * h[i])
                s_trial = s + 2.0 * h[i] # s_new = s - 2*old = s - 2*(-new) = s + 2*new
                
                if abs(s_trial) < 1e-9:
                    continue
                    
                score_trial = _score_from_spectrum(H_trial, s_trial)
                
                if score_trial < current_score:
                    H = H_trial
                    s = s_trial
                    current_score = score_trial
                    improved = True
                else:
                    # Revert
                    h[i] = -h[i]
                    
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()

    return best_h
