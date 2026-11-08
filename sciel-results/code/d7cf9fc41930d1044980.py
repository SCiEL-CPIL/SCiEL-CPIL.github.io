import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    N = 200  # number of steps
    
    def compute_overlaps(h):
        """Compute overlap O(t) = sum_i h[i]*(1-h[i+t]) for all valid shifts t >= 0,
        using linear (non-periodic) interpretation where h is 0 outside [0, N-1]."""
        overlaps = []
        for t in range(N):
            # h[i] for i in [0, N-1-t], h[i+t] for i in [0, N-1-t]
            # But 1-h[i+t] where h[i+t] is the shifted version
            # O(t) = sum_{i=0}^{N-1-t} h[i] * (1 - h[i+t])
            # But also need to consider that 1-h is defined on [0,2]
            # Actually for linear: O(t) = sum_i h[i]*(1-h[i+t]) where both are in range
            # For i+t >= N, h[i+t] = 0, so 1-h[i+t] = 1
            # So O(t) = sum_{i=0}^{N-1-t} h[i]*(1-h[i+t]) + sum_{i=N-t}^{N-1} h[i]*1
            if t == 0:
                o = np.sum(h * (1.0 - h))
            else:
                part1 = np.sum(h[:N-t] * (1.0 - h[t:]))
                part2 = np.sum(h[N-t:])  # these have h[i+t]=0, so 1-h[i+t]=1
                o = part1 + part2
            overlaps.append(o)
        return np.array(overlaps)
    
    def max_overlap(h):
        overlaps = compute_overlaps(h)
        return np.max(overlaps)
    
    # Initial: constant 0.5
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = max_overlap(best_h)
    
    # Try various structured patterns
    # 1. Low-frequency cosine
    for freq in [1, 2, 3]:
        for amp in [0.1, 0.2, 0.3, 0.4]:
            t = np.arange(N)
            h_trial = 0.5 + amp * np.cos(2 * np.pi * freq * t / N)
            h_trial = np.clip(h_trial, 0, 1)
            # Normalize to have sum = N/2
            s = np.sum(h_trial)
            if s > 0:
                h_trial = h_trial * (N / 2.0) / s
                h_trial = np.clip(h_trial, 0, 1)
            score = max_overlap(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    # 2. Two-level pattern: alternate between a and b
    for a in np.linspace(0, 1, 21):
        b = (N/2.0 - a * (N//2)) / (N - N//2) if N - N//2 > 0 else 0
        if 0 <= b <= 1:
            h_trial = np.zeros(N)
            h_trial[0::2] = a
            h_trial[1::2] = b
            # Adjust to get exact sum
            s = np.sum(h_trial)
            target = N / 2.0
            if abs(s) > 1e-10:
                h_trial = h_trial + (target - s) / N
                h_trial = np.clip(h_trial, 0, 1)
            score = max_overlap(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    # 3. Block pattern: high in first half, low in second
    for split in range(1, N):
        for val_high in np.linspace(0.5, 1.0, 11):
            total_high = val_high * split
            remaining = N/2.0 - total_high
            if remaining < 0 or remaining > (N - split) * 1.0:
                continue
            val_low = remaining / (N - split)
            if val_low < 0 or val_low > 1:
                continue
            h_trial = np.zeros(N)
            h_trial[:split] = val_high
            h_trial[split:] = val_low
            score = max_overlap(h_trial)
            if score < best_score:
                best_score = score
                best_h = h_trial.copy()
    
    # 4. Gradient-based refinement on best_h
    if time.time() < deadline - 30:
        h_curr = best_h.copy()
        lr = 0.01
        for iteration in range(500):
            if time.time() > deadline - 10:
                break
            
            overlaps = compute_overlaps(h_curr)
            max_ov = np.max(overlaps)
            
            # Compute gradient of max overlap w.r.t. h
            # For simplicity, use subgradient: gradient of the max term
            t_max = np.argmax(overlaps)
            if t_max == 0:
                # O(0) = sum h[i](1-h[i]) = sum h[i] - sum h[i]^2
                # dO/dh[i] = 1 - 2*h[i]
                grad = 1.0 - 2.0 * h_curr
            else:
                # O(t) = sum_{i=0}^{N-1-t} h[i]*(1-h[i+t]) + sum_{i=N-t}^{N-1} h[i]
                # dO/dh[i] for i < N-t: (1-h[i+t]) - h[i] = 1 - h[i+t] - h[i]
                # dO/dh[i] for i >= N-t: 1
                grad = np.zeros(N)
                for i in range(N - t_max):
                    grad[i] = 1.0 - h_curr[i + t_max] - h_curr[i]
                for i in range(N - t_max, N):
                    grad[i] = 1.0
            
            # Project gradient step to stay in [0,1] and maintain sum = N/2
            h_new = h_curr - lr * grad
            h_new = np.clip(h_new, 0, 1)
            # Adjust sum
            s = np.sum(h_new)
            target = N / 2.0
            if abs(s - target) > 1e-10:
                # Simple correction
                h_new = h_new + (target - s) / N
                h_new = np.clip(h_new, 0, 1)
            
            new_score = max_overlap(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
                h_curr = h_new
            else:
                h_curr = h_curr - 0.5 * lr * grad
                h_curr = np.clip(h_curr, 0, 1)
                s = np.sum(h_curr)
                if abs(s - target) > 1e-10:
                    h_curr = h_curr + (target - s) / N
                    h_curr = np.clip(h_curr, 0, 1)
    
    # Final check and return
    final_score = max_overlap(best_h)
    if final_score > best_score:
        # Shouldn't happen, but just in case
        best_score = final_score
    
    return best_h

# EVOLVE-BLOCK-END
