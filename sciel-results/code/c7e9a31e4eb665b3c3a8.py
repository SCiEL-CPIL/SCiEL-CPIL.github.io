"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def _overlap_max(h, N):
    """Compute max over shifts of sum_i h[i] * (1 - h[(i+t) % N]) / N."""
    h = np.asarray(h, dtype=np.float64)
    comp = 1.0 - h
    H = np.fft.fft(h)
    C = np.fft.fft(comp)
    overlap = np.real(np.fft.ifft(np.conj(H) * C))
    return np.max(overlap)


def construct():
    N = 100

    def project(h):
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        h += diff / N
        for _ in range(50):
            h = np.clip(h, 0.0, 1.0)
            diff = N / 2.0 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        return h

    # Try a family of structured patterns and optimize each with projected gradient descent
    # The key insight: we want h[i] * (1-h[j]) small when i and j are close in the circular sense
    # for all shifts t. This suggests h should be "spread out" - no large contiguous blocks.
    
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # Generate diverse initial patterns
    initial_patterns = []
    
    # Pattern 1: Alternating high/low
    h1 = np.zeros(N)
    h1[::2] = 0.9
    h1[1::2] = 0.1
    initial_patterns.append(h1)
    
    # Pattern 2: Three-level pattern
    h2 = np.zeros(N)
    for i in range(N):
        if i % 3 == 0:
            h2[i] = 0.8
        elif i % 3 == 1:
            h2[i] = 0.2
        else:
            h2[i] = 0.5
    initial_patterns.append(h2)
    
    # Pattern 3: Smooth multi-harmonic
    h3 = np.full(N, 0.5)
    for k in [1, 2, 3, 5, 7]:
        amp = 0.12 / np.sqrt(k)
        phase = np.random.uniform(0, 2 * np.pi)
        h3 += amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
    initial_patterns.append(h3)
    
    # Pattern 4: Random binary-ish
    h4 = np.random.choice([0.0, 1.0], size=N)
    initial_patterns.append(h4)
    
    # Pattern 5: Golden-ratio spacing
    phi = (1 + np.sqrt(5)) / 2
    h5 = np.zeros(N)
    for i in range(N):
        frac = (i * phi) % 1.0
        h5[i] = 1.0 if frac < 0.5 else 0.0
    initial_patterns.append(h5)
    
    # Pattern 6: Block pattern
    h6 = np.zeros(N)
    for i in range(N):
        block = i // 10
        if block % 2 == 0:
            h6[i] = 0.7
        else:
            h6[i] = 0.3
    initial_patterns.append(h6)
    
    # Pattern 7: Sine wave
    h7 = 0.5 + 0.4 * np.sin(2 * np.pi * 5 * np.arange(N) / N)
    initial_patterns.append(h7)
    
    # Pattern 8: Two-scale pattern
    h8 = np.zeros(N)
    for i in range(N):
        h8[i] = 0.5 + 0.3 * np.sin(2 * np.pi * i / N) + 0.15 * np.sin(2 * np.pi * 7 * i / N)
    initial_patterns.append(h8)
    
    for h_init in initial_patterns:
        h = project(h_init.copy())
        
        # Projected gradient descent with subgradients
        def get_grad(h, top_k=3):
            comp = 1.0 - h
            H = np.fft.fft(h)
            C = np.fft.fft(comp)
            overlap = np.real(np.fft.ifft(np.conj(H) * C))
            
            idx = np.argsort(overlap)[-top_k:]
            weights = np.exp((overlap[idx] - np.max(overlap)) * 10)
            weights /= np.sum(weights)
            
            grad = np.zeros(N)
            for w, t in zip(weights, idx):
                grad += w * (1.0 - h[(np.arange(N) + t) % N] - h[(np.arange(N) - t) % N])
            return grad, np.max(overlap)
        
        lr = 0.15
        for step in range(3000):
            h = project(h)
            grad, curr_score = get_grad(h, top_k=3)
            h = project(h - lr * grad)
            
            if step % 500 == 0:
                lr *= 0.7
            
            if curr_score < best_score:
                best_score = curr_score
                best_h = h.copy()
        
        # Final check
        h_final = project(h)
        score_final = _overlap_max(h_final, N)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final.copy()
    
    # Additional refinement: local search on the best solution
    h = project(best_h.copy())
    lr = 0.05
    for step in range(2000):
        h = project(h)
        comp = 1.0 - h
        H = np.fft.fft(h)
        C = np.fft.fft(comp)
        overlap = np.real(np.fft.ifft(np.conj(H) * C))
        
        idx = np.argsort(overlap)[-3:]
        weights = np.exp((overlap[idx] - np.max(overlap)) * 10)
        weights /= np.sum(weights)
        
        grad = np.zeros(N)
        for w, t in zip(weights, idx):
            grad += w * (1.0 - h[(np.arange(N) + t) % N] - h[(np.arange(N) - t) % N])
        
        h = project(h - lr * grad)
        
        if step % 400 == 0:
            lr *= 0.7
        
        curr_score = _overlap_max(h, N)
        if curr_score < best_score:
            best_score = curr_score
            best_h = h.copy()
    
    h = project(best_h)
    # Ensure exact constraints
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    return h
# EVOLVE-BLOCK-END
