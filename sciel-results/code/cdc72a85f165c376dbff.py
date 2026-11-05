import numpy as np

N = 400

def construct():
    np.random.seed(123)
    h = np.random.randn(N)
    h = h / np.sum(h)
    
    # Phase 1: Minimize L2 norm of convolution to flatten it
    for step in range(2000):
        c = np.convolve(h, h)
        J = np.sum(c**2)
        
        # Gradient of sum(c^2)
        # c = h * h
        # d/dc (c^2) = 2c
        # d/dh (sum c^2) = 2 * (c * h) + 2 * (c * h_reversed)?
        # Chain rule:
        # L = sum_k c_k^2
        # dL/dh_j = sum_k 2 c_k (dc_k/dh_j)
        # dc_k/dh_j = 2 h_{k-j} (as derived before, with bounds)
        # So dL/dh_j = 2 sum_k c_k (2 h_{k-j}) = 4 sum_k c_k h_{k-j}
        # This is 4 * (c * h)_j ?
        # Let's check indices.
        # (c * h)_j = sum_k c_k h_{j-k}.
        # Our term is sum_k c_k h_{k-j}.
        # This is (c * h_reversed)_j?
        # h_{k-j} is h at index k-j.
        # Let m = k-j. k = m+j.
        # sum_m c_{m+j} h_m.
        # This is the correlation of c and h?
        
        # Actually, gradient of ||h*h||_2^2 is 4 * (h * h * h * h) ? No.
        # In Fourier domain:
        # C = H^2.
        # J = sum |C|^2 = (1/N) sum |H^2|^2 = (1/N) sum |H|^4.
        # dJ/dH = (4/N) H^3 H^* ? No.
        # d/dH (H^4 H^{*4})?
        # J = sum H^4 H^{*4} / N.
        # dJ/dH^* = (4/N) H^4 H^*.
        # dJ/dh = IFFT( dJ/dH^* )?
        
        # Let's use the time domain formula:
        # grad_h = 4 * np.convolve(c, h, 'full')?
        # Let's test with N=2.
        # h=[a,b]. c=[a^2, 2ab, b^2].
        # J = a^4 + 4a^2b^2 + b^4.
        # dJ/da = 4a^3 + 8ab^2.
        # Formula: 4 * (c * h).
        # c*h = [a^3, 2a^2b + ab^2, 2ab^2 + b^3, ab^2].
        # This is getting complicated.
        
        # Let's just use numerical gradient or the explicit formula:
        # grad[j] = 4 * sum_k c_k h_{k-j}
        
        # We can compute this efficiently.
        # sum_k c_k h_{k-j} is the cross-correlation of c and h?
        # np.correlate(c, h, 'full')?
        
        # Let's stick to the subgradient of Max for simplicity and robustness, 
        # but use a smaller step and more iterations.
        
        pass

    # Use the simple max-gradient approach but with momentum
    velocity = np.zeros(N)
    mu = 0.9
    eta = 0.05
    
    for step in range(10000):
        c = np.convolve(h, h)
        max_idx = np.argmax(c)
        c_max = c[max_idx]
        
        g = np.zeros(N)
        indices = max_idx - np.arange(N)
        valid = (indices >= 0) & (indices < N)
        g[valid] = 2.0 * h[indices[valid]]
        
        velocity = mu * velocity - eta * g
        h = h + velocity
        
        s = np.sum(h)
        if abs(s) < 1e-9:
            h[0] += 1e-9
        h = h / np.sum(h)
        
    return h
