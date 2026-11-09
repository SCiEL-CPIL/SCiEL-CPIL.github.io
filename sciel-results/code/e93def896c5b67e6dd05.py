import math
import numpy as np

def find_root(f, x0):
    x0 = float(x0)
    
    # Step 1: Sample f at points around x0 to characterize oscillation
    # Use a window that's likely to contain a zero
    # Start with a modest window
    window = max(1.0, abs(x0) * 0.1)
    
    # Sample at N points
    N = 16
    xs = x0 + window * np.linspace(-1, 1, N)
    ys = np.array([f(xi) for xi in xs])
    
    # Check if any sample is already close to zero
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        if abs(yi) <= 1e-8:
            return float(xi)
    
    # Step 2: Fit a sinusoid y = A*cos(w*x + phi) + B using least squares
    # Use DFT to estimate dominant frequency
    # Compute DFT
    dft = np.fft.fft(ys)
    freqs = np.fft.fftfreq(N, d=xs[1]-xs[0])
    
    # Find dominant frequency (exclude DC)
    magnitudes = np.abs(dft)
    # Exclude index 0 (DC)
    magnitudes[0] = 0
    # Use positive frequencies
    pos_mask = freqs > 0
    if not np.any(pos_mask):
        # Fallback: bisection
        return _bisect(f, xs, ys, x0)
    
    # Find peak in positive frequencies
    pos_mags = np.where(pos_mask, magnitudes, 0)
    peak_idx = np.argmax(pos_mags)
    w_est = freqs[peak_idx] * 2 * np.pi  # angular frequency
    
    if abs(w_est) < 1e-12:
        return _bisect(f, xs, ys, x0)
    
    # Fit A, B, phi using linear least squares
    # y = A*cos(w*x) - A*sin(w)*sin(phi) + A*sin(w)*cos(phi) + B
    # Let c = A*cos(phi), s = A*sin(phi)
    # y = c*cos(w*x) - s*sin(w*x) + B
    X = np.column_stack([np.cos(w_est * xs), -np.sin(w_est * xs), np.ones(N)])
    coeffs, _, _, _ = np.linalg.lstsq(X, ys, rcond=None)
    c, s, B = coeffs
    A = math.sqrt(c**2 + s**2)
    phi = math.atan2(-s, c)  # since y = A*cos(w*x + phi) = A*cos(w*x)*cos(phi) - A*sin(w*x)*sin(phi)
    # Actually: A*cos(w*x + phi) = A*cos(w*x)*cos(phi) - A*sin(w*x)*sin(phi)
    # So c = A*cos(phi), -s = -A*sin(phi) => s = A*sin(phi)
    # phi = atan2(s, c)
    phi = math.atan2(s, c)
    
    # Step 3: Find nearest zero of A*cos(w*x + phi) + B
    # Zero when cos(w*x + phi) = -B/A
    if abs(A) < 1e-15:
        return _bisect(f, xs, ys, x0)
    
    ratio = -B / A
    if ratio > 1.0 or ratio < -1.0:
        # No real zero from this fit; use bisection on samples
        return _bisect(f, xs, ys, x0)
    
    # cos(theta) = ratio => theta = ±arccos(ratio) + 2*k*pi
    theta0 = math.acos(ratio)
    
    # w*x + phi = ±theta0 + 2*k*pi
    # x = (±theta0 - phi + 2*k*pi) / w
    # Find k such that x is nearest to x0
    
    best_x = None
    best_dist = float('inf')
    
    for sign in [1, -1]:
        for k in range(-5, 6):
            x_candidate = (sign * theta0 - phi + 2 * k * math.pi) / w_est
            dist = abs(x_candidate - x0)
            if dist < best_dist:
                best_dist = dist
                best_x = x_candidate
    
    if best_x is None:
        return _bisect(f, xs, ys, x0)
    
    # Step 4: Polish with Newton's method (secant variant to save calls)
    x = float(best_x)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Use secant method with one more point
    h = 1e-6 * max(1.0, abs(x))
    x2 = x + h
    fx2 = f(x2)
    if abs(fx2) <= 1e-8:
        return x2
    
    # Secant iterations
    for _ in range(10):
        if fx2 == fx:
            break
        x_new = x2 - fx2 * (x2 - x) / (fx2 - fx)
        x, fx = x2, fx2
        x2 = x_new
        fx2 = f(x2)
        if abs(fx2) <= 1e-8:
            return x2
        if not math.isfinite(x2):
            break
    
    # Return best candidate
    if abs(fx) <= abs(fx2):
        return x
    else:
        return x2

def _bisect(f, xs, ys, x0):
    """Bisection fallback using sampled points."""
    # Find sign change in samples
    for i in range(len(xs) - 1):
        if ys[i] * ys[i+1] < 0:
            a, b = float(xs[i]), float(xs[i+1])
            fa, fb = float(ys[i]), float(ys[i+1])
            for _ in range(60):
                m = 0.5 * (a + b)
                fm = f(m)
                if abs(fm) <= 1e-8:
                    return m
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            return 0.5 * (a + b)
    # No sign change found; return point with smallest |f|
    idx = int(np.argmin(np.abs(ys)))
    return float(xs[idx])
