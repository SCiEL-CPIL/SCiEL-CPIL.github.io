"""Program for the third autocorrelation inequality task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400
DOMAIN = [-0.25, 0.25]
DX = (DOMAIN[1] - DOMAIN[0]) / N
GRID = np.linspace(DOMAIN[0], DOMAIN[1], N + 1)[:-1] + DX / 2


def _autoconvolution_peak(x):
    """
    Compute the peak of the autoconvolution of the step function defined by heights x.
    The step function is f(t) = x[i] for t in [t_i, t_{i+1}).
    The autoconvolution is (f * f)(s) = ∫ f(t) f(s-t) dt.
    
    Since f is a step function, we can compute the convolution efficiently.
    For a step function with N intervals of width dx, the convolution at shift s 
    can be computed by considering the overlap of the two shifted step functions.
    
    However, a simpler approach for the peak (which occurs at s=0 for symmetric functions,
    but we need the max over all s) is to use FFT.
    
    Let's use FFT to compute the full convolution and then find its maximum.
    """
    # To compute the linear convolution of a signal with itself, we can zero-pad.
    # The length of the convolution of two signals of length N is 2N-1.
    # We need to pad to at least 2N to avoid circular convolution effects.
    # Let's pad to 2*N for simplicity (next power of 2 might be faster, but 2*N is fine).
    
    # Actually, for the step function, the values are constant over intervals.
    # The convolution integral is:
    # (f*f)(s) = sum_{i,j} x[i] x[j] * length(intersection of [t_i, t_{i+1}] and [s-t_j, s-t_{j+1}])
    
    # This is equivalent to the discrete convolution of the sequence x with itself,
    # scaled by dx, but only if we interpret the step function as a sequence of impulses
    # with weights x[i]*dx.
    
    # Let's define a discrete signal y[i] = x[i] * dx.
    # Then the continuous autoconvolution at the grid points (shifts that are multiples of dx)
    # is approximately the discrete convolution of y with y.
    
    # More precisely, if we define g[k] = x[k] for k=0..N-1,
    # then the continuous function f(t) is g[k] for t in [k*dx, (k+1)*dx).
    # The convolution (f*f)(s) for s = m*dx (integer m) is:
    # sum_{k} g[k] * (integral of f(s-t) over t in [k*dx, (k+1)*dx))
    # = sum_{k} g[k] * dx * g[m-k]  (if m-k is in [0, N-1])
    
    # So the discrete convolution of g with g, scaled by dx, gives the values at s=m*dx.
    # The peak will be at some s, which might not be exactly on the grid, but for step functions,
    # the maximum of the convolution is typically at a grid point or near it.
    # Actually, the convolution of two step functions is a piecewise linear function.
    # Its maximum occurs at a grid point (shift that is a multiple of dx) if the steps are aligned.
    
    # Let's compute the full linear convolution using FFT.
    n_fft = 2 * N
    X = np.fft.rfft(x, n_fft)
    conv = np.fft.irfft(X * X, n_fft)
    
    # The convolution result conv[m] corresponds to the sum x[i]*x[j] where i+j = m.
    # This is the unnormalized discrete convolution.
    # The actual continuous autoconvolution at shift s = m*dx is conv[m] * dx.
    # Wait, let's re-derive.
    # (f*f)(s) = ∫ f(t) f(s-t) dt
    # Let t = k*dx + u, where u in [0, dx).
    # Then f(t) = x[k].
    # f(s-t) = f(m*dx - k*dx - u) = f((m-k)*dx - u).
    # If m-k is an integer j, then f(j*dx - u) = x[j-1] for u in (0, dx) and x[j] for u=0.
    # This is getting complicated. The standard result is that the convolution of two step functions
    # with step size dx is the discrete convolution of the step heights, multiplied by dx.
    
    # So, peak = max(conv) * dx.
    # But we need to be careful with the scaling.
    
    # Let's test with a simple case.
    # If x = [1, 1, 1, 1] (N=4), dx = 1.
    # f(t) = 1 for t in [0, 4).
    # (f*f)(s) = length of intersection of [0,4] and [s, s+4].
    # For s=0, intersection is [0,4], length 4.
    # Discrete convolution of [1,1,1,1] with itself:
    # [1, 2, 3, 4, 3, 2, 1]
    # max is 4. 4 * dx = 4 * 1 = 4. Correct.
    
    # So the formula is: peak = max(np.fft.irfft(np.fft.rfft(x, n_fft)**2, n_fft)) * dx
    
    peak = np.max(conv) * DX
    return peak


def _score(x):
    """Compute the score (upper bound on C3). Lower is better."""
    # The task says "score is the upper bound on C3".
    # The autoconvolution peak is a key component.
    # Let's assume the score is simply the peak of the autoconvolution.
    return _autoconvolution_peak(x)


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4]
    to minimize the autoconvolution peak.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for the 150s limit
    
    # Initial guess: alternating +1 and -1 to break symmetry
    x0 = np.ones(N)
    x0[::2] = -1.0
    
    best_x = x0.copy()
    best_score = _score(best_x)
    
    # Use L-BFGS-B to minimize the autoconvolution peak
    # We treat the step heights as continuous variables first, then round to ±1
    
    def objective(x):
        return _autoconvolution_peak(x)
    
    def gradient(x):
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(N):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad
    
    # Try a few different initial conditions
    initial_guesses = [
        np.ones(N),
        -np.ones(N),
        np.ones(N) * np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)),
        np.ones(N) * np.sin(np.linspace(0, 2*np.pi, N, endpoint=False)),
        np.random.choice([-1.0, 1.0], N),
    ]
    
    for x_init in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            result = minimize(
                objective,
                x_init,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Round to nearest sign
            x_rounded = np.sign(result.x)
            # Handle zeros by keeping the sign from the continuous solution
            x_rounded[result.x == 0] = 1.0
            
            score_rounded = _score(x_rounded)
            
            if score_rounded < best_score:
                best_score = score_rounded
                best_x = x_rounded.copy()
                
        except Exception:
            continue
    
    return best_x


# Test the construct function
if __name__ == "__main__":
    x = construct()
    print(f"Score: {_score(x)}")
