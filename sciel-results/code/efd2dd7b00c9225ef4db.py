<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_all_corrs_fft(h):
    """Compute all circular cross-correlations using FFT.
    
    For circular correlation: R_k = sum_i h[i]*h[(i+k)%n]
    We use FFT: R = ifft(fft(h) * conj(fft(h)))
    """
    n = len(h)
    H = np.fft.fft(h)
    R = np.fft.ifft(H * np.conj(H)).real
    # R[k] = sum_i h[i]*h[(i-k)%n], so we need to flip
    # Actually, let's be careful:
    # corr[k] = sum_i h[i] * h[(i+k) % n]
    # Using FFT: ifft(fft(h) * conj(fft(h))) gives autocorrelation
    # The result at index k is sum_i h[i] * h[(i-k) % n]
    # So we need R_flipped where R_flipped[k] = R[-k % n]
    R = np.fft.ifft(H * np.conj(H)).real
    # R[k] = sum_i h[i] * h[(i-k)%n]
    # We want S[k] = sum_i h[i] * h[(i+k)%n] = R[-k % n]
    S = np.zeros(n)
    S[0] = R[0]
    for k in range(1, n):
        S[k] = R[n-k]
    return S

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s) = ∫ h(x)h(x+s)dx.
    
    We evaluate at s = 2k/n for k = 1, ..., n-1 (shifts within (0,2)).
    R(2k/n) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    
    We want to maximize min_k R(2k/n).
    """
    n = len(h)
    min_corr = float('inf')
    for k in range(1, n):
        corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
        if corr < min_corr:
            min_corr = corr
    return min_corr

def _soft_min_cross_correlation(h, beta=100.0):
    """Compute a smooth approximation of min_k R(2k/n) using log-sum-exp.
    
    soft_min = -log(sum(exp(-beta * R_k))) / beta
    
    This is differentiable and approximates the hard min as beta -> inf.
    """
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    
    # Use log-sum-exp for numerical stability
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min, corrs

def _objective_soft(x, beta=100.0):
    """Objective: minimize 1 - soft_min_corr, approximating max overlap."""
    h = x
    soft_min, _ = _soft_min_cross_correlation(h, beta)
    obj = 1.0 - soft_min
    return obj

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient using soft-min approximation."""
    h = x
    n = len(h)
    
    # Compute all correlations
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    
    # Soft min
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient of soft_min w.r.t. corrs:
    # d(soft_min)/d(corrs_k) = (1/beta) * exp(-beta*(corrs_k - min_val)) / sum_exp
    # But we also need d(corrs_k)/d(h)
    
    # Actually, let's compute gradient numerically or analytically
    # d(soft_min)/d(h) = sum_k [d(soft_min)/d(corrs_k)] * d(corrs_k)/d(h)
    
    # d(soft_min)/d(corrs_k) = -exp(-beta*(corrs_k - min_val)) / (beta * sum_exp)
    # Wait, let me re-derive:
    # soft_min = min_val - (1/beta) * log(sum_j exp(-beta*(corrs_j - min_val)))
    # This is tricky because min_val depends on h too.
    
    # Let's use a cleaner formulation:
    # soft_min = -(1/beta) * log(sum_k exp(-beta * corrs_k))
    # d(soft_min)/d(corrs_k) = exp(-beta*corrs_k) / sum_j exp(-beta*corrs_j)
    
    exp_neg_beta = np.exp(-beta * corrs)
    sum_exp_neg = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp_neg  # These are the softmax weights (inverted)
    
    # d(soft_min)/d(h) = sum_k weights_k * d(corrs_k)/d(h)
    # corrs_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    # d(corrs_k)/d(h[j]) = (2/n) * [h[j+k] if j < n-k] + (2/n) * [h[j-k] if j >= k]
    
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        # For corrs_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
        # d/dh[j] = (2/n) * h[j+k] for j in [0, n-k-1]
        #           + (2/n) * h[j-k] for j in [k, n-1]
        if w != 0:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    # d(obj)/d(h) = -d(soft_min)/d(h) = -grad
    grad_obj = -grad
    
    return obj, grad_obj
=======
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_linear_corrs(h):
    """Compute all linear cross-correlations R_k = sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1..n-1.
    Uses FFT for O(n log n) complexity.
    """
    n = len(h)
    # Zero-pad to length 2n-1 to avoid circular convolution
    size = 2 * n - 1
    H = np.fft.fft(h, size=size)
    # Correlation is ifft(fft(h) * conj(fft(h)))
    # Let's check the indexing.
    # r[k] = sum_i h[i] h[i+k]
    # This is equivalent to cross-correlation.
    # np.correlate(h, h, mode='full') gives correlation.
    # Let's use ifft(fft(h) * conj(fft(h)))
    # The result c[m] = sum_i h[i] h[i-m] ? No.
    # Let's verify with a small example.
    # h = [1, 2, 3]
    # H = fft([1,2,3,0,0])
    # R = ifft(H * conj(H))
    # R[0] = 1+4+9 = 14
    # R[1] = 1*2 + 2*3 = 8 ? Or 2*1 + 3*2 = 8?
    # R[2] = 1*3 = 3
    # R[-1] (i.e., R[4]) = 2*1 + 3*2 = 8
    # R[-2] (i.e., R[3]) = 3*1 = 3
    
    # We want C_k = sum_{i=0}^{n-k-1} h[i] h[i+k] for k=1..n-1
    # This corresponds to R[-k] in the zero-padded FFT result if we define R as autocorrelation.
    # Let's compute R = ifft(fft(h) * conj(fft(h))).real
    # R[0] = sum h[i]^2
    # R[1] = sum h[i] h[i-1] (with wrap) -> with zero padding, R[1] = sum_{i=1}^{n-1} h[i] h[i-1]
    # This is C_1.
    # R[k] = sum_{i=k}^{n-1} h[i] h[i-k] = sum_{j=0}^{n-k-1} h[j+k] h[j] = C_k.
    # So R[k] for k=1..n-1 is exactly what we want, scaled by 1 (if we don't normalize).
    # Wait, FFT normalization: np.fft.ifft includes 1/N factor.
    # So R = ifft(fft(h) * conj(fft(h))).real
    # R[k] = sum_{i} h[i] h[i-k] (circular). With zero padding, it becomes linear.
    # Specifically, R[k] = sum_{i=0}^{n-1} h[i] h[i-k] where h is extended with zeros.
    # For k>0, i-k must be >=0 for non-zero product. So i>=k.
    # R[k] = sum_{i=k}^{n-1} h[i] h[i-k]. Let j = i-k. Sum_{j=0}^{n-k-1} h[j+k] h[j].
    # This is exactly C_k.
    
    H = np.fft.fft(h, size=size)
    R = np.fft.ifft(H * np.conj(H)).real
    # R[1:n] contains C_1, ..., C_{n-1}
    return R[1:n] * (2.0 / n)

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s) = ∫ h(x)h(x+s)dx.
    
    We evaluate at s = 2k/n for k = 1, ..., n-1 (shifts within (0,2)).
    R(2k/n) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    
    We want to maximize min_k R(2k/n).
    """
    n = len(h)
    if n <= 1:
        return 0.0
    corrs = _compute_linear_corrs(h)
    return np.min(corrs)

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient using soft-min approximation.
    
    Objective: minimize 1 - soft_min_corr
    soft_min_corr = -(1/beta) * log(sum_k exp(-beta * C_k))
    """
    h = x
    n = len(h)
    if n <= 1:
        return 1.0, np.zeros(n)
    
    # Compute all correlations
    corrs = _compute_linear_corrs(h)
    
    # Soft min
    # For numerical stability, subtract max
    max_val = np.max(corrs)
    shifted = corrs - max_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = max_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient of soft_min w.r.t. corrs
    # d(soft_min)/d(corrs_k) = exp(-beta * (corrs_k - max_val)) / sum_exp
    # Note: max_val depends on h, but its derivative is zero almost everywhere (subgradient 0 at non-min, undefined at min).
    # Using the standard log-sum-exp gradient formula which is valid for the smooth approximation:
    # g_k = exp(-beta * (corrs_k - max_val)) / sum_exp
    
    weights = exp_terms / sum_exp
    
    # d(soft_min)/d(h) = sum_k weights_k * d(corrs_k)/d(h)
    # corrs_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    # d(corrs_k)/d(h[j]) = (2/n) * (h[j+k] if j < n-k else 0) + (2/n) * (h[j-k] if j >= k else 0)
    
    grad = np.zeros(n)
    # Vectorized approach might be tricky due to varying lengths.
    # Loop over k is O(n^2) but with small constant. n=100 is fine.
    # We can optimize this by noting that we are summing weighted shifts.
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w > 1e-10:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    # d(obj)/d(h) = -d(soft_min)/d(h)
    grad_obj = -grad
    
    return obj, grad_obj
>>>>>>> REPLACE

Now, let's update the `construct` function to use the faster correlation and potentially higher beta values, and improve the restarts.

