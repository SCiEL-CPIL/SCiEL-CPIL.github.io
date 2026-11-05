import numpy as np

N = 400

def construct():
    # Frequency domain approach to flatten the autocorrelation
    # We want h * h to be approximately constant.
    # This implies H(f)^2 is proportional to sinc(f).
    
    # Create frequency bins
    # np.fft.fftfreq gives frequencies in cycles/sample
    freqs = np.fft.fftfreq(N)
    
    # np.sinc(x) is sin(pi*x)/(pi*x)
    # The DTFT of a constant sequence is a sinc function.
    # Specifically, if h is constant, H(f) is sinc-like.
    # We want the inverse: H(f) such that H(f)^2 ~ sinc(f).
    
    # Note: np.sinc(0) = 1.
    # For frequencies where sinc is negative, we clip to 0.
    sinc_vals = np.sinc(freqs)
    # Clip negative values to avoid complex square roots
    sinc_clipped = np.clip(sinc_vals, 0, None)
    
    # H(f) = sqrt(sinc(f))
    H = np.sqrt(sinc_clipped)
    
    # Inverse FFT to get time domain sequence
    h = np.fft.ifft(H).real
    
    # The sum of h is H(0) * N? No.
    # Sum(h) = H[0] * N?
    # Actually, ifft(H)[0]... 
    # Let's just compute sum.
    s = np.sum(h)
    
    # If sum is close to zero, we have a problem.
    # H[0] = sqrt(sinc(0)) = 1.
    # So sum should be significant.
    
    # Ensure finite
    if not np.all(np.isfinite(h)):
        return np.ones(N)
        
    if abs(s) < 1e-10:
        return np.ones(N)
        
    return h
