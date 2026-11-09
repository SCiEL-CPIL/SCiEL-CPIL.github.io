import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # p0: constant
    # p1: linear coeff
    # p2, p3: amp1, freq1
    # p4, p5: amp2, freq2
    # p6, p7: amp3, freq3
    # p8, p9: amp4, freq4
    
    p = params
    # Clip frequencies to prevent numerical instability and high-frequency noise
    # that BFGS might chase but which don't generalize or cause convergence issues.
    # Range [-20, 20] allows periods from ~0.3 to ~120.
    f1 = np.clip(p[3], -20.0, 20.0)
    f2 = np.clip(p[5], -20.0, 20.0)
    f3 = np.clip(p[7], -20.0, 20.0)
    f4 = np.clip(p[9], -20.0, 20.0)
    
    a1 = p[2]
    a2 = p[4]
    a3 = p[6]
    a4 = p[8]
    
    c0 = p[0]
    c1 = p[1]
    
    # Compute terms
    # Using sin and cos directly
    term1 = a1 * np.sin(f1 * x)
    term2 = a2 * np.sin(f2 * x)
    term3 = a3 * np.sin(f3 * x)
    term4 = a4 * np.sin(f4 * x)
    
    # Wait, the inspiration used sin AND cos for each frequency?
    # "y = p0 + p1*x + p2*sin(p3*x) + p4*cos(p5*x) + ..."
    # That uses 2 params per term if freq is shared? No, inspiration had:
    # p2: amp sin 1, p3: freq sin 1
    # p4: amp cos 1, p5: freq cos 1
    # This decouples sin and cos frequencies, which is redundant and harder to optimize.
    # A standard Fourier term is A*sin(w*x) + B*cos(w*x) or R*cos(w*x - phi).
    # With 10 params:
    # p0: c
    # p1: lin
    # Remaining 8 params -> 4 terms.
    # If we do 4 pairs of (amp, freq) for sin only, we lose the cos phase shift capability unless freq=0 or we use both.
    # Better: 4 pairs of (freq, phase)? No, phase is hard for BFGS.
    # Better: 4 pairs of (amp_sin, freq) and 4 pairs of (amp_cos, freq)? That's 8 params + 2 linear = 10.
    # But sharing the frequency for sin and cos of the same "harmonic" is physically more sensible.
    # Let's try:
    # p0: const
    # p1: linear
    # p2: amp1, p3: freq1  -> a1*sin(w1*x) + b1*cos(w1*x)? No, that needs 3 params per harmonic.
    # 10 params total.
    # Option A: 2 harmonics.
    #   p0: c, p1: lin
    #   p2: a1_sin, p3: a1_cos, p4: w1
    #   p5: a2_sin, p6: a2_cos, p7: w2
    #   p8: a3_sin, p9: a3_cos ... wait, w3 is missing.
    #   This leaves p8, p9 unused or we drop linear.
    
    # Let's look at the inspiration again.
    # p0: c, p1: lin
    # p2: a_sin1, p3: f_sin1
    # p4: a_cos1, p5: f_cos1
    # p6: a_sin2, p7: f_sin2
    # p8: a_cos2, p9: f_cos2
    # This allows independent frequencies for sin and cos. It's flexible but potentially redundant.
    # However, it scored better. I will stick close to this but maybe enforce that f_sin and f_cos are close?
    # No, BFGS will figure it out.
    
    # Let's refine the inspiration's structure slightly for robustness.
    # The inspiration's score was 45k. Current is 112M.
    # The jump is massive. The structure is likely correct.
    
    # I will implement the inspiration's structure exactly, but with slightly tighter clipping if needed,
    # and ensure no overflow.
    
    result = c0 + c1 * x + term1 + a2 * np.cos(f2 * x) + term3 + a4 * np.cos(f4 * x)
    
    # Wait, I defined term1 as a1*sin(f1*x).
    # Inspiration:
    # term1 = a2 * np.sin(f1 * x)  (using p2, p3)
    # term2 = a3 * np.cos(f2 * x)  (using p4, p5)
    # term3 = a4 * np.sin(f3 * x)  (using p6, p7)
    # term4 = a5 * np.cos(f4 * x)  (using p8, p9)
    
    # This allows the model to fit any combination of sines and cosines with different frequencies.
    # It is effectively a sum of 4 sinusoids with independent frequencies and amplitudes (sign determined by amp).
    # This is a very powerful basis for periodic functions.
    
    return np.where(np.isfinite(result), result, 0.0)
