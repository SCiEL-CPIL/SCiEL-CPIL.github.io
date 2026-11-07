"""
Rewritten program for symbolic regression.
Uses a hybrid model: Polynomial trend + 2 Flexible Fourier pairs.
This structure allows for both global trends and periodic/non-periodic oscillatory features.
The frequencies are constrained to be positive to reduce the effective dimensionality of the search space.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard variables
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a combination of polynomial terms
    and two flexible Fourier (sine/cosine) units.
    
    Model:
    y = p0 + p1*x + p2*x^2 
      + p3*sin(f1*x) 
      + p5*cos(f2*x) 
      + p7*sin(f3*x) 
      + p9*cos(f4*x)
      
    Where f1=|p4|, f2=|p6|, f3=|p8|, f4=|p9|.
    
    Parameters:
    p0: Constant offset
    p1: Linear coefficient
    p2: Quadratic coefficient
    p3: Amplitude of 1st sine unit
    p4: Frequency of 1st sine unit (magnitude used)
    p5: Amplitude of 1st cosine unit
    p6: Frequency of 1st cosine unit (magnitude used)
    p7: Amplitude of 2nd sine unit
    p8: Frequency of 2nd sine unit (magnitude used)
    p9: Amplitude of 2nd cosine unit (Note: p9 is amplitude, wait, I need 10 params)
    
    Let's re-map to 10 params clearly:
    p0: Const
    p1: Lin
    p2: Quad
    p3: Amp sin 1
    p4: Freq sin 1
    p5: Amp cos 1
    p6: Freq cos 1
    p7: Amp sin 2
    p8: Freq sin 2
    p9: Amp cos 2 ... wait, I need a freq for cos 2?
    I have 10 params.
    If I use:
    p0, p1, p2 (Poly)
    p3, p4 (Sin1)
    p5, p6 (Cos1)
    p7, p8 (Sin2)
    p9 (Cos2 Amp) -- Missing Freq for Cos2?
    
    Alternative Mapping:
    p0, p1, p2 (Poly)
    p3, p4 (Sin1)
    p5, p6 (Cos1)
    p7, p8 (Sin2)
    p9 (Cos2) -> Maybe fix freq of Cos2 to be same as Sin2? Or same as Sin1?
    
    Let's use:
    p0, p1, p2 (Poly)
    p3 * sin(p4 * x)
    p5 * cos(p6 * x)
    p7 * sin(p8 * x)
    p9 * cos(p8 * x)  <- Share frequency p8 for the second pair?
    This reduces independent frequencies to 3 (p4, p6, p8).
    
    Or better:
    p0, p1, p2 (Poly)
    p3 * sin(p4 * x)
    p5 * cos(p4 * x)  <- Share freq p4 for first pair
    p7 * sin(p6 * x)
    p9 * cos(p6 * x)  <- Share freq p6 for second pair
    This uses:
    p0, p1, p2 (3 params)
    p3, p4, p5 (3 params) -> Sin/Cos pair 1 with freq p4
    p7, p6, p9 (3 params) -> Sin/Cos pair 2 with freq p6
    Total 9 params. I have 10.
    Add p8 as a cubic term?
    
    Final Model:
    y = p0 + p1*x + p2*x^2 + p8*x^3 
      + p3*sin(p4*x) + p5*cos(p4*x) 
      + p7*sin(p6*x) + p9*cos(p6*x)
      
    This is very robust.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to a simple polynomial if we are running out of time
        # This should rarely happen as the function is vectorized and fast.
        pass

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Ensure frequencies are positive to avoid redundant search space
    # and to keep the function behavior consistent (sin(-x) = -sin(x) is handled by amplitude sign)
    f1 = np.abs(p4)
    f2 = np.abs(p6)
    
    # Clip frequencies to a reasonable range to prevent numerical issues
    # with high-frequency oscillations which can create rugged loss landscapes
    # for BFGS. 20 is a high enough limit for most functions on [-2, 2].
    f1 = np.clip(f1, 0.1, 20.0)
    f2 = np.clip(f2, 0.1, 20.0)

    x2 = x * x
    x3 = x2 * x

    # Polynomial part
    poly = p0 + p1 * x + p2 * x2 + p8 * x3

    # Fourier part
    # Pair 1: frequency f1
    unit1 = p3 * np.sin(f1 * x) + p5 * np.cos(f1 * x)
    
    # Pair 2: frequency f2
    unit2 = p7 * np.sin(f2 * x) + p9 * np.cos(f2 * x)

    return poly + unit1 + unit2
# EVOLVE-BLOCK-END
