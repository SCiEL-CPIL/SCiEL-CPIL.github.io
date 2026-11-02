import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import factorial

# Deadline guard: 210s limit, use 195s for safety
_DEADLINE = time.time() + 195

def construct():
    """
    Returns three coefficients [c0, c4, c8] for the Hermite combination.
    The degree-12 coefficient c12 is derived to ensure g(0) = 0.
    The score is the upper bound on C4, which is minimized by optimizing
    the ratio I2/I0 where I0 and I2 are quadratic forms in the coefficients.
    """
    
    # Weights for I0 and I2 derived from integrals of Hermite polynomials
    # I0 terms: c_n^2 * 2^n * n!
    # I2 terms: c_n^2 * 2^n * n! * (2n+1) ... wait, let's use the values from the current program
    # which were likely correct for the specific definition of A(f) in the task.
    
    # From current program:
    w0_0 = 1.0
    w0_4 = 384.0
    w0_8 = 10321920.0
    w0_12 = 1961988556800.0
    
    w2_4 = 1536.0
    w2_8 = 82575360.0
    w2_12 = 23543862681600.0
    
    # Constraint: P(0) = 0
    # P(x) = c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)
    # H0(0)=1, H4(0)=12, H8(0)=3024, H12(0)=10321920
    # c0 + 12*c4 + 3024*c8 + 10321920*c12 = 0
    # Wait, the current program used: c0 = -12 c4 - 1680 c8 - 479040 c12
    # Let's check H8(0) and H12(0) values.
    # Physicist's Hermite:
    # H0(x) = 1
    # H1(x) = 2x
    # H2(x) = 4x^2 - 2
    # H3(x) = 8x^3 - 12x
    # H4(x) = 16x^4 - 48x^2 + 12 -> H4(0) = 12
    # H5(x) = 32x^5 - 160x^3 + 120x
    # H6(x) = 64x^6 - 480x^4 + 720x^2 - 120
    # H7(x) = 128x^7 - 1344x^5 + 3360x^3 - 1680x
    # H8(x) = 256x^8 - 4480x^6 + 26880x^4 - 67200x^2 + 40320 -> H8(0) = 40320?
    # Let's re-verify H8(0).
    # Recurrence: H_n(x) = 2x H_{n-1}(x) - 2(n-1) H_{n-2}(x)
    # H_0(0) = 1
    # H_1(0) = 0
    # H_2(0) = -2
    # H_3(0) = 0
    # H_4(0) = 2(0)(0) - 2(3)(-2) = 12. Correct.
    # H_5(0) = 0
    # H_6(0) = 2(0)(0) - 2(5)(0) = 0? No.
    # H_6(x) = 2x H_5(x) - 10 H_4(x). At x=0, H_6(0) = -10 H_4(0) = -10(12) = -120.
    # H_7(0) = 0
    # H_8(x) = 2x H_7(x) - 14 H_6(x). At x=0, H_8(0) = -14 H_6(0) = -14(-120) = 1680.
    # H_9(0) = 0
    # H_10(x) = 2x H_9(x) - 18 H_8(x). At x=0, H_10(0) = -18(1680) = -30240.
    # H_11(0) = 0
    # H_12(x) = 2x H_11(x) - 22 H_10(x). At x=0, H_12(0) = -22(-30240) = 665280.
    
    # The current program used:
    # c0 = -12 c4 - 1680 c8 - 479040 c12
    # This implies H8(0) = 1680 and H12(0) = 479040.
    # My calculation gave H12(0) = 665280.
    # Let's re-calculate H12(0) carefully.
    # H_0(0) = 1
    # H_2(0) = -2
    # H_4(0) = 12
    # H_6(0) = -120
    # H_8(0) = 1680
    # H_10(0) = -30240
    # H_12(0) = 665280
    
    # Why did the current program use 479040?
    # Maybe it's a different normalization or I made a mistake?
    # Let's check the seed: [-113/100, 1/25, 1/3240]
    # c0 = -1.13, c4 = 0.04, c8 = 0.000308642
    # If H8(0)=1680, H12(0)=665280:
    # c12 = -(c0 + 12c4 + 1680c8) / 665280
    # c12 = -(-1.13 + 0.48 + 0.5185) / 665280 = -(-0.1315) / 665280 = 0.1315 / 665280 ≈ 1.97e-7
    
    # If the current program's constraint was correct, it likely had a different set of weights or definitions.
    # However, the current program scored 0.76, which is worse than the seed (1.0).
    # The seed program simply returned the fixed values.
    # The task is to *reduce* the upper bound. The seed is the baseline.
    # If the current program's optimization led to a worse score, maybe the objective function or constraint was slightly off, or the optimization got stuck.
    
    # Let's use the correct Hermite values for the constraint.
    # H0(0) = 1
    # H4(0) = 12
    # H8(0) = 1680
    # H12(0) = 665280
    
    # Constraint: c0 + 12*c4 + 1680*c8 + 665280*c12 = 0
    # c0 = -12*c4 - 1680*c8 - 665280*c12
    
    # Let's re-verify the weights for I0 and I2.
    # I0 = sum c_n^2 * 2^n * n!
    # 2^0*0! = 1
    # 2^4*4! = 16*24 = 384
    # 2^8*8! = 256*40320 = 10321920
    # 2^12*12! = 4096*479001600 = 1961988556800
    # These match the current program's w0 values.
    
    # I2 = sum c_n^2 * 2^n * n! * (2n+1)? Or something else?
    # The current program used:
    # w2_4 = 1536 = 384 * 4
    # w2_8 = 82575360 = 10321920 * 8
    # w2_12 = 23543862681600 = 1961988556800 * 12
    # It seems I2 weight is 2*n * (2^n * n!)?
    # For n=4: 2*4*384 = 3072? No, 1536 = 4*384.
    # For n=8: 8*10321920 = 82575360. Yes.
    # For n=12: 12*1961988556800 = 23543862681600. Yes.
    # So I2 = sum c_n^2 * 2^n * n! * n ?
    # Or maybe I2 = sum c_n^2 * 2^n * n! * (n)?
    # Let's assume the current program's weights for I0 and I2 are correct for the specific "C4 upper bound" definition.
    
    # The issue might be the constraint coefficient for c12.
    # If the current program used 479040 instead of 665280, that's a significant error.
    # Let's use the correct H12(0) = 665280.
    
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    def objective(params):
        c4, c8, c12 = params
        c0 = -H4_0 * c4 - H8_0 * c8 - H12_0 * c12
        
        I0 = w0_0 * c0**2 + w0_4 * c4**2 + w0_8 * c8**2 + w0_12 * c12**2
        I2 = w2_4 * c4**2 + w2_8 * c8**2 + w2_12 * c12**2
        
        if I0 == 0:
            return 1e10
        return I2 / I0
    
    # Initial guess from seed
    seed_c0, seed_c4, seed_c8 = -113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0
    seed_c12 = -(seed_c0 + H4_0 * seed_c4 + H8_0 * seed_c8) / H12_0
    
    best_val = np.inf
    best_params = None
    
    # Try a few starting points, including the seed
    starts = [
        [seed_c4, seed_c8, seed_c12],
        [0.04, 0.0003, 0.0],
        [0.0, 0.0, 1e-6],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    
    for s in starts:
        if time.time() > _DEADLINE - 10:
            break
        s = np.array(s)
        norm = np.linalg.norm(s)
        if norm > 0:
            s = s / norm
        
        try:
            res = minimize(objective, s, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-15, 'fatol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        return np.array([seed_c0, seed_c4, seed_c8])
        
    c4, c8, c12 = best_params
    c0 = -H4_0 * c4 - H8_0 * c8 - H12_0 * c12
    
    return np.array([c0, c4, c8])
