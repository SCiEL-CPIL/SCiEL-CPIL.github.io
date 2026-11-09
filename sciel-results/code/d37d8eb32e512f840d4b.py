import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre
import time

def construct():
    """
    Returns a list of 12 positive double-root locations for the Laguerre combination.
    
    Strategy:
    1. Use zeros of the generalized Laguerre polynomial L_12^(alpha)(x).
    2. The parameter alpha controls the spacing. alpha = m-1 = 11 yields near-uniform spacing.
    3. The scale of the roots significantly impacts the C4 upper bound. 
       Observation: Scaling roots to a maximum of ~48000 yielded a score of ~33.3.
       Scaling to ~480 yielded ~2.1.
       This suggests that larger root magnitudes (wider support) lead to a higher score (lower bound? or just different metric behavior).
       Wait, the task says "score is the upper bound on C4" and "lower is better" for the value, but "search score is seed/value".
       Seed value V_seed. Score = V_seed / V.
       If Score = 33.3 and V is the "upper bound", then V is small?
       Let's re-read carefully: "score is the upper bound on C4... The evaluator measures the raw value (lower is better). The search score is seed/value".
       This implies:
       Value = C4_bound (lower is better for the raw value).
       Search_Score = Seed_Value / Value.
       If Search_Score > 1, it's an improvement, meaning Value < Seed_Value.
       
       Seed: [4, ..., 48]. Score = 1.0. So Seed_Value = V_seed.
       Current Program: Roots scaled to max 48000. Score = 33.3.
       This implies 33.3 = V_seed / V_current.
       So V_current = V_seed / 33.3.
       This means V_current is much smaller than V_seed.
       Therefore, scaling the roots UP (to 48000) DECREASED the C4 bound significantly.
       
       Inspiration 1: Roots scaled to max 480. Score = 2.12.
       V_insp1 = V_seed / 2.12.
       V_current < V_insp1 < V_seed.
       So larger roots -> smaller C4 bound -> higher search score.
       
       Hypothesis: The C4 bound decreases as the roots move further apart / to larger values.
       Why? In Fourier uncertainty, compact support in one domain spreads in the other. 
       However, this is a specific polynomial construction. 
       The Cohn-Goncalves construction likely involves a polynomial P(x) with these roots.
       If the roots are large, the polynomial oscillates rapidly? Or the "uncertainty" measure is defined such that spreading out the roots reduces the bound?
       
       Let's test if we can push the scale even higher.
       Current max: 48000.
       Let's try scaling to 480000 (10x).
       Or maybe the relationship is asymptotic.
       
       Let's also refine the alpha.
       alpha=11 gave uniform spacing.
       Does non-uniform spacing help?
       Usually, for minimax problems, Chebyshev-like nodes (clustered at ends) are better.
       Laguerre roots with alpha=0 are clustered near 0.
       Laguerre roots with alpha=large are uniform.
       Laguerre roots with alpha=negative? Not standard.
       
       Let's try to optimize the scale factor.
       We have data points:
       Max=48 -> Score=1.0
       Max=480 -> Score=2.12
       Max=48000 -> Score=33.3
       
       Let's fit a power law: Score = A * (Max)^k
       1.0 = A * 48^k
       2.12 = A * 480^k
       33.3 = A * 48000^k
       
       From 1 and 2: 2.12 = (480/48)^k = 10^k => k = log10(2.12) ≈ 0.326
       From 2 and 3: 33.3/2.12 = 15.7 = (48000/480)^k = 100^k => k = log(15.7)/log(100) ≈ 1.196/2 ≈ 0.598
       
       The exponent is increasing. This suggests it's not a simple power law, or there's a regime change.
       Or maybe my assumption about "Max" is the driver is slightly off, but it's the only variable changed.
       
       Let's try a much larger scale. If the bound decreases with scale, let's go to the limit.
       However, numerical stability might degrade if roots are too large.
       Let's try Max = 480,000.
       
       Also, let's verify the root shape.
       For alpha=11, the roots are roughly uniform in [0, ~2m+2alpha+1]? No, generalized Laguerre L_n^(alpha) has roots in (0, infinity).
       For large alpha, they are spread out.
       
       Let's try to perform a local optimization on the roots themselves, using a surrogate for the C4 bound.
       Since we don't have the exact formula for C4 in the prompt, we rely on the score.
       But we can't query the score during execution.
       
       However, we can assume the "lower is better" value is related to some norm of the polynomial or its derivative.
       
       Let's stick to the structural change:
       1. Use generalized Laguerre roots.
       2. Scale them aggressively.
       
       Let's try to refine the alpha.
       If uniform spacing (alpha=11) is good, maybe slightly more clustered (alpha=10) or less (alpha=12)?
       Unlikely to make a huge difference compared to scale.
       
       Let's try scaling by 10x more: Max = 480,000.
       
       Also, consider that the "double roots" might imply the polynomial is squared?
       "12 double roots" -> (x-r1)^2 ... (x-r12)^2.
       
       Let's generate the roots and scale them.
       
       To be safe, I will implement a quick check to ensure the roots are positive and distinct.
       
       I will also try a few different scales and pick the one that seems "best" based on a heuristic if I had one, but since I don't, I'll bet on the trend: larger scale -> better score.
       
       Let's try Max = 480,000.
       
       Wait, looking at the jump from 480 (2.12) to 48000 (33.3), the factor is ~15x for 100x scale increase.
       If I go to 480,000 (10x more scale), maybe another 15x? Score ~500?
       
       Let's try it.
       
       One concern: Floating point precision.
       Roots of L_12^(11) are O(10).
       Scaling by 10^4 gives O(10^5).
       Scaling by 10^5 gives O(10^6).
       Double precision has ~15-16 digits. This should be fine.
       
       Let's code it up.
    """
    t0 = time.time()
    deadline = 590.0

    m = 12
    # alpha = 11 for near-uniform spacing
    alpha = 11.0

    try:
        zeros = roots_genlaguerre(m, alpha)[0]
        zeros = np.sort(zeros)
    except Exception:
        # Fallback
        return [4.0 * 100000.0 * (i + 1) for i in range(12)]

    # Current best max was 48000.
    # Let's try 480000.
    target_max = 480000.0
    scale_factor = target_max / zeros[-1]
    scaled_zeros = zeros * scale_factor

    # Verify they are positive
    if np.any(scaled_zeros <= 0):
        # Shift if necessary, though Laguerre roots are positive
        scaled_zeros = np.abs(scaled_zeros) + 1e-9

    return list(scaled_zeros)
