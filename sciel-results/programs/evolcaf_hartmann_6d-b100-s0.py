"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Hypar Improvement (EHI) acquisition function.
    
    EHI explicitly models the trade-off between function improvement and
    evaluation cost by maximizing the expected hyperbolic tangent of the
    normalized improvement. This avoids UCB's over-exploration of high-cost,
    high-uncertainty regions while maintaining robustness in low-noise regimes.
    
    The improvement is normalized by cost so that cheaper evaluations receive
    proportionally higher credit for the same absolute improvement.
    """
    # Ensure sigma is positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Normalized improvement: (f_best - mu) / cost
    # This scales the improvement by the evaluation cost
    imp = (f_best - mu) / cost
    
    # Normalized uncertainty: sigma / cost
    unc = sigma / cost
    
    # For large positive imp, tanh(imp) ≈ 1, so E[tanh(imp)] ≈ 1
    # For imp ~ 0, E[tanh(imp)] depends on unc
    # For large negative imp, tanh(imp) ≈ -1, so E[tanh(imp)] ≈ -1
    
    # Expected tanh of a normal random variable N(imp, unc)
    # E[tanh(X)] where X ~ N(m, s) can be approximated using the fact that
    # tanh is bounded and we can use a numerical approximation.
    
    # Use a closed-form approximation:
    # E[tanh(X)] ≈ tanh(m / sqrt(1 + s^2)) * (1 - 0.5 * s^2 * (1 - tanh(m)^2))
    # This is a second-order approximation that captures the main behavior.
    
    # Simpler and more robust approach: use the identity
    # E[tanh(X)] = E[sinh(X)/cosh(X)]
    # For practical purposes, we can use:
    # E[tanh(X)] ≈ tanh(m) * (1 - 0.5 * s^2 * (1 - tanh(m)^2))
    # which is a delta-method approximation.
    
    t = np.tanh(imp)
    # Delta method: Var(tanh(X)) ≈ (1 - tanh(m)^2)^2 * s^2
    # E[tanh(X)] ≈ tanh(m) - 0.5 * (1 - tanh(m)^2)^2 * s^2 * tanh(m)
    # Actually, using the second derivative: d²/dx² tanh(x) = -2*tanh(x)*(1-tanh(x)²)
    # So E[tanh(X)] ≈ tanh(m) + 0.5 * (-2*tanh(m)*(1-tanh(m)²)) * s²
    #                 = tanh(m) * (1 - s² * (1 - tanh(m)²))
    
    ehi = t * (1.0 - unc**2 * (1.0 - t**2))
    
    # Clip to ensure finite values
    ehi = np.clip(ehi, -1.0, 1.0)
    
    return ehi
# EVOLVE-BLOCK-END
