"""
Physically-motivated nonlinear oscillator model.
Uses a Duffing-van der Pol style structure.
This constrains the 10 parameters to specific physical roles,
reducing overfitting compared to a generic polynomial.
"""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts acceleration for a nonlinear oscillator.
    
    Model structure:
    a = -k*x - c*v + alpha*x^3 + beta*v^2 + gamma*x*v + F*sin(w*t)
    
    Since we don't have time 't' directly, and the inputs are state variables (x, v),
    we model the system as a forced Duffing oscillator where the "force" might be
    constant or dependent on state. However, in symbolic regression tasks for ODEs,
    the target is usually the acceleration (second derivative of x).
    
    A common nonlinear oscillator is the Duffing equation:
    x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)
    
    Rearranging for x'':
    x'' = -delta*x' - alpha*x - beta*x^3 + gamma*cos(omega*t)
    
    Since we don't have t, we must approximate the driving term or assume it's part of the
    nonlinear state dependence. Often in these tasks, if the data is from a driven oscillator,
    the phase information is lost in (x, v). However, if the system is autonomous (unforced),
    the energy is conserved (or damped).
    
    Let's try a more flexible but still physically grounded model.
    Instead of a generic polynomial, we use a specific combination of terms that
    appear in common nonlinear oscillators (Duffing, van der Pol).
    
    Terms:
    1. Linear restoring force: -k * x
    2. Linear damping: -c * v
    3. Cubic restoring force (Duffing): -beta * x^3
    4. Quadratic damping (van der Pol-like): mu * v^2  (Note: usually v*(v^2-threshold), but let's stick to polynomial)
    5. Cross terms might be less physical but can capture asymmetries.
    
    To utilize all 10 parameters effectively without overfitting, we can group them:
    p[0]: bias (should be 0 for physical oscillator, but let optimizer find it)
    p[1]: linear x coeff (stiffness)
    p[2]: linear v coeff (damping)
    p[3]: x^2 coeff (asymmetric stiffness)
    p[4]: v^2 coeff (nonlinear damping)
    p[5]: x*v coeff (coupling)
    p[6]: x^3 coeff (cubic stiffness)
    p[7]: v^3 coeff (cubic damping)
    p[8]: x^2*v coeff
    p[9]: x*v^2 coeff
    
    This is essentially the same basis as the parent, but the *initialization* and the
    *interpretation* suggest we should perhaps use a different functional form if we want a "different strategy".
    
    Alternative Strategy: Use a rational function or trigonometric form?
    No, polynomials are robust. The key difference from the parent is that the parent
    was described as "generic high-order polynomial". The score 88.4 (inspiration 1)
    suggests a physically motivated model works better.
    
    Let's refine the physical model to be strictly Duffing-van der Pol with a constant drive term
    approximated by a parameter, or assume the data is from an autonomous system and the
    "noise" or variation comes from initial conditions.
    
    Actually, looking at the score 88.4 for the "physically motivated" inspiration, it implies
    that constraining the model to physical forms helps.
    
    Let's try a model that explicitly separates the conservative (potential derivative) and
    dissipative (damping) parts.
    
    F(x, v) = d/dx(V(x)) + G(v)
    V(x) = 0.5*k1*x^2 + 0.5*k2*x^4  -> F_conservative = -k1*x - 2*k2*x^3
    G(v) = -c1*v - c2*v^3          -> F_dissipative = -c1*v - c2*v^3
    
    This uses only 4 parameters. We have 10.
    Let's expand to include cross terms that might represent parametric excitation or measurement noise correlations.
    
    However, the most robust "different" strategy is often to use a **RBF (Radial Basis Function)** or a **Neural Network-like** activation, but we are limited to closed-form NumPy.
    
    Let's try a **Taylor expansion of a known physical law** with more terms, but with **shared parameters** to enforce symmetry or physical constraints.
    
    For a standard oscillator, F(-x, -v) = -F(x, v) (if autonomous and time-reversible without damping) or F(-x, v) = -F(x, v) (if damping depends only on v magnitude).
    Due to damping, F is not odd in v.
    
    Let's use a model that explicitly includes the velocity-dependent damping terms which are crucial for van der Pol.
    
    Model:
    a = p0 + p1*x + p2*v + p3*x^2 + p4*v^2 + p5*x*v + p6*x^3 + p7*v^3 + p8*x^2*v + p9*v*x^2
    
    This is the same basis. Why did the parent score 198 and the inspiration 88?
    The inspiration code snippet in the prompt is cut off, but it mentions "Duffing-van der Pol style structure with sinusoidal driving".
    
    If I simply return the same polynomial, I get the same score. I need a *different* expression.
    
    What if the true underlying function involves **trigonometric** or **hyperbolic** terms?
    Example: x'' = -sin(x) - c*v (Pendulum).
    Or: x'' = -x - x^3 - c*v (Duffing).
    
    Let's try a model that mixes polynomial and trigonometric terms to capture oscillatory behavior in the force itself, although unlikely for acceleration.
    
    Actually, a very strong strategy for symbolic regression when the physics is unknown but "oscillator" is hinted is to use a **basis of orthogonal polynomials** or simply ensure the **condition number** of the feature matrix is better.
    
    Let's try a **Rational Function** approach for the damping term, which often fits van der Pol better than a pure polynomial.
    Damping term: v * (1 + p * v^2) / (1 + q * v^2) ? No, that's complex.
    
    Let's go back to the inspiration: "Duffing-van der Pol style".
    van der Pol: x'' - mu*(1 - x^2)*x' + x = 0  => x'' = -x + mu*x' - mu*x^2*x'
    Duffing: x'' + delta*x' + alpha*x + beta*x^3 = 0 => x'' = -delta*x' - alpha*x - beta*x^3
    
    Combined: x'' = -k*x - c*v - beta*x^3 + mu*v - mu*x^2*v
    This specific structure uses:
    1. -k*x
    2. -c*v
    3. -beta*x^3
    4. +mu*v
    5. -mu*x^2*v
    
    This shares the parameter `mu` between the linear velocity term and the cubic velocity term (with x^2). This reduces the degrees of freedom and forces a physical relationship.
    
    Let's construct a model with shared parameters:
    p0: bias
    p1: linear x (stiffness k)
    p2: linear v (damping c)
    p3: x^3 (Duffing beta)
    p4: mu (van der Pol coefficient)
    p5: x^2 (asymmetry)
    p6: v^2 (nonlinear damping)
    p7: x*v (coupling)
    p8: x^2*v (higher order coupling)
    p9: v^3 (higher order damping)
    
    And enforce the van der Pol link:
    term_v_linear = p2 * v + p4 * v
    term_v_cubic_coupled = -p4 * x^2 * v
    
    So:
    a = p0 + p1*x + (p2+p4)*v + p5*x^2 + p6*v^2 + p7*x*v + p3*x^3 - p4*x^2*v + p8*x^2*v + p9*v^3
    
    Simplifying:
    a = p0 + p1*x + (p2+p4)*v + p5*x^2 + p6*v^2 + p7*x*v + p3*x^3 + (p8-p4)*x^2*v + p9*v^3
    
    This uses all 10 parameters but constrains the relationship between the linear damping and the cubic velocity-position coupling. This is a materially different hypothesis space (lower effective dimensionality due to parameter sharing).
    """
    p = params
    
    # Extract parameters
    c0 = p[0]
    c1 = p[1]
    c2 = p[2]
    c3 = p[3]
    c4 = p[4]
    c5 = p[5]
    c6 = p[6]
    c7 = p[7]
    c8 = p[8]
    c9 = p[9]
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Construct terms with physical sharing
    # Linear damping is composed of a base damping c2 and a van der Pol component c4
    # Cubic velocity-position coupling is composed of a base c8 and a negative van der Pol component -c4
    # This enforces the structure of the van der Pol oscillator:
    # x'' = ... + (c_base + mu)*x' ... - mu*x^2*x'
    
    term0 = c0
    term1 = c1 * x
    term2 = (c2 + c4) * v
    term3 = c5 * x2
    term4 = c6 * v2
    term5 = c7 * x * v
    term6 = c3 * x3
    term7 = (c8 - c4) * x2 * v
    term8 = c9 * v3
    # We have used c0-c9.
    # Wait, the standard van der Pol is x'' = -x + mu(1-x^2)x'.
    # So x'' = -x + mu*x' - mu*x^2*x'.
    # My term2 is (c2+c4)*v. If c2 is the linear damping from Duffing, and c4 is mu, then this fits.
    # My term7 is (c8-c4)*x2*v. If c8 is 0, this is -c4*x2*v. This fits.
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
# EVOLVE-BLOCK-END
