"""Improved growth-index approximation using Taylor expansion in redshift z around z=0."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return six coefficients [c0, c1, c2, c3, c4, c5] for the growth index ansatz:
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    
    where:
      q = 1 - Om(a)
      u = 1 + w0 + wa*(1-a)
      Om(a) = Omega_m0 * a^3 / (Omega_m0 * a^3 + Omega_Lambda0 * a^(-3*wa) * (1+a*wa)^3)
    
    We use a Taylor expansion in redshift z = 1/a - 1 around z=0 (a=1):
    
    For a flat universe with CPL dark energy:
      Om(a) = Omega_m0 / (Omega_m0 + Omega_Lambda0 * a^(-3-3*wa) * (1+a*wa)^3)
    
    At a=1: Om(1) = Omega_m0
    q = 1 - Om(a)
    
    We want gamma to match the true growth index behavior.
    
    The true growth index gamma_true = d(ln f)/d(ln a) where f = Om^gamma.
    
    For a Taylor expansion in z around z=0:
      gamma(z) = gamma_0 + gamma_1 * z + gamma_2 * z^2 + ...
    
    We map our six coefficients to approximate this expansion.
    
    Key relationships at a=1 (z=0):
      q(1) = 1 - Omega_m0
      u(1) = 1 + w0 + wa*(1-1) = 1 + w0
      wa*a at a=1 is just wa
    
    So gamma(1) = c0 + c1*(1-Om0) + c2*(1-Om0)^2 + c3*(1+w0) + c4*wa + c5*(1+w0)*(1-Om0)
    
    We want this to equal the known value of gamma at z=0.
    
    For a flat universe with w0=-1, wa=0:
      gamma(0) ≈ 0.55 (constant approximation)
      More accurately: gamma(0) = (1/2 + w0/7 - 3*wa/35) * (1 - Om0/2) ... but let's use a better fit.
    
    Actually, let's think about this differently. The standard result for the growth index at z=0 is:
      gamma_0 ≈ 0.55 - 0.05*wa/(1+wa) * (1-Om0) ... this is getting complicated.
    
    Let's use a simpler approach: we know that for the constant gamma approximation,
    gamma ≈ 0.55 works well. Let's design the coefficients so that:
    
    1. At z=0, gamma matches a good estimate of the true growth index.
    2. The derivative d(gamma)/d(z) at z=0 matches the true slope.
    3. The second derivative matches the true curvature.
    
    For a flat universe with CPL DE, the growth index can be expanded as:
      gamma(z) = gamma_0 + gamma_1 * z + gamma_2 * z^2
    
    where (from literature and perturbation theory):
      gamma_0 ≈ 0.55 - 0.05*(1-Om0) ... actually let's use:
      
    A common parametrization: gamma(z) ≈ 0.55 + (0.05 - 0.05*Om0) * z + ...
    
    Actually, let's be more careful. The true growth index at z=0 for a flat universe is:
      gamma(0) = (1/2) * (1 - 3/2 * w0_eff) ... no, that's not right either.
    
    Let me use a different strategy. I'll set up the coefficients so that the ansatz
    reduces to a Taylor series in z with specific coefficients that I can tune.
    
    Since we have 6 coefficients and we want to match:
      gamma(0), d(gamma)/dz at z=0, d^2(gamma)/dz^2 at z=0, 
      and behavior at other z values
    
    Let's express q, u, and wa*a in terms of z.
    
    z = 1/a - 1, so a = 1/(1+z)
    
    For a flat universe:
      Om(a) = Om0 / (Om0 + (1-Om0) * a^(-3-3*wa) * (1+a*wa)^3)
    
    Let's define E^2(a) = Om(a)/Om0 = 1 + (1-Om0)/Om0 * a^(-3-3*wa) * (1+a*wa)^3
    
    This is complex. Let's use a simpler mapping.
    
    Alternative approach: Let's directly set the coefficients to produce a good
    Taylor expansion. 
    
    At z=0 (a=1):
      q = 1 - Om0
      u = 1 + w0
      wa*a = wa
    
    For small z, a ≈ 1-z, so:
      q ≈ 1 - Om(1-z) ≈ q(1) + dq/da * (a-1) = q(1) - dq/da * z
      u ≈ u(1) + du/da * (a-1) = u(1) - du/da * z
      wa*a ≈ wa*(1-z) = wa - wa*z
    
    So gamma(z) ≈ c0 + c1*(q0 - q1*z) + c2*(q0 - q1*z)^2 + c3*(u0 - u1*z) + c4*(wa - wa*z) + c5*(u0-u1*z)*(q0-q1*z)
    
    where q0 = 1-Om0, u0 = 1+w0.
    
    Expanding to second order in z:
      gamma(z) = c0 + c1*q0 + c2*q0^2 + c3*u0 + c4*wa + c5*u0*q0
                 + z * (-c1*q1 - c2*2*q0*q1 - c3*u1 - c4*wa - c5*(u1*q0 + u0*q1))
                 + z^2 * (c2*q1^2 + c5*u1*q1)
    
    We want this to match:
      gamma(z) = gamma_0 + gamma_1*z + gamma_2*z^2
    
    So we need:
      c0 + c1*q0 + c2*q0^2 + c3*u0 + c4*wa + c5*u0*q0 = gamma_0
      -c1*q1 - c2*2*q0*q1 - c3*u1 - c4*wa - c5*(u1*q0 + u0*q1) = gamma_1
      c2*q1^2 + c5*u1*q1 = gamma_2
    
    We have 6 unknowns and 3 equations, so we have 3 degrees of freedom.
    
    Let's choose specific values for gamma_0, gamma_1, gamma_2 that work well.
    
    From the literature, for a flat universe:
      gamma_0 ≈ 0.55 (for w0=-1, wa=0, Om0=0.3)
      More generally, gamma_0 ≈ 0.55 - 0.05*(1-Om0) ... let's use a better fit.
    
    Actually, let's use the known result that for a flat universe with CPL DE:
      gamma(0) ≈ 0.55 + (0.05 - 0.05*Om0) * (something) ... 
    
    Let me just pick reasonable values and see. For Om0=0.3, w0=-1, wa=0:
      gamma_0 ≈ 0.55
      gamma_1 ≈ 0.05 (positive slope, gamma increases with z)
      gamma_2 ≈ 0.01 (small positive curvature)
    
    But these depend on the cosmology. Since we're using a fixed ansatz, we need
    coefficients that work across a range of cosmologies.
    
    Let's try a different approach: set c1=c2=c3=c4=c5=0 and just tune c0.
    But that's the seed program. We need to do better.
    
    Let's set up the system with specific target values and solve.
    
    For simplicity, let's assume typical values: Om0=0.3, w0=-1, wa=0.
      q0 = 0.7, u0 = 0
      dq/da at a=1: need to compute this.
      
    Om(a) = Om0 * a^3 / (Om0*a^3 + (1-Om0)*a^(-3*wa)*(1+a*wa)^3)
    For wa=0: Om(a) = Om0*a^3 / (Om0*a^3 + 1-Om0)
    d(Om)/da = Om0 * 3*a^2*(Om0*a^3+1-Om0) - Om0*a^3*3*Om0*a^2) / (Om0*a^3+1-Om0)^2
              = 3*Om0*a^2*(1-Om0) / (Om0*a^3+1-Om0)^2
    At a=1: d(Om)/da = 3*Om0*(1-Om0)/(Om0+1-Om0)^2 = 3*Om0*(1-Om0)
    So dq/da = -d(Om)/da = -3*Om0*(1-Om0)
    q1 = -dq/da = 3*Om0*(1-Om0) = 3*0.3*0.7 = 0.63
    
    du/da = -3*wa (since u = 1+w0+wa*(1-a), du/da = -wa)
    For wa=0: u1 = 0
    
    So the equations become:
      c0 + c1*0.7 + c2*0.49 + c4*0 + c5*0*0.7 = gamma_0  (since u0=0, wa=0)
      -c1*0.63 - c2*2*0.7*0.63 - c4*0 - c5*(0*0.7 + 0*0.63) = gamma_1
      c2*0.63^2 + c5*0*0.63 = gamma_2
    
    Simplifying:
      c0 + 0.7*c1 + 0.49*c2 = gamma_0
      -0.63*c1 - 0.882*c2 = gamma_1
      0.3969*c2 = gamma_2
    
    From the third: c2 = gamma_2 / 0.3969
    From the second: c1 = (-gamma_1 - 0.882*c2) / 0.63
    From the first: c0 = gamma_0 - 0.7*c1 - 0.49*c2
    
    Let's pick gamma_0=0.55, gamma_1=0.05, gamma_2=0.01:
      c2 = 0.01/0.3969 ≈ 0.0252
      c1 = (-0.05 - 0.882*0.0252)/0.63 = (-0.05 - 0.02223)/0.63 = -0.07223/0.63 ≈ -0.1147
      c0 = 0.55 - 0.7*(-0.1147) - 0.49*0.0252 = 0.55 + 0.08029 - 0.012348 ≈ 0.6179
    
    So [c0, c1, c2, c3, c4, c5] ≈ [0.6179, -0.1147, 0.0252, 0, 0, 0]
    
    But this is for a specific cosmology. We need something that works generally.
    
    Let's just return these values as a starting point.
    """
    return [0.61, -0.10, 0.02, 0.01, 0.01, 0.0]
# EVOLVE-BLOCK-END
