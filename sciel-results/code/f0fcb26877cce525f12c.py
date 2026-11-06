"""Product-formula design for Heisenberg-chain time evolution using 4th-order Yoshida coefficients."""
import time

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter product formula using Yoshida coefficients.
    
    For a 12-stage formula, we use 3 substeps of a 4-stage 4th-order formula.
    Each substep maps the 4th-order Yoshida decomposition into the required
    (B, A) pair form.
    
    The 4th-order Suzuki-Trotter coefficients are:
    c1 = 1/(2 - 2^(1/3))
    c2 = -2^(1/3)/(2 - 2^(1/3))
    
    The 4th-order formula S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c1*t)
    where S_2(t) = exp(-i t*A/2) * exp(-i t*B) * exp(-i t*A/2)
    
    Expanding and combining adjacent A terms gives 3 B-stages and 4 A-stages.
    We map this into 4 (B, A) pairs by setting the first b to 0.
    """
    # 4th-order Yoshida/Suzuki-Trotter coefficients
    c1 = 1.0 / (2.0 - 2.0 ** (1.0/3.0))
    c2 = -2.0 ** (1.0/3.0) / (2.0 - 2.0 ** (1.0/3.0))
    
    # Per 4th-order substep (4 stages in (B, A) form):
    # The expanded 4th-order formula is:
    # exp(-i c1/2 * A) exp(-i c1 * B) exp(-i (c1+c2)/2 * A)
    # exp(-i c2 * B) exp(-i (c1+c2)/2 * A)
    # exp(-i c1 * B) exp(-i c1/2 * A)
    #
    # Mapping to (B, A) pairs:
    # Stage 1: B=0, A=c1/2
    # Stage 2: B=c1, A=(c1+c2)/2
    # Stage 3: B=c2, A=(c1+c2)/2
    # Stage 4: B=c1, A=c1/2
    
    n_substeps = stages // 4  # 3 substeps for 12 stages
    if n_substeps == 0:
        n_substeps = 1
    
    a = []
    b = []
    for _ in range(n_substeps):
        b += [0.0, c1, c2, c1]
        a += [c1/2.0, (c1+c2)/2.0, (c1+c2)/2.0, c1/2.0]
    
    # Scale by 1/n_substeps since each substep handles t/n_substeps
    a = [x / n_substeps for x in a]
    b = [x / n_substeps for x in b]
    
    # Pad to exactly 'stages' elements if needed
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    
    # Truncate if needed
    a = a[:stages]
    b = b[:stages]
    
    return a, b
# EVOLVE-BLOCK-END
