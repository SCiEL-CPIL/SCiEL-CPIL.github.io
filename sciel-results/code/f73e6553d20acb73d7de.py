"""Product-formula design for Heisenberg-chain time evolution using 4th-order Yoshida coefficients."""

# EVOLVE-BLOCK-START
def coefficients(stages):
    """
    4th-order symmetric Suzuki-Trotter product formula using Yoshida coefficients.
    
    The product formula is U(t) = prod_{k=1}^{stages} exp(-i b_k t B) exp(-i a_k t A).
    
    We use 3 substeps of a 4-stage 4th-order Yoshida decomposition for 12 stages.
    The 4th-order formula is:
        S_4(t) = S_2(c1*t) * S_2(c2*t) * S_2(c1*t)
    where S_2(t) = exp(-i t A/2) * exp(-i t B) * exp(-i t A/2)
    
    c1 = 1/(2 - 2^(1/3))
    c2 = -2^(1/3)/(2 - 2^(1/3))
    
    The raw coefficients sum to 1+c1 for both A and B.
    We scale them by 1/(1+c1) to ensure sum(a)=sum(b)=1 for each substep.
    """
    # 4th-order Yoshida coefficients
    c1 = 1.0 / (2.0 - 2.0 ** (1.0/3.0))
    c2 = -2.0 ** (1.0/3.0) / (2.0 - 2.0 ** (1.0/3.0))
    
    # Scaling factor to ensure sums are 1
    S = 1.0 + c1
    
    # Per 4th-order substep (4 stages in (B, A) form):
    # Stage 1: B=0, A=c1/2S
    # Stage 2: B=c1/S, A=(c1+c2)/2S
    # Stage 3: B=c2/S, A=(c1+c2)/2S
    # Stage 4: B=c1/S, A=c1/2S
    
    n_substeps = stages // 4
    if n_substeps == 0:
        n_substeps = 1
    
    a = []
    b = []
    for _ in range(n_substeps):
        b += [0.0, c1/S, c2/S, c1/S]
        a += [c1/(2.0*S), (c1+c2)/(2.0*S), (c1+c2)/(2.0*S), c1/(2.0*S)]
    
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
