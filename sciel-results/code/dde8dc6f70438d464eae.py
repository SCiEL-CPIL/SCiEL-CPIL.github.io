"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def priority(vector, n):
    """
    Priority for greedy cap set construction in {0,1,2}^n.
    Higher priority = added earlier in greedy process.
    
    Uses deviation-from-uniformity as primary signal:
    vectors far from the (n/3, n/3, n/3) count profile are less risky
    as they have fewer potential AP partners.
    
    Adds a smooth secondary term based on 1-count (AP-middle risk)
    and a deterministic hash for tie-breaking.
    """
    vec = tuple(vector)
    
    c0 = sum(1 for x in vec if x == 0)
    c1 = sum(1 for x in vec if x == 1)
    c2 = sum(1 for x in vec if x == 2)
    
    target = n / 3.0
    
    # Squared deviation from uniform counts - primary signal
    dev = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    
    # Normalize dev to [0, 1]: max dev occurs at (n,0,0) or similar
    # Max possible dev: when one count is n, others 0: (n-n/3)^2 + 2*(0-n/3)^2 = (2n/3)^2 + 2*(n/3)^2 = 4n^2/9 + 2n^2/9 = 6n^2/9 = 2n^2/3
    max_dev = 2.0 * n * n / 3.0 if n > 0 else 1.0
    dev_norm = dev / max_dev  # [0, 1], higher = more extreme = less risky
    
    # Secondary: smooth penalty for many 1s (AP-middle risk)
    # Use a soft measure: exp(-alpha * c1/n) gives smooth decay
    # Vectors with c1=0 get ~1.0, c1=n get ~exp(-alpha)
    alpha = 2.0
    ones_ratio = c1 / max(n, 1)
    ones_smooth = 1.0 / (1.0 + alpha * ones_ratio)  # smooth surrogate, range (1/(1+alpha), 1]
    
    # Tertiary: deterministic hash for stable tie-breaking
    h = 0
    for i, v in enumerate(vec):
        h = (h * 31 + v) & 0x7FFFFFFF
    hash_norm = (h % 10000) / 10000.0  # [0, 1)
    
    # Weighted combination
    # Primary weight on deviation (proven effective), secondary on smooth ones penalty
    priority_val = 0.7 * dev_norm + 0.2 * ones_smooth + 0.1 * hash_norm
    
    return priority_val
# EVOLVE-BLOCK-END
