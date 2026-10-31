import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if time.time() - _start_time > _DEADLINE:
        return 0.0
    
    # Count occurrences of each symbol
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Shannon entropy of the coordinate distribution
    total = n
    entropy = 0.0
    for c in (c0, c1, c2):
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    
    # Max possible entropy for ternary is log2(3) ≈ 1.585
    max_entropy = math.log2(3)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Number of distinct symbols present
    distinct = (1 if c0 > 0 else 0) + (1 if c1 > 0 else 0) + (1 if c2 > 0 else 0)
    norm_distinct = distinct / 3.0
    
    # Pairwise disagreement: count positions where adjacent coords differ
    disagreement = 0
    for i in range(n - 1):
        if vector[i] != vector[i + 1]:
            disagreement += 1
    if n > 1:
        norm_disagree = disagreement / (n - 1)
    else:
        norm_disagree = 0.0
    
    # Line-risk heuristic: count "dangerous" coordinate patterns
    # A position is "balanced" if it's 1 (middle value) — these are more likely
    # to complete affine lines. Also, runs of same values create structure.
    # Count the number of positions that are 1 (center of {0,1,2})
    center_count = c1
    norm_center = center_count / n if n > 0 else 0.0
    
    # Count maximal runs of identical values — longer runs = more structure
    # which can create arithmetic progressions
    run_penalty = 0.0
    if n > 1:
        run_len = 1
        for i in range(1, n):
            if vector[i] == vector[i-1]:
                run_len += 1
            else:
                if run_len >= 3:
                    run_penalty += run_len - 2  # excess over minimum
                run_len = 1
        if run_len >= 3:
            run_penalty += run_len - 2
        # Normalize: max possible run penalty is roughly n/2
        norm_run = min(run_penalty / (n * 0.5), 1.0) if n > 0 else 0.0
    else:
        norm_run = 0.0
    
    # Hamming weight relative to all-ones vector: count of 2s
    # This breaks symmetry and forces exploration beyond {0,1}^n
    norm_twos = c2 / n if n > 0 else 0.0
    
    # Hash-based tiebreaker for deterministic diversity
    h = 0
    for x in vector:
        h = (h * 31 + x) & 0xFFFFFFFF
    hash_frac = (h % 100000) / 100000.0 * 0.005  # very small perturbation
    
    # Use a smooth combination: weighted geometric mean (via log-sum-exp trick)
    # This avoids hard thresholds and creates smoother gradients
    # Primary: entropy (structural balance)
    # Secondary: distinct symbols (uses all three values)
    # Tertiary: disagreement (mixing)
    # New: twos (forces full space exploration)
    # Penalty: center concentration and run structure
    
    # Soft-max style: sum of weighted terms with slight nonlinearity
    # Use tanh-like softening on each component to avoid sharp transitions
    w_ent = 8.0
    w_dist = 3.0
    w_dis = 2.0
    w_twos = 2.5
    w_center = 1.5
    w_run = 1.0
    
    # Apply gentle saturation to prevent any single term from dominating
    s_ent = norm_entropy
    s_dist = norm_distinct
    s_dis = norm_disagree
    s_twos = norm_twos
    s_center = norm_center
    s_run = norm_run
    
    score = (w_ent * s_ent + w_dist * s_dist + w_dis * s_dis + w_twos * s_twos 
             - w_center * s_center - w_run * s_run + hash_frac)
    
    return float(score)
