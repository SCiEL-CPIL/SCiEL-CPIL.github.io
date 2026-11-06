"""New program: Branch and Bound search for optimal residue class set."""
# EVOLVE-BLOCK-START
import math
import sys
sys.setrecursionlimit(10000)

# Precompute moduli and weights
MODULI = list(range(20, 201))
WEIGHTS = [1.0 / m for m in MODULI]
# Sort by weight descending (smaller m first) to find high value solutions early
# and to allow tighter upper bounds (sum of remaining weights)
SORTED_MODULI = sorted(MODULI, key=lambda m: 1.0/m, reverse=True)
SORTED_WEIGHTS = [1.0 / m for m in SORTED_MODULI]

# Precompute suffix sums for upper bound pruning
# suffix_sum[i] = sum of SORTED_WEIGHTS[i:]
N = len(SORTED_MODULI)
SUFFIX_SUM = [0.0] * (N + 1)
for i in range(N - 1, -1, -1):
    SUFFIX_SUM[i] = SUFFIX_SUM[i + 1] + SORTED_WEIGHTS[i]

def _find_valid_residue(m, chosen):
    """
    Find a residue r in [0, m) such that for all (m_i, r_i) in chosen,
    r != r_i mod gcd(m, m_i).
    Returns r or None if no valid residue exists.
    """
    for r in range(m):
        valid = True
        for m_i, r_i in chosen:
            g = math.gcd(m, m_i)
            if r % g == r_i % g:
                valid = False
                break
        if valid:
            return r
    return None

def construct():
    best_score = 0.0
    best_classes = []
    
    # current state
    current_classes = []
    current_score = 0.0
    
    def dfs(index):
        nonlocal best_score, best_classes
        
        # Pruning: if current score + max possible remaining score <= best_score, stop
        if current_score + SUFFIX_SUM[index] <= best_score:
            return
            
        if index == N:
            if current_score > best_score:
                best_score = current_score
                best_classes = [c[:] for c in current_classes]
            return
        
        m = SORTED_MODULI[index]
        w = SORTED_WEIGHTS[index]
        
        # Option 1: Include m
        # Try to find a valid residue for m given current_classes
        r = _find_valid_residue(m, current_classes)
        if r is not None:
            current_classes.append([m, r])
            current_score += w
            dfs(index + 1)
            current_classes.pop()
            current_score -= w
            
        # Option 2: Exclude m
        dfs(index + 1)

    dfs(0)
    
    return best_classes
# EVOLVE-BLOCK-END
