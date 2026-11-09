"""Cap set priority: structural F_3^n cap-set awareness."""
import time
import math

_T0 = time.time()
_DEADLINE = _T0 + 280.0  # 50s safety margin before 330s kill

def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n for greedy cap-set construction.
    
    DIRECT structural approach:
    - In F_3^n, a 3-term AP {a,b,c} has a+c = 2b (mod 3).
    - A coordinate with value 1 in b means b is the midpoint of (0,2) there.
    - Vectors with FEWER 1s are "corners/edges" and are structurally safer:
      they participate in fewer potential APs as the middle element.
    - So: PRIMARY term = -(count of 1s).  Corners (0 ones) rank highest.
    - SECONDARY: among equal 1-counts, balance 0s vs 2s to spread.
    - TERTIARY: deterministic weighted hash for full tie-break.
    """
    if time.time() > _DEADLINE:
        return 0.0

    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1

    # PRIMARY: balance toward n/3 per symbol. Vectors with symbol counts
    # close to n/3 are "interior" points that avoid the corner/edge
    # structures which form small 3-term APs. This lets the greedy
    # algorithm accept more vectors before hitting the cap-set constraint.
    target = n / 3.0
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    score = -penalty

    # SECONDARY: probabilistic collision-density estimate.
    # For a vector v in F_3^n, the number of 3-term APs containing v as
    # the middle element is 2^n (one per pair of opposite directions).
    # As an endpoint, each coordinate contributes differently:
    #   - If v[i]=0, APs where v is endpoint with midpoint having 1 at i.
    #   - If v[i]=1, APs where v is endpoint with midpoint having 0 or 2.
    #   - If v[i]=2, APs where v is endpoint with midpoint having 1 at i.
    #
    # A smooth surrogate for "collision density": estimate the expected
    # number of already-accepted vectors that would form an AP with v,
    # weighted by coordinate-wise entropy. Vectors whose symbol distribution
    # is more "spread out" (higher entropy) have lower per-coordinate
    # concentration, reducing the probability of sharing an AP with any
    # particular accepted vector.
    #
    # Compute Shannon entropy of the symbol distribution:
    #   H = -sum(p_k * log(p_k)) for k in {0,1,2}, p_k = c_k/n
    # Higher entropy => more uniform => lower collision risk.
    # We subtract a scaled entropy term (so higher entropy = higher score).
    if n > 0:
        ent = 0.0
        for c in (c0, c1, c2):
            if c > 0:
                p = c / n
                ent -= p * math.log(p)
        # Entropy ranges from 0 (all one symbol) to log(3) ~ 1.0986.
        # Scale so it provides a meaningful secondary signal without
        # dominating the balance penalty.
        score += 0.5 * ent

    # TERTIARY: among equally balanced+entropic vectors, prefer those with
    # more 1s (midpoints are structurally useful for blocking APs).
    score += 0.01 * c1

    # QUATERNARY: deterministic position-weighted hash for stable tie-break.
    tie = 0.0
    for i, v in enumerate(vector):
        w = math.sin((i + 1) * 0.6180339887)
        tie += w * v
    score += tie * 1e-8

    return score
