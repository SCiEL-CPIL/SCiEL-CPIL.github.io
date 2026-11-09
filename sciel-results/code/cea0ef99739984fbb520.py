"""Cap set task: rank vectors so greedy descending insertion maximizes cap set size.

Strategy:
  * Prioritize vectors with few 1s. Vectors with 1s are more likely to be the
    middle term of a 3-term arithmetic progression (AP) because $2 \cdot 1 = 2$
    allows pairs $(0,2)$ and $(1,1)$ to sum to 2. Vectors with 0s are safer
    ($2 \cdot 0 = 0$ requires $a=c=0$, so no distinct AP with $v$ as middle).
    Vectors with 2s are intermediate ($2 \cdot 2 = 4 \equiv 1$ allows $(0,1)$).
  * Use variance as a secondary structural feature to prefer spread-out vectors.
  * Use a deterministic hash for tie-breaking to avoid lexicographic bias.
  * Include a deadline guard to prevent timeouts.
"""
import time

_T0 = time.time()
_DEADLINE = _T0 + 290.0  # Safety margin for the 330s limit

_HASH_BASE = 31
_HASH_MASK = 0x7FFFFFFF


def priority(vector, n):
    """Priority for greedy cap set construction in {0,1,2}^n.
    
    Higher return value => picked first.
    """
    if time.time() > _DEADLINE:
        return 0.0

    v = vector
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    h = 0
    
    for x in v:
        if x == 0:
            c0 += 1
        elif x == 1:
            c1 += 1
        else:
            c2 += 1
        s += x
        s2 += x * x
        h = (h * _HASH_BASE + x) & _HASH_MASK

    mean = s / n
    var = (s2 / n) - mean * mean
    if var < 0.0:
        var = 0.0

    # Fewer 1s is better.
    # Higher variance is better (spikiness).
    # Jitter for tie-breaking.
    
    # Weights:
    # -5.0 * c1: Strong penalty for 1s.
    # +1.0 * var: Bonus for spread.
    # +0.01 * jitter: Tiny tie-breaker.
    
    jitter = (h & 0xFFFF) / 65536.0
    
    score = -5.0 * c1 + 1.0 * var + 0.01 * jitter
    
    return float(score)
