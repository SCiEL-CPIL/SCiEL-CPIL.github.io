"""Cap set task: rank vectors so greedy descending insertion maximizes cap set size.

DIRECT approach with refined structural heuristics:
- A cap set in {0,1,2}^n is a set with no 3-term arithmetic progression.
- In {0,1,2}, a 3-term AP {a,b,c} exists iff a+c = 2*b (mod 3) with a,b,c not all equal.
- The greedy algorithm adds vectors in descending priority order, keeping only those
  that don't create a 3-AP with any two already-added vectors.

Key structural insights:
1. Vectors far from the "center" (value 1) in many coordinates are less likely to be
   midpoints of 3-APs, making them safer to add early.
2. High variance across coordinates indicates structural diversity.
3. Coordinate diversity (using all three symbols) reduces local clustering.
4. A structured deterministic tiebreaker breaks ties in a way that spreads vectors
   across the cube.

We combine:
- Squared edge distance from center (primary, quadratic penalty for centrality)
- Variance (secondary, structural spread)
- Symbol diversity bonus (tertiary, reward using all 3 symbols)
- Structured hash tiebreaker (quaternary, stable ordering)
"""
import time

# Explicit deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 280.0  # 50s safety margin before 330s kill

_SEED = 987654321


def _xorshift(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= (state >> 7)
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF


def _better_hash(vector, n):
    """Structured hash that mixes coordinates with position-dependent weights."""
    h = _SEED
    for i, x in enumerate(vector):
        # Mix coordinate value with position to create structured variation
        pos_weight = (i + 1) * 0x9E3779B97F4A7C15
        h = _xorshift(h ^ (x * 0x85EBCA6B) ^ pos_weight)
    return (h & 0xFFFFFFFF) / 4294967296.0


def priority(vector, n):
    # Deadline guard
    if time.time() > _DEADLINE:
        return 0.0

    if n == 0:
        return 0.0

    v = vector
    c0 = 0
    c1 = 0
    c2 = 0
    s = 0
    s2 = 0
    # Squared edge score: sum of (x-1)^2 for each coordinate
    # This gives quadratic penalty for being near center
    sq_edge = 0
    for x in v:
        if x == 0:
            c0 += 1
            s += 0
            s2 += 0
            sq_edge += 1  # (0-1)^2
        elif x == 1:
            c1 += 1
            s += 1
            s2 += 1
            sq_edge += 0  # (1-1)^2
        else:
            c2 += 1
            s += 2
            s2 += 4
            sq_edge += 1  # (2-1)^2

    mean = s / n
    var = (s2 / n) - mean * mean
    if var < 0:
        var = 0.0

    third = n / 3.0
    bal = (abs(c0 - third) + abs(c1 - third) + abs(c2 - third)) / n

    # Normalized squared edge score: range [0, n], normalize to [0,1]
    sq_edge_norm = sq_edge / n if n > 0 else 0.0

    # Symbol diversity: reward using all three symbols
    # Max diversity = 1.0 when all three are present, less if any symbol is missing
    diversity = 0.0
    if c0 > 0 and c1 > 0 and c2 > 0:
        diversity = 1.0
    elif c0 > 0 and c1 > 0:
        diversity = 0.5
    elif c0 > 0 and c2 > 0:
        diversity = 0.5
    elif c1 > 0 and c2 > 0:
        diversity = 0.5
    else:
        diversity = 0.25

    # Structured deterministic tiebreaker
    rnd = _better_hash(v, n)

    # Combine: prefer high squared edge (far from center), high variance, 
    # high diversity, low balance deviation, structured tiebreak
    score = (
        2.0 * sq_edge_norm
        + 0.6 * var
        + 0.3 * diversity
        - 0.8 * bal
        + 0.02 * rnd
    )
    return float(score)
