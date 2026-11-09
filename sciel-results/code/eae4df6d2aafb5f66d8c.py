"""Cap set greedy construction with balanced-digit priority and deadline guard."""
import time
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

_DEADLINE_SECONDS = 320.0  # safety margin before 330s kill
_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.monotonic()
    return time.monotonic() - _start_time

def _remaining():
    return _DEADLINE_SECONDS - _elapsed()

def _is_cap_set(vectors):
    """Check if a list of vectors in {0,1,2}^n forms a cap set (no 3-term AP).
    A 3-term AP: a, b, c with a + c = 2b component-wise (mod 3).
    Equivalently, for any two distinct vectors a, c in the set, 
    the midpoint b = ((a_i + c_i) // 2) mod 3 must NOT be in the set.
    """
    vec_set = set(vectors)
    n = len(vectors[0]) if vectors else 0
    for i, a in enumerate(vectors):
        for j in range(i + 1, len(vectors)):
            c = vectors[j]
            # Compute midpoint mod 3
            mid = tuple((a[k] + c[k]) % 3 for k in range(n))
            # Midpoint is valid if it's distinct from a and c
            if mid != a and mid != c and mid in vec_set:
                return False
    return True

def _adds_cap_set(vec, existing_set, n):
    """Check if adding vec to existing_set (a set of tuples) keeps it a cap set."""
    if vec in existing_set:
        return False
    for a in existing_set:
        # For each a in existing, check if midpoint of a and vec is in existing
        mid = tuple((a[k] + vec[k]) % 3 for k in range(n))
        if mid != a and mid != vec and mid in existing_set:
            return False
    return True

def _balanced_digit_priority(vector):
    """Priority that favors vectors with balanced digit distribution.
    We want to avoid forming APs early. Vectors with roughly equal counts of 0,1,2
    are 'more uniform' and less likely to create APs with other balanced vectors.
    
    The idea: if we pick vectors that are all very similar in distribution, they're
    more likely to form APs. So we want diversity. But actually, the measured finding
    says balanced digit distribution helps. Let's use: negative of variance of digit counts,
    or simply reward having all three digits present.
    
    Actually, let's think: a cap set in F_3^n has no 3-term AP. The greedy approach
    picks vectors in order of priority. If we pick vectors with balanced digits first,
    they are 'spread out' in the space, which may reduce the chance of forming APs
    with later additions.
    
    Let's define priority as: sum of (count_d - n/3)^2 for d in {0,1,2}, negated.
    So more balanced = higher priority.
    """
    counts = [0, 0, 0]
    for v in vector:
        counts[v] += 1
    n = len(vector)
    expected = n / 3.0
    variance = sum((c - expected) ** 2 for c in counts)
    return -variance  # higher priority for more balanced

def _lex_priority(vector):
    """Lexicographic priority: earlier in lex order gets higher priority."""
    # Convert to a number for sorting (higher number = higher priority for descending sort)
    # Actually, we want descending priority, so we want earlier lex vectors to have higher priority
    # Let's assign priority based on negative lex rank
    n = len(vector)
    # Compute the rank in lex order (0 to 3^n - 1)
    rank = 0
    for i, v in enumerate(vector):
        rank = rank * 3 + v
    # Higher priority for smaller rank (earlier in lex order)
    return -rank

def _combined_priority(vector, n):
    """Combine balanced digit priority with a small lex component for tie-breaking."""
    balanced = _balanced_digit_priority(vector)
    # Add a small lex component to break ties
    rank = 0
    for i, v in enumerate(vector):
        rank = rank * 3 + v
    lex_component = -rank / (3 ** n)  # small negative value
    return balanced + lex_component

def priority(vector, n):
    """Return priority for a vector. Higher priority = added earlier in greedy order."""
    return _combined_priority(vector, n)

# EVOLVE-BLOCK-END

# The evaluator will call priority(vector, n) to get priorities, then sort all vectors
# in descending priority and greedily add them if they keep the set a cap set.
# The score is the size of the resulting cap set.

# We need to make sure the function is fast since it will be called for all 3^n vectors.
# For large n, 3^n can be huge, so we need to be careful.
# However, the problem says vectors of {0,1,2}^n, and the evaluator will call priority
# for each vector. We just need to return a float.

# Let's optimize the priority function for speed.
# Precompute nothing since n varies.

def priority(vector, n):
    """Optimized priority function with balanced counts and positional spread."""
    # Count digits
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
    
    expected = n / 3.0
    var = (c0 - expected) ** 2 + (c1 - expected) ** 2 + (c2 - expected) ** 2
    
    # Positional spread: for each digit value, measure how evenly its positions are distributed
    # A vector with digits spread across coordinates is less likely to form APs
    # We compute the variance of position indices for each digit group
    pos0 = []
    pos1 = []
    pos2 = []
    for i, v in enumerate(vector):
        if v == 0:
            pos0.append(i)
        elif v == 1:
            pos1.append(i)
        else:
            pos2.append(i)
    
    def _pos_spread_penalty(positions, n):
        """Penalty for clustered positions. Returns 0 if positions are evenly spread."""
        if len(positions) <= 1:
            return 0.0
        # Expected spacing if evenly distributed
        expected_spacing = n / len(positions)
        # Compute actual spacings between consecutive positions
        spacings = []
        for k in range(len(positions) - 1):
            spacings.append(positions[k + 1] - positions[k])
        # Variance of spacings relative to expected
        mean_spacing = sum(spacings) / len(spacings)
        if mean_spacing == 0:
            return 0.0
        var_spacing = sum((s - mean_spacing) ** 2 for s in spacings) / len(spacings)
        # Normalize by expected spacing squared
        return var_spacing / (expected_spacing ** 2)
    
    spread_penalty = (_pos_spread_penalty(pos0, n) + 
                      _pos_spread_penalty(pos1, n) + 
                      _pos_spread_penalty(pos2, n)) / 3.0
    
    # Lex rank for tie-breaking
    rank = 0
    for v in vector:
        rank = rank * 3 + v
    
    # Combine: balanced count priority dominates, spread is secondary, lex breaks ties
    # Weight: count balance (1.0), positional spread (0.3), lex (tiny)
    return -var - 0.3 * spread_penalty - rank / (3 ** n)
