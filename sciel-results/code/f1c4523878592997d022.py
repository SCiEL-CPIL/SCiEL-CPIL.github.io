import time
import math

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s

# Global state to track accepted vectors for adaptive priority
_accepted_vectors = []
_accepted_set = set()

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy: Adaptive Conflict Avoidance.
    1. Primary: Minimize the number of 'near-miss' arithmetic progressions.
       A vector is 'dangerous' if it is close (Hamming distance <= 2) to existing vectors.
       This is because cap set violations (x, x+y, x+2y) often involve vectors
       that are structurally similar. By penalizing vectors that are close to
       already accepted ones, we encourage a more spread-out, diverse set.
    2. Secondary: Prefer vectors with balanced symbol counts (0,1,2) to avoid
       clustering in subcubes like {0,1}^n which are prone to cap set violations.
    3. Tertiary: Deterministic hash for tie-breaking.
    
    This approach is stateful: the priority of a vector changes as the set grows.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    # Convert to tuple for hashing
    v_tuple = tuple(vector)
    
    # If this vector is already accepted, it shouldn't be considered again,
    # but the greedy algorithm handles that. We just need a good priority.
    
    # 1. Adaptive Conflict Score
    # Count how many accepted vectors are within Hamming distance 2.
    # This is O(N_accepted * n). To keep it fast, we only do this for the first ~1000 accepted vectors
    # or if the set is small. For larger sets, we approximate or skip.
    # However, since the greedy process adds vectors one by one, and we are in the 'priority' call,
    # we can afford a slightly more expensive check if it significantly improves the score.
    
    # Optimization: Only check against a subset of accepted vectors if the set is large.
    # But for cap sets, the density is low, so N_accepted is likely small relative to 3^n.
    
    conflict_score = 0.0
    if _accepted_set:
        # Check against all accepted vectors for accuracy in the early/mid game.
        # If performance is an issue, we can sample.
        for acc_v in _accepted_set:
            # Fast Hamming distance check
            # If distance <= 2, it's a potential conflict zone.
            # We count the number of such vectors.
            dist = sum(1 for a, b in zip(v_tuple, acc_v) if a != b)
            if dist <= 2:
                conflict_score += 1.0
            elif dist == 3:
                conflict_score += 0.5  # Less dangerous, but still nearby
    
    # The more conflicts, the lower the priority.
    # We use an exponential decay to make the penalty stronger for high conflict counts.
    conflict_penalty = -math.exp(conflict_score * 0.5)
    
    # 2. Balance Score
    # Prefer vectors with all three symbols present.
    c0 = v_tuple.count(0)
    c1 = v_tuple.count(1)
    c2 = v_tuple.count(2)
    
    # Reward having all 3 symbols
    if c0 > 0 and c1 > 0 and c2 > 0:
        distinct_score = 10.0
    else:
        distinct_score = 0.0
        
    # Reward balance
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -dev * 0.1
    
    # 3. Hash for tie-breaking
    h = 0
    for i, val in enumerate(v_tuple):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF * 0.01  # Small weight

    total_score = distinct_score + balance_score + conflict_penalty + hash_score
    
    return float(total_score)

# We need a way to update the global state when a vector is accepted.
# However, the interface only provides `priority`. 
# The FunSearch evaluator typically calls `priority` for all vectors, sorts, and then adds the top one.
# It does NOT call a callback when a vector is added.
# This means my stateful approach `_accepted_set` will NOT be updated by the evaluator.
# 
# WAIT. The standard FunSearch interface for this task is:
# `priority(vector, n) -> float`
# The evaluator generates all vectors, calls `priority` for each, sorts by priority, and then greedily adds them if they don't violate the cap set property.
# My `priority` function is called BEFORE the set is built. So I cannot use the actual accepted set.
#
# This is a critical realization. The current program is stateless.
# My "stateful" idea fails because I don't have access to the current set during the priority calculation.
#
# I must revert to a stateless strategy but make it STRUCTURALLY DIFFERENT.
#
# NEW STRATEGY: Geometric Potential Energy.
# Instead of counting symbols, we map each vector to a point in a 2D plane using two orthogonal basis vectors in {0,1,2}^n.
# Then, we compute the "potential energy" of the vector relative to a fixed origin (e.g., all zeros).
# Vectors that are "far" from the origin in this geometric sense are less likely to form short APs with the origin?
# No, that's not robust.
#
# Let's look at Inspiration 1: "global spectral representation by scoring vectors based on the magnitude of their Discrete Fourier Transform over Z_3^n".
# This is a strong candidate. It uses the structure of the group Z_3^n.
# A cap set is a set with no non-trivial solutions to x+y+z=0.
# In Fourier analysis, large cap sets have specific spectral properties.
# Specifically, we can score a vector by its "spectral norm" or how "random" it looks in the frequency domain.
#
# However, computing the full DFT for every vector is expensive O(n * 3^n) or similar.
# But we can use a *local* spectral feature.
#
# Let's try a different structural change: **Randomized Geometric Hashing with Conflict Estimation**.
# Since we can't see the set, we must estimate which vectors are likely to conflict with *any* other vector in a large cap set.
# Vectors with high "autocorrelation" or specific symmetries are more likely to be part of APs.
#
# Let's try: **Variance-based Priority**.
# Vectors that are "extreme" (e.g., all 0s or all 1s) are dangerous because they are in many APs.
# Vectors that are "central" (balanced) are safer.
# The current program does this with L1 distance.
#
# Let's try a **Quadratic Penalty** instead of L1.
# The current program uses `abs(c0 - ideal) + ...`.
# Let's use `(c0 - ideal)^2 + (c1 - ideal)^2 + (c2 - ideal)^2`.
# This penalizes extreme imbalances more heavily.
#
# ALSO: Add a **Symmetry Breaking Term**.
# In {0,1,2}^n, the vector (0,0,...,0) is very dangerous.
# We can penalize vectors that are close to the all-0, all-1, or all-2 vectors.
#
# Let's combine:
# 1. Quadratic Balance Score.
# 2. Distance from "Dangerous Corners" (all 0, all 1, all 2).
# 3. Hash.

def priority(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        return 0.0

    v_tuple = tuple(vector)
    
    c0 = v_tuple.count(0)
    c1 = v_tuple.count(1)
    c2 = v_tuple.count(2)
    
    # 1. Quadratic Balance Score
    ideal = n / 3.0
    dev0 = c0 - ideal
    dev1 = c1 - ideal
    dev2 = c2 - ideal
    balance_score = -(dev0*dev0 + dev1*dev1 + dev2*dev2)
    
    # 2. Distance from Dangerous Corners
    # Penalize vectors that are close to (0,0,...,0), (1,1,...,1), (2,2,...,2)
    # These vectors are in many arithmetic progressions.
    dist_to_0 = c0 # Number of 0s is distance from all-1s? No.
    # Distance from all-0s is number of non-zeros.
    dist_to_0 = n - c0
    dist_to_1 = n - c1
    dist_to_2 = n - c2
    
    # We want to be FAR from these corners.
    # So we reward large distances.
    corner_score = (dist_to_0 + dist_to_1 + dist_to_2) * 0.1
    
    # 3. Hash
    h = 0
    for i, val in enumerate(v_tuple):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    hash_score = h / 0x7FFFFFFF * 0.01

    return float(balance_score + corner_score + hash_score)
