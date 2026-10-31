"""Cap set greedy construction with geometric conflict avoidance."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _is_cap_set_check(a, b, c):
    """Check if three vectors form a cap set (no three collinear)."""
    # In F3^n, three points a, b, c are collinear iff a + b + c = 0 mod 3
    # i.e., for each coordinate, a[i] + b[i] + c[i] == 0 mod 3
    # Equivalently, c = -a - b mod 3 = (3 - a) - b mod 3 = (2 - a + 3 - b) % 3
    # More simply: c[i] = (2 - a[i] - b[i]) % 3
    # Actually: a + b + c = 0 mod 3 means c = -a-b mod 3
    # -a-b mod 3 = (3 - a - b) % 3
    return np.all((np.array(a) + np.array(b) + np.array(c)) % 3 != 0)

def _third_point(a, b):
    """Given two distinct points a, b in {0,1,2}^n, find the third point c on the line.
    c = -a - b mod 3 (element-wise)"""
    return tuple((3 - a[i] - b[i]) % 3 for i in range(len(a)))

def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    
    The evaluator calls this to get a priority score for each vector.
    Vectors are added in descending priority order.
    We need to design a priority that, when vectors are greedily added
    (skipping those that would create a 3-term arithmetic progression),
    results in a large cap set.
    
    Strategy: Assign priority based on how "spread out" the vector is.
    Vectors with more diverse coordinate values tend to avoid forming lines
    with other diverse vectors.
    
    However, since we don't know the current set state during priority assignment,
    we use a static heuristic that promotes vectors likely to be compatible
    with many others.
    
    Key insight for cap sets in F3^n:
    - A line is {a, a+d, a+2d} for direction d
    - Three points a,b,c form a line iff a+b+c = 0 mod 3 (coordinate-wise)
    - To maximize cap set size, we want to avoid having any triple sum to 0
    
    Heuristic: Prefer vectors where the sum of coordinates mod 3 is balanced,
    and where the vector has a "generic" direction pattern.
    
    Actually, let me think about this differently. The greedy algorithm adds
    vectors in order of priority. When it considers a vector, it checks if
    adding it would create a 3-term AP with any two already-added vectors.
    
    For a vector v to NOT create an AP with two existing vectors a, b,
    we need: v != (3-a-b) mod 3 for all pairs (a,b) in the current set.
    
    A good strategy: prioritize vectors that are "far" from creating APs.
    Since we can't know the current set, use a static ordering that tends
    to work well.
    
    Empirically, for cap sets, a good greedy strategy is to prefer vectors
    with coordinate sum congruent to a specific value mod 3, or to use
    a space-filling curve that spreads vectors out.
    
    Let me use: priority = -sum of squared deviations from 1 (center)
    This prefers vectors centered around 1, which might be more "generic".
    
    Actually, a better approach: use the number of coordinates equal to 0, 1, 2.
    Vectors with balanced counts might be better.
    
    Let me try: priority = -number of distinct values in the vector
    (prefer vectors with all same value? No, that's bad for cap sets)
    
    Let me try a different approach: use a hash-like function based on
    the vector's position in a structured ordering.
    
    For n dimensions over {0,1,2}, there are 3^n vectors.
    A good cap set heuristic: prefer vectors where no coordinate pattern
    repeats too much.
    
    I'll use: priority = sum of (coord - 1)^2, minimized (so negated for max)
    This prefers vectors close to the all-ones vector.
    """
    v = np.array(vector)
    # Prefer vectors with balanced coordinate distribution
    # Count how many 0s, 1s, 2s
    n0 = np.sum(v == 0)
    n1 = np.sum(v == 1)
    n2 = np.sum(v == 2)
    
    # Reward balance: maximize min(n0, n1, n2) or minimize variance
    # Variance of the counts
    counts = np.array([n0, n1, n2], dtype=float)
    mean_count = counts.mean()
    variance = np.sum((counts - mean_count) ** 2)
    
    # Also consider the coordinate sum mod 3
    coord_sum = np.sum(v) % 3
    
    # Combine: lower variance is better, and prefer certain residue classes
    # In F3, vectors with sum 0 mod 3 are in the "hyperplane" through origin
    # Cap sets avoid having 3 collinear, so maybe avoid the hyperplane?
    
    # Let's use: priority = -variance + small bonus for sum mod 3 == 1
    bonus = 0.1 if coord_sum == 1 else 0.0
    
    return -variance + bonus

# EVOLVE-BLOCK-END
