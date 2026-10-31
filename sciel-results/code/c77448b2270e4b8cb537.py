"""Cap set greedy construction with line-blocking potential heuristic."""
import time
import itertools

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Priority for a vector in {0,1,2}^n.
    
    The greedy algorithm sorts all vectors by descending priority and adds them
    if they don't create a 3-term arithmetic progression (affine line) with
    already-added vectors.
    
    Strategy: Maximize "line-blocking potential" — vectors that, if added, would
    form affine lines with the most other vectors in the space. This aggressively
    eliminates future conflict options early.
    
    We compute this as: for candidate v, count the number of pairs (a, b) such that
    a, v, b form an affine line (a + b = 2v mod 3, i.e., a, v, b are collinear in
    F_3^n). The more such lines v participates in, the more "blocking" it is.
    
    However, since we don't know the current set at priority time (priority is
    static per vector), we use a structural proxy: vectors with more coordinates
    equal to 1 (the midpoint value) participate in more affine lines through them
    as the middle point. Specifically, a vector v is the middle of an affine line
    a, v, b iff a_i + b_i = 2*v_i mod 3 for all i. For a fixed v, the number of
    (a,b) pairs where v is the middle is 3^n (one choice per coordinate for the
    offset). But the number of lines where v is an endpoint varies.
    
    Better structural proxy: Count the number of distinct directions d in F_3^n
    such that v + d and v - d are both in the space (always true for d != 0).
    All non-zero vectors have the same number of such lines passing through them.
    
    Alternative: Use coordinate balance. Vectors with more 1s are "central" and
    block more lines. Also use a tie-breaker based on Hamming weight to spread
    out the selection.
    
    We'll use: priority = 2 * (number of 1s) - (number of 2s) + small random-like
    tie-breaker based on lexicographic position to break symmetry.
    """
    v = tuple(vector)
    count_1 = sum(1 for x in v if x == 1)
    count_2 = sum(1 for x in v if x == 2)
    
    # Central vectors (more 1s) have higher line-blocking potential
    # Vectors with many 0s or 2s are more "peripheral"
    base_priority = 2.0 * count_1 - 1.5 * count_2
    
    # Add a small perturbation to break ties and avoid degenerate symmetry
    # Use a deterministic hash-like function of the vector
    h = 0
    for i, x in enumerate(v):
        h = (h * 31 + x * (i + 1)) % 1000003
    perturbation = (h / 1000003.0 - 0.5) * 0.1
    
    return base_priority + perturbation

# EVOLVE-BLOCK-END
