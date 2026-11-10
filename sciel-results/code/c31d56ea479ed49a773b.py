# EVOLVE-BLOCK-START
import time
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set N in F_q^2 with q = p^2, p = 7.
    Maximize |N^c| = |S|.
    """
    p = 7
    q = p * p
    NPTS = q * q  # 2401

    start = time.time()
    deadline = start + 580.0  # safety margin

    # --- Field F_p[alpha], alpha^2 = k ---
    k_const = 3  # 3 is a non-residue mod 7 (3^3 = 27 = 6 = -1)

    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * k_const) % p,
                (a[0] * b[1] + a[1] * b[0]) % p)

    def pt_to_idx(pt):
        xa, xb = pt[0]
        ya, yb = pt[1]
        return xa * 343 + xb * 49 + ya * 7 + yb

    def idx_to_pt(idx):
        xa = idx // 343
        idx %= 343
        xb = idx // 49
        idx %= 49
        ya = idx // 7
        yb = idx % 7
        return [[xa, xb], [ya, yb]]

    # --- Directions: (1, m) for all m in F_q, plus (0, 1) ---
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)  # 50

    # --- Precompute affected[p, d, t] ---
    # affected[i, d, j] is the index of the point on the line through i in direction d, at offset j (j=0..48)
    # The point i itself is not included.
    affected = np.zeros((NPTS, num_dirs, q - 1), dtype=np.int32)
    for p_idx in range(NPTS):
        px, py = idx_to_pt(p_idx)[0], idx_to_pt(p_idx)[1]
        for d_idx in range(num_dirs):
            vx, vy = directions[d_idx]
            count = 0
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    tvx = mul((ta, tb), vx)
                    tvy = mul((ta, tb), vy)
                    nx = add(px, tvx)
                    ny = add(py, tvy)
                    affected[p_idx, d_idx, count] = pt_to_idx([[nx[0], nx[1]], [ny[0], ny[1]]])
                    count += 1

    # --- State ---
    # in_S[i] is 1 if point i is in S (complement of N)
    in_S = np.zeros(NPTS, dtype=np.int32)
    
    # line_count[i, d] is the number of points in S on the line through i in direction d, excluding i itself.
    line_count = np.zeros((NPTS, num_dirs), dtype=np.int32)
    
    # zero_counts[i] is the number of directions d such that line_count[i, d] == 0.
    # For validity, we need zero_counts[i] == 0 for all i.
    zero_counts = np.zeros(NPTS, dtype=np.int32)
    
    # To update efficiently, we need to know for each point i and direction d, 
    # which points j are on the line through i in direction d.
    # affected[i, d, :] gives these points.
    # When we add/remove a point j, it affects line_count[i, d] for all i such that j is in affected[i, d, :].
    # Let's precompute reverse_affected: for each point j, and direction d, which points i have j in their line?
    # Actually, the relation is symmetric: j is on the line through i in direction d iff i is on the line through j in direction d.
    # So affected[j, d, :] contains i.
    # Thus, when j changes, we need to update line_count[i, d] for all i in affected[j, d, :].
    
    # --- Initialization ---
    # Start with a small valid S. 
    # A single point S={0} is valid?
    # For i=0, line_count[0, d] = 0 for all d. zero_counts[0] = 50. Invalid.
    # We need S to be a blocking set for all lines.
    # Let's start with S = all points. This is valid (zero_counts all 0).
    # Then we try to remove points to minimize |S|.
    
    in_S[:] = 1
    for i in range(NPTS):
        for d in range(num_dirs):
            line_count[i, d] = q - 1  # All other q-1 points on the line are in S
    zero_counts[:] = 0
    
    # Objective: maximize sum(in_S) is wrong. We want to MINIMIZE sum(in_S).
    # Wait, the task says "Metric (higher is better): the complement size 2401 - |N|".
    # N is the Nikodym set. S is the complement.
    # So we want to MAXIMIZE |S|.
    # My current formulation: in_S[i] = 1 if i in S.
    # Validity condition: For every point x, there is a direction v such that the punctured line is in N.
    # This means for every x, there is a direction d such that NO point on the punctured line is in S.
    # i.e., line_count[x, d] == 0 for some d.
    # So zero_counts[x] must be > 0 for all x.
    # My previous note "zero_counts[i] == 0 for all i" was WRONG.
    # Let's re-verify.
    # "every point x ... must lie on a line ... whose points other than x all belong to N"
    # Points other than x belong to N means they are NOT in S.
    # So the count of points in S on that line (excluding x) must be 0.
    # So line_count[x, d] == 0 for some d.
    # So zero_counts[x] (count of d where line_count[x, d]==0) must be >= 1.
    # Correct.
    
    # So we start with S = all points.
    # line_count[i, d] = 49 for all i, d.
    # zero_counts[i] = 0 for all i.
    # This is INVALID.
    
    # We need a valid starting set.
    # The seed program started with a random point and grew N (so shrank S).
    # Here I want to start with a large S and shrink it? No, start with small S and grow it?
    # If S is small, line_counts are small, so zero_counts are large. Likely valid.
    # Let's start with S = empty set.
    # in_S[:] = 0
    # line_count[:] = 0
    # zero_counts[:] = num_dirs (50)
    # This is valid.
    # Then we add points to S to maximize |S| while maintaining validity.
    
    # Reset state for S = empty
    in_S[:] = 0
    line_count[:] = 0
    zero_counts[:] = num_dirs
    
    def is_valid():
        return np.all(zero_counts > 0)
    
    def try_add(p_idx):
        """
        Try to add p_idx to S.
        Returns list of changes to apply, or None if invalid.
        Adding p_idx to S increases line_count[i, d] for all i such that p_idx is on the line through i in direction d.
        i.e., for all i in affected[p_idx, d, :] (which is symmetric).
        Specifically, for each direction d, and each i in affected[p_idx, d, :]:
            line_count[i, d] increases by 1.
            If it was 0, zero_counts[i] decreases by 1.
            If zero_counts[i] becomes 0, it's invalid.
        """
        changes = []
        for d in range(num_dirs):
            for i in affected[p_idx, d, :]:
                if line_count[i, d] == 0:
                    if zero_counts[i] == 1:
                        return None
                    changes.append((i, d))
        return changes

    def apply_add(p_idx, changes):
        in_S[p_idx] = 1
        for i, d in changes:
            line_count[i, d] += 1
            zero_counts[i] -= 1

    def try_remove(p_idx):
        """
        Try to remove p_idx from S.
        Returns list of changes to apply, or None if invalid.
        Removing p_idx from S decreases line_count[i, d] for all i in affected[p_idx, d, :].
        If line_count[i, d] was 1, it becomes 0, so zero_counts[i] increases by 1.
        This always maintains validity (or improves it).
        Wait, removing a point from S makes it easier to satisfy the condition "some line has 0 points in S".
        So removal is always safe?
        Yes. If S is valid, S\{p} is also valid.
        Proof: For any x, there was a direction d with line_count[x, d] == 0.
        Removing a point can only decrease line counts.
        So line_count[x, d] remains 0.
        So validity is preserved.
        """
        changes = []
        for d in range(num_dirs):
            for i in affected[p_idx, d, :]:
                if line_count[i, d] > 0:
                    if line_count[i, d] == 1:
                        changes.append((i, d))
        return changes

    def apply_remove(p_idx, changes):
        in_S[p_idx] = 0
        for i, d in changes:
            line_count[i, d] -= 1
            if line_count[i, d] == 0:
                zero_counts[i] += 1

    # Strategy:
    # 1. Start with S = empty.
    # 2. Greedily add points to S.
    #    Choose a point p that can be added without breaking validity.
    #    To maximize |S|, we just add any valid point.
    #    But we might get stuck in a local optimum where no single point can be added.
    # 3. To escape local optima, use local search (simulated annealing or hill climbing with restarts).
    #    Move: Add a point (if valid), or Remove a point (always valid), or Swap (Remove p, Add q).
    #    Since we want to maximize |S|, we prefer Add.
    #    If no Add is possible, try Swap?
    #    Or just restart with a different random seed/order.
    
    # The seed program used random seeds and greedy add.
    # Let's enhance the greedy add with a scoring function to pick the "best" point to add.
    # Score: How many new zero_counts does it create? (i.e., how many i, d pairs go from 1->0? No, 0->1 decreases zero_counts).
    # We want to minimize the decrease in "slack".
    # Slack = sum(zero_counts).
    # Adding p decreases slack by the number of (i, d) where line_count[i, d] was 0.
    # Let's pick the p that minimizes this decrease.
    
    def get_add_score(p_idx):
        """
        Returns the number of (i, d) pairs where line_count[i, d] == 0.
        Lower is better (less reduction in slack).
        """
        score = 0
        for d in range(num_dirs):
            for i in affected[p_idx, d, :]:
                if line_count[i, d] == 0:
                    score += 1
        return score

    best_S_size = 0
    best_in_S = None
    best_line_count = None
    best_zero_counts = None
    
    np.random.seed(42)
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Reset state
        in_S[:] = 0
        line_count[:] = 0
        zero_counts[:] = num_dirs
        
        # Randomize the order of points to break symmetry
        order = np.random.permutation(NPTS)
        
        # Greedy phase
        added = 0
        for p_idx in order:
            if time.time() > deadline - 5:
                break
            changes = try_add(p_idx)
            if changes is not None:
                apply_add(p_idx, changes)
                added += 1
        
        # Local search phase: Try to add more points by swapping?
        # Or just try to add points that were previously blocked?
        # Let's do a few passes of "try to add any point".
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            progress = False
            for p_idx in np.random.permutation(NPTS):
                if in_S[p_idx]:
                    continue
                changes = try_add(p_idx)
                if changes is not None:
                    apply_add(p_idx, changes)
                    progress = True
            if not progress:
                break
        
        if np.sum(in_S) > best_S_size:
            best_S_size = np.sum(in_S)
            best_in_S = in_S.copy()
            best_line_count = line_count.copy()
            best_zero_counts = zero_counts.copy()

    # Use the best found S
    if best_in_S is not None:
        in_S = best_in_S
    else:
        # Fallback: S = empty
        in_S = np.zeros(NPTS, dtype=np.int32)

    # Construct N = complement of S
    # N contains points where in_S[i] == 0
    arr = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = pt_to_idx([[xa, xb], [ya, yb]])
                    if in_S[idx] == 0:
                        arr.append([[xa % p, xb % p], [ya % p, yb % p]])
    
    return np.array(arr, dtype=np.int64)

# EVOLVE-BLOCK-END
