import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    n_points = p ** 4  # 2401
    n_dirs = p * p + 1  # 50
    
    # Field operations for F_49 = F_7[alpha] with alpha^2 = 3
    k = 3
    
    def field_mul(a, b):
        a0, a1 = a
        b0, b1 = b
        c0 = (a0 * b0 + a1 * b1 * k) % p
        c1 = (a0 * b1 + a1 * b0) % p
        return (c0, c1)
        
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def field_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)

    # Encode point (xa, xb, ya, yb) to index
    def encode(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
        
    # Precompute line IDs for each point and direction
    # line_id[point_idx, dir_idx] = integer ID of the line containing point_idx in direction dir_idx
    line_id = np.zeros((n_points, n_dirs), dtype=np.int32)
    
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = encode(xa, xb, ya, yb)
                    x = (xa, xb)
                    y = (ya, yb)
                    for ma in range(p):
                        for mb in range(p):
                            d_idx = ma * p + mb
                            m = (ma, mb)
                            # Line equation: y = m*x + c  => c = y - m*x
                            c = field_add(y, field_neg(field_mul(m, x)))
                            line_id[idx, d_idx] = c[0] * p + c[1]
                    # Vertical direction: x = const
                    line_id[idx, n_dirs - 1] = xa * p + xb

    # We want to find S (complement of N) such that for every point x:
    # There exists a direction v such that:
    #   if x in S: count of S on L_v(x) == 1 (the point x itself)
    #   if x not in S: count of S on L_v(x) == 0
    #
    # Let S be a boolean array of size n_points.
    # For a given S and direction d, let counts_d[line] = sum(S[points on line]).
    # For point x with line_id l, the condition for direction d is satisfied if:
    #   (S[x] and counts_d[l] == 1) or (not S[x] and counts_d[l] == 0)
    #
    # A point x is "satisfied" if ANY direction d satisfies the condition.
    # We want to maximize |S| such that ALL points are satisfied.
    #
    # This is a Max-SAT problem. We use Simulated Annealing.
    # State: S (boolean array)
    # Energy: Number of unsatisfied points. We want to minimize this to 0.
    # Objective: Among states with Energy=0, maximize |S|.
    
    def compute_energy(S):
        """
        Returns (num_unsatisfied, size_S).
        num_unsatisfied: number of points x for which no direction v satisfies the condition.
        """
        satisfied = np.zeros(n_points, dtype=bool)
        
        for d in range(n_dirs):
            lids = line_id[:, d]
            # Count S points on each line for direction d
            # We can use bincount on the line IDs of points in S
            s_indices = np.where(S)[0]
            if len(s_indices) > 0:
                s_lids = lids[s_indices]
                counts = np.bincount(s_lids, minlength=q)
            else:
                counts = np.zeros(q, dtype=np.int32)
            
            counts_for_points = counts[lids]
            
            # Condition:
            # If S[x] is True, we need counts_for_points[x] == 1
            # If S[x] is False, we need counts_for_points[x] == 0
            valid_for_d = (S & (counts_for_points == 1)) | (~S & (counts_for_points == 0))
            
            satisfied |= valid_for_d
            
        num_unsatisfied = np.sum(~satisfied)
        size_S = np.sum(S)
        return num_unsatisfied, size_S

    def compute_energy_delta(S, idx):
        """
        Compute the change in energy if we flip S[idx].
        This is tricky to do incrementally without recomputing everything because
        the "satisfied" status of a point depends on the existence of ANY valid direction.
        
        However, we can approximate or just recompute if it's fast enough.
        Let's check the cost of compute_energy.
        n_dirs=50. For each, bincount is O(N). Total O(50*N) = 120k ops.
        In Python/NumPy, this is very fast, likely < 1ms.
        So we can just recompute energy for every flip in SA.
        """
        # Flip
        old_val = S[idx]
        S[idx] = not S[idx]
        
        new_unsat, new_size = compute_energy(S)
        
        # Revert for return value calculation if needed, but SA will keep the flip if accepted.
        # We return the new state's metrics.
        return new_unsat, new_size, old_val

    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 600  # 10s margin
    
    # Initial state: Random S with low density
    # Or start with empty S (which is valid, energy=0, size=0)
    # Let's start with a random valid state if possible, or just empty.
    # Empty S is always valid.
    current_S = np.zeros(n_points, dtype=bool)
    current_unsat, current_size = compute_energy(current_S)
    
    best_S = current_S.copy()
    best_size = current_size
    best_unsat = current_unsat
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random point to flip
        idx = random.randint(0, n_points - 1)
        
        # Flip it
        old_val = current_S[idx]
        current_S[idx] = not current_S[idx]
        
        new_unsat, new_size = compute_energy(current_S)
        
        delta_unsat = new_unsat - current_unsat
        delta_size = new_size - current_size
        
        # We want to minimize unsat. If unsat decreases, always accept.
        # If unsat increases, accept with probability exp(-delta_unsat / T).
        # If unsat is same, we prefer larger size.
        
        if delta_unsat < 0:
            accept = True
        elif delta_unsat == 0:
            # Prefer larger S
            if delta_size > 0:
                accept = True
            else:
                # Accept with some probability to escape local optima of size
                accept = random.random() < 0.5
        else:
            # delta_unsat > 0
            prob = np.exp(-delta_unsat / T)
            accept = random.random() < prob
        
        if accept:
            current_unsat = new_unsat
            current_size = new_size
            if current_unsat < best_unsat or (current_unsat == best_unsat and current_size > best_size):
                best_unsat = current_unsat
                best_size = current_size
                best_S = current_S.copy()
        else:
            # Revert
            current_S[idx] = old_val
            
        T *= cooling
        
        # If we found a valid set (unsat=0), we can try to increase size greedily or continue SA
        # to find a larger valid set.
        
    # If best_unsat > 0, we don't have a valid set.
    # We need to return a valid set.
    # Fallback: If we didn't find a valid set, return the empty set (N = all points).
    # But we want to maximize complement size, so we want to minimize N size?
    # No, Metric is 2401 - |N|. So we want |N| to be small.
    # |N| = n_points - |S|.
    # So we want |S| to be large.
    # The task asks for N, the Nikodym set.
    # Validity: N is a Nikodym set.
    # The current program logic:
    # S is the complement of N.
    # The condition in the prompt: "every point x ... lie on a line ... whose points other than x all belong to N."
    # This means for every x, there is a line L through x such that L \ {x} subset N.
    # This is equivalent to saying: For every x, there is a direction v such that
    # the line L_v(x) contains NO points from S \ {x}.
    # If x in S: L_v(x) \ {x} must be subset of N => L_v(x) \ {x} disjoint from S => count of S on L_v(x) is 1 (only x).
    # If x not in S: L_v(x) must be subset of N => L_v(x) disjoint from S => count of S on L_v(x) is 0.
    # This matches the `compute_energy` logic.
    
    # If best_unsat > 0, we must fix it.
    # A simple valid set is N = all points (S = empty).
    # Is there a smaller N?
    # If SA failed to find a valid S, we can try a greedy repair on best_S.
    
    if best_unsat > 0:
        # Greedy repair: Remove points from S until valid.
        # This will reduce |S|, increasing |N|, lowering score.
        # But it's better than invalid.
        current_S = best_S.copy()
        unsat, size = compute_energy(current_S)
        # Points that are unsatisfied must be handled.
        # If a point x is unsatisfied, it means for ALL directions v, the condition fails.
        # If x in S: for all v, count_S(L_v(x)) != 1.
        # If x not in S: for all v, count_S(L_v(x)) != 0.
        
        # Simple strategy: Start with empty S (valid), and greedily add points from best_S.
        current_S = np.zeros(n_points, dtype=bool)
        unsat, size = compute_energy(current_S)
        
        # Try to add points from best_S that are True
        candidates = np.where(best_S)[0]
        random.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline:
                break
            current_S[idx] = True
            new_unsat, new_size = compute_energy(current_S)
            if new_unsat == 0:
                # Keep it
                unsat, size = new_unsat, new_size
                best_S = current_S.copy()
                best_size = size
            else:
                # Reject
                current_S[idx] = False
                
        # If still not valid (shouldn't happen since empty is valid), fallback to empty
        if unsat > 0:
            best_S = np.zeros(n_points, dtype=bool)
            best_size = 0

    # Convert best_S to N (complement)
    N_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = encode(xa, xb, ya, yb)
                    if not best_S[idx]:
                        N_points.append([[xa, xb], [ya, yb]])
                        
    if not N_points:
        # Fallback: return all points
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)
