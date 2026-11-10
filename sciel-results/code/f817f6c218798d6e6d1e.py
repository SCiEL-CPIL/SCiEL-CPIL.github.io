# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 with q = 49 (p=7).
    Uses local search with simulated annealing for better optimization.
    """
    p = 7
    q = p * p
    total_points = q * q
    deadline = time.time() + 580.0  # Safety margin

    k = 3  # alpha^2 = 3

    def mul(u, v):
        ua, ub = u
        va, vb = v
        return ((ua*va + ub*vb*k) % p, (ua*vb + ub*va) % p)
        
    def add(u, v):
        return ((u[0]+v[0])%p, (u[1]+v[1])%p)
        
    def get_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute all lines
    lines = []
    line_membership = [[] for _ in range(total_points)]
    
    # Vertical lines: v = (0, 1)
    for xa in range(p):
        for xb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    idx = get_idx(xa, xb, ya, yb)
                    pts.append(idx)
            lines.append(pts)
            for pt in pts:
                line_membership[pt].append(len(lines) - 1)
                
    # Lines with v = (1, m)
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ba in range(p):
                for bb in range(p):
                    pts = []
                    for ta in range(p):
                        for tb in range(p):
                            t = (ta, tb)
                            x = t
                            y = add((ba, bb), mul(m, t))
                            idx = get_idx(x[0], x[1], y[0], y[1])
                            pts.append(idx)
                    lines.append(pts)
                    for pt in pts:
                        line_membership[pt].append(len(lines) - 1)

    num_lines = len(lines)
    lines_per_point = [len(line_membership[x]) for x in range(total_points)]
    
    # For each point, store which lines pass through it
    # line_membership[x] = list of line indices through point x

    # State: in_S[x] = True if x is in S (not in N)
    in_S = np.zeros(total_points, dtype=bool)
    
    # For each line, count how many points are in S
    line_S_count = np.zeros(num_lines, dtype=np.int32)
    
    # For each point, count how many lines through it have S_count == 0 or 1
    # A point x is "satisfied" if:
    #   if x in S: exists line L through x with line_S_count[L] == 1 (only x in S on L)
    #   if x not in S: exists line L through x with line_S_count[L] == 0 (no S on L)
    
    # satisfied[x] = True if x's constraint is met
    satisfied = np.zeros(total_points, dtype=bool)
    
    # For efficiency, track for each point how many "good" lines it has
    # good_lines[x] = number of lines through x that satisfy the condition for x
    good_lines = np.zeros(total_points, dtype=np.int32)
    
    def update_good_lines_for_point(x):
        """Update good_lines[x] based on current in_S and line_S_count."""
        if in_S[x]:
            # Need a line through x with line_S_count == 1
            cnt = 0
            for l in line_membership[x]:
                if line_S_count[l] == 1:
                    cnt += 1
            good_lines[x] = cnt
        else:
            # Need a line through x with line_S_count == 0
            cnt = 0
            for l in line_membership[x]:
                if line_S_count[l] == 0:
                    cnt += 1
            good_lines[x] = cnt
        satisfied[x] = good_lines[x] > 0
    
    def init_state(seed_S=None):
        nonlocal in_S, line_S_count, good_lines, satisfied
        in_S.fill(False)
        line_S_count.fill(0)
        good_lines.fill(0)
        satisfied.fill(False)
        
        if seed_S is not None:
            for x in seed_S:
                in_S[x] = True
                for l in line_membership[x]:
                    line_S_count[l] += 1
        
        for x in range(total_points):
            update_good_lines_for_point(x)
    
    def is_fully_valid():
        return np.all(satisfied)
    
    def count_unsatisfied():
        return np.sum(~satisfied)
    
    def try_add_to_S(x):
        """Try adding x to S. Returns (success, list_of_affected_points)."""
        if in_S[x]:
            return False, []
        
        # Find all affected points: points on any line through x
        affected = set()
        for l in line_membership[x]:
            for pt in lines[l]:
                affected.add(pt)
        
        # Save old state
        old_in_S = in_S[x]
        old_line_counts = {}
        for l in line_membership[x]:
            old_line_counts[l] = line_S_count[l]
        old_good = {}
        for pt in affected:
            old_good[pt] = good_lines[pt]
        old_satisfied = {}
        for pt in affected:
            old_satisfied[pt] = satisfied[pt]
        
        # Apply
        in_S[x] = True
        for l in line_membership[x]:
            line_S_count[l] += 1
        
        # Update affected points
        for pt in affected:
            update_good_lines_for_point(pt)
        
        # Check if any point became unsatisfied
        valid = True
        for pt in affected:
            if not satisfied[pt]:
                valid = False
                break
        
        if not valid:
            # Rollback
            in_S[x] = old_in_S
            for l in line_membership[x]:
                line_S_count[l] = old_line_counts[l]
            for pt in affected:
                good_lines[pt] = old_good[pt]
                satisfied[pt] = old_satisfied[pt]
        
        return valid, list(affected)
    
    def try_remove_from_S(x):
        """Try removing x from S. Returns (success, list_of_affected_points)."""
        if not in_S[x]:
            return False, []
        
        affected = set()
        for l in line_membership[x]:
            for pt in lines[l]:
                affected.add(pt)
        
        old_in_S = in_S[x]
        old_line_counts = {}
        for l in line_membership[x]:
            old_line_counts[l] = line_S_count[l]
        old_good = {}
        for pt in affected:
            old_good[pt] = good_lines[pt]
        old_satisfied = {}
        for pt in affected:
            old_satisfied[pt] = satisfied[pt]
        
        # Apply
        in_S[x] = False
        for l in line_membership[x]:
            line_S_count[l] -= 1
        
        for pt in affected:
            update_good_lines_for_point(pt)
        
        valid = True
        for pt in affected:
            if not satisfied[pt]:
                valid = False
                break
        
        if not valid:
            in_S[x] = old_in_S
            for l in line_membership[x]:
                line_S_count[l] = old_line_counts[l]
            for pt in affected:
                good_lines[pt] = old_good[pt]
                satisfied[pt] = old_satisfied[pt]
        
        return valid, list(affected)
    
    # Local search with simulated annealing
    def local_search(initial_S, max_iters=50000, start_temp=10.0, end_temp=0.1):
        nonlocal in_S, line_S_count, good_lines, satisfied
        init_state(initial_S)
        
        if count_unsatisfied() > 0:
            # Try to fix by removing points from S
            for x in range(total_points):
                if in_S[x]:
                    ok, _ = try_remove_from_S(x)
                    if ok and count_unsatisfied() < total_points:
                        continue
            if count_unsatisfied() > 0:
                # Reset to empty
                init_state(set())
        
        best_size = np.sum(in_S)
        best_state = (in_S.copy(), line_S_count.copy(), good_lines.copy(), satisfied.copy())
        
        current_size = np.sum(in_S)
        current_unsat = count_unsatisfied()
        
        if current_unsat > 0:
            # Find a valid starting point
            init_state(set())
            current_size = 0
            current_unsat = 0
        
        temp = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            temp *= cooling
            
            # Choose a random move: add to S or remove from S
            if random.random() < 0.5:
                # Try to add a random point to S
                x = random.randint(0, total_points - 1)
                if in_S[x]:
                    continue
                ok, affected = try_add_to_S(x)
                if ok:
                    new_size = current_size + 1
                    new_unsat = count_unsatisfied()
                    if new_unsat == 0:
                        # Always accept if valid
                        current_size = new_size
                        current_unsat = 0
                        if new_size > best_size:
                            best_size = new_size
                            best_state = (in_S.copy(), line_S_count.copy(), good_lines.copy(), satisfied.copy())
                    else:
                        # Accept with probability exp(-delta / temp)
                        delta = new_unsat - current_unsat  # should be >= 0
                        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                            current_size = new_size
                            current_unsat = new_unsat
                            if new_size > best_size:
                                best_size = new_size
                                best_state = (in_S.copy(), line_S_count.copy(), good_lines.copy(), satisfied.copy())
                        else:
                            # Rollback
                            in_S[x] = False
                            for l in line_membership[x]:
                                line_S_count[l] -= 1
                            for pt in affected:
                                update_good_lines_for_point(pt)
                else:
                    pass
            else:
                # Try to remove a random point from S
                if current_size == 0:
                    continue
                # Pick a random point in S
                s_points = np.where(in_S)[0]
                if len(s_points) == 0:
                    continue
                x = s_points[random.randint(0, len(s_points)-1)]
                ok, affected = try_remove_from_S(x)
                if ok:
                    new_size = current_size - 1
                    new_unsat = count_unsatisfied()
                    # Removing from S might make some points unsatisfied
                    # But it might allow other additions later
                    delta = new_unsat - current_unsat
                    if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                        current_size = new_size
                        current_unsat = new_unsat
                    else:
                        # Rollback
                        in_S[x] = True
                        for l in line_membership[x]:
                            line_S_count[l] += 1
                        for pt in affected:
                            update_good_lines_for_point(pt)
                else:
                    pass
        
        # Restore best state
        in_S, line_S_count, good_lines, satisfied = best_state
        return best_size
    
    # Strategy: Try multiple random initial seeds
    best_S_size = 0
    best_state = None
    
    # Strategy 1: Empty start
    size1 = local_search(set(), max_iters=20000, start_temp=5.0, end_temp=0.01)
    if size1 > best_S_size:
        best_S_size = size1
        best_state = (in_S.copy(), line_S_count.copy(), good_lines.copy(), satisfied.copy())
    
    if time.time() > deadline:
        in_S, line_S_count, good_lines, satisfied = best_state
    else:
        # Strategy 2: Random sparse seeds
        for trial in range(10):
            if time.time() > deadline:
                break
            seed_size = random.randint(0, 50)
            seed = set(random.sample(range(total_points), seed_size))
            size_t = local_search(seed, max_iters=10000, start_temp=5.0, end_temp=0.01)
            if size_t > best_S_size:
                best_S_size = size_t
                best_state = (in_S.copy(), line_S_count.copy(), good_lines.copy(), satisfied.copy())
    
    if best_state is not None:
        in_S, line_S_count, good_lines, satisfied = best_state
    else:
        init_state(set())
    
    # Return N = F_q^2 \ S
    points = []
    for x in range(total_points):
        if not in_S[x]:
            yb = x % p
            ya = (x // p) % p
            xb = (x // (p*p)) % p
            xa = (x // (p*p*p)) % p
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
