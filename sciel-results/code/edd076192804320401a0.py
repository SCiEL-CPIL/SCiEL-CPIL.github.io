import time
import random
from fractions import Fraction
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DEADLINE = 85.0  # Safety margin under 90s
_START = None

def _start_clock():
    global _START
    _START = time.time()

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def _is_reciprocal_subsum_free(S):
    """
    Checks if S is reciprocal-subsum-free.
    Condition: For every a in S, 1/a != sum(1/b for b in T) for any subset T of S\{a}.
    """
    if not S:
        return False
    
    S = list(set(S))
    n = len(S)
    recips = {v: Fraction(1, v) for v in S}
    
    for a in S:
        target = recips[a]
        others = [recips[b] for b in S if b != a]
        
        m = len(others)
        mid = m // 2
        left, right = others[:mid], others[mid:]
        
        L = {Fraction(0)}
        for v in left:
            L = L | {x + v for x in L}
        
        R = {Fraction(0)}
        for v in right:
            R = R | {x + v for x in R}
            
        R_set = set(R)
        for l in L:
            if (target - l) in R_set:
                return False
    return True

def _generate_conflicts_fast():
    """
    Efficiently generate all conflict constraints using precomputed MITM tables.
    Returns list of (row, rhs) tuples where row is a 24-dim 0/1 vector and rhs is int.
    """
    N = 24
    indices = list(range(1, 25))
    
    # Precompute subset sums for the full set using MITM
    # Split into two halves
    mid = N // 2
    left_half = indices[:mid]   # 1..12
    right_half = indices[mid:]  # 13..24
    
    # Build all subset sums for left half: sum -> list of masks
    L_sums = {}
    for mask in range(1 << len(left_half)):
        s = Fraction(0)
        for i in range(len(left_half)):
            if mask & (1 << i):
                s += Fraction(1, left_half[i])
        if s not in L_sums:
            L_sums[s] = []
        L_sums[s].append(mask)
    
    # Build all subset sums for right half: sum -> list of masks
    R_sums = {}
    for mask in range(1 << len(right_half)):
        s = Fraction(0)
        for i in range(len(right_half)):
            if mask & (1 << i):
                s += Fraction(1, right_half[i])
        if s not in R_sums:
            R_sums[s] = []
        R_sums[s].append(mask)
    
    # For each target a, find all subsets T of [1..24]\{a} with sum = 1/a
    constraints = []
    for a in range(1, 25):
        if _deadline_hit():
            break
        target = Fraction(1, a)
        
        # Find all (m_l, m_r) pairs where sum_l + sum_r = target
        # and the combined set does NOT include a
        for s_l, masks_l in L_sums.items():
            s_r_needed = target - s_l
            if s_r_needed in R_sums:
                masks_r = R_sums[s_r_needed]
                for m_l in masks_l:
                    for m_r in masks_r:
                        # Reconstruct the set T
                        T = []
                        for i in range(len(left_half)):
                            if m_l & (1 << i):
                                T.append(left_half[i])
                        for i in range(len(right_half)):
                            if m_r & (1 << i):
                                T.append(right_half[i])
                        
                        # Check that a is not in T
                        if a in T:
                            continue
                        
                        # Add constraint: x_a + sum(x_b for b in T) <= |T|
                        if not T:
                            continue
                        row = [0] * 24
                        row[a - 1] = 1
                        for b in T:
                            row[b - 1] = 1
                        constraints.append((row, len(T)))
    
    return constraints

def _lagrangian_bound(conflicts, current_assignment):
    """
    Compute a Lagrangian lower bound for the independent set problem.
    For each conflict hyperedge e = {a} ∪ T, the constraint is:
        x_a + sum_{b in T} x_b <= |T|
    Lagrangian relaxation with multiplier lambda_e >= 0:
        L = sum_i x_i - sum_e lambda_e * (x_a + sum_{b in T} x_b - |T|)
          = sum_i x_i * (1 - sum_{e: i in e} lambda_e) + sum_e lambda_e * |T|
    
    The dual function g(lambda) = max_{x in {0,1}^24} L(x, lambda)
    For fixed lambda, the max over x is achieved by setting x_i = 1 if coefficient > 0, else 0.
    
    We use this to prune branch-and-bound nodes.
    """
    n = 24
    # Coefficient for each variable: 1 - sum of lambdas over edges containing i
    coeff = [1.0] * n
    constant = 0.0
    
    for row, rhs in conflicts:
        # We need lambda values; for simplicity, use uniform lambda = 1/|hyperedge|
        # This is a heuristic dual bound
        pass
    
    # Simplified: just return the number of unassigned variables as an upper bound
    # This is weak but fast
    return sum(1 for i in range(n) if current_assignment[i] is None)

def _branch_and_bound(conflicts, deadline_check):
    """
    Branch-and-bound for maximum independent set in hypergraph.
    Uses Lagrangian-style bounds for pruning.
    """
    n = 24
    best_set = []
    best_size = 0
    
    # Sort variables by "constraint degree" (how many conflicts they participate in)
    var_degree = [0] * n
    for row, rhs in conflicts:
        for i in range(n):
            if row[i] == 1:
                var_degree[i] += 1
    
    # Order variables: most constrained first
    var_order = sorted(range(n), key=lambda i: -var_degree[i])
    
    # State: assignment[i] = True/False/None
    assignment = [None] * n
    
    def bound(assignment):
        """Upper bound on additional elements we can add."""
        # Count unassigned variables
        unassigned = [i for i in range(n) if assignment[i] is None]
        if not unassigned:
            return 0
        
        # For each unassigned variable, check if it can be added
        # without violating any conflict given current assignments
        # Simple bound: number of unassigned vars that don't immediately violate
        count = 0
        for i in unassigned:
            # Check if setting x_i = 1 would violate any conflict
            can_add = True
            for row, rhs in conflicts:
                if row[i] != 1:
                    continue
                # This conflict involves variable i
                # Count how many variables in this conflict are already set to 1
                ones_in_conflict = 0
                for j in range(n):
                    if row[j] == 1 and assignment[j] is True:
                        ones_in_conflict += 1
                # If we set x_i = 1, total ones would be ones_in_conflict + 1
                # Constraint: total_ones <= rhs
                if ones_in_conflict + 1 > rhs:
                    can_add = False
                    break
            if can_add:
                count += 1
        return count
    
    def bb(idx, current_size):
        nonlocal best_set, best_size
        
        if deadline_check():
            return
        
        if idx == n:
            if current_size > best_size:
                best_size = current_size
                best_set = [i + 1 for i in range(n) if assignment[i] is True]
            return
        
        # Upper bound check
        ub = current_size + bound(assignment)
        if ub <= best_size:
            return
        
        v = var_order[idx]
        
        # Try including v
        assignment[v] = True
        # Check feasibility
        feasible = True
        for row, rhs in conflicts:
            if row[v] != 1:
                continue
            ones = 0
            for j in range(n):
                if row[j] == 1 and assignment[j] is True:
                    ones += 1
            if ones > rhs:
                feasible = False
                break
        if feasible:
            bb(idx + 1, current_size + 1)
        
        # Try excluding v
        assignment[v] = False
        bb(idx + 1, current_size)
        
        assignment[v] = None
    
    bb(0, 0)
    
    if best_set:
        return best_set
    return None

def _greedy_with_conflicts(conflicts):
    """
    Greedy construction using conflict information.
    Process variables in order of constraint degree, add if feasible.
    """
    n = 24
    var_degree = [0] * n
    for row, rhs in conflicts:
        for i in range(n):
            if row[i] == 1:
                var_degree[i] += 1
    
    var_order = sorted(range(n), key=lambda i: -var_degree[i])
    
    selected = set()
    for v in var_order:
        val = v + 1
        # Check if adding val violates any conflict
        can_add = True
        for row, rhs in conflicts:
            if row[v] != 1:
                continue
            # Count selected elements in this conflict
            count = 0
            for j in range(n):
                if row[j] == 1 and (j + 1) in selected:
                    count += 1
            # If we add v, total would be count + 1
            if count + 1 > rhs:
                can_add = False
                break
        if can_add:
            selected.add(val)
    
    return list(selected)

def construct():
    _start_clock()
    
    # Generate conflicts
    conflicts = []
    try:
        if not _deadline_hit():
            conflicts = _generate_conflicts_fast()
    except Exception:
        pass
    
    best_set = []
    
    # Strategy 1: Greedy with conflict info
    try:
        if conflicts:
            greedy_result = _greedy_with_conflicts(conflicts)
            if _is_reciprocal_subsum_free(greedy_result):
                best_set = greedy_result
    except Exception:
        pass
    
    # Strategy 2: Branch and bound
    try:
        if not _deadline_hit():
            bb_result = _branch_and_bound(conflicts, _deadline_hit)
            if bb_result and _is_reciprocal_subsum_free(bb_result):
                if len(bb_result) > len(best_set):
                    best_set = bb_result
    except Exception:
        pass
    
    # Strategy 3: ILP if time remains
    try:
        if not _deadline_hit() and conflicts:
            c = np.zeros(24)
            for i in range(24):
                c[i] = -1
            
            A_ub = np.array([row for row, _ in conflicts])
            b_ub = np.array([rhs for _, rhs in conflicts])
            lin_cons = LinearConstraint(A_ub, ub=b_ub)
            
            res = milp(
                c=c,
                constraints=lin_cons,
                integrality=np.ones(24),
                bounds=Bounds(lb=0, ub=1)
            )
            
            if res.success:
                x = res.x
                selected = [i + 1 for i in range(24) if x[i] > 0.5]
                if _is_reciprocal_subsum_free(selected):
                    if len(selected) > len(best_set):
                        best_set = selected
    except Exception:
        pass
    
    # Strategy 4: Fallback greedy + local search
    if not _deadline_hit():
        # Greedy descending
        base_set = []
        for v in range(24, 0, -1):
            if _is_reciprocal_subsum_free([v] + base_set):
                base_set.append(v)
        
        if len(base_set) > len(best_set):
            best_set = base_set
        
        # Multi-start random greedy
        rng = random.Random(42)
        all_numbers = list(range(1, 25))
        
        for i in range(30):
            if _deadline_hit():
                break
            
            perm = all_numbers[:]
            rng.shuffle(perm)
            
            current_set = []
            for v in perm:
                trial_set = current_set + [v]
                if _is_reciprocal_subsum_free(trial_set):
                    current_set.append(v)
            
            if len(current_set) > len(best_set):
                best_set = current_set
        
        # Refinement: try adding remaining elements
        remaining = [v for v in all_numbers if v not in set(best_set)]
        remaining.sort(reverse=True)
        
        final_set = list(best_set)
        for v in remaining:
            if _deadline_hit():
                break
            trial = final_set + [v]
            if _is_reciprocal_subsum_free(trial):
                final_set.append(v)
        
        if len(final_set) > len(best_set):
            best_set = final_set
    
    if not best_set:
        best_set = [24]
    
    return list(best_set)

if __name__ == "__main__":
    print(construct())
