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

# Precompute LCM of 1..24 to convert fractions to integers
# LCM(1, ..., 24) = 26771144400
_LCM_24 = 26771144400
_WEIGHTS = [_LCM_24 // i for i in range(1, 25)]

def _is_reciprocal_subsum_free(S):
    """
    Checks if S is reciprocal-subsum-free.
    Condition: For every a in S, 1/a != sum(1/b for b in T) for any subset T of S\{a}.
    Uses integer weights for speed.
    """
    if not S:
        return False
    
    S = list(set(S))
    n = len(S)
    # Map value to index in S? No, just use values.
    # Weights are indexed by value: W[v] = LCM/v.
    
    # For each element a in S, check if W[a] is a subset sum of {W[b] for b in S, b != a}
    for a in S:
        target_w = _WEIGHTS[a-1]
        others_weights = [_WEIGHTS[b-1] for b in S if b != a]
        
        m = len(others_weights)
        if m == 0:
            continue
            
        mid = m // 2
        left, right = others_weights[:mid], others_weights[mid:]
        
        # Generate subset sums for left
        L_sums = {0}
        for v in left:
            L_sums = L_sums | {x + v for x in L_sums}
        
        # Generate subset sums for right
        R_sums = {0}
        for v in right:
            R_sums = R_sums | {x + v for x in R_sums}
            
        R_set = set(R_sums)
        for l in L_sums:
            if (target_w - l) in R_set:
                return False
    return True

def _generate_conflicts():
    """
    Generates all linear constraints of the form x_a + sum(x_b for b in T) <= |T|
    where 1/a == sum(1/b for b in T).
    Returns a list of tuples (row, rhs) where row is a list of 0s and 1s of length 24.
    """
    constraints = []
    indices = list(range(1, 25))
    
    for a in range(1, 25):
        if _deadline_hit():
            break
        target_w = _WEIGHTS[a-1]
        others = [v for v in indices if v != a]
        others_weights = [_WEIGHTS[v-1] for v in others]
        
        # MITM to find all subsets of 'others' summing to 'target_w'
        m = len(others)
        mid = m // 2
        left_vals, right_vals = others[:mid], others[mid:]
        left_ws, right_ws = others_weights[:mid], others_weights[mid:]
        
        # Generate all subset sums for left part: (sum, mask)
        # We need to track masks to reconstruct T
        L_dict = {}
        for mask in range(1 << len(left_vals)):
            s = 0
            for i in range(len(left_vals)):
                if mask & (1 << i):
                    s += left_ws[i]
            if s <= target_w:
                if s not in L_dict:
                    L_dict[s] = []
                L_dict[s].append(mask)
                
        R_dict = {}
        for mask in range(1 << len(right_vals)):
            s = 0
            for i in range(len(right_vals)):
                if mask & (1 << i):
                    s += right_ws[i]
            if s <= target_w:
                if s not in R_dict:
                    R_dict[s] = []
                R_dict[s].append(mask)
                
        # Combine
        found_Ts = []
        for s_l, masks_l in L_dict.items():
            s_r_needed = target_w - s_l
            if s_r_needed in R_dict:
                for m_l in masks_l:
                    for m_r in R_dict[s_r_needed]:
                        # Reconstruct T
                        T = []
                        for i in range(len(left_vals)):
                            if m_l & (1 << i):
                                T.append(left_vals[i])
                        for i in range(len(right_vals)):
                            if m_r & (1 << i):
                                T.append(right_vals[i])
                        found_Ts.append(T)
                        
        # Add constraints for each T
        # Constraint: x_a + sum(x_b for b in T) <= |T|
        # Variables are indexed 0..23 corresponding to values 1..24.
        # x_a is at index a-1.
        # x_b is at index b-1.
        
        for T in found_Ts:
            if not T: continue
            row = [0] * 24
            row[a-1] = 1
            for b in T:
                row[b-1] = 1
            constraints.append((row, len(T)))
            
    return constraints

def construct():
    _start_clock()
    
    # Try ILP approach
    try:
        if not _deadline_hit():
            constraints = _generate_conflicts()
            if not _deadline_hit():
                c = np.zeros(24)
                for i in range(24):
                    c[i] = -1 # Maximize sum x_i -> Minimize -sum x_i
                    
                if constraints:
                    A_ub = np.array([c[0] for c in constraints])
                    b_ub = np.array([c[1] for c in constraints])
                    lin_cons = LinearConstraint(A_ub, ub=b_ub)
                else:
                    lin_cons = LinearConstraint(np.zeros((0, 24)), ub=np.zeros(0))
                    
                res = milp(
                    c=c,
                    constraints=lin_cons,
                    integrality=np.ones(24),
                    bounds=Bounds(lb=0, ub=1)
                )
                
                if res.success:
                    x = res.x
                    selected = [i+1 for i in range(24) if x[i] > 0.5]
                    # Verify result
                    if _is_reciprocal_subsum_free(selected):
                        return selected
                    else:
                        # Fallback if ILP gave invalid result (shouldn't happen with exact constraints, but safety)
                        pass
                else:
                    # ILP failed, fallback to heuristic
                    pass
    except Exception:
        pass

    # Fallback: Original heuristic logic
    # 1. Generate a strong baseline using greedy descending order
    base_set = []
    for v in range(24, 0, -1):
        if _is_reciprocal_subsum_free([v] + base_set):
            base_set.append(v)
    
    best_set = set(base_set)
    
    # 2. Local Search / Swap Optimization
    rng = random.Random(42)
    all_numbers = list(range(1, 25))
    
    # Multi-start Greedy
    for i in range(50):
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
            best_set = set(current_set)
            
    # 3. Refinement
    remaining = [v for v in all_numbers if v not in best_set]
    remaining.sort(reverse=True)
    
    final_set = list(best_set)
    for v in remaining:
        if _deadline_hit():
            break
        trial = final_set + [v]
        if _is_reciprocal_subsum_free(trial):
            final_set.append(v)
            
    best_set = set(final_set)
    
    if not best_set:
        best_set = {24}
        
    return list(best_set)

if __name__ == "__main__":
    print(construct())
