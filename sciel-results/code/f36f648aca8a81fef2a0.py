"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _count_nonzero_square_fast(coeffs):
    """Count nonzero terms in the exact square using NumPy convolution."""
    n = len(coeffs)
    if n == 0:
        return 0
    c = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(c, c)
    return int(np.count_nonzero(sq))

def _validate(coeffs):
    """Check all constraints on the coefficient list."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    for c in coeffs:
        if not isinstance(c, int) or c < -64 or c > 64:
            return False
    nonzero = sum(1 for c in coeffs if c != 0)
    if nonzero != 12:
        return False
    return True

def _sparse_to_dense(sparse, max_deg=40):
    """Convert sparse list of (exponent, coeff) to dense coefficient list."""
    coeffs = [0] * (max_deg + 1)
    for exp, c in sparse:
        if 0 <= exp <= max_deg:
            coeffs[exp] = c
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs

def _dense_to_sparse(coeffs):
    """Convert dense coefficient list to sparse representation."""
    return [(i, c) for i, c in enumerate(coeffs) if c != 0]

def _square_nonzero_count(sparse):
    """Count nonzero terms in square from sparse representation directly."""
    dense = _sparse_to_dense(sparse)
    return _count_nonzero_square_fast(dense)

def _pairwise_cancellation_score(sparse):
    """
    Estimate how many cross-terms cancel.
    For each pair (i,j) with i<j, the cross term at degree e_i+e_j is 2*c_i*c_j.
    If two pairs have the same sum degree, their cross terms add up.
    We want many pairs to share sum degrees so cancellations can happen.
    """
    from collections import defaultdict
    pair_sums = defaultdict(int)
    n = len(sparse)
    for i in range(n):
        e_i, c_i = sparse[i]
        for j in range(i + 1, n):
            e_j, c_j = sparse[j]
            pair_sums[e_i + e_j] += 2 * c_i * c_j
    # Count how many pair-sum degrees have zero total
    cancellations = sum(1 for v in pair_sums.values() if v == 0)
    return cancellations

def _hill_climb(sparse_init, deadline, coeff_choices=None):
    """
    Focused hill-climbing with pairwise-cancellation-aware moves.
    """
    if coeff_choices is None:
        coeff_choices = [-2, -1, 1, 2]
    
    current_sparse = list(sparse_init)
    current_sq = _square_nonzero_count(current_sparse)
    
    best_sparse = list(current_sparse)
    best_sq = current_sq
    
    random.seed(42)
    max_exp = 40
    stall_count = 0
    max_stall = 5000
    
    while time.time() < deadline - 3.0 and stall_count < max_stall:
        move_type = random.choices(
            ['flip_sign', 'shift_exp', 'change_coeff', 'shift_pair', 'coeff_pair', 'swap_exps'],
            weights=[25, 25, 15, 15, 10, 10]
        )[0]
        
        trial_sparse = list(current_sparse)
        valid = True
        
        if move_type == 'flip_sign':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            trial_sparse[idx] = (e, -c)
            
        elif move_type == 'shift_exp':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_e = e + shift
            if 0 <= new_e <= max_exp:
                existing_exps = set(x[0] for x in trial_sparse if x[0] != e)
                if new_e not in existing_exps:
                    trial_sparse[idx] = (new_e, c)
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'change_coeff':
            idx = random.randrange(len(trial_sparse))
            e, c = trial_sparse[idx]
            new_c = random.choice(coeff_choices)
            if new_c != c:
                trial_sparse[idx] = (e, new_c)
            else:
                valid = False
                
        elif move_type == 'shift_pair':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                shift_i = random.choice([-2, -1, 1, 2])
                shift_j = random.choice([-2, -1, 1, 2])
                new_ei = ei + shift_i
                new_ej = ej + shift_j
                if 0 <= new_ei <= max_exp and 0 <= new_ej <= max_exp:
                    other_exps = set(x[0] for k, x in enumerate(trial_sparse) if k not in (i, j))
                    if new_ei not in other_exps and new_ej not in other_exps and new_ei != new_ej:
                        trial_sparse[i] = (new_ei, ci)
                        trial_sparse[j] = (new_ej, cj)
                    else:
                        valid = False
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'coeff_pair':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                new_ci = random.choice(coeff_choices)
                new_cj = random.choice(coeff_choices)
                if new_ci != ci or new_cj != cj:
                    trial_sparse[i] = (ei, new_ci)
                    trial_sparse[j] = (ej, new_cj)
                else:
                    valid = False
            else:
                valid = False
                
        elif move_type == 'swap_exps':
            if len(trial_sparse) >= 2:
                i, j = random.sample(range(len(trial_sparse)), 2)
                ei, ci = trial_sparse[i]
                ej, cj = trial_sparse[j]
                # Swap exponents
                trial_sparse[i] = (ej, ci)
                trial_sparse[j] = (ei, cj)
        
        if not valid:
            continue
        
        if len(trial_sparse) != 12:
            continue
        exps = [x[0] for x in trial_sparse]
        if len(exps) != len(set(exps)):
            continue
        
        trial_dense = _sparse_to_dense(trial_sparse)
        if not _validate(trial_dense):
            continue
        
        trial_sq = _square_nonzero_count(trial_sparse)
        
        if trial_sq < current_sq:
            current_sparse = trial_sparse
            current_sq = trial_sq
            stall_count = 0
            if trial_sq < best_sq:
                best_sq = trial_sq
                best_sparse = list(trial_sparse)
        else:
            stall_count += 1
            # Accept worse moves with small probability for exploration
            if random.random() < 0.03:
                current_sparse = trial_sparse
                current_sq = trial_sq
    
    return best_sparse, best_sq

def _structured_search(deadline):
    """
    Structured search: try specific algebraic constructions.
    Key insight: (a+b+c+...)^2 has many cross terms. We want to arrange
    exponents and coefficients so cross terms cancel.
    
    Strategy: Group terms into pairs that share sum degrees.
    If we have terms at exponents e1, e2, e3, e4 where e1+e4 = e2+e3,
    then the cross terms at that degree are 2*c1*c4 + 2*c2*c3.
    If c1*c4 = -c2*c3, they cancel.
    """
    best_sparse = None
    best_sq = float('inf')
    
    random.seed(99)
    
    # Construction 1: Symmetric pairs
    # Place terms in pairs (a, b) where a+b = S for some constant S
    # Then cross terms within a pair cancel if coefficients are opposite
    # Actually, for 12 terms, we can have 6 pairs all with the same sum S
    # But we need distinct exponents. Let's try:
    # Terms at: (S-10, c1), (S-9, c2), ..., (S-1, c6), (S+1, c7), ..., (S+10, c12)
    # No, that's 12 terms but not pairs.
    
    # Better: 6 pairs, each pair (e, S-e) for distinct e values
    # Cross term at degree S from pair i: 2*c_i * c_{S-e_i}
    # If we set c_i = -c_{S-e_i}, the cross term at S is 0.
    # But there are also cross terms at other degrees.
    
    # Let's try a specific construction:
    # 12 terms, 6 pairs with the same sum S
    # Pair k: exponents (S - 2k - 1, S - 2k) for k=0..5
    # Coefficients: (1, -1) for each pair
    # This gives cancellation at degree S from each pair.
    
    for S in range(20, 40):
        sparse = []
        valid = True
        for k in range(6):
            e1 = S - 2 * k - 1
            e2 = S - 2 * k
            if e1 < 0 or e2 > 40:
                valid = False
                break
            sparse.append((e1, 1))
            sparse.append((e2, -1))
        if not valid:
            continue
        if len(sparse) != 12:
            continue
        exps = [x[0] for x in sparse]
        if len(exps) != len(set(exps)):
            continue
        dense = _sparse_to_dense(sparse)
        if not _validate(dense):
            continue
        sq = _square_nonzero_count(sparse)
        if sq < best_sq:
            best_sq = sq
            best_sparse = list(sparse)
    
    # Construction 2: Three groups of 4, with internal cancellation
    # Group at exponents {a, a+1, a+2, a+3} with coeffs {1, -1, -1, 1}
    # Square of this group: (x^a - x^{a+1} - x^{a+2} + x^{a+3})^2
    # = x^{2a} - 2x^{2a+1} - 2x^{2a+2} + 2x^{2a+3} + x^{2a+2} - 2x^{2a+3} + x^{2a+4} + 2x^{2a+3} - 2x^{2a+4} + x^{2a+4}
    # Let me compute more carefully:
    # (x^a - x^{a+1} - x^{a+2} + x^{a+3})^2
    # = x^{2a} + x^{2a+2} + x^{2a+4} + x^{2a+6} - 2x^{2a+1} - 2x^{2a+2} + 2x^{2a+3} - 2x^{2a+3} - 2x^{2a+4} + 2x^{2a+5}
    # Wait, let me just compute it.
    
    # For each group, the square has some nonzero terms. Cross terms between groups add more.
    # We want cross terms between groups to cancel.
    
    # Let's try: 3 groups, each with 4 terms, arranged so that cross-group terms cancel.
    # Group 1 at exponents {0, 1, 2, 3} with coeffs {1, -1, -1, 1}
    # Group 2 at exponents {10, 11, 12, 13} with coeffs {1, -1, -1, 1}
    # Group 3 at exponents {20, 21, 22, 23} with coeffs {1, -1, -1, 1}
    
    for offset in range(0, 15):
        sparse = []
        valid = True
        for g in range(3):
            base = offset + g * 10
            for k, c in enumerate([1, -1, -1, 1]):
                e = base + k
                if e > 40:
                    valid = False
                    break
                sparse.append((e, c))
            if not valid:
                break
        if not valid:
            continue
        if len(sparse) != 12:
            continue
        exps = [x[0] for x in sparse]
        if len(exps) != len(set(exps)):
            continue
        dense = _sparse_to_dense(sparse)
        if not _validate(dense):
            continue
        sq = _square_nonzero_count(sparse)
        if sq < best_sq:
            best_sq = sq
            best_sparse = list(sparse)
    
    # Construction 3: Random with pairwise cancellation targeting
    for _ in range(500):
        sparse = []
        # Choose 12 random exponents
        exps = sorted(random.sample(range(0, 41), 12))
        for e in exps:
            c = random.choice([-2, -1, 1, 2])
            sparse.append((e, c))
        dense = _sparse_to_dense(sparse)
        if not _validate(dense):
            continue
        sq = _square_nonzero_count(sparse)
        if sq < best_sq:
            best_sq = sq
            best_sparse = list(sparse)
    
    if best_sparse is not None:
        return best_sparse, best_sq
    return None, float('inf')

def _multi_start_hill_climb(deadline):
    """
    Run hill-climbing from multiple diverse starting points.
    """
    best_sparse = None
    best_sq = float('inf')
    
    random.seed(777)
    
    # Generate diverse starting points
    start_points = []
    
    # Start 1: All ones
    start_points.append([(i, 1) for i in range(12)])
    
    # Start 2: Alternating signs, spread out
    start_points.append([(i * 3, 1 if i % 2 == 0 else -1) for i in range(12) if i * 3 < 41])
    while len(start_points[-1]) < 12:
        start_points[-1].append((len(start_points[-1]) * 3, 1))
    start_points[-1] = start_points[-1][:12]
    
    # Start 3: Random
    for _ in range(10):
        exps = sorted(random.sample(range(0, 41), 12))
        sparse = [(e, random.choice([-2, -1, 1, 2])) for e in exps]
        start_points.append(sparse)
    
    # Start 4: Symmetric pairs
    for S in [20, 25, 30]:
        sparse = []
        for k in range(6):
            e1 = S - 2 * k - 1
            e2 = S - 2 * k
            if 0 <= e1 <= 40 and 0 <= e2 <= 40 and e1 != e2:
                sparse.append((e1, 1))
                sparse.append((e2, -1))
        if len(sparse) == 12:
            exps = [x[0] for x in sparse]
            if len(exps) == len(set(exps)):
                start_points.append(sparse)
    
    # Run hill-climb from each start point with allocated time
    num_starts = len(start_points)
    time_per_start = max(2.0, (deadline - time.time() - 5.0) / num_starts)
    
    for i, sp in enumerate(start_points):
        if time.time() >= deadline - 5.0:
            break
        if time_per_start < 1.0:
            time_per_start = 1.0
        
        # Validate start point
        dense = _sparse_to_dense(sp)
        if not _validate(dense):
            continue
        
        sq = _square_nonzero_count(sp)
        if sq < best_sq:
            best_sq = sq
            best_sparse = list(sp)
        
        # Run hill-climb
        try:
            found_sparse, found_sq = _hill_climb(sp, time.time() + time_per_start)
            if found_sq < best_sq:
                best_sq = found_sq
                best_sparse = list(found_sparse)
        except Exception:
            pass
    
    if best_sparse is not None:
        return best_sparse, best_sq
    return None, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initial candidate
    initial_sparse = [(i * 3, 1 if i % 2 == 0 else -1) for i in range(12)]
    initial_sparse = [(e, c) for e, c in initial_sparse if e < 41]
    while len(initial_sparse) < 12:
        initial_sparse.append((len(initial_sparse) * 3, 1))
    initial_sparse = initial_sparse[:12]
    
    initial_dense = _sparse_to_dense(initial_sparse)
    if not _validate(initial_dense):
        initial_dense = [1] * 12
        initial_sparse = _dense_to_sparse(initial_dense)
    
    best_sparse = list(initial_sparse)
    best_sq = _square_nonzero_count(best_sparse)
    
    # Phase 1: Structured search
    if time.time() < deadline - 40.0:
        try:
            struct_sparse, struct_sq = _structured_search(deadline - 35.0)
            if struct_sparse is not None and struct_sq < best_sq:
                best_sq = struct_sq
                best_sparse = list(struct_sparse)
        except Exception:
            pass
    
    # Phase 2: Multi-start hill climbing
    if time.time() < deadline - 10.0:
        try:
            found_sparse, found_sq = _multi_start_hill_climb(deadline)
            if found_sparse is not None and found_sq < best_sq:
                best_sq = found_sq
                best_sparse = list(found_sparse)
        except Exception:
            pass
    
    # Final: One more hill-climb from best
    if time.time() < deadline - 5.0:
        try:
            final_sparse, final_sq = _hill_climb(best_sparse, deadline)
            if final_sq < best_sq:
                best_sq = final_sq
                best_sparse = list(final_sparse)
        except Exception:
            pass
    
    # Convert to dense and validate
    best_coeffs = _sparse_to_dense(best_sparse)
    if not _validate(best_coeffs):
        best_coeffs = [1] * 12
    
    return best_coeffs
# EVOLVE-BLOCK-END
