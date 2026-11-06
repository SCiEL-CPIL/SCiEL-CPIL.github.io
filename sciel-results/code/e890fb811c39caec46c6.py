import math
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s limit

    n = 256
    max_sum = 2 * n  # 512

    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)

    # Filter out numbers where 2*a is a square (diagonal constraint)
    # If 2*a is a square, then a+a is a square, so a cannot be in the set.
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}

    # Build conflict list
    # We need x_a + x_b <= 1 for all a < b such that a+b is a square.
    # Note: The problem statement says "a+b is never a perfect square for a<=b in the set, INCLUDING a=b".
    # We already filtered a=b cases (diagonal).
    # So we only need to check a < b.
    
    constraints_rows = []
    constraints_cols = []
    constraints_vals = []
    
    for i, a in enumerate(valid_numbers):
        for s in squares:
            b = s - a
            if b > a and b in val_to_idx:
                j = val_to_idx[b]
                # Constraint: x_i + x_j <= 1
                constraints_rows.append(len(constraints_vals))
                constraints_cols.append(i)
                constraints_vals.append(1)
                
                constraints_rows.append(len(constraints_vals))
                constraints_cols.append(j)
                constraints_vals.append(1)

    c = -np.ones(num_valid)  # Maximize sum x_i => Minimize -sum x_i
    
    # Convert to dense matrix for scipy (num_valid is small enough, ~200)
    # A sparse matrix would be better for large N, but 200x200 is fine dense
    A = np.zeros((len(constraints_rows), num_valid))
    for r, c_idx, v in zip(constraints_rows, constraints_cols, constraints_vals):
        A[r, c_idx] += v
        
    # Remove empty rows if any (shouldn't happen, but safe)
    # Actually, scipy milp handles this.
    
    lb = np.zeros(num_valid)
    ub = np.ones(num_valid)
    integrality = np.ones(num_valid)
    
    # Define constraints: 0 <= x_i <= 1
    # And A x <= 1
    # We can combine or keep separate. scipy milp takes LinearConstraint.
    
    # Bounds
    bounds = Bounds(lb, ub)
    
    # Linear Constraints
    # 0 <= x <= 1 is handled by bounds.
    # Conflict constraints: A x <= 1
    # We need A_lower = -inf, A_upper = 1
    A_lower = -np.inf * np.ones(len(constraints_rows))
    A_upper = np.ones(len(constraints_rows))
    
    lin_cons = LinearConstraint(A, A_lower, A_upper)
    
    # Solve
    try:
        res = milp(
            c=c,
            constraints=[lin_cons],
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0 and res.x is not None:
            # res.x contains the optimal values (0 or 1)
            # Threshold to handle floating point errors
            x = (res.x > 0.5).astype(int)
            
            # Map back to values
            result = []
            for i in range(num_valid):
                if x[i] == 1:
                    result.append(valid_numbers[i])
            
            if result:
                # Verify validity just in case
                result_set = set(result)
                valid = True
                for a in result:
                    if 2 * a in squares:
                        valid = False
                        break
                    for b in result:
                        if a <= b and (a + b) in squares:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    return sorted(result)
                else:
                    # Fallback if verification fails (shouldn't happen)
                    return [1]
            else:
                return [1]
        else:
            # Fallback to greedy if MILP fails
            pass
            
    except Exception as e:
        pass

    # Fallback: Greedy Min-Degree
    # Rebuild adjacency for fallback
    adj = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                j = val_to_idx[b]
                adj[i] |= (1 << j)

    all_mask = (1 << num_valid) - 1
    remaining = all_mask
    selected = 0
    while remaining:
        best_v = -1
        best_deg = float('inf')
        v = remaining
        while v:
            lowbit = v & (-v)
            idx = lowbit.bit_length() - 1
            deg = (adj[idx] & remaining).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = idx
            v ^= lowbit
        if best_v != -1:
            selected |= (1 << best_v)
            remaining &= ~(adj[best_v] | (1 << best_v))
        else:
            break

    result = []
    v = selected
    while v:
        lowbit = v & (-v)
        idx = lowbit.bit_length() - 1
        result.append(valid_numbers[idx])
        v ^= lowbit
        
    if not result:
        return [1]
        
    return sorted(result)
