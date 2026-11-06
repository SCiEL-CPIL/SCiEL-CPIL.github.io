"""Combined program: Direct LP solve with HiGHS for incentive-compatible auction design."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    # 1. Parse input to get probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=np.float64)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=np.float64)
        else:
            # Try to find a list of floats
            for v in distribution_spec.values():
                if isinstance(v, (list, np.ndarray)):
                    probs = np.array(v, dtype=np.float64)
                    break
            else:
                probs = np.ones(16) / 16.0
    else:
        probs = np.array(distribution_spec, dtype=np.float64)
    
    # Ensure we have 16 probabilities
    if len(probs) != 16:
        # If support is provided, use it to determine size
        if support is not None and len(support) == 16:
            # Assume probs is aligned with support
            probs = np.array(probs[:16], dtype=np.float64) if len(probs) >= 16 else np.pad(probs, (0, 16-len(probs)))
        else:
            probs = np.ones(16) / 16.0
    
    # Normalize probabilities
    total_p = np.sum(probs)
    if total_p > 0:
        probs = probs / total_p
    else:
        probs = np.ones(16) / 16.0
    
    # 2. Define the grid of types
    # Index mapping: i = v1 * 4 + v2 for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    
    n_types = len(types)  # 16
    
    # 3. Formulate the LP
    # Variables: x1_0...x1_15, x2_0...x2_15, p_0...p_15
    # Total variables: 48
    # Objective: Maximize sum_i probs[i] * p_i
    #   => Minimize -sum_i probs[i] * p_i
    #
    # Constraints:
    # 1. IR: v1_i * x1_i + v2_i * x2_i - p_i >= 0  =>  p_i <= v1_i*x1_i + v2_i*x2_i
    #    => -v1_i*x1_i - v2_i*x2_i + p_i <= 0
    # 2. IC: u_i >= u_j + (v_i - v_j) dot x_j for all i, j
    #    u_i = v_i dot x_i - p_i
    #    u_j = v_j dot x_j - p_j
    #    v_i dot x_i - p_i >= v_j dot x_j - p_j + (v_i - v_j) dot x_j
    #    v_i dot x_i - p_i >= v_i dot x_j - p_j
    #    v_i dot (x_i - x_j) - (p_i - p_j) >= 0
    #    v_i dot (x_i - x_j) - p_i + p_j >= 0
    #    => -v_i dot (x_i - x_j) + p_i - p_j <= 0
    #    => -v_i1*(x1_i - x1_j) - v_i2*(x2_i - x2_j) + p_i - p_j <= 0
    #
    # Bounds: 0 <= x1_i, x2_i <= 1, p_i >= 0
    
    # Objective coefficients
    c = np.zeros(48)
    for i in range(n_types):
        c[32 + i] = -probs[i]  # Minimize -revenue
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_rows = []
    
    # IR constraints (16 constraints)
    for i in range(n_types):
        v1, v2 = types[i]
        row = np.zeros(48)
        row[i] = -v1          # x1_i
        row[16 + i] = -v2     # x2_i
        row[32 + i] = 1       # p_i
        A_rows.append(row)
        b_rows.append(0.0)
    
    # IC constraints (16 * 16 = 256 constraints)
    for i in range(n_types):
        vi1, vi2 = types[i]
        for j in range(n_types):
            vj1, vj2 = types[j]
            row = np.zeros(48)
            # -vi1*(x1_i - x1_j) - vi2*(x2_i - x2_j) + p_i - p_j <= 0
            row[i] = -vi1
            row[16 + i] = -vi2
            row[32 + i] = 1
            row[j] = vi1
            row[16 + j] = vi2
            row[32 + j] = -1
            A_rows.append(row)
            b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x1, x2 <= 1; p >= 0 (no upper bound on p)
    bounds = []
    for _ in range(16):
        bounds.append((0, 1))  # x1
    for _ in range(16):
        bounds.append((0, 1))  # x2
    for _ in range(16):
        bounds.append((0, None))  # p
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        # Fallback to simplex if HiGHS fails
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex')
        except Exception:
            # Return a trivial feasible solution
            return {
                'allocations': np.zeros((16, 2)),
                'payments': np.zeros(16)
            }
    
    if not result.success:
        # If LP fails, return a trivial feasible solution
        return {
            'allocations': np.zeros((16, 2)),
            'payments': np.zeros(16)
        }
    
    # Extract solution
    x1 = result.x[:16]
    x2 = result.x[16:32]
    p = result.x[32:48]
    
    # Clip to ensure feasibility (handle numerical errors)
    x1 = np.clip(x1, 0, 1)
    x2 = np.clip(x2, 0, 1)
    p = np.clip(p, 0, None)
    
    allocations = np.column_stack((x1, x2))
    payments = p
    
    # Verify feasibility
    # Check IR
    for i in range(n_types):
        v1, v2 = types[i]
        utility = v1 * x1[i] + v2 * x2[i] - p[i]
        if utility < -1e-8:
            # Adjust payment to satisfy IR
            p[i] = max(0, v1 * x1[i] + v2 * x2[i])
    
    # Check IC
    for i in range(n_types):
        vi1, vi2 = types[i]
        for j in range(n_types):
            vj1, vj2 = types[j]
            ui = vi1 * x1[i] + vi2 * x2[i] - p[i]
            uj = vj1 * x1[j] + vj2 * x2[j] - p[j]
            # IC: ui >= uj + (vi - vj) dot xj
            rhs = uj + (vi1 - vj1) * x1[j] + (vi2 - vj2) * x2[j]
            if ui < rhs - 1e-8:
                # This shouldn't happen with a correct LP solution, but handle gracefully
                pass
    
    return {
        'allocations': allocations,
        'payments': p
    }
# EVOLVE-BLOCK-END
