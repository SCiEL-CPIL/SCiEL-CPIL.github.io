"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Cubic B-spline basis with 10 control points over x in [-3, 3].
    The 10 params are the control-point values.
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)

    # Knot vector for cubic B-splines on [-3, 3] with 10 control points
    # Uniform knots: 10 control points need 10+4=14 knots (with multiplicity at boundaries)
    # Interior knots: 10 - 4 = 6 interior knots
    # Total knots: 4 + 6 + 4 = 14
    # Boundary knots at -3 and 3 with multiplicity 4
    # Interior knots evenly spaced in (-3, 3)
    
    # Actually, for n_ctrl=10 cubic B-splines, we need n_ctrl + 4 = 14 knots
    # With clamped boundaries: knots = [-3, -3, -3, -3, k1, k2, ..., k6, 3, 3, 3, 3]
    # That's 4 + 6 + 4 = 14 knots. Good.
    
    interior_knots = np.linspace(-3.0, 3.0, 8)[1:-1]  # 6 interior knots
    knots = np.concatenate([np.full(4, -3.0), interior_knots, np.full(4, 3.0)])
    
    # Compute cubic B-spline basis values using de Boor's algorithm / direct evaluation
    # For efficiency, we'll compute the basis matrix directly.
    # Number of basis functions = 10
    # For each x_i, find which knot span it belongs to, then compute the 4 non-zero basis values.
    
    # Precompute knot indices for each x point
    # np.searchsorted gives the index such that knots[idx-1] <= x < knots[idx]
    # For B-spline evaluation, we need the span index i such that knots[i] <= x < knots[i+1]
    # The non-zero basis functions for span i are N_{i-3}, N_{i-2}, N_{i-1}, N_{i}
    
    # Clip x to be within [knots[0], knots[-2]] to handle boundary cases
    x_clipped = np.clip(x, knots[0], knots[-2] - 1e-15)
    
    # Find span index for each x
    # span[i] = j such that knots[j] <= x[i] < knots[j+1]
    span_idx = np.searchsorted(knots, x_clipped, side='right') - 1
    # Clip span_idx to valid range [3, len(knots)-5] = [3, 9] for 14 knots
    span_idx = np.clip(span_idx, 3, len(knots) - 5)
    
    # Initialize basis matrix (n, 10)
    basis = np.zeros((n, 10), dtype=np.float64)
    
    # For each unique span, compute the 4 basis values
    # Process in vectorized fashion per span
    unique_spans = np.unique(span_idx)
    
    for s in unique_spans:
        mask = (span_idx == s)
        if not np.any(mask):
            continue
        x_s = x_clipped[mask]
        
        # Compute the 4 non-zero cubic B-spline values for span s
        # N_{s-3,3}, N_{s-2,3}, N_{s-1,3}, N_{s,3}
        
        # Using the standard Cox-de Boor recursion, but we can use the explicit formula
        # for cubic B-splines in terms of the local parameter.
        
        # Let's use the direct formula with knot differences
        # For a point x in [knots[s], knots[s+1]):
        # The four non-zero basis functions are N_{s-3}, N_{s-2}, N_{s-1}, N_{s}
        
        # Compute using the explicit cubic B-spline formula
        # Reference: Piegl & Tiller, The NURBS Book
        
        # d1 = knots[s+1] - knots[s-2]  (for N_{s-3})
        # Actually, let me use the standard recursive evaluation which is more reliable.
        
        # Let me use a simpler approach: evaluate using the known explicit formulas
        # for cubic B-splines with uniform interior structure.
        
        # Actually, let's just use the recursive Cox-de Boor algorithm in a vectorized way.
        
        # N_{i,0}(x) = 1 if knots[i] <= x < knots[i+1], else 0
        # N_{i,k}(x) = (x - knots[i])/(knots[i+k] - knots[i]) * N_{i,k-1}(x) + (knots[i+k+1] - x)/(knots[i+k+1] - knots[i+1]) * N_{i+1,k-1}(x)
        
        # For span s, the non-zero N_{i,0} are for i in {s-3, s-2, s-1, s}
        # We need N_{s-3,3}, N_{s-2,3}, N_{s-1,3}, N_{s,3}
        
        # Let me compute step by step for the relevant indices.
        
        # Level 0: N_{i,0}
        # For x in [knots[s], knots[s+1]):
        # N_{s-3,0}: knots[s-3] <= x < knots[s-2]? Generally no, since s-3 < s
        # Let me just use a direct evaluation function.
        
        # Simpler: use scipy if available, but let's implement directly.
        # I'll use the explicit formula for cubic B-splines.
        
        # For a point t in [u_j, u_{j+1}), the four non-zero basis functions are:
        # N_{j-3,3}(t) = (t - u_{j-3})^3 / ((u_j - u_{j-3})(u_{j+1} - u_{j-3})(u_{j+2} - u_{j-3}))
        #               + 3(u_{j+1} - t)(t - u_{j-2})^2 / ((u_j - u_{j-2})(u_{j+1} - u_{j-2})(u_{j+2} - u_{j-2}))
        #               + 3(u_{j+1} - t)^2(t - u_{j-1}) / ((u_j - u_{j-1})(u_{j+1} - u_{j-1})(u_{j+2} - u_{j-1}))
        #               + (u_{j+1} - t)^3 / ((u_j - u_j)(u_{j+1} - u_j)(u_{j+2} - u_j))  -- this is getting complex
        
        # Let me use a cleaner recursive approach.
        
        # Compute N_{i,k} for i in [s-3, s] and k in 0..3
        # Only need values at x_s
        
        # Level 0
        n0 = np.zeros((4, len(x_s)))
        for i_offset in range(4):
            i = s - 3 + i_offset
            n0[i_offset] = ((knots[i] <= x_s) & (x_s < knots[i+1])).astype(np.float64)
        
        # Level 1
        n1 = np.zeros((4, len(x_s)))
        for i_offset in range(3):
            i = s - 3 + i_offset
            denom1 = knots[i+1] - knots[i]
            denom2 = knots[i+2] - knots[i+1]
            if denom1 > 1e-15:
                n1[i_offset] += (x_s - knots[i]) / denom1 * n0[i_offset]
            if denom2 > 1e-15:
                n1[i_offset] += (knots[i+2] - x_s) / denom2 * n0[i_offset + 1]
        
        # Level 2
        n2 = np.zeros((4, len(x_s)))
        for i_offset in range(2):
            i = s - 3 + i_offset
            denom1 = knots[i+2] - knots[i]
            denom2 = knots[i+3] - knots[i+1]
            if denom1 > 1e-15:
                n2[i_offset] += (x_s - knots[i]) / denom1 * n1[i_offset]
            if denom2 > 1e-15:
                n2[i_offset] += (knots[i+3] - x_s) / denom2 * n1[i_offset + 1]
        
        # Level 3
        n3 = np.zeros((4, len(x_s)))
        i = s - 3
        denom1 = knots[i+3] - knots[i]
        denom2 = knots[i+4] - knots[i+1]
        if denom1 > 1e-15:
            n3[0] += (x_s - knots[i]) / denom1 * n2[0]
        if denom2 > 1e-15:
            n3[0] += (knots[i+4] - x_s) / denom2 * n2[1]
        
        # Assign to basis matrix
        basis[mask, s-3] = n3[0]
        basis[mask, s-2] = 0  # Wait, I only computed N_{s-3,3}. Let me re-examine.
        
        # Actually, I need N_{s-3,3}, N_{s-2,3}, N_{s-1,3}, N_{s,3}
        # My computation above only gives N_{s-3,3}. Let me redo this more carefully.
        
        # Let me use a different, cleaner approach: evaluate all 10 basis functions
        # for each x using the recursive definition, but only for the 4 that can be non-zero.
        
        # Actually, let me just use a well-tested direct evaluation.
        pass
    
    # Let me use a simpler and more robust approach:
    # Use the fact that we can evaluate cubic B-splines via the de Boor algorithm
    # or just use a precomputed lookup / direct formula.
    
    # Alternative: use scipy.interpolate's BSpline if available, but let's keep it self-contained.
    
    # Let me restart with a cleaner implementation using the explicit Cox-de Boor recursion
    # for all relevant basis functions.
    
    # Clear and redo
    basis = np.zeros((n, 10), dtype=np.float64)
    
    # For each x point, compute the 4 non-zero cubic B-spline values
    # We'll do this in a loop over unique spans for efficiency
    
    for s in unique_spans:
        mask = (span_idx == s)
        if not np.any(mask):
            continue
        x_s = x_clipped[mask]
        m = len(x_s)
        
        # The 4 non-zero basis functions for span s are N_{s-3,3}, N_{s-2,3}, N_{s-1,3}, N_{s,3}
        # Compute using Cox-de Boor recursion
        
        # Level 0: N_{i,0}(x) = 1 if u_i <= x < u_{i+1}
        # For i = s-3, s-2, s-1, s
        n0 = np.zeros((4, m))
        for k in range(4):
            i = s - 3 + k
            n0[k] = ((knots[i] <= x_s) & (x_s < knots[i+1])).astype(np.float64)
        
        # Level 1: N_{i,1}(x)
        n1 = np.zeros((4, m))
        for k in range(3):
            i = s - 3 + k
            d1 = knots[i+1] - knots[i]
            d2 = knots[i+2] - knots[i+1]
            if d1 > 1e-15:
                n1[k] += (x_s - knots[i]) / d1 * n0[k]
            if d2 > 1e-15:
                n1[k] += (knots[i+2] - x_s) / d2 * n0[k+1]
        
        # Level 2: N_{i,2}(x)
        n2 = np.zeros((4, m))
        for k in range(2):
            i = s - 3 + k
            d1 = knots[i+2] - knots[i]
            d2 = knots[i+3] - knots[i+1]
            if d1 > 1e-15:
                n2[k] += (x_s - knots[i]) / d1 * n1[k]
            if d2 > 1e-15:
                n2[k] += (knots[i+3] - x_s) / d2 * n1[k+1]
        
        # Level 3: N_{i,3}(x)
        n3 = np.zeros((4, m))
        # N_{s-3,3}
        i = s - 3
        d1 = knots[i+3] - knots[i]
        d2 = knots[i+4] - knots[i+1]
        if d1 > 1e-15:
            n3[0] += (x_s - knots[i]) / d1 * n2[0]
        if d2 > 1e-15:
            n3[0] += (knots[i+4] - x_s) / d2 * n2[1]
        
        # N_{s-2,3}
        i = s - 2
        d1 = knots[i+3] - knots[i]
        d2 = knots[i+4] - knots[i+1]
        if d1 > 1e-15:
            n3[1] += (x_s - knots[i]) / d1 * n2[0]  # Wait, n2 indices need to be shifted
        # I'm getting confused with indices. Let me use a cleaner recursive function.
        
        # Let me just use a direct evaluation for each of the 4 basis functions.
        # For N_{j,3}(x) where j is in {s-3, s-2, s-1, s}, compute via recursion.
        
        # I'll define a helper that computes N_{j,3}(x) for a given j and x array.
        # But to keep it vectorized, let me just do it for all 4 at once with proper indexing.
        
        # Actually, let me simplify: compute all N_{i,k} for i in range(s-3, s+1) and k in 0..3
        # using arrays of size 4 x m.
        
        # N_{i,0} for i = s-3, s-2, s-1, s
        N0 = np.zeros((4, m))
        for k in range(4):
            i = s - 3 + k
            N0[k] = ((knots[i] <= x_s) & (x_s < knots[i+1])).astype(np.float64)
        
        # N_{i,1} for i = s-3, s-2, s-1, s
        # N_{i,1} = (x-u_i)/(u_{i+1}-u_i) * N_{i,0} + (u_{i+2}-x)/(u_{i+2}-u_{i+1}) * N_{i+1,0}
        N1 = np.zeros((4, m))
        for k in range(4):
            i = s - 3 + k
            term1 = 0.0
            term2 = 0.0
            d1 = knots[i+1] - knots[i]
            d2 = knots[i+2] - knots[i+1]
            if d1 > 1e-15:
                term1 = (x_s - knots[i]) / d1 * N0[k]
            if d2 > 1e-15 and k+1 < 4:
                term2 = (knots[i+2] - x_s) / d2 * N0[k+1]
            N1[k] = term1 + term2
        
        # N_{i,2} for i = s-3, s-2, s-1, s
        # N_{i,2} = (x-u_i)/(u_{i+2}-u_i) * N_{i,1} + (u_{i+3}-x)/(u_{i+3}-u_{i+1}) * N_{i+1,1}
        N2 = np.zeros((4, m))
        for k in range(4):
            i = s - 3 + k
            term1 = 0.0
            term2 = 0.0
            d1 = knots[i+2] - knots[i]
            d2 = knots[i+3] - knots[i+1]
            if d1 > 1e-15:
                term1 = (x_s - knots[i]) / d1 * N1[k]
            if d2 > 1e-15 and k+1 < 4:
                term2 = (knots[i+3] - x_s) / d2 * N1[k+1]
            N2[k] = term1 + term2
        
        # N_{i,3} for i = s-3, s-2, s-1, s
        # N_{i,3} = (x-u_i)/(u_{i+3}-u_i) * N_{i,2} + (u_{i+4}-x)/(u_{i+4}-u_{i+1}) * N_{i+1,2}
        N3 = np.zeros((4, m))
        for k in range(4):
            i = s - 3 + k
            term1 = 0.0
            term2 = 0.0
            d1 = knots[i+3] - knots[i]
            d2 = knots[i+4] - knots[i+1]
            if d1 > 1e-15:
                term1 = (x_s - knots[i]) / d1 * N2[k]
            if d2 > 1e-15 and k+1 < 4:
                term2 = (knots[i+4] - x_s) / d2 * N2[k+1]
            N3[k] = term1 + term2
        
        # Assign to basis
        basis[mask, s-3] = N3[0]
        basis[mask, s-2] = N3[1]
        basis[mask, s-1] = N3[2]
        basis[mask, s] = N3[3]
    
    # Linear combination
    result = basis @ params
    
    if time.time() > deadline:
        pass
    
    return result
# EVOLVE-BLOCK-END
