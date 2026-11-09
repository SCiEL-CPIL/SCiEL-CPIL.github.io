# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _incidence_count(points, lines):
    """Compute number of (point, line) incidences. points: (n,2), lines: (m,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = lines[0:1, :]
    B = lines[1:2, :]
    C = lines[2:3, :]
    total = px * A + py * B + C
    return np.sum(total == 0)

def _gcd3(a, b, c):
    """Compute gcd of three integers."""
    g = math.gcd(abs(int(a)), abs(int(b)))
    g = math.gcd(g, abs(int(c)))
    return g if g != 0 else 1

def _normalize_line(A, B, C):
    """Normalize a line so it's unique."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    Uses a 6x9 grid and selects the 27 lines with the highest point coverage.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # 1. Define Points: 6x9 grid
    # x in [1, 6], y in [1, 9]
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate Candidate Lines
    # We look for lines A*x + B*y + C = 0 with small integer coefficients.
    # To cover the grid effectively, A and B should be small.
    # C determines the position.
    candidates = []
    
    # Range for A and B
    # Small slopes/intercepts are best for grid points.
    # Let's try A, B in [-3, 3]
    for A in range(-3, 4):
        for B in range(-3, 4):
            if A == 0 and B == 0:
                continue
            
            # Determine range of C such that the line intersects the grid
            # A*x + B*y + C = 0  => C = -A*x - B*y
            # x in [1, 6], y in [1, 9]
            # Min C = -A*max(x) - B*max(y) if A,B > 0? No, just check bounds.
            # Actually, we can just iterate C in a reasonable range and filter.
            # Max |A*x| is 3*6=18. Max |B*y| is 3*9=27.
            # So C is roughly in [-45, 45].
            
            # More precise:
            # x_min, x_max = 1, 6
            # y_min, y_max = 1, 9
            # C = -A*x - B*y
            # If A>0, min C_x = -A*6, max C_x = -A*1
            # If A<0, min C_x = -A*1, max C_x = -A*6
            # Same for B.
            
            cx_min = 0
            cx_max = 0
            if A > 0:
                cx_min = -A * 6
                cx_max = -A * 1
            elif A < 0:
                cx_min = -A * 1
                cx_max = -A * 6
            
            cy_min = 0
            cy_max = 0
            if B > 0:
                cy_min = -B * 9
                cy_max = -B * 1
            elif B < 0:
                cy_min = -B * 1
                cy_max = -B * 9
            
            C_min = cx_min + cy_min
            C_max = cx_max + cy_max
            
            for C in range(C_min, C_max + 1):
                line = (A, B, C)
                # Check if line actually hits any grid point
                # We can do a quick check:
                # A*x + B*y + C == 0 for some x in [1,6], y in [1,9]
                # Instead of checking every point, we can just add it and filter later?
                # No, better to check.
                # But since the range is small, we can just compute incidence for all candidates at once?
                # Or just store them.
                candidates.append(line)
                
    # 3. Evaluate Incidences for Candidates
    # Vectorized incidence count is fast.
    # We have ~ (7*7 - 1) * ~30 C's = ~1500 candidates.
    # 54 points * 1500 lines is small.
    
    if not candidates:
        # Fallback
        lines = np.zeros((27, 3), dtype=np.int64)
        return points, lines
        
    cand_array = np.array(candidates, dtype=np.int64)
    
    # Compute incidence matrix: (n_points, n_candidates)
    # This might be large if many candidates, but 1500*54 is fine.
    px = points[:, 0:1]
    py = points[:, 1:2]
    A = cand_array[:, 0:1].T
    B = cand_array[:, 1:2].T
    C = cand_array[:, 2:3].T
    
    # total[i, j] = A_j * x_i + B_j * y_i + C_j
    total = px * A + py * B + C
    incidences = np.sum(total == 0, axis=0)
    
    # 4. Select Top 27 Unique Lines
    # We need to normalize lines to check for duplicates.
    # Sort candidates by incidence count descending.
    sorted_indices = np.argsort(incidences)[::-1]
    
    selected_lines = []
    seen_normalized = set()
    
    for idx in sorted_indices:
        if len(selected_lines) >= 27:
            break
        A, B, C = cand_array[idx]
        norm = _normalize_line(A, B, C)
        if norm not in seen_normalized:
            seen_normalized.add(norm)
            selected_lines.append(norm)
            
    # If we didn't find enough, we might need to expand search space
    # But 27 lines from ~1500 candidates with high coverage should be sufficient.
    # The grid 6x9 has 6 verticals (9 pts each), 9 horizontals (6 pts each),
    # plus diagonals.
    # Verticals: x=1..6 -> (1, 0, -c). Incidence 9.
    # Horizontals: y=1..9 -> (0, 1, -c). Incidence 6.
    # Diagonals x-y=c: (1, -1, -c). Incidence up to 6.
    # We have 6+9+~12 = 27 lines easily.
    # Sum of incidences: 6*9 + 9*6 + ...
    # Just verticals and horizontals give 54 + 54 = 108.
    # Adding diagonals adds more.
    # The prompt says 160 is achievable.
    
    if len(selected_lines) < 27:
        # Pad with any valid lines if needed (should not happen)
        for A in range(-5, 6):
            for B in range(-5, 6):
                if A == 0 and B == 0: continue
                for C in range(-50, 51):
                    norm = _normalize_line(A, B, C)
                    if norm not in seen_normalized:
                        seen_normalized.add(norm)
                        selected_lines.append(norm)
                        if len(selected_lines) >= 27:
                            break
            if len(selected_lines) >= 27:
                break
            if time.time() > deadline - 10:
                break
        if len(selected_lines) < 27:
             # Last resort: just take first 27 from candidates without dedup check? No, must be valid.
             # Force fill with verticals/horizontals if we missed them?
             # The loop above should have found them.
             pass

    lines = np.array(selected_lines[:27], dtype=np.int64)
    
    # Safety check: ensure exactly 27 lines
    if len(lines) < 27:
        # This is a critical failure state, pad with zeros (invalid but better than exception?)
        # No, invalid scores 0. We must return valid.
        # If we are short, we made a mistake in generation.
        # Let's just return the best we have, padded with (0,1,0) if needed? 
        # (0,1,0) is y=0.
        # But duplicates are invalid.
        # Given the density, we should have found 27.
        # Fallback: return empty or raise. Let's raise to indicate failure.
        raise ValueError("Failed to generate valid 27 points")
        
    return points, lines

# EVOLVE-BLOCK-END
