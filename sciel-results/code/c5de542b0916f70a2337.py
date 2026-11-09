"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 630
    safety_margin = 30

    def elapsed():
        return time.time() - start_time

    def time_remaining():
        return deadline - time.time() - safety_margin

    # Primary candidate: arithmetic sequence known to score 1.0
    z_arithmetic = [4.0 * (i + 1) for i in range(12)]

    # Generate additional compact candidate sets
    candidate_sets = [z_arithmetic]

    # Try variations around the arithmetic sequence
    # Slightly compressed
    for scale in [3.0, 3.5, 4.0, 4.5, 5.0]:
        z = [scale * (i + 1) for i in range(12)]
        candidate_sets.append(z)

    # Geometric with small ratio
    for ratio in [1.05, 1.1, 1.15, 1.2]:
        for start in [2.0, 3.0, 4.0, 5.0]:
            z = [start * (ratio ** i) for i in range(12)]
            if all(z_i > 0 for z_i in z) and len(set(z)) == 12 and max(z) < 100:
                candidate_sets.append(z)

    # Log-spaced compact
    for lo, hi in [(2, 50), (3, 40), (4, 35), (2, 40), (1, 30)]:
        z = list(np.logspace(np.log10(lo), np.log10(hi), 12))
        if all(z_i > 0 for z_i in z) and len(set(z)) == 12:
            candidate_sets.append(z)

    # Uniform spaced compact
    for lo, hi in [(2, 50), (3, 45), (4, 48), (1, 40), (2, 40), (5, 50)]:
        z = list(np.linspace(lo, hi, 12))
        if all(z_i > 0 for z_i in z) and len(set(z)) == 12:
            candidate_sets.append(z)

    # Bessel J_{1/2} zeros for reference (z_i = i^2 * pi^2)
    z_bessel = [float(i**2 * math.pi**2) for i in range(1, 13)]
    candidate_sets.append(z_bessel)

    # Deduplicate
    unique_sets = []
    seen = set()
    for cs in candidate_sets:
        key = tuple(round(v, 6) for v in sorted(cs))
        if key not in seen:
            seen.add(key)
            unique_sets.append(cs)

    # Sort by max value (smaller max likely better)
    unique_sets.sort(key=lambda z: max(z))

    def evaluate(z_list):
        """Evaluate B/(2*pi) for given z_list using sympy exact arithmetic."""
        import sympy as sp
        x = sp.Symbol('x')
        alpha_val = sp.Rational(-1, 2)

        # Convert z_i to rational
        z_rats = [sp.Rational(float(zi)) for zi in z_list]
        z_rats_sorted = sorted(z_rats)

        # Check distinct and positive
        if len(set(z_rats_sorted)) != 12:
            return float('inf')
        if any(z <= 0 for z in z_rats_sorted):
            return float('inf')

        # Build L_2k^(alpha)(x) for k=0..25
        L_terms = []
        for k in range(26):
            L = sp.laguerre(2 * k, x, alpha_val)
            L_terms.append(L)

        # Build the linear system
        A_rows = []
        b_vec = []

        # g(0) = 0
        row = []
        for k in range(26):
            row.append(L_terms[k].subs(x, 0))
        A_rows.append(row)
        b_vec.append(0)

        # g'(0) = 1
        row = []
        for k in range(26):
            row.append(sp.diff(L_terms[k], x).subs(x, 0))
        A_rows.append(row)
        b_vec.append(1)

        for zi in z_rats_sorted:
            # g(z_i) = 0
            row = []
            for k in range(26):
                row.append(L_terms[k].subs(x, zi))
            A_rows.append(row)
            b_vec.append(0)

            # g'(z_i) = 0
            row = []
            for k in range(26):
                row.append(sp.diff(L_terms[k], x).subs(x, zi))
            A_rows.append(row)
            b_vec.append(0)

        A_mat = sp.Matrix(A_rows)
        b_mat = sp.Matrix(b_vec)

        # Check if singular
        try:
            if A_mat.det() == 0:
                return float('inf')
        except:
            return float('inf')

        # Solve
        try:
            c_sol = A_mat.LUsolve(b_mat)
        except:
            return float('inf')

        # Build g(x)
        g_sol = sum(c_sol[k] * L_terms[k] for k in range(26))
        g_sol = sp.expand(g_sol)

        # Build denominator: x * prod_i (x - z_i)^2
        denom = x
        for zi in z_rats_sorted:
            denom = denom * (x - zi) ** 2
        denom = sp.expand(denom)

        # q(x) = g(x) / denom, exact polynomial division
        q_poly, remainder = sp.div(g_sol, denom, x)
        if remainder != 0:
            return float('inf')

        # Find real roots
        try:
            q_poly_sp = sp.Poly(q_poly, x)
            real_roots = q_poly_sp.real_roots()
        except:
            return float('inf')

        # For each real root r, check sign change
        B = 0
        for r in real_roots:
            if r > 0:
                r_float = float(r)
                delta = 1e-198
                try:
                    val_neg = q_poly.subs(x, r - delta)
                    val_pos = q_poly.subs(x, r + delta)
                    vn = float(val_neg)
                    vp = float(val_pos)
                    if vn * vp < 0:
                        if r_float > B:
                            B = r_float
                except:
                    pass

        return B / (2 * np.pi)

    # Evaluate candidates, starting with the most promising (smallest max)
    best_score = float('inf')
    best_z = z_arithmetic  # Fallback

    for i, cs in enumerate(unique_sets):
        if time_remaining() < 90:
            break
        try:
            score = evaluate(cs)
            if score < best_score:
                best_score = score
                best_z = cs
                print(f"Candidate {i}: score={score:.6f}, max_z={max(cs):.4f}")
        except Exception as e:
            pass

    return best_z

# EVOLVE-BLOCK-END
