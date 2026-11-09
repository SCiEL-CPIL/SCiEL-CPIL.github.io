"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 600
    best_score = float('inf')
    best_z = None

    # Precompute Laguerre zeros for various degrees using numpy
    def laguerre_zeros(n, alpha):
        """Compute zeros of L_n^(alpha)(x) using companion matrix eigenvalues."""
        # For L_n^(alpha), we can use the three-term recurrence
        # Build the Jacobi matrix for Laguerre polynomials
        # L_n^(alpha) has zeros that can be found via the Jacobi matrix
        # The Jacobi matrix for Laguerre L_n^(alpha) is:
        # subdiagonal: sqrt(k*(k+alpha)) for k=1..n-1
        # diagonal: 2*k-1+alpha for k=1..n
        n = int(n)
        a = alpha
        diag = np.array([2.0*k - 1.0 + a for k in range(1, n+1)])
        subdiag = np.array([np.sqrt(k*(k+a)) for k in range(1, n)])
        J = np.diag(diag)
        for i in range(n-1):
            J[i, i+1] = subdiag[i]
            J[i+1, i] = subdiag[i]
        eigvals = np.linalg.eigvalsh(J)
        return np.sort(eigvals)

    # Try different configurations of z_i based on Laguerre zeros
    # Strategy: pick z_i from zeros of L_n^(alpha) for suitable n

    alpha = -0.5

    # Try various candidate sets
    candidate_sets = []

    # Set 1: Zeros of L_12^(-0.5)(x)
    try:
        z12 = laguerre_zeros(12, alpha)
        if len(z12) == 12 and all(z > 0 for z in z12) and len(set(np.round(z12, 10))) == 12:
            candidate_sets.append(list(np.round(z12, 10)))
    except:
        pass

    # Set 2: Zeros of L_13^(-0.5)(x), take first 12
    try:
        z13 = laguerre_zeros(13, alpha)
        if len(z13) >= 12:
            z13_12 = z13[:12]
            if all(z > 0 for z in z13_12) and len(set(np.round(z13_12, 10))) == 12:
                candidate_sets.append(list(np.round(z13_12, 10)))
    except:
        pass

    # Set 3: Geometric sequence
    for ratio in [1.1, 1.2, 1.3, 1.5, 2.0, 0.8, 0.7, 0.6]:
        for start in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            z = [start * (ratio ** i) for i in range(12)]
            if all(z_i > 0 for z_i in z) and len(set(z)) == 12:
                candidate_sets.append(z)

    # Set 4: Log-spaced
    for lo, hi in [(0.1, 100), (0.5, 50), (1, 20), (0.1, 10), (1, 10), (5, 100)]:
        z = list(np.logspace(np.log10(lo), np.log10(hi), 12))
        if all(z_i > 0 for z_i in z) and len(set(z)) == 12:
            candidate_sets.append(z)

    # Set 5: Uniform spaced
    for lo, hi in [(0.1, 100), (0.5, 50), (1, 20), (0.1, 10), (1, 10), (5, 100), (0.1, 50)]:
        z = list(np.linspace(lo, hi, 12))
        if all(z_i > 0 for z_i in z) and len(set(z)) == 12:
            candidate_sets.append(z)

    # Set 6: Zeros of L_n^(-0.5) for various n, take 12
    for n in [12, 13, 14, 15, 16, 17, 18, 20, 25, 30, 40, 50]:
        try:
            zn = laguerre_zeros(n, alpha)
            if len(zn) >= 12:
                # Take 12 from the zeros - try different subsets
                # First 12
                sub = zn[:12]
                if all(z > 0 for z in sub) and len(set(np.round(sub, 10))) == 12:
                    candidate_sets.append(list(np.round(sub, 10)))
                # Last 12
                sub2 = zn[-12:]
                if all(z > 0 for z in sub2) and len(set(np.round(sub2, 10))) == 12:
                    candidate_sets.append(list(np.round(sub2, 10)))
                # Every other, take 12
                if n >= 24:
                    sub3 = zn[::2][:12]
                    if len(sub3) == 12 and all(z > 0 for z in sub3) and len(set(np.round(sub3, 10))) == 12:
                        candidate_sets.append(list(np.round(sub3, 10)))
        except:
            pass

    # Deduplicate candidate sets
    unique_sets = []
    seen = set()
    for cs in candidate_sets:
        key = tuple(round(v, 6) for v in sorted(cs))
        if key not in seen:
            seen.add(key)
            unique_sets.append(cs)

    # Now evaluate each candidate set
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
            L = sp.laguerre(2*k, x, alpha_val)
            L_terms.append(L)

        # Coefficients c_0..c_25
        c = sp.symbols('c0:26')

        # g(x) = sum c_k * L_2k(x)
        g = sum(c[k] * L_terms[k] for k in range(26))
        g = sp.expand(g)

        # Conditions:
        # g(0) = 0
        # g'(0) = 1
        # g(z_i) = 0 for each i
        # g'(z_i) = 0 for each i
        # Total: 2 + 12 + 12 = 26 equations, 26 unknowns

        gp = sp.diff(g, x)

        eqs = []
        eqs.append(g.subs(x, 0))  # g(0) = 0
        eqs.append(gp.subs(x, 0) - 1)  # g'(0) = 1
        for zi in z_rats_sorted:
            eqs.append(g.subs(x, zi))  # g(z_i) = 0
            eqs.append(gp.subs(x, zi))  # g'(z_i) = 0

        # Set up linear system
        A = sp.Matrix([[eq.coeff(ci) for ci in c] for eq in eqs])
        b = sp.Matrix([0]*26)
        # Actually, all equations are homogeneous in c except g'(0)=1
        # g(0)=0: sum c_k L_2k(0) = 0
        # g'(0)=1: sum c_k L_2k'(0) = 1
        # g(z_i)=0: sum c_k L_2k(z_i) = 0
        # g'(z_i)=0: sum c_k L_2k'(z_i) = 0

        # So A is 26x26, b has 0 except one entry is 1
        # Let's rebuild properly
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
        if A_mat.det() == 0:
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
            denom = denom * (x - zi)**2
        denom = sp.expand(denom)

        # q(x) = g(x) / denom, exact polynomial division
        # g should be divisible by denom since g has double roots at z_i and g(0)=0
        q_poly, remainder = sp.div(g_sol, denom, x)
        if remainder != 0:
            # Try to proceed anyway
            pass

        # q_poly is a polynomial of degree 50 - (1+24) = 25
        # Find real roots
        try:
            q_poly_sp = sp.Poly(q_poly, x)
            real_roots = q_poly_sp.real_roots()
        except:
            return float('inf')

        # For each real root r, check sign change at r - 1e-198 and r + 1e-198
        B = 0
        for r in real_roots:
            if r > 0:
                # Evaluate q at r - 1e-198 and r + 1e-198
                r_float = float(r)
                delta = 1e-198
                try:
                    val_neg = q_poly.subs(x, r - delta)
                    val_pos = q_poly.subs(x, r + delta)
                    # Convert to float for sign check
                    vn = float(val_neg)
                    vp = float(val_pos)
                    if vn * vp < 0:
                        # Strictly opposite signs
                        if r_float > B:
                            B = r_float
                except:
                    pass

        return B / (2 * np.pi)

    # Evaluate candidates
    for i, cs in enumerate(unique_sets):
        if time.time() > deadline - 60:
            break
        try:
            score = evaluate(cs)
            if score < best_score:
                best_score = score
                best_z = cs
                print(f"Candidate {i}: score={score:.6f}")
        except Exception as e:
            pass

    if best_z is not None:
        return best_z
    else:
        # Fallback
        return [4.0 * (i + 1) for i in range(12)]

# EVOLVE-BLOCK-END
