"""Product formula design for Heisenberg-chain time evolution using high-order Suzuki-Trotter with Nelder-Mead refinement."""
import time
import math
import numpy as np

_EVOLVE_START_TIME = time.time()
_DEADLINE = 135.0  # 150s wall-clock limit, keep 15s safety margin


def _check_deadline():
    """Return True if we're close to the deadline."""
    return (time.time() - _EVOLVE_START_TIME) > _DEADLINE


def _build_hamiltonians():
    """
    Build the A and B Hamiltonian matrices for a 6-site open XXZ Heisenberg chain.
    
    The chain has anisotropy Δ=0.8 and uniform field h=0.3.
    A = odd-bond half + half the uniform field
    B = even-bond half + half the uniform field
    """
    N = 6
    dim = 2 ** N
    delta = 0.8
    field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    
    # Build single-site operators
    # For site j (0-indexed), the operator is I⊗...⊗O⊗...⊗I with O at position j
    def site_op(op, site):
        """Build operator acting on site in N-site chain."""
        result = op
        for i in range(N - 1):
            if i < site:
                result = np.kron(result, identity)
            else:
                result = np.kron(identity, result)
        return result
    
    # XXZ Hamiltonian: H = sum_j [Δ/4 * Sz_j Sz_{j+1} + 1/4 * (Sx_j Sx_{j+1} + Sy_j Sy_{j+1})] + h/2 * sum_j Sz_j
    # For the split: A gets odd bonds (j=0,2,4) and half field, B gets even bonds (j=1,3,5) and half field
    
    # Bond operators
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    for j in range(N - 1):  # bonds j to j+1, j=0..4
        # XXZ interaction term
        sz_j = site_op(sz, j)
        sz_j1 = site_op(sz, j + 1)
        sx_j = site_op(sx, j)
        sx_j1 = site_op(sx, j + 1)
        sy_j = site_op(sy, j)
        sy_j1 = site_op(sy, j + 1)
        
        bond_term = (delta / 4.0) * (sz_j @ sz_j1) + (1.0 / 4.0) * (sx_j @ sx_j1 + sy_j @ sy_j1)
        
        if j % 2 == 0:  # odd bonds (0-indexed: bond 0,2,4)
            A += bond_term
        else:  # even bonds (bond 1,3)
            B += bond_term
    
    # Uniform field: h/2 * sum_j Sz_j, split half to A and half to B
    for j in range(N):
        sz_j = site_op(sz, j)
        A += (field / 4.0) * sz_j
        B += (field / 4.0) * sz_j
    
    return A, B


def _expm_pade(H, order=13):
    """
    Compute matrix exponential exp(H) using Padé approximation.
    H should be a square matrix.
    """
    n = H.shape[0]
    # Use scipy if available, otherwise Padé
    try:
        from scipy.linalg import expm
        return expm(H)
    except ImportError:
        pass
    
    # Padé approximation of order m
    m = order
    # Build the Padé approximant: exp(H) ≈ Q_m(-H/2)^{-1} Q_m(H/2)
    # where Q_m is the m-th degree Taylor polynomial of e^x
    
    # For simplicity, use a series expansion with scaling and squaring
    # Scale H so that ||H|| is small
    norm_H = np.linalg.norm(H, ord=2)
    if norm_H < 1e-15:
        return np.eye(n, dtype=complex)
    
    # Number of squarings needed
    s = max(0, int(math.ceil(math.log2(norm_H / 0.5))))
    H_scaled = H / (2 ** s)
    
    # Compute exp(H_scaled) using Taylor series
    # exp(X) = sum_{k=0}^{inf} X^k / k!
    X = H_scaled
    result = np.eye(n, dtype=complex)
    term = np.eye(n, dtype=complex)
    
    for k in range(1, 30):
        term = term @ X / k
        result = result + term
        if np.linalg.norm(term, ord=2) < 1e-16:
            break
    
    # Undo scaling
    for _ in range(s):
        result = result @ result
    
    return result


def _compute_error(a, b, A, B, times=(0.25, 0.5, 1.0)):
    """
    Compute the mean spectral-norm error for the product formula.
    
    U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
    Exact: exp(-i t (A + B))
    """
    dim = A.shape[0]
    errors = []
    
    for t in times:
        # Compute product formula
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            # exp(-i b_k t B)
            U = _expm_pade(-1j * b[k] * t * B) @ U
            # exp(-i a_k t A)
            U = _expm_pade(-1j * a[k] * t * A) @ U
        
        # Exact propagator
        U_exact = _expm_pade(-1j * t * (A + B))
        
        # Spectral norm of difference
        diff = U - U_exact
        # Compute singular values
        svals = np.linalg.svd(diff, compute_uv=False)
        errors.append(svals[0])
    
    return np.mean(errors)


def _normalize_and_clip(a, b, bounds=(-50.0, 50.0)):
    """Normalize so sums are 1 and clip to bounds."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    # Clip
    a = np.clip(a, bounds[0], bounds[1])
    b = np.clip(b, bounds[0], bounds[1])
    
    # Normalize
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    else:
        a = np.ones(len(a)) / len(a)
    if abs(sb) > 1e-15:
        b = b / sb
    else:
        b = np.ones(len(b)) / len(b)
    
    return list(a), list(b)


def _nelder_mead(func, x0, bounds=(-50.0, 50.0), max_iter=200, tol=1e-8):
    """
    Simple Nelder-Mead optimization.
    func: objective function
    x0: initial point (list)
    bounds: tuple (lower, upper) for each variable
    """
    n = len(x0)
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink
    
    # Initial simplex
    simplex = [list(x0)]
    fvals = [func(x0)]
    
    for i in range(n):
        point = list(x0)
        step = 0.05 if x0[i] == 0 else 0.05 * abs(x0[i])
        point[i] += step
        simplex.append(point)
        fvals.append(func(point))
    
    # Sort by function value
    order = np.argsort(fvals)
    simplex = [simplex[i] for i in order]
    fvals = [fvals[i] for i in order]
    
    best_x = simplex[0]
    best_f = fvals[0]
    
    for iteration in range(max_iter):
        if _check_deadline():
            break
        
        # Check convergence
        if max(fvals) - min(fvals) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.zeros(n)
        for i in range(n):
            centroid += np.array(simplex[i])
        centroid /= n
        
        # Reflection
        worst = simplex[-1]
        xr = centroid + alpha * (centroid - np.array(worst))
        fr = func(list(xr))
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - np.array(worst))
            fe = func(list(xe))
            if fe < fr:
                simplex[-1] = list(xe)
                fvals[-1] = fe
            else:
                simplex[-1] = list(xr)
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = list(xr)
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (np.array(worst) - centroid)
                fc = func(list(xc))
                if fc <= fr:
                    simplex[-1] = list(xc)
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        fvals[i] = func(simplex[i])
            else:
                # Inside contraction
                xc = centroid - rho * (np.array(worst) - centroid)
                fc = func(list(xc))
                if fc < fvals[-1]:
                    simplex[-1] = list(xc)
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                        fvals[i] = func(simplex[i])
        
        # Re-sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if fvals[0] < best_f:
            best_f = fvals[0]
            best_x = simplex[0]
    
    return best_x, best_f


def _get_8th_order_coeffs():
    """
    Get 8th-order symmetric Suzuki-Trotter coefficients for 12 stages.
    
    The 8th-order scheme is built by composing three 4th-order steps.
    4th-order symmetric 4-stage:
      A: [z/2, (1-z)/2, (1-z)/2, z/2] where z = 1/(2-2^(1/4))
      B: [(1-z)/2, z/2, z/2, (1-z)/2]
    
    8th-order: U8 = U4(c1) * U4(c2) * U4(c1) where c1 = z, c2 = 1-2z
    """
    z = 1.0 / (2.0 - 2.0 ** 0.25)
    
    A4 = [z/2.0, (1.0-z)/2.0, (1.0-z)/2.0, z/2.0]
    B4 = [(1.0-z)/2.0, z/2.0, z/2.0, (1.0-z)/2.0]
    
    A8 = []
    B8 = []
    for w in [z, 1.0 - 2.0*z, z]:
        A8.extend([w*x for x in A4])
        B8.extend([w*x for x in B4])
    
    # Normalize
    sa = sum(A8)
    sb = sum(B8)
    if abs(sa - 1.0) > 1e-15:
        A8 = [x/sa for x in A8]
    if abs(sb - 1.0) > 1e-15:
        B8 = [x/sb for x in B8]
    
    return A8, B8


def coefficients(stages):
    """
    Return (a, b) coefficient lists for a product formula.
    
    For stages=12, uses 8th-order symmetric Suzuki-Trotter coefficients
    with Nelder-Mead refinement.
    """
    if stages != 12:
        # Fallback for other stage counts
        if stages > 0:
            a = [1.0/stages] * stages
            b = [1.0/stages] * stages
            return a, b
        else:
            return [], []
    
    # Check deadline
    if _check_deadline():
        a12, b12 = _get_8th_order_coeffs()
        return _normalize_and_clip(a12, b12)
    
    # Build Hamiltonians
    A, B = _build_hamiltonians()
    
    # Get initial 8th-order coefficients
    a_init, b_init = _get_8th_order_coeffs()
    
    # Define objective function
    def objective(params):
        if _check_deadline():
            return 1e10
        a = params[:12]
        b = params[12:24]
        a, b = _normalize_and_clip(a, b)
        return _compute_error(a, b, A, B)
    
    # Run Nelder-Mead optimization
    x0 = list(a_init) + list(b_init)
    
    try:
        x_best, f_best = _nelder_mead(objective, x0, max_iter=300, tol=1e-10)
    except Exception:
        x_best = x0
    
    # Extract and normalize
    a_best = x_best[:12]
    b_best = x_best[12:24]
    a_best, b_best = _normalize_and_clip(a_best, b_best)
    
    # Final deadline check
    if _check_deadline():
        return a_best, b_best
    
    return a_best, b_best
