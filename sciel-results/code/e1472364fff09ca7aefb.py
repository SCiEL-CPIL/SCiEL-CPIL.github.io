"""Program for spectral graph lifts on prism graphs using circulant structure."""
import numpy as np
from collections import deque


def _build_signed_adj(graph, signs):
    """Build the signed adjacency matrix of the base graph from signs.

    graph: {'n': 2m, 'edges': [(u,v), ...]}
    signs: array of shape (n_edges,) with values in {-1,+1} (or continuous).
    Returns the signed adjacency matrix A (n x n), symmetric.
    """
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=float)
    for (u, v), s in zip(edges, signs):
        A[u, v] = s
        A[v, u] = s
    return A


def _spectral_radius(A):
    """Max absolute eigenvalue of symmetric A."""
    vals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(vals)))


def _is_connected_lift(graph, signs):
    """Check whether the 2-lift is connected.

    The 2-lift is connected iff there exists at least one cycle in the base
    graph whose product of edge signs is -1. We detect this by attempting to
    assign each vertex a side label in {0,1} such that for every edge (u,v)
    with sign s, the labels satisfy label[v] == label[u] if s>0 else 
    1-label[u]. If such a labeling exists, the lift is disconnected.
    """
    n = graph['n']
    edges = graph['edges']
    adj = {i: [] for i in range(n)}
    for (u, v), s in zip(edges, signs):
        adj[u].append((v, s))
        adj[v].append((u, s))

    label = [-1] * n
    label[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for (v, s) in adj[u]:
            expected = label[u] if s > 0 else 1 - label[u]
            if label[v] == -1:
                label[v] = expected
                q.append(v)
            else:
                if label[v] != expected:
                    return True
    return False


def _build_circulant_prism(m):
    """
    Build the prism graph structure with circulant-friendly indexing.
    
    Prism graph: two m-cycles (top: 0..m-1, bottom: m..2m-1) joined by matching.
    Edges:
      - Top cycle: (i, i+1) for i=0..m-2, and (m-1, 0)
      - Bottom cycle: (m+i, m+i+1) for i=0..m-2, and (m+m-1, m+0)
      - Rungs: (i, m+i) for i=0..m-1
    
    Returns: (edges_list, edge_types) where edge_types indicates which type each edge is
    """
    edges = []
    edge_types = []  # 0=top, 1=bottom, 2=rung
    
    # Top cycle
    for i in range(m):
        j = (i + 1) % m
        edges.append((i, j))
        edge_types.append(0)
    
    # Bottom cycle
    for i in range(m):
        j = (i + 1) % m
        edges.append((m + i, m + j))
        edge_types.append(1)
    
    # Rungs
    for i in range(m):
        edges.append((i, m + i))
        edge_types.append(2)
    
    return edges, edge_types


def _build_circulant_signed_adj(m, top_signs, bottom_signs, rung_signs):
    """
    Build signed adjacency matrix exploiting circulant structure.
    
    top_signs: array of m values (continuous or +/-1) for top cycle edges
    bottom_signs: array of m values for bottom cycle edges  
    rung_signs: array of m values for rung edges
    
    The signed adjacency matrix has block structure:
    A = [[T, R],
         [R', B]]
    where T and B are circulant-like (with one edge wrapping around),
    R and R' are diagonal with rung signs.
    """
    n = 2 * m
    A = np.zeros((n, n), dtype=float)
    
    # Top cycle: edge i connects i to (i+1)%m
    for i in range(m):
        j = (i + 1) % m
        s = top_signs[i]
        A[i, j] = s
        A[j, i] = s
    
    # Bottom cycle: edge i connects m+i to m+(i+1)%m
    for i in range(m):
        j = (i + 1) % m
        s = bottom_signs[i]
        A[m + i, m + j] = s
        A[m + j, m + i] = s
    
    # Rungs: edge i connects i to m+i
    for i in range(m):
        s = rung_signs[i]
        A[i, m + i] = s
        A[m + i, i] = s
    
    return A


def _circulant_eigenvalues(m, signs):
    """
    Compute eigenvalues of a circulant matrix defined by first row signs.
    For a cycle graph with edge signs, the matrix is almost circulant.
    
    Actually, for a cycle with edge weights w_0, w_1, ..., w_{m-1} where
    edge i connects i to (i+1)%m, the adjacency is:
    A[i, i+1] = w_i, A[i+1, i] = w_i (with indices mod m)
    
    This is NOT exactly circulant unless all w_i are equal. But we can still
    use FFT-based approaches or just compute directly for small m.
    """
    n = m
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = signs[i]
        A[j, i] = signs[i]
    return np.linalg.eigvalsh(A)


def sign_edges(graph, budget):
    """Return one sign (-1 or +1) per edge, minimizing spectral radius
    of the signed adjacency matrix, subject to the 2-lift being connected.
    
    Uses phase-space continuous optimization exploiting circulant structure.
    """
    rng = np.random.default_rng(3)
    n_edges = len(graph['edges'])
    m = graph['n'] // 2
    
    # Extract edge structure
    edges = graph['edges']
    
    # Classify edges into top cycle, bottom cycle, rungs
    top_edges = []
    bottom_edges = []
    rung_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Top cycle edge
            # Determine which position in the cycle
            if v == (u + 1) % m:
                pos = u
            else:
                # Edge from m-1 to 0
                pos = m - 1
            top_edges.append((idx, pos))
        elif u >= m and v >= m:
            # Bottom cycle edge
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                pos = u_local
            else:
                pos = m - 1
            bottom_edges.append((idx, pos))
        else:
            # Rung
            i = min(u, v)
            rung_edges.append((idx, i))
    
    # Sort by position for consistency
    top_edges.sort(key=lambda x: x[1])
    bottom_edges.sort(key=lambda x: x[1])
    rung_edges.sort(key=lambda x: x[1])
    
    def eval_continuous(top_s, bot_s, rung_s):
        """Evaluate spectral radius for continuous sign values."""
        A = _build_circulant_signed_adj(m, top_s, bot_s, rung_s)
        return _spectral_radius(A)
    
    def eval_discrete(signs):
        """Evaluate spectral radius for discrete signs."""
        A = _build_signed_adj(graph, signs)
        return _spectral_radius(A)
    
    def is_valid(signs):
        """Check if the lift is connected."""
        return _is_connected_lift(graph, signs)
    
    # ---- Phase-space optimization ----
    # Parameterize: top_signs[i] = cos(theta_top[i]), bottom_signs[i] = cos(theta_bot[i]),
    # rung_signs[i] = cos(theta_rung[i])
    # Optimize over theta in [0, pi] to get values in [-1, 1]
    
    best_obj = np.inf
    best_theta_top = None
    best_theta_bot = None
    best_theta_rung = None
    
    n_trials = 12
    
    for trial in range(n_trials):
        # Random initialization
        if trial == 0:
            # Start with alternating pattern (often good for spectral radius)
            theta_top = np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)])
            theta_bot = np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)])
            theta_rung = np.zeros(m)
        elif trial == 1:
            # Start with all +1 except one -1 for connectivity
            theta_top = np.zeros(m)
            theta_bot = np.zeros(m)
            theta_rung = np.zeros(m)
            theta_rung[0] = np.pi
        else:
            theta_top = rng.uniform(0, np.pi, size=m)
            theta_bot = rng.uniform(0, np.pi, size=m)
            theta_rung = rng.uniform(0, np.pi, size=m)
        
        # Gradient descent with adaptive learning rate
        lr = 0.1
        momentum = 0.9
        vel_top = np.zeros(m)
        vel_bot = np.zeros(m)
        vel_rung = np.zeros(m)
        
        for step in range(500):
            # Compute current values
            top_s = np.cos(theta_top)
            bot_s = np.cos(theta_bot)
            rung_s = np.cos(theta_rung)
            
            # Objective: spectral radius + penalty for not being at +/-1
            A = _build_circulant_signed_adj(m, top_s, bot_s, rung_s)
            vals = np.linalg.eigvalsh(A)
            lam_max = np.max(np.abs(vals))
            
            # Penalty to push toward +/-1 (encourage |cos(theta)| -> 1)
            penalty = 0.01 * (np.sum((1 - top_s**2)**2) + 
                            np.sum((1 - bot_s**2)**2) + 
                            np.sum((1 - rung_s**2)**2))
            obj = lam_max + penalty
            
            if obj < best_obj:
                best_obj = obj
                best_theta_top = theta_top.copy()
                best_theta_bot = theta_bot.copy()
                best_theta_rung = theta_rung.copy()
            
            # Gradient via finite differences
            h = 1e-4
            grad_top = np.zeros(m)
            grad_bot = np.zeros(m)
            grad_rung = np.zeros(m)
            
            for i in range(m):
                # Top
                tp = theta_top.copy(); tp[i] += h
                tm = theta_top.copy(); tm[i] -= h
                op = eval_continuous(np.cos(tp), bot_s, rung_s) + 0.01 * (np.sum((1 - np.cos(tp)**2)**2) + np.sum((1 - bot_s**2)**2) + np.sum((1 - rung_s**2)**2))
                om = eval_continuous(np.cos(tm), bot_s, rung_s) + 0.01 * (np.sum((1 - np.cos(tm)**2)**2) + np.sum((1 - bot_s**2)**2) + np.sum((1 - rung_s**2)**2))
                grad_top[i] = (op - om) / (2 * h)
                
                # Bottom
                bp = theta_bot.copy(); bp[i] += h
                bm = theta_bot.copy(); bm[i] -= h
                op = eval_continuous(top_s, np.cos(bp), rung_s) + 0.01 * (np.sum((1 - top_s**2)**2) + np.sum((1 - np.cos(bp)**2)**2) + np.sum((1 - rung_s**2)**2))
                om = eval_continuous(top_s, np.cos(bm), rung_s) + 0.01 * (np.sum((1 - top_s**2)**2) + np.sum((1 - np.cos(bm)**2)**2) + np.sum((1 - rung_s**2)**2))
                grad_bot[i] = (op - om) / (2 * h)
                
                # Rung
                rp = theta_rung.copy(); rp[i] += h
                rm = theta_rung.copy(); rm[i] -= h
                op = eval_continuous(top_s, bot_s, np.cos(rp)) + 0.01 * (np.sum((1 - top_s**2)**2) + np.sum((1 - bot_s**2)**2) + np.sum((1 - np.cos(rp)**2)**2))
                om = eval_continuous(top_s, bot_s, np.cos(rm)) + 0.01 * (np.sum((1 - top_s**2)**2) + np.sum((1 - bot_s**2)**2) + np.sum((1 - np.cos(rm)**2)**2))
                grad_rung[i] = (op - om) / (2 * h)
            
            # Update with momentum
            vel_top = momentum * vel_top - lr * grad_top
            vel_bot = momentum * vel_bot - lr * grad_bot
            vel_rung = momentum * vel_rung - lr * grad_rung
            
            theta_top = theta_top + vel_top
            theta_bot = theta_bot + vel_bot
            theta_rung = theta_rung + vel_rung
            
            # Keep in [0, pi]
            theta_top = np.clip(theta_top, 0, np.pi)
            theta_bot = np.clip(theta_bot, 0, np.pi)
            theta_rung = np.clip(theta_rung, 0, np.pi)
            
            # Reduce learning rate
            lr *= 0.995
        
        # Final evaluation
        top_s = np.cos(best_theta_top)
        bot_s = np.cos(best_theta_bot)
        rung_s = np.cos(best_theta_rung)
        A = _build_circulant_signed_adj(m, top_s, bot_s, rung_s)
        vals = np.linalg.eigvalsh(A)
        obj = np.max(np.abs(vals)) + 0.01 * (np.sum((1 - top_s**2)**2) + 
                                            np.sum((1 - bot_s**2)**2) + 
                                            np.sum((1 - rung_s**2)**2))
        if obj < best_obj:
            best_obj = obj
    
    # Round to discrete signs
    best_top_s = np.cos(best_theta_top)
    best_bot_s = np.cos(best_theta_bot)
    best_rung_s = np.cos(best_theta_rung)
    
    top_signs = np.where(best_top_s >= 0, 1, -1).astype(int)
    bot_signs = np.where(best_bot_s >= 0, 1, -1).astype(int)
    rung_signs = np.where(best_rung_s >= 0, 1, -1).astype(int)
    
    # Map back to edge order
    signs = np.zeros(n_edges, dtype=int)
    for idx, pos in top_edges:
        signs[idx] = top_signs[pos]
    for idx, pos in bottom_edges:
        signs[idx] = bot_signs[pos]
    for idx, pos in rung_edges:
        signs[idx] = rung_signs[pos]
    
    # Ensure connectivity
    if not is_valid(signs):
        # Try flipping individual edges to achieve connectivity
        improved = False
        for i in range(n_edges):
            cand = signs.copy()
            cand[i] = -cand[i]
            if is_valid(cand):
                signs = cand
                improved = True
                break
        if not improved:
            # Random search for connected configuration
            for _ in range(5000):
                cand = rng.choice([-1, 1], size=n_edges).astype(int)
                if is_valid(cand):
                    signs = cand
                    break
    
    # Local search to improve spectral radius
    current_val = eval_discrete(signs)
    improved = True
    max_iters = 30000
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(n_edges):
            cand = signs.copy()
            cand[i] = -cand[i]
            if not is_valid(cand):
                continue
            v = eval_discrete(cand)
            if v < current_val - 1e-12:
                signs = cand
                current_val = v
                improved = True
                break
    
    # Multiple random restarts with local search
    best_signs = signs.copy()
    best_val = current_val
    
    for restart in range(40):
        cand = rng.choice([-1, 1], size=n_edges).astype(int)
        if not is_valid(cand):
            for i in range(n_edges):
                c2 = cand.copy()
                c2[i] = -c2[i]
                if is_valid(c2):
                    cand = c2
                    break
            else:
                continue
        
        val = eval_discrete(cand)
        
        # Local search
        for _ in range(3000):
            imp = False
            for i in range(n_edges):
                c2 = cand.copy()
                c2[i] = -c2[i]
                if not is_valid(c2):
                    continue
                v2 = eval_discrete(c2)
                if v2 < val - 1e-12:
                    cand = c2
                    val = v2
                    imp = True
                    break
            if not imp:
                break
        
        if val < best_val:
            best_val = val
            best_signs = cand.copy()
    
    return best_signs.astype(int)
