"""Program for spectral graph lifts on prism graphs using circulant structure."""
import numpy as np
from collections import deque


def _build_signed_adj(graph, signs):
    """Build the signed adjacency matrix of the base graph from signs."""
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
    """Check whether the 2-lift is connected."""
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


def _build_circulant_signed_adj(m, top_signs, bottom_signs, rung_signs):
    """Build signed adjacency matrix exploiting circulant structure."""
    n = 2 * m
    A = np.zeros((n, n), dtype=float)
    for i in range(m):
        j = (i + 1) % m
        s = top_signs[i]
        A[i, j] = s
        A[j, i] = s
    for i in range(m):
        j = (i + 1) % m
        s = bottom_signs[i]
        A[m + i, m + j] = s
        A[m + j, m + i] = s
    for i in range(m):
        s = rung_signs[i]
        A[i, m + i] = s
        A[m + i, i] = s
    return A


def _fast_eigvalsh(m, top_s, bot_s, rung_s):
    """Compute eigvalsh of prism signed adjacency matrix efficiently."""
    n = 2 * m
    A = np.zeros((n, n), dtype=float)
    for i in range(m):
        j = (i + 1) % m
        A[i, j] = top_s[i]
        A[j, i] = top_s[i]
    for i in range(m):
        j = (i + 1) % m
        A[m + i, m + j] = bot_s[i]
        A[m + j, m + i] = bot_s[i]
    for i in range(m):
        A[i, m + i] = rung_s[i]
        A[m + i, i] = rung_s[i]
    return np.linalg.eigvalsh(A)


def sign_edges(graph, budget):
    """Return one sign (-1 or +1) per edge, minimizing spectral radius
    of the signed adjacency matrix, subject to the 2-lift being connected."""
    rng = np.random.default_rng(3)
    n_edges = len(graph['edges'])
    m = graph['n'] // 2
    edges = graph['edges']

    top_edges = []
    bottom_edges = []
    rung_edges = []
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            if v == (u + 1) % m:
                pos = u
            else:
                pos = m - 1
            top_edges.append((idx, pos))
        elif u >= m and v >= m:
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                pos = u_local
            else:
                pos = m - 1
            bottom_edges.append((idx, pos))
        else:
            i = min(u, v)
            rung_edges.append((idx, i))

    top_edges.sort(key=lambda x: x[1])
    bottom_edges.sort(key=lambda x: x[1])
    rung_edges.sort(key=lambda x: x[1])

    def eval_continuous(top_s, bot_s, rung_s):
        vals = _fast_eigvalsh(m, top_s, bot_s, rung_s)
        return float(np.max(np.abs(vals)))

    def eval_discrete(signs):
        A = _build_signed_adj(graph, signs)
        return _spectral_radius(A)

    def is_valid(signs):
        return _is_connected_lift(graph, signs)

    # ---- Phase-space optimization with Adam ----
    best_obj = np.inf
    best_theta_top = None
    best_theta_bot = None
    best_theta_rung = None

    # Diverse initializations
    inits = []
    # 0: alternating
    inits.append((np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)]),
                  np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)]),
                  np.zeros(m)))
    # 1: all +1, one -1 rung
    inits.append((np.zeros(m), np.zeros(m), np.array([np.pi] + [0.0]*(m-1))))
    # 2: all -1, one +1 rung
    inits.append((np.full(m, np.pi), np.full(m, np.pi), np.array([0.0] + [np.pi]*(m-1))))
    # 3: alternating top, all + bottom, alternating rungs
    inits.append((np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)]),
                  np.zeros(m),
                  np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)])))
    # 4: all -1, two +1 rungs
    inits.append((np.full(m, np.pi), np.full(m, np.pi),
                  np.array([0.0, 0.0] + [np.pi]*(m-2)) if m >= 2 else np.zeros(m)))
    # 5: all +1, alternating rungs
    inits.append((np.zeros(m), np.zeros(m),
                  np.array([0.0 if i % 2 == 0 else np.pi for i in range(m)])))
    # 6-11: random
    for _ in range(6):
        inits.append((rng.uniform(0, np.pi, size=m),
                      rng.uniform(0, np.pi, size=m),
                      rng.uniform(0, np.pi, size=m)))

    for trial, (tt0, tb0, tr0) in enumerate(inits):
        theta_top = tt0.copy()
        theta_bot = tb0.copy()
        theta_rung = tr0.copy()

        # Adam optimizer
        lr = 0.03
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m_top, v_top = np.zeros(m), np.zeros(m)
        m_bot, v_bot = np.zeros(m), np.zeros(m)
        m_rung, v_rung = np.zeros(m), np.zeros(m)
        t = 0
        h = 1e-3

        for step in range(600):
            top_s = np.cos(theta_top)
            bot_s = np.cos(theta_bot)
            rung_s = np.cos(theta_rung)

            lam_max = eval_continuous(top_s, bot_s, rung_s)
            penalty = 0.005 * (np.sum((1 - top_s**2)**2) +
                              np.sum((1 - bot_s**2)**2) +
                              np.sum((1 - rung_s**2)**2))
            obj = lam_max + penalty

            if obj < best_obj:
                best_obj = obj
                best_theta_top = theta_top.copy()
                best_theta_bot = theta_bot.copy()
                best_theta_rung = theta_rung.copy()

            # Finite difference gradients
            grad_top = np.zeros(m)
            grad_bot = np.zeros(m)
            grad_rung = np.zeros(m)

            for i in range(m):
                tp = theta_top.copy(); tp[i] += h
                tm = theta_top.copy(); tm[i] -= h
                op = eval_continuous(np.cos(tp), bot_s, rung_s) + 0.005*(np.sum((1-np.cos(tp)**2)**2)+np.sum((1-bot_s**2)**2)+np.sum((1-rung_s**2)**2))
                om = eval_continuous(np.cos(tm), bot_s, rung_s) + 0.005*(np.sum((1-np.cos(tm)**2)**2)+np.sum((1-bot_s**2)**2)+np.sum((1-rung_s**2)**2))
                grad_top[i] = (op - om) / (2*h)

                bp = theta_bot.copy(); bp[i] += h
                bm = theta_bot.copy(); bm[i] -= h
                op = eval_continuous(top_s, np.cos(bp), rung_s) + 0.005*(np.sum((1-top_s**2)**2)+np.sum((1-np.cos(bp)**2)**2)+np.sum((1-rung_s**2)**2))
                om = eval_continuous(top_s, np.cos(bm), rung_s) + 0.005*(np.sum((1-top_s**2)**2)+np.sum((1-np.cos(bm)**2)**2)+np.sum((1-rung_s**2)**2))
                grad_bot[i] = (op - om) / (2*h)

                rp = theta_rung.copy(); rp[i] += h
                rm = theta_rung.copy(); rm[i] -= h
                op = eval_continuous(top_s, bot_s, np.cos(rp)) + 0.005*(np.sum((1-top_s**2)**2)+np.sum((1-bot_s**2)**2)+np.sum((1-np.cos(rp)**2)**2))
                om = eval_continuous(top_s, bot_s, np.cos(rm)) + 0.005*(np.sum((1-top_s**2)**2)+np.sum((1-bot_s**2)**2)+np.sum((1-np.cos(rm)**2)**2))
                grad_rung[i] = (op - om) / (2*h)

            t += 1
            m_top = beta1*m_top + (1-beta1)*grad_top
            v_top = beta2*v_top + (1-beta2)*grad_top**2
            theta_top -= lr * (m_top/(1-beta1**t)) / (np.sqrt(v_top/(1-beta2**t)) + eps)

            m_bot = beta1*m_bot + (1-beta1)*grad_bot
            v_bot = beta2*v_bot + (1-beta2)*grad_bot**2
            theta_bot -= lr * (m_bot/(1-beta1**t)) / (np.sqrt(v_bot/(1-beta2**t)) + eps)

            m_rung = beta1*m_rung + (1-beta1)*grad_rung
            v_rung = beta2*v_rung + (1-beta2)*grad_rung**2
            theta_rung -= lr * (m_rung/(1-beta1**t)) / (np.sqrt(v_rung/(1-beta2**t)) + eps)

            theta_top = np.clip(theta_top, 0, np.pi)
            theta_bot = np.clip(theta_bot, 0, np.pi)
            theta_rung = np.clip(theta_rung, 0, np.pi)
            lr *= 0.999

    # Round to discrete signs
    best_top_s = np.cos(best_theta_top)
    best_bot_s = np.cos(best_theta_bot)
    best_rung_s = np.cos(best_theta_rung)

    top_signs = np.where(best_top_s >= 0, 1, -1).astype(int)
    bot_signs = np.where(best_bot_s >= 0, 1, -1).astype(int)
    rung_signs = np.where(best_rung_s >= 0, 1, -1).astype(int)

    signs = np.zeros(n_edges, dtype=int)
    for idx, pos in top_edges:
        signs[idx] = top_signs[pos]
    for idx, pos in bottom_edges:
        signs[idx] = bot_signs[pos]
    for idx, pos in rung_edges:
        signs[idx] = rung_signs[pos]

    # Ensure connectivity
    if not is_valid(signs):
        for i in range(n_edges):
            cand = signs.copy()
            cand[i] = -cand[i]
            if is_valid(cand):
                signs = cand
                break
        else:
            for _ in range(10000):
                cand = rng.choice([-1, 1], size=n_edges).astype(int)
                if is_valid(cand):
                    signs = cand
                    break

    # Local search
    current_val = eval_discrete(signs)
    improved = True
    it = 0
    while improved and it < 50000:
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

    # Random restarts with local search
    best_signs = signs.copy()
    best_val = current_val

    for restart in range(50):
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
        for _ in range(4000):
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
