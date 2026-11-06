# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """Find a high-quality 2-partition of the graph.

    Combines fast O(n) delta-update local search (maintaining As = A @ s)
    with multiple initializations: spectral (Laplacian Fiedler),
    SDP-inspired (eigenvector of D - A), and random.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    adj = np.asarray(adjacency, dtype=np.float64)
    # Ensure symmetry and no self-loops
    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0.0)

    total_edges = 0.5 * float(np.sum(adj))

    def cut_from_As(As, s):
        # cut = total_edges - 0.5 * s^T A s = total_edges - 0.5 * s . As
        return total_edges - 0.5 * float(np.dot(s, As))

    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    best_labels = None
    best_cut = -1.0

    def polish(s):
        """Fast greedy polish using vectorized gain computation.
        s is a -1/+1 array. Returns improved s and its cut."""
        s = s.copy()
        As = adj @ s
        current_cut = cut_from_As(As, s)

        # Repeat: compute gains for all vertices, flip all improving ones
        for _ in range(200):
            if time.time() > deadline:
                break
            # gain[v] = 2 * s[v] * As[v]  (change in s^T A s if we flip v)
            # Flipping v: s[v] -> -s[v], delta in s^T A s = -2 * s[v] * As[v]
            # We want to maximize cut = total - 0.5 * s^T A s, so minimize s^T A s
            # Flipping v changes s^T A s by -2 * s[v] * As[v]
            # We flip if -2 * s[v] * As[v] < 0, i.e., s[v] * As[v] > 0
            gains = s * As  # if gains[v] > 0, flipping v decreases s^T A s (improves cut)
            to_flip = gains > 0
            if not np.any(to_flip):
                break
            # Flip all improving vertices simultaneously
            s[to_flip] = -s[to_flip]
            # Update As: As -= 2 * old_s[v] * adj[v] for each flipped v
            # old_s[v] = -s[v] after flip (since we flipped)
            # Actually: before flip, s_old[v]; after flip s_new[v] = -s_old[v]
            # As_new = As_old + (s_new - s_old) @ adj = As_old - 2 * s_old @ adj
            # s_old[v] = -s[v] (current), so:
            old_s_flipped = -s[to_flip]
            As -= 2.0 * (old_s_flipped @ adj)
            current_cut = cut_from_As(As, s)

        return s, current_cut

    def simulated_annealing(s, As, current_cut, max_time=30.0):
        """SA with O(n) delta updates."""
        s = s.copy()
        As = As.copy()
        current_cut = cut_from_As(As, s)
        best_s = s.copy()
        best_cut_local = current_cut

        sa_deadline = time.time() + max_time
        T = 5.0
        T_min = 0.01
        alpha = 0.99995

        while time.time() < sa_deadline and time.time() < deadline:
            v = np.random.randint(n)
            # delta in cut if we flip v:
            # new s^T A s = old - 2 * s[v] * As[v]
            # new cut = total - 0.5 * new_sAs = old_cut + s[v] * As[v]
            delta = float(s[v] * As[v])

            if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / T)):
                old_s_v = s[v]
                s[v] = -old_s_v
                As -= 2.0 * old_s_v * adj[v]
                current_cut += delta

                if current_cut > best_cut_local:
                    best_cut_local = current_cut
                    best_s = s.copy()

            T = max(T * alpha, T_min)

        return best_s, best_cut_local

    # --- Initialization strategies ---
    inits = []

    # 1. Spectral: Fiedler vector of Laplacian L = D - A
    try:
        deg = np.sum(adj, axis=1)
        L = np.diag(deg) - adj
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector: index 1 (second smallest eigenvalue)
        fiedler = eigvecs[:, 1]
        s_fiedler = np.where(fiedler >= 0, 1.0, -1.0)
        inits.append(s_fiedler)
    except Exception:
        pass

    # 2. SDP-inspired: eigenvector of (D - A) = L, use the one with largest
    #    spread (often the Fiedler vector, but try a few)
    try:
        # Try eigenvectors with indices 1, 2, 3 of L
        for idx in [1, 2, 3]:
            if idx < eigvecs.shape[1]:
                vec = eigvecs[:, idx]
                s_vec = np.where(vec >= 0, 1.0, -1.0)
                inits.append(s_vec)
    except Exception:
        pass

    # 3. Spectral: eigenvector of A corresponding to largest eigenvalue
    try:
        eigvals_a, eigvecs_a = np.linalg.eigh(adj)
        # Largest eigenvalue eigenvector
        idx_max = np.argmax(eigvals_a)
        s_spec = np.where(eigvecs_a[:, idx_max] >= 0, 1.0, -1.0)
        inits.append(s_spec)
    except Exception:
        pass

    # 4. Random initialization
    s_rand = np.random.choice([-1.0, 1.0], size=n)
    inits.append(s_rand)

    # 5. Alternating initialization
    s_alt = np.ones(n)
    s_alt[1::2] = -1.0
    inits.append(s_alt)

    # --- Process each initialization: polish, then optionally SA ---
    for i, s_init in enumerate(inits):
        if time.time() > deadline:
            break

        # Greedy polish
        s_polished, cut_polished = polish(s_init)

        if cut_polished > best_cut:
            best_cut = cut_polished
            best_labels = ((s_polished + 1) / 2).astype(int)

        # Simulated annealing from polished result if time permits
        if time.time() < deadline - 25.0:
            As_polished = adj @ s_polished
            s_sa, cut_sa = simulated_annealing(s_polished, As_polished, cut_polished,
                                               max_time=min(25.0, deadline - time.time() - 10.0))
            if cut_sa > best_cut:
                best_cut = cut_sa
                best_labels = ((s_sa + 1) / 2).astype(int)

    # Fallback
    if best_labels is None:
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
