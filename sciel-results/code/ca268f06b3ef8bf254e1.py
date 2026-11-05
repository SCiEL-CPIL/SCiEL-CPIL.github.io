"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
import scipy.sparse.linalg as spla

DEADLINE_MARGIN = 5.0


def _cut_size(labels, adj):
    """Compute the cut size (number of crossing edges) for a labeling."""
    diff = labels[None, :] != labels[:, None]
    return int(np.sum(adj & diff))


def _vectorized_gains(labels, adj):
    """
    Compute flip gain for every vertex in a vectorized way.
    gain[v] = change in cut size if we flip vertex v.
    """
    deg = adj.sum(axis=1)
    neighbor_label_sum = adj @ labels
    sign = 2 * labels - 1
    gains = sign * (2 * neighbor_label_sum - deg)
    return gains


def _greedy_polish(labels, adj, deadline):
    """
    Greedy local search: repeatedly flip the vertex with the best positive gain.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_cut = current_cut
    best_labels = labels.copy()

    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        gains = _vectorized_gains(labels, adj)
        max_gain = int(gains.max())
        if max_gain > 0:
            best_v = int(np.argmax(gains))
            labels[best_v] = 1 - labels[best_v]
            current_cut += max_gain
            improved = True
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

    return best_labels, best_cut


def _simulated_annealing(labels, adj, deadline, rng):
    """
    Simulated annealing on the cut problem with adaptive iteration count.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_labels = labels.copy()
    best_cut = current_cut

    T_start = 10.0
    T_end = 0.01
    # Estimate how many iterations we can afford
    elapsed = time.time() - (deadline - 150.0 + 5.0)  # approximate start
    remaining = max(deadline - time.time(), 0.1)
    # Each iteration takes roughly O(n) for gain computation + O(1) for update
    # Empirically ~0.1ms per iteration for n=100, so we can do ~10000 in 1s
    # Be conservative: assume 2ms per iteration
    n_iter = int(min(3000, max(100, remaining * 500)))

    for it in range(n_iter):
        if time.time() > deadline:
            break

        if it < n_iter - 1:
            T = T_start * (T_end / T_start) ** (it / max(n_iter - 1, 1))
        else:
            T = T_end

        gains = _vectorized_gains(labels, adj)

        v = rng.integers(0, n)
        delta = gains[v]

        if delta >= 0:
            labels[v] = 1 - labels[v]
            current_cut += int(delta)
        else:
            if T > 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    labels[v] = 1 - labels[v]
                    current_cut += int(delta)

        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()

    return best_labels, best_cut


def _goemans_maxwfn(adj, deadline):
    """
    Goemans-Williamson style rounding via random hyperplane.
    For MaxCut, we solve the SDP relaxation approximately.
    Since we don't have an SDP solver, we use a heuristic:
    Use the eigenvector of the Laplacian or adjacency matrix as a proxy for the SDP solution.
    """
    n = adj.shape[0]
    # Use the Fiedler vector (2nd smallest eigenvector of Laplacian) as a starting point
    # This is a common heuristic for graph partitioning
    try:
        L = csgraph_laplacian(adj, normed=False)
        # Get the 2nd smallest eigenvector
        vals, vecs = spla.eigsh(L, k=2, which='SM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler = vecs[:, idx[1]]
        # Round to 0/1 based on sign
        labels = (fiedler >= 0).astype(int)
    except Exception:
        # Fallback to random
        labels = np.random.randint(0, 2, size=n)
    
    return labels


def _ilp_polish(labels, adj, deadline):
    """
    Final high-precision polish: single-flip greedy + improved double-flip search.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(labels, adj)
    best_cut = current_cut
    best_labels = labels.copy()

    # First do single-flip greedy
    improved = True
    while improved:
        if time.time() > deadline:
            return best_labels, best_cut
        improved = False
        gains = _vectorized_gains(labels, adj)
        max_gain = int(gains.max())
        if max_gain > 0:
            best_v = int(np.argmax(gains))
            labels[best_v] = 1 - labels[best_v]
            current_cut += max_gain
            improved = True
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

    # Then try double-flip moves with better pair selection
    if time.time() > deadline - 1.0:
        return best_labels, best_cut
    
    rng = np.random.default_rng(123)
    for round_num in range(10):  # up to 10 rounds of double-flip
        if time.time() > deadline - 0.3:
            break
        best_double_gain = 0
        best_pair = None
        
        # Compute gains once for this round
        gains = _vectorized_gains(labels, adj)
        
        # Try pairs where at least one has positive gain, plus random sampling
        # This is more targeted than pure random
        num_samples = 2000
        for _ in range(num_samples):
            if time.time() > deadline - 0.3:
                break
            i, j = rng.choice(n, size=2, replace=False)
            if i == j:
                continue
            
            # Quick estimate: if both have negative gain, unlikely to help
            # but still try some random pairs
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            new_cut = _cut_size(labels, adj)
            delta = new_cut - current_cut
            if delta > best_double_gain:
                best_double_gain = delta
                best_pair = (i, j)
            # Revert
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
        
        if best_pair is not None and best_double_gain > 0:
            i, j = best_pair
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            current_cut += best_double_gain
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # No improvement in this round, try a few more random restarts of double-flip
            for extra in range(3):
                if time.time() > deadline - 0.3:
                    break
                best_double_gain = 0
                best_pair = None
                for _ in range(500):
                    if time.time() > deadline - 0.3:
                        break
                    i, j = rng.choice(n, size=2, replace=False)
                    if i == j:
                        continue
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    new_cut = _cut_size(labels, adj)
                    delta = new_cut - current_cut
                    if delta > best_double_gain:
                        best_double_gain = delta
                        best_pair = (i, j)
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                
                if best_pair is not None and best_double_gain > 0:
                    i, j = best_pair
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
                    current_cut += best_double_gain
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
                else:
                    break
            break

    return best_labels, best_cut


def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses multiple strategies: Fiedler vector initialization, simulated annealing,
    greedy polish, and double-flip local search.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)

    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN

    best_labels = None
    best_cut = -1

    rng = np.random.default_rng(42)

    # Phase 1: Fiedler vector initialization + greedy polish
    if time.time() < deadline - 10.0:
        labels_fiedler = _goemans_maxwfn(adj, deadline)
        labels_fiedler, cut_fiedler = _greedy_polish(labels_fiedler, adj, deadline)
        best_labels = labels_fiedler.copy()
        best_cut = cut_fiedler

    # Phase 2: Alternating start + greedy polish
    labels_alt = np.arange(n) % 2
    labels_alt, cut_alt = _greedy_polish(labels_alt, adj, deadline)
    if cut_alt > best_cut:
        best_cut = cut_alt
        best_labels = labels_alt.copy()

    # Phase 3: Random restarts with simulated annealing
    n_restarts_max = 50
    for r in range(n_restarts_max):
        if time.time() > deadline - 15.0:
            break

        # Random initial labeling
        labels = rng.integers(0, 2, size=n)
        labels, cut = _simulated_annealing(labels, adj, deadline, rng)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Also try a greedy polish after SA
        if time.time() > deadline - 15.0:
            continue
        labels, cut = _greedy_polish(labels, adj, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Final: high-precision polish on the best found
    if time.time() < deadline - 3.0:
        best_labels, best_cut = _ilp_polish(best_labels, adj, deadline)
        
        # One more round of greedy polish after double-flip
        if time.time() < deadline - 1.0:
            best_labels, best_cut = _greedy_polish(best_labels, adj, deadline)

    return best_labels
# EVOLVE-BLOCK-END
