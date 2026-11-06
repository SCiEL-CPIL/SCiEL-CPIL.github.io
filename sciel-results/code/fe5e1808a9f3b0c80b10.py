"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    cross_mask = labels[:, None] != labels[None, :]
    return int((adj * cross_mask).sum() // 2)


def _goemans_willson_sdp_init(adj, n, num_roundings=10, max_sdp_iters=50):
    """
    Goemans-Williamson style SDP relaxation for Max-Cut.
    1. Solve SDP relaxation approximately using projected gradient descent.
    2. Use randomized hyperplane rounding to generate diverse partitions.
    Returns a list of 0/1 label arrays.
    """
    A = adj.astype(np.float64)
    
    # Initialize Y as a random PSD matrix
    # Start with identity + random symmetric perturbation
    Y = np.eye(n)
    for _ in range(10):
        R = np.random.randn(n, n)
        Y += R @ R.T
    Y = (Y + Y.T) / 2.0
    
    # Project to PSD
    eigvals, eigvecs = np.linalg.eigh(Y)
    eigvals = np.maximum(eigvals, 0)
    Y = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Normalization: ensure trace(Y) = n
    trace_Y = np.trace(Y)
    if trace_Y > 0:
        Y = Y * (n / trace_Y)
    
    # Projected gradient descent for Max-Cut SDP
    # Objective: maximize sum_{i,j} A_{ij} Y_{ij}
    # Subject to: Y >= 0, Y_{ii} = 1
    
    learning_rate = 0.1
    best_Y = Y.copy()
    best_obj = -np.inf
    
    for it in range(max_sdp_iters):
        # Gradient of objective
        grad = 2.0 * A  # d/dY sum A_ij Y_ij = 2A for symmetric Y
        
        # Update
        Y_new = Y + learning_rate * grad
        
        # Enforce diagonal = 1
        Y_new[np.diag_indices(n)] = 1.0
        
        # Symmetrize
        Y_new = (Y_new + Y_new.T) / 2.0
        
        # Project to PSD
        eigvals, eigvecs = np.linalg.eigh(Y_new)
        eigvals = np.maximum(eigvals, 0)
        Y_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Re-enforce diagonal = 1 after projection
        Y_new[np.diag_indices(n)] = 1.0
        
        Y = Y_new
        
        # Compute objective
        obj = np.sum(A * Y)
        if obj > best_obj:
            best_obj = obj
            best_Y = Y.copy()
        
        # Adaptive learning rate
        if obj < best_obj * 0.99:
            learning_rate *= 0.5
    
    Y = best_Y
    
    # Randomized hyperplane rounding
    # For each vertex i, we have a unit vector v_i (row of Y)
    # Generate random unit vector r, assign i to side 0 if <v_i, r> >= 0
    
    labels_list = []
    for _ in range(num_roundings):
        # Generate random unit vector
        r = np.random.randn(n)
        r = r / np.linalg.norm(r)
        
        # Extract vectors from Y
        # v_i is the i-th row of Y (since Y_ii = 1, these are unit vectors)
        # Actually, Y = V V^T where V is n x n, V_i = v_i
        # We can use Cholesky or eigendecomposition to get V
        # But for rounding, we just need signs of <v_i, r>
        # <v_i, r> = sum_j Y_{ij} V_{jr} ... this is complex
        # Simpler: use the fact that if Y = V V^T, then Y r = V (V^T r)
        # So <v_i, r> = (Y r)_i / ||v_i|| ... but ||v_i|| = 1
        # Actually <v_i, r> is not simply (Yr)_i
        
        # Let's use eigendecomposition: Y = U D U^T
        # v_i = U_i * sqrt(D_i) ... no
        # Y_{ij} = v_i . v_j
        # Let V be the matrix with rows v_i. Then Y = V V^T.
        # We want sign(v_i . r) for each i.
        # v_i . r = (V r)_i
        # So we need V. From Y = V V^T, we can get V via Cholesky or SVD.
        
        # Use SVD: Y = U S U^T (since symmetric)
        # V = U * sqrt(S)
        eigvals, eigvecs = np.linalg.eigh(Y)
        eigvals = np.maximum(eigvals, 0)
        V = eigvecs * np.sqrt(eigvals)  # This gives V where Y = V V^T
        
        # Now v_i . r = (V^T r)_i ... wait
        # V has rows v_i, so V r is a vector with components (V r)_i = v_i . r
        # But V is n x n, V[i, :] = v_i
        # So V @ r gives vector with i-th component = v_i . r
        
        scores = V @ r
        labels = (scores >= 0).astype(np.int8)
        labels_list.append(labels)
    
    return labels_list


def _spectral_init(adj, n):
    """
    Initialize using the second eigenvector of the graph Laplacian.
    Returns a 0/1 label array.
    """
    # Build Laplacian L = D - A
    degrees = np.sum(adj, axis=1).astype(float)
    D = np.diag(degrees)
    A = adj.astype(float)
    L = D - A

    try:
        # Use numpy's eigvalsh/eigh for symmetric matrix
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The second smallest eigenvector (index 1) is the Fiedler vector
        fiedler = eigenvectors[:, 1]
        # Threshold at 0 to create partition
        labels = (fiedler > 0).astype(np.int8)
    except Exception:
        # Fallback to random
        labels = np.random.randint(0, 2, size=n).astype(np.int8)

    return labels


def _random_init(n):
    """Random 0/1 initialization."""
    return np.random.randint(0, 2, size=n).astype(np.int8)


def _greedy_improve(adj, labels, max_passes=100):
    """
    Greedy local search: repeatedly flip vertices that improve the cut.
    Returns improved labels.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()

    for _ in range(max_passes):
        improved = False
        for v in range(n):
            same = adj_int[v][labels == labels[v]].sum()
            other = adj_int[v][labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
                improved = True
        if not improved:
            break

    return labels


def _goemans_maxwell(adj, labels):
    """
    Goemans-Williamson style local improvement:
    For each vertex, compute the net gain of flipping it.
    Use a priority queue approach to find the best flip.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    
    # Precompute for each vertex:
    # gain[v] = (edges to opposite side) - (edges to same side)
    # If gain[v] > 0, flipping v increases the cut by gain[v]
    
    for _ in range(50):
        # Compute gains for all vertices
        gains = np.zeros(n, dtype=np.int32)
        for v in range(n):
            same_count = adj_int[v][labels == labels[v]].sum()
            other_count = adj_int[v][labels != labels[v]].sum()
            gains[v] = other_count - same_count
        
        # Find vertex with maximum positive gain
        max_gain_idx = np.argmax(gains)
        max_gain = gains[max_gain_idx]
        
        if max_gain <= 0:
            break
        
        # Flip the vertex
        v = max_gain_idx
        labels[v] = 1 - labels[v]
        current_cut += max_gain
        
        # Update gains for neighbors of v
        neighbors = np.where(adj_int[v])[0]
        for u in neighbors:
            if u == v:
                continue
            # Recompute gain for u
            same_count = adj_int[u][labels == labels[u]].sum()
            other_count = adj_int[u][labels != labels[u]].sum()
            gains[u] = other_count - same_count
            # Also update gain for v (its neighbors changed relative to v's side)
        
        # Recompute gain for v
        same_count = adj_int[v][labels == labels[v]].sum()
        other_count = adj_int[v][labels != labels[v]].sum()
        gains[v] = other_count - same_count
    
    return labels, current_cut


def _simulated_annealing(adj, labels, start_temp=10.0, cool_rate=0.99, max_steps=5000):
    """
    Simple simulated annealing to escape local optima.
    Returns improved labels.
    """
    n = len(adj)
    adj_int = adj.astype(np.int8)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    temp = start_temp

    for step in range(max_steps):
        v = np.random.randint(0, n)
        old_label = labels[v]
        new_label = 1 - old_label

        # Compute change in cut size
        same_count = adj_int[v][labels == old_label].sum()
        other_count = adj_int[v][labels != old_label].sum()

        # If we flip v from old_label to new_label:
        # Edges to same-side vertices become crossing (+1 each)
        # Edges to other-side vertices become non-crossing (-1 each)
        delta = same_count - other_count

        new_cut = current_cut + delta

        if new_cut > current_cut:
            accept = True
        else:
            if temp < 1e-10:
                accept = False
            else:
                accept = np.random.random() < np.exp(delta / temp)

        if accept:
            labels[v] = new_label
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()

        temp *= cool_rate

    return best_labels


def partition(adjacency):
    """
    Partition the graph into two sets to maximize the cut.
    Uses spectral initialization + GW-SDP rounding + greedy local search + simulated annealing.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)

    # Generate multiple initializations
    inits = []

    # Spectral initialization
    spec_labels = _spectral_init(adj, n)
    inits.append(spec_labels)

    # Alternating initialization
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)

    # Goemans-Williamson SDP rounding initializations
    if time.time() < deadline - 30:
        try:
            gw_labels = _goemans_willson_sdp_init(adj, n, num_roundings=8, max_sdp_iters=30)
            inits.extend(gw_labels)
        except Exception:
            pass

    # Several random initializations
    for _ in range(3):
        inits.append(_random_init(n))

    best_labels = None
    best_cut = -1

    for init_labels in inits:
        if time.time() > deadline:
            break

        # Greedy local search
        greedy_labels = _greedy_improve(adj, init_labels)
        greedy_cut = _cut_size(adj, greedy_labels)

        if greedy_cut > best_cut:
            best_cut = greedy_cut
            best_labels = greedy_labels.copy()

        # Simulated annealing from the greedy result
        if time.time() < deadline - 15:
            sa_labels = _simulated_annealing(adj, greedy_labels, start_temp=5.0, cool_rate=0.995, max_steps=2000)
            sa_cut = _cut_size(adj, sa_labels)

            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()

    # Final greedy pass on the best found
    if best_labels is not None and time.time() < deadline:
        final_labels = _greedy_improve(adj, best_labels)
        final_cut = _cut_size(adj, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels.copy()

    if best_labels is None:
        # Fallback
        best_labels = np.arange(n) % 2

    return best_labels
# EVOLVE-BLOCK-END
