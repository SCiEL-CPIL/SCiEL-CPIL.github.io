"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _cut_size_fast(adjacency, labels):
    """Fast cut size using matrix operations."""
    A = adjacency.astype(np.int32)
    # Convert labels to +1/-1 for efficient computation
    s = 2 * labels - 1  # +1 for label 1, -1 for label 0
    # Cut size = (n*sum(A) - s^T A s) / 4 for symmetric adjacency
    # Actually: cut = sum over edges (i,j) where s[i]*s[j] < 0
    # = (total_edges - sum_{i,j} A[i,j] * s[i] * s[j] / 2) ... let me just use direct
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _gain_for_flip(adjacency, labels, v):
    """Compute the gain in cut size if we flip vertex v."""
    lv = labels[v]
    A = adjacency.astype(np.int32)
    # Number of neighbors in same partition
    same_count = A[v, labels == lv].sum()
    # Number of neighbors in opposite partition
    opp_count = A[v, labels != lv].sum()
    # If we flip, same becomes opp and vice versa
    # New cut contribution from v: opp_count (edges to same now cross)
    # Old cut contribution from v: opp_count (edges to opp already cross)
    # Wait, let me think again.
    # Currently, edges from v to opposite partition are cut.
    # If we flip v, edges from v to same partition become cut.
    # So the change is: new_cut_edges_from_v - old_cut_edges_from_v
    # = same_count - opp_count
    return int(same_count - opp_count)


def _spectral_partition(adjacency):
    """Use spectral method (Fiedler vector) to get initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.random.randint(0, 2, size=n)
    
    idx = np.argsort(eigvals)
    fiedler = eigvecs[:, idx[1]]
    
    labels = (fiedler >= 0).astype(np.int32)
    
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.random.randint(0, 2, size=n)
    
    return labels


def _gwg_spectral_partition(adjacency):
    """Goemans-Williamson style: use Fiedler vector with threshold."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.random.randint(0, 2, size=n)
    
    idx = np.argsort(eigvals)
    fiedler = eigvecs[:, idx[1]]
    
    # Try multiple thresholds around median
    best_labels = None
    best_cut = -1
    median_val = np.median(fiedler)
    
    for frac in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        threshold = np.percentile(fiedler, frac * 100)
        labels = (fiedler >= threshold).astype(np.int32)
        if labels.sum() == 0 or labels.sum() == n:
            continue
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels
    
    if best_labels is None:
        best_labels = (fiedler >= median_val).astype(np.int32)
        if best_labels.sum() == 0 or best_labels.sum() == n:
            best_labels = np.random.randint(0, 2, size=n)
    
    return best_labels


def _random_labels(n):
    """Generate random initial labels."""
    return np.random.randint(0, 2, size=n)


def _vectorized_local_search(adjacency, labels, deadline, max_rounds=100):
    """Vectorized local search: compute all gains at once, flip best improving vertices."""
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Compute gain for flipping each vertex
        # For vertex v with label lv:
        #   same_count = A[v, labels==lv].sum()
        #   opp_count = A[v, labels!=lv].sum()
        #   gain = same_count - opp_count (if we flip, this many more edges cross)
        
        # Vectorized: for each v, compute A[v, labels==labels[v]].sum() and A[v, labels!=labels[v]].sum()
        # Create mask for same partition
        same_mask = (labels[:, None] == labels[None, :])  # n x n boolean
        opp_mask = ~same_mask
        
        # For each row v, sum A[v, same_mask[v]] and A[v, opp_mask[v]]
        # This is expensive. Let's do it differently.
        
        # For vertex v with label 0: same = neighbors with label 0, opp = neighbors with label 1
        # For vertex v with label 1: same = neighbors with label 1, opp = neighbors with label 0
        
        # Let's compute using matrix multiplication
        # A @ (labels == 0) gives for each v: number of neighbors with label 0
        # A @ (labels == 1) gives for each v: number of neighbors with label 1
        
        count_0 = (A @ (labels == 0).astype(np.int32))  # neighbors with label 0
        count_1 = (A @ (labels == 1).astype(np.int32))  # neighbors with label 1
        
        # For vertex v with label 0: same_count = count_0[v], opp_count = count_1[v]
        # For vertex v with label 1: same_count = count_1[v], opp_count = count_0[v]
        
        same_counts = np.where(labels == 0, count_0, count_1)
        opp_counts = np.where(labels == 0, count_1, count_0)
        
        gains = same_counts - opp_counts
        
        # Find vertices with positive gain
        improving = gains > 0
        if not improving.any():
            break
        
        # Flip all improving vertices simultaneously (can cause oscillation, but let's try)
        # Actually, flipping simultaneously may not be valid. Let's flip one at a time in order of best gain.
        order = np.argsort(-gains)
        
        for v in order:
            if time.time() > deadline:
                break
            if gains[v] <= 0:
                break
            labels[v] = 1 - labels[v]
            improved = True
            # Update gains approximately - this is complex. Let's just do one pass.
        
        if not improved:
            break
    
    return labels


def _local_search_fast(adjacency, labels, deadline, max_rounds=200):
    """Fast local search with vectorized gain computation."""
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Compute gains for all vertices at once
        count_0 = (A @ (labels == 0).astype(np.int32))
        count_1 = (A @ (labels == 1).astype(np.int32))
        
        same_counts = np.where(labels == 0, count_0, count_1)
        opp_counts = np.where(labels == 0, count_1, count_0)
        
        gains = same_counts - opp_counts
        
        # Get vertices with positive gain, sorted by gain descending
        improving_mask = gains > 0
        if not improving_mask.any():
            break
        
        improving_indices = np.where(improving_mask)[0]
        # Sort by gain descending
        sorted_idx = improving_indices[np.argsort(-gains[improving_indices])]
        
        for v in sorted_idx:
            if time.time() > deadline:
                break
            # Re-check gain (may have changed due to previous flips)
            lv = labels[v]
            same_count = A[v, labels == lv].sum()
            opp_count = A[v, labels != lv].sum()
            if same_count > opp_count:
                labels[v] = 1 - lv
                improved = True
        
        if not improved:
            break
    
    return labels


def _simulated_annealing_fast(adjacency, labels, deadline, max_iterations=10000):
    """Simulated annealing with fast gain computation."""
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    T_start = 5.0
    T_end = 0.01
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        progress = it / max_iterations
        T = T_start * (T_end / T_start) ** progress
        
        v = np.random.randint(n)
        old_label = labels[v]
        
        # Compute gain for flipping v
        lv = labels[v]
        same_count = A[v, labels == lv].sum()
        opp_count = A[v, labels != lv].sum()
        delta = same_count - opp_count
        
        if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-10)):
            labels[v] = 1 - old_label
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            pass  # Don't flip
    
    return best_labels, best_cut


def _walksat(adjacency, labels, deadline, max_iterations=20000, bias=0.5):
    """WalkSAT-style stochastic local search for Max-Cut.
    
    At each step:
    - If there's a vertex whose flip improves the cut, flip it (greedy).
    - Otherwise, pick a vertex involved in an uncut edge (with some probability) 
      or randomly, and flip it.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    
    current_cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Compute gains for all vertices
        count_0 = (A @ (labels == 0).astype(np.int32))
        count_1 = (A @ (labels == 1).astype(np.int32))
        
        same_counts = np.where(labels == 0, count_0, count_1)
        opp_counts = np.where(labels == 0, count_1, count_0)
        
        gains = same_counts - opp_counts
        
        # Find vertices with positive gain
        improving = np.where(gains > 0)[0]
        
        if len(improving) > 0:
            # Greedy: pick the best improving vertex
            v = improving[np.argmax(gains[improving])]
        else:
            # No improving move: pick a vertex from an uncut edge with probability bias
            # Uncut edges are those where both endpoints have the same label
            # Find vertices that are part of at least one uncut edge
            
            # For each vertex v, count uncut edges incident to it
            # An edge (v, u) is uncut if labels[v] == labels[u]
            # uncut_degree[v] = sum over u where labels[u] == labels[v] of A[v,u]
            
            # Actually, same_counts[v] is exactly the number of neighbors with same label
            # So vertices with same_counts[v] > 0 have at least one uncut edge
            
            uncut_vertices = np.where(same_counts > 0)[0]
            
            if len(uncut_vertices) > 0 and np.random.random() < bias:
                # Pick from uncut vertices, weighted by same_counts
                weights = same_counts[uncut_vertices].astype(np.float64)
                weights /= weights.sum()
                v = uncut_vertices[np.random.choice(len(uncut_vertices), p=weights)]
            else:
                # Pick randomly
                v = np.random.randint(n)
        
        # Flip vertex v
        labels[v] = 1 - labels[v]
        current_cut += gains[v]  # Note: gains[v] was computed before flip
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def _polish(adjacency, labels, deadline):
    """Final high-precision polish: multiple passes of local search with tighter criteria."""
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int32)
    
    # Multiple passes
    for pass_num in range(5):
        if time.time() > deadline:
            break
        
        improved = False
        # Try multiple orderings
        for order_type in ['random', 'degree_desc']:
            if time.time() > deadline:
                break
            
            if order_type == 'random':
                order = np.random.permutation(n)
            else:
                degrees = A.sum(axis=1)
                order = np.argsort(-degrees)
            
            for v in order:
                if time.time() > deadline:
                    break
                lv = labels[v]
                same_count = A[v, labels == lv].sum()
                opp_count = A[v, labels != lv].sum()
                if same_count > opp_count:
                    labels[v] = 1 - lv
                    improved = True
        
        if not improved:
            break
    
    return labels


def partition(adjacency):
    """Main partition function using spectral initialization, local search, WalkSAT, and polish."""
    n = len(adjacency)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate multiple initial candidates
    candidates = []
    
    # 1. Spectral partition
    spec_labels = _spectral_partition(adjacency)
    candidates.append(spec_labels)
    
    # 2. GWG spectral partition (try multiple thresholds)
    gwg_labels = _gwg_spectral_partition(adjacency)
    candidates.append(gwg_labels)
    
    # 3. Random partitions
    for _ in range(5):
        rand_labels = _random_labels(n)
        candidates.append(rand_labels)
    
    # 4. Alternating partition
    alt_labels = np.arange(n) % 2
    candidates.append(alt_labels)
    
    # Run local search on each candidate
    best_labels = None
    best_cut = -1
    
    for cand in candidates:
        if time.time() > deadline - 20.0:  # Leave time for SA/WalkSAT
            break
        
        ls_labels = _local_search_fast(adjacency, cand, deadline - 20.0)
        ls_cut = _cut_size(adjacency, ls_labels)
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_cut = _cut_size(adjacency, best_labels)
    
    # Run simulated annealing from best local search result
    if time.time() < deadline - 10.0:
        sa_labels, sa_cut = _simulated_annealing_fast(adjacency, best_labels, deadline - 10.0)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Run WalkSAT from best result
    if time.time() < deadline - 5.0:
        ws_labels, ws_cut = _walksat(adjacency, best_labels, deadline - 5.0)
        if ws_cut > best_cut:
            best_cut = ws_cut
            best_labels = ws_labels
    
    # Final polish
    if time.time() < deadline - 2.0:
        final_labels = _polish(adjacency, best_labels, deadline - 1.0)
        final_cut = _cut_size(adjacency, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    return best_labels
# EVOLVE-BLOCK-END
