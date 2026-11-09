# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    diff = np.abs(labels[:, np.newaxis] - labels[np.newaxis, :])
    return np.sum(adj * diff) / 2.0


def _all_flip_gains(adj, labels):
    """Compute the gain from flipping each vertex, vectorized."""
    s = 2 * labels - 1
    As = adj @ s
    gains = s * As
    return gains


def _local_search_1flip(adj, labels, deadline=None):
    """Improve partition by greedy 1-flip until no improvement."""
    n = len(labels)
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        gains = _all_flip_gains(adj, labels)
        max_gain = np.max(gains)
        if max_gain <= 0:
            break
        v = np.argmax(gains)
        labels[v] = 1 - labels[v]
        improved = True
    return labels


def _local_search_2swap(adj, labels, deadline=None):
    """Improve partition by greedy 2-swap until no improvement (vectorized)."""
    n = len(labels)
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        s = 2 * labels - 1
        As = adj @ s
        sAs = s * As
        
        # For pairs (i, j) with labels[i] != labels[j], the gain from swapping is:
        # sAs[i] + sAs[j] + 2*adj[i,j]
        sAs_sum = sAs[:, np.newaxis] + sAs[np.newaxis, :]
        gain_matrix = sAs_sum + 2 * adj
        
        # Zero out diagonal and same-label pairs
        np.fill_diagonal(gain_matrix, -np.inf)
        same_label_mask = (labels[:, np.newaxis] == labels[np.newaxis, :])
        gain_matrix[same_label_mask] = -np.inf
        
        max_gain = np.max(gain_matrix)
        if max_gain <= 0:
            break
        
        idx = np.unravel_index(np.argmax(gain_matrix), gain_matrix.shape)
        i, j = idx
        if i > j:
            i, j = j, i
        
        labels[i], labels[j] = labels[j], labels[i]
        improved = True
    return labels


def _full_local_search(adj, labels, deadline=None):
    """Alternate 1-flip and 2-swap local search with more rounds."""
    if deadline is None:
        deadline = time.time() + 10
    labels = _local_search_1flip(adj, labels, deadline)
    if time.time() > deadline:
        return labels
    labels = _local_search_2swap(adj, labels, deadline)
    if time.time() > deadline:
        return labels
    labels = _local_search_1flip(adj, labels, deadline)
    if time.time() > deadline:
        return labels
    labels = _local_search_2swap(adj, labels, deadline)
    if time.time() > deadline:
        return labels
    labels = _local_search_1flip(adj, labels, deadline)
    return labels


def _tabu_search(adj, labels, tabu_len=10, max_iter=500, deadline=None):
    """Tabu search with 1-flip moves."""
    n = len(labels)
    if deadline is None:
        deadline = time.time() + 10
    
    current_labels = labels.copy()
    current_cut = _cut_size(adj, current_labels)
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    tabu = {}
    iteration = 0
    
    while iteration < max_iter:
        if time.time() > deadline:
            break
        
        gains = _all_flip_gains(adj, current_labels)
        
        best_gain = -np.inf
        best_v = None
        for v in range(n):
            if v in tabu and tabu[v] > iteration - tabu_len:
                continue
            if gains[v] > best_gain:
                best_gain = gains[v]
                best_v = v
        
        if best_v is None:
            break
        
        # Aspiration: allow tabu move if it improves best
        if best_v in tabu and tabu[best_v] > iteration - tabu_len:
            if best_gain <= best_cut - current_cut:
                found = False
                for v in range(n):
                    if v not in tabu or tabu[v] <= iteration - tabu_len:
                        if gains[v] > best_gain:
                            best_gain = gains[v]
                            best_v = v
                            found = True
                if not found:
                    break
        
        current_labels[best_v] = 1 - current_labels[best_v]
        tabu[best_v] = iteration + 1
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = current_labels.copy()
        
        iteration += 1
    
    return best_labels


def _simulated_annealing(adj, labels, deadline=None):
    """Simulated annealing with 1-flip and 2-swap moves."""
    n = len(labels)
    if deadline is None:
        deadline = time.time() + 10
    
    current_labels = labels.copy()
    current_cut = _cut_size(adj, current_labels)
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    temperature = 5.0
    cooling = 0.95
    
    while temperature > 0.01:
        if time.time() > deadline:
            break
        
        v = np.random.randint(n)
        gain = _all_flip_gains(adj, current_labels)[v]
        
        if gain > 0:
            current_labels[v] = 1 - current_labels[v]
            current_cut += gain
        else:
            if np.random.random() < np.exp(gain / temperature):
                current_labels[v] = 1 - current_labels[v]
                current_cut += gain
        
        if np.random.random() < 0.3:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j and current_labels[i] != current_labels[j]:
                s = 2 * current_labels - 1
                As = adj @ s
                sAs = s * As
                gain2 = sAs[i] + sAs[j] + 2 * adj[i, j]
                if gain2 > 0:
                    current_labels[i], current_labels[j] = current_labels[j], current_labels[i]
                    current_cut += gain2
                else:
                    if np.random.random() < np.exp(gain2 / temperature):
                        current_labels[i], current_labels[j] = current_labels[j], current_labels[i]
                        current_cut += gain2
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = current_labels.copy()
        
        temperature *= cooling
    
    return best_labels


def _spectral_init(adj, n):
    """Initialize labels using the Fiedler vector (eigenvector of Laplacian)."""
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        labels = (fiedler > 0).astype(int)
    except:
        labels = np.arange(n) % 2
    return labels


def _admm_maxcut(adj, n, max_iter=200, rho=1.0, deadline=None):
    """
    ADMM for Max-Cut.
    
    Formulation:
      max  -1/2 * s^T A s   (equivalent to maximizing cut size)
      s.t.  s_i in {-1, 1}
    
    Relax to:
      max  -1/2 * s^T A s
      s.t.  s = x,  x_i in {-1, 1}
    
    ADMM updates:
      s^{k+1} = argmin  -1/2 s^T A s + rho/2 * ||s - x^k + z^k||^2
      = argmin  1/2 s^T (A - rho I) s - rho * (x^k - z^k)^T s
      This is a convex quadratic if A - rho I is PSD, which requires rho >= lambda_max(A).
      For safety, we use a larger rho.
      
      Actually, let me reformulate more carefully.
      
      The objective to minimize is: -1/2 s^T A s (since we want to maximize s^T A s)
      Wait, that's not convex. Let me think again.
      
      Max-Cut: max sum_{(i,j) in E} (1 - s_i s_j) / 2
              = |E|/2 - 1/2 sum_{i,j} A_{ij} s_i s_j
              = |E|/2 - 1/2 s^T A s
      
      So maximizing cut = minimizing 1/2 s^T A s.
      
      But 1/2 s^T A s is not convex in general (A is PSD, so this IS convex).
      Wait, A is the adjacency matrix, which is PSD. So 1/2 s^T A s is convex.
      But we want to MINIMIZE it, subject to s_i in {-1,1}.
      That's a convex minimization with discrete constraints.
      
      ADMM:
        min  1/2 s^T A s
        s.t.  s = x,  x_i in {-1, 1}
        
        Augmented Lagrangian:
          L_rho = 1/2 s^T A s + rho/2 * ||s - x + z||^2
        
        s-update: min_s 1/2 s^T A s + rho/2 * ||s - x + z||^2
                 = min_s 1/2 s^T (A + rho I) s - rho * (x - z)^T s
                 s* = (A + rho I)^{-1} (x - z)
        
        x-update: min_x rho/2 * ||s - x + z||^2  s.t. x_i in {-1, 1}
                 x_i = clip(s_i + z_i, -1, 1)
        
        z-update: z = z + s - x
    """
    if deadline is None:
        deadline = time.time() + 5
    
    A = adj.copy()
    E = np.sum(A) / 2.0
    
    # Initialize
    s = np.random.randn(n)
    x = np.sign(s)
    z = np.zeros(n)
    
    # Precompute (A + rho I)^{-1}
    A_plus_rho = A + rho * np.eye(n)
    try:
        A_inv = np.linalg.inv(A_plus_rho)
    except:
        A_inv = np.linalg.pinv(A_plus_rho)
    
    best_labels = None
    best_cut = -1
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
        
        # s-update
        rhs = x - z
        s = A_inv @ rhs
        
        # x-update
        x = np.clip(s + z, -1, 1)
        
        # z-update
        z = z + s - x
        
        # Check convergence / evaluate
        if k % 10 == 0 or k == max_iter - 1:
            labels = (s > 0).astype(int)
            cut = _cut_size(adj, labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = (s > 0).astype(int)
    
    return best_labels


def _admm_polish(adj, labels, deadline=None):
    """Run ADMM then polish with local search."""
    n = len(labels)
    admm_labels = _admm_maxcut(adj, n, max_iter=100, rho=5.0, deadline=deadline)
    polished = _full_local_search(adj, admm_labels, deadline)
    return polished


def partition(adjacency):
    n = len(adjacency)
    adj = np.array(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Generate initial solutions
    inits = []
    
    # Spectral init
    spec_labels = _spectral_init(adj, n)
    inits.append(spec_labels)
    
    # Alternating init
    alt_labels = np.arange(n) % 2
    inits.append(alt_labels)
    
    # ADMM init
    np.random.seed(42)
    admm_labels = _admm_maxcut(adj, n, max_iter=50, rho=5.0, deadline=deadline)
    inits.append(admm_labels)
    
    # Random inits
    for _ in range(6):
        rand_labels = np.random.randint(0, 2, n)
        inits.append(rand_labels)
    
    for init_labels in inits:
        if time.time() > deadline:
            break
        
        labels = init_labels.copy()
        
        # Full local search
        labels = _full_local_search(adj, labels, deadline)
        cut = _cut_size(adj, labels)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Tabu search
        if time.time() > deadline:
            break
        tabu_labels = _tabu_search(adj, labels, tabu_len=10, max_iter=300, deadline=deadline)
        tabu_cut = _cut_size(adj, tabu_labels)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()
        
        # Simulated annealing
        if time.time() > deadline:
            break
        sa_labels = _simulated_annealing(adj, labels, deadline=deadline)
        sa_cut = _cut_size(adj, sa_labels)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # ADMM polish from current best
        if time.time() > deadline:
            break
        admm_polished = _admm_polish(adj, best_labels.copy(), deadline)
        admm_pol_cut = _cut_size(adj, admm_polished)
        if admm_pol_cut > best_cut:
            best_cut = admm_pol_cut
            best_labels = admm_polished.copy()
        
        # Final polish: run more 1-flip and 2-swap iterations on the best so far
        if time.time() > deadline:
            break
        polished = _local_search_1flip(adj, best_labels.copy(), deadline)
        polished = _local_search_2swap(adj, polished, deadline)
        polished = _local_search_1flip(adj, polished, deadline)
        pol_cut = _cut_size(adj, polished)
        if pol_cut > best_cut:
            best_cut = pol_cut
            best_labels = polished.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
