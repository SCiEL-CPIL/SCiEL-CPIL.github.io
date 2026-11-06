"""Maximum Cut: spectral init + fast greedy + SA with incremental gain tracking."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute cut size efficiently using matrix operations."""
    # cut = sum_v labels[v] * deg[v] - labels^T A labels
    labels_f = labels.astype(np.float64)
    deg = adj.sum(axis=1).astype(np.float64)
    return float(labels_f @ deg - labels_f @ adj @ labels_f)


def _compute_gains(adj, labels):
    """
    Compute gain[v] = delta in cut if we flip vertex v.
    gain[v] = (neighbors of v on opposite side) - (neighbors of v on same side)
             = diff[v] - same[v]
    """
    n = len(labels)
    labels_f = labels.astype(np.float64)
    # For each v: same[v] = sum of adj[v, u] where labels[u] == labels[v]
    #             diff[v] = sum of adj[v, u] where labels[u] != labels[v]
    # gain[v] = diff[v] - same[v]
    # Note: deg[v] = same[v] + diff[v]
    # So gain[v] = diff[v] - same[v] = deg[v] - 2*same[v]
    
    # same[v] = adj[v] @ (labels == labels[v]) but that's per-row boolean
    # Vectorized: create a matrix M where M[v,u] = 1 if labels[v]==labels[u] else 0
    # same = M @ adj ... no, same[v] = sum_u adj[v,u] * (labels[v]==labels[u])
    # = (adj * (labels[:,None] == labels[None,:])).sum(axis=1)
    
    same_mask = (labels[:, None] == labels[None, :])  # n x n bool
    same_counts = (adj & same_mask).sum(axis=1)  # n
    deg = adj.sum(axis=1)
    gains = deg - 2 * same_counts  # gain[v] = diff - same = (deg - same) - same = deg - 2*same
    return gains, same_counts, deg


def _greedy_improve_fast(adj, labels, deadline):
    """
    Fast greedy local search using incremental gain updates.
    When we flip vertex v, we update gains for all neighbors of v.
    """
    n = len(labels)
    labels = labels.copy()
    gains, same_counts, deg = _compute_gains(adj, labels)
    
    # adj_int for fast neighbor lookup
    adj_int = adj.astype(np.int8)
    
    improved = True
    passes = 0
    while improved and time.time() < deadline:
        improved = False
        passes += 1
        # Process vertices in random order
        order = list(range(n))
        random.shuffle(order)
        for v in order:
            if time.time() >= deadline:
                return labels
            if gains[v] > 0:
                # Flip v
                labels[v] = 1 - labels[v]
                improved = True
                
                # Update gains for all neighbors of v
                # When v flips, for each neighbor u of v:
                #   u's gain changes because v moved from one side to the other
                #   If u and v were on same side, after flip they're on different sides
                #   The gain for u changes by: -2 * (was_same ? +1 : -1) ... 
                #   Actually: gain[u] = deg[u] - 2*same[u]
                #   When v flips, same[u] changes by +1 if v was same as u, -1 if different
                #   So gain[u] changes by -2 * (1 if v was same as u, -1 if different)
                #   = -2 * (labels_old[v] == labels[u] ? 1 : -1)
                #   But labels_old[v] is the old label of v before flip
                #   After flip, labels[v] is new. Before flip, labels[v] was 1-labels[v].
                #   So: if (1 - labels[v]) == labels[u], then same[u] increased by 1, gain[u] -= 2
                #       else same[u] decreased by 1, gain[u] += 2
                #   Simplify: gain[u] += 2 * (1 if (1-labels[v]) != labels[u] else -1)
                #   = 2 * (1 if labels[v] == labels[u] else -1)
                #   Wait, let me re-derive:
                #   Before flip: v has label old_v = 1 - labels[v] (after flip)
                #   For neighbor u: was_same = (old_v == labels[u])
                #   After flip: v has label labels[v]
                #   Now_same = (labels[v] == labels[u]) = (1 - old_v == labels[u]) = not was_same
                #   So same[u] changes by: +1 if was_same (now different, so same decreases)... 
                #   Wait: same[u] = count of neighbors with same label as u.
                #   Before: if old_v == labels[u], v contributed to same[u]. After: if labels[v] == labels[u], v contributes.
                #   Since labels[v] = 1 - old_v, labels[v] == labels[u] iff old_v != labels[u].
                #   So: if old_v == labels[u] (was same), now different -> same[u] decreases by 1
                #       if old_v != labels[u] (was different), now same -> same[u] increases by 1
                #   gain[u] = deg[u] - 2*same[u], so:
                #   if was same: gain[u] += 2
                #   if was different: gain[u] -= 2
                #   was_same = (1 - labels[v]) == labels[u] = (labels[v] != labels[u])
                #   So: if labels[v] != labels[u]: gain[u] += 2
                #       if labels[v] == labels[u]: gain[u] -= 2
                #   gain[u] += 2 * (1 if labels[v] != labels[u] else -1)
                #   = 2 * (1 - 2*int(labels[v] == labels[u]))
                
                neighbors = np.where(adj[v])[0]
                for u in neighbors:
                    if u == v:
                        continue
                    if labels[v] == labels[u]:
                        gains[u] -= 2
                    else:
                        gains[u] += 2
                
                # Also update gain for v itself: after flip, gain[v] becomes -gain[v]
                # Because the situation is symmetric: what was diff is now same and vice versa
                gains[v] = -gains[v]
    
    return labels


def _sa_with_gains(adj, labels, deadline, start_temp=8.0, cooling=0.9995):
    """Simulated annealing with incremental gain tracking."""
    n = len(labels)
    labels = labels.copy()
    gains, same_counts, deg = _compute_gains(adj, labels)
    
    current_cut = _cut_size(adj, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    temp = start_temp
    adj_int = adj.astype(np.int8)
    
    while time.time() < deadline and temp > 0.01:
        # Pick random vertex
        v = random.randint(0, n - 1)
        
        delta = gains[v]
        
        if delta > 0:
            # Accept
            labels[v] = 1 - labels[v]
            current_cut += delta
            
            # Update gains
            neighbors = np.where(adj[v])[0]
            for u in neighbors:
                if u == v:
                    continue
                if labels[v] == labels[u]:
                    gains[u] -= 2
                else:
                    gains[u] += 2
            gains[v] = -gains[v]
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # Accept with probability exp(delta/temp)
            if random.random() < np.exp(delta / temp):
                labels[v] = 1 - labels[v]
                current_cut += delta
                
                neighbors = np.where(adj[v])[0]
                for u in neighbors:
                    if u == v:
                        continue
                    if labels[v] == labels[u]:
                        gains[u] -= 2
                    else:
                        gains[u] += 2
                gains[v] = -gains[v]
        
        temp *= cooling
    
    return best_labels


def _spectral_init(adj):
    """Spectral initialization using Fiedler vector."""
    n = len(adj)
    adj_f = adj.astype(np.float64)
    deg = adj_f.sum(axis=1)
    L = np.diag(deg) - adj_f
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]
    except:
        fiedler = np.random.randn(n)
    
    threshold = np.mean(fiedler)
    labels = (fiedler > threshold).astype(int)
    
    if labels.sum() == 0 or labels.sum() == n:
        threshold = np.median(fiedler)
        labels = (fiedler > threshold).astype(int)
        if labels.sum() == 0 or labels.sum() == n:
            labels = np.random.randint(0, 2, size=n)
    
    return labels


def _two_flip_improve(adj, labels, deadline, max_rounds=50):
    """Try 2-flip moves to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    gains, same_counts, deg = _compute_gains(adj, labels)
    
    current_cut = _cut_size(adj, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    for rnd in range(max_rounds):
        if time.time() >= deadline:
            break
        
        # Try a few random 2-flip moves
        for attempt in range(5):
            if time.time() >= deadline:
                break
            
            i, j = random.sample(range(n), 2)
            if i == j:
                continue
            
            # Compute delta for flipping both i and j
            # delta_ij = gain[i] + gain[j] + correction
            # When we flip i, gain[j] changes. The combined effect:
            # If i and j are neighbors, flipping both affects each other's gain
            # delta_total = gains[i] + gains[j] + (2 if adj[i,j] and labels[i]==labels[j] else 
            #                                    -2 if adj[i,j] and labels[i]!=labels[j] else 0)
            # Actually, let me just compute it directly for correctness
            # Save state
            old_li, old_lj = labels[i], labels[j]
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            new_cut = _cut_size(adj, labels)
            delta = new_cut - current_cut
            
            if delta > 0:
                current_cut = new_cut
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                # Revert
                labels[i] = old_li
                labels[j] = old_lj
    
    return best_labels


def partition(adjacency):
    """
    Solve Maximum Cut using spectral initialization, fast greedy local search,
    simulated annealing, and 2-flip improvements.
    """
    start_time = time.time()
    DEADLINE = start_time + 115.0
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adjacency = np.array(adjacency, dtype=bool)
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral initialization + fast greedy
    spectral_labels = _spectral_init(adjacency)
    refined = _greedy_improve_fast(adjacency, spectral_labels.copy(), DEADLINE)
    cut = _cut_size(adjacency, refined)
    best_labels = refined.copy()
    best_cut = cut
    
    # Phase 2: Simulated annealing from greedy solution
    remaining = DEADLINE - time.time()
    if remaining > 10.0:
        sa_labels = _sa_with_gains(adjacency, refined.copy(), DEADLINE - 5.0, start_temp=8.0, cooling=0.9995)
        sa_cut = _cut_size(adjacency, sa_labels)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    # Phase 3: 2-flip improvement
    remaining = DEADLINE - time.time()
    if remaining > 10.0:
        tf_labels = _two_flip_improve(adjacency, best_labels.copy(), DEADLINE - 5.0)
        tf_cut = _cut_size(adjacency, tf_labels)
        if tf_cut > best_cut:
            best_cut = tf_cut
            best_labels = tf_labels.copy()
    
    # Phase 4: Multiple random restarts with fast greedy + short SA
    num_restarts = 0
    while time.time() < DEADLINE:
        num_restarts += 1
        
        # Random initial assignment
        labels = np.random.randint(0, 2, size=n)
        
        # Fast greedy refinement
        refined = _greedy_improve_fast(adjacency, labels.copy(), DEADLINE)
        cut = _cut_size(adjacency, refined)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = refined.copy()
        
        # Short SA if time permits
        remaining = DEADLINE - time.time()
        if remaining > 15.0:
            sa_labels = _sa_with_gains(adjacency, refined.copy(), DEADLINE - 5.0, 
                                       start_temp=5.0, cooling=0.999)
            sa_cut = _cut_size(adjacency, sa_labels)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
        
        # 2-flip if time permits
        remaining = DEADLINE - time.time()
        if remaining > 15.0:
            tf_labels = _two_flip_improve(adjacency, best_labels.copy(), DEADLINE - 5.0, max_rounds=20)
            tf_cut = _cut_size(adjacency, tf_labels)
            if tf_cut > best_cut:
                best_cut = tf_cut
                best_labels = tf_labels.copy()
        
        # Limit restarts
        if num_restarts >= 20:
            break
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
