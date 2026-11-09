"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute cut size: number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adj & diff).sum() // 2)


def _all_flip_gains(adj, labels):
    """Compute flip gains for all vertices at once (vectorized)."""
    same_mask = (labels[:, None] == labels[None, :])
    opp_mask = ~same_mask
    same_counts = (adj & same_mask).sum(axis=1)
    opp_counts = (adj & opp_mask).sum(axis=1)
    gains = same_counts - opp_counts
    return gains.astype(np.int32)


def _local_search(adj, labels, deadline, max_rounds=200):
    """Iterative local search: flip all vertices with positive gain per round."""
    n = len(labels)
    labels = labels.copy().astype(np.int8)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        gains = _all_flip_gains(adj, labels)
        positive = np.where(gains > 0)[0]
        if len(positive) == 0:
            break
        labels[positive] = 1 - labels[positive]
    
    return labels


def _spectral_partition(adj, n):
    """Use Fiedler vector of Laplacian for initial partition."""
    A = adj.astype(np.float64)
    D = np.diag(A.sum(axis=1))
    L = D - A
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        labels = (fiedler >= 0).astype(np.int8)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n).astype(np.int8)


def _greedy_partition(adj, n, deadline):
    """Greedy construction: add vertices by descending degree, assign to maximize cut."""
    labels = np.zeros(n, dtype=np.int8)
    degrees = adj.sum(axis=1)
    order = np.argsort(-degrees)
    
    for v in order:
        if time.time() > deadline:
            break
        cut_if_0 = adj[v][labels == 1].sum()
        cut_if_1 = adj[v][labels == 0].sum()
        labels[v] = 0 if cut_if_0 >= cut_if_1 else 1
    
    return labels


def _simulated_annealing_2flip(adj, labels, deadline, rng, 
                                  start_temp=50.0, end_temp=0.1, 
                                  cooling=0.999, adapt_rate=0.05):
    """Simulated annealing with 2-flip moves and adaptive cooling."""
    n = len(labels)
    labels = labels.copy().astype(np.int8)
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Precompute adjacency as int
    adj_int = adj.astype(np.int8)
    
    # Initialize gain tracking: gain[v] = same_count[v] - opp_count[v]
    gains = _all_flip_gains(adj, labels).astype(np.int32)
    
    temp = start_temp
    accept_count = 0
    total_count = 0
    window = 1000  # Window for adaptive cooling
    
    while time.time() < deadline and temp > end_temp:
        total_count += 1
        
        # Choose 2-flip move: pick two random vertices
        # But sometimes do single flip for exploration
        if rng.random() < 0.3:
            # Single flip
            v = int(rng.integers(0, n))
            gain = int(gains[v])
            
            if gain >= 0 or (rng.random() < np.exp(-gain / temp)):
                # Accept
                accept_count += 1
                # Flip v
                labels[v] = 1 - labels[v]
                current_cut += gain
                
                # Update gains for all neighbors of v
                neighbors = np.where(adj_int[v] == 1)[0]
                for nb in neighbors:
                    # When v flips, for neighbor nb:
                    # If nb and v were same, now they're opp: nb's same decreases by 1, opp increases by 1
                    # So nb's gain changes by -2
                    # If nb and v were opp, now they're same: nb's same increases by 1, opp decreases by 1
                    # So nb's gain changes by +2
                    if labels[nb] == labels[v]:
                        # They are now same (v just flipped to match nb or vice versa)
                        # Before flip: they were opposite
                        # After flip: they are same
                        # nb's same_count increases by 1, opp_count decreases by 1
                        # gain change = +1 - (-1) = +2
                        gains[nb] += 2
                    else:
                        # Before flip: they were same
                        # After flip: they are opposite
                        # nb's same_count decreases by 1, opp_count increases by 1
                        # gain change = -1 - (+1) = -2
                        gains[nb] -= 2
                
                # v's own gain: after flip, v's role reverses
                # gain[v] was same_v - opp_v. After flip, v's same becomes old opp, v's opp becomes old same
                # But we also need to account for the fact that we already updated neighbors
                # Actually, v's gain after flip = old_opp_v - old_same_v = -old_gain_v
                # But wait, the neighbors' updates already accounted for the edge (v, nb)
                # Let me reconsider: gains[v] should be recomputed or updated
                # The gain for v after flip: v is now in the other partition
                # For each neighbor nb:
                #   If nb is in same partition as new v: contributes +1 to v's same
                #   If nb is in opposite partition as new v: contributes +1 to v's opp
                # This is equivalent to: new_gain[v] = -old_gain[v] + 2*(number of neighbors that changed relationship)
                # Actually simpler: just recompute v's gain from scratch using the updated labels
                # But that's O(deg(v)). Let's do it properly.
                # After flipping v, for each neighbor nb:
                #   If labels[nb] == labels[v] (new): v's same_count includes nb
                #   If labels[nb] != labels[v] (new): v's opp_count includes nb
                # We can compute: new_same_v = sum(adj_int[v] & (labels == labels[v]))
                # But that's O(n). Instead, use the fact that:
                # old_same_v + old_opp_v = deg(v)
                # new_same_v = old_opp_v (since v flipped, what was opp is now same, except for neighbors we updated)
                # Wait, we updated neighbors' gains but not v's. Let me just recompute v's gain.
                new_same_v = adj_int[v][labels == labels[v]].sum()
                new_opp_v = adj_int[v][labels != labels[v]].sum()
                gains[v] = int(new_same_v - new_opp_v)
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        else:
            # 2-flip move
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            if u == v:
                continue
            
            gain = int(gains[u] + gains[v])
            
            if gain >= 0 or (rng.random() < np.exp(-gain / temp)):
                # Accept
                accept_count += 1
                # Flip both u and v
                labels[u] = 1 - labels[u]
                labels[v] = 1 - labels[v]
                current_cut += gain
                
                # Update gains for neighbors of u and v
                neighbors_u = np.where(adj_int[u] == 1)[0]
                neighbors_v = np.where(adj_int[v] == 1)[0]
                
                # Build set of affected vertices
                affected = set(neighbors_u.tolist()) | set(neighbors_v.tolist())
                affected.discard(u)
                affected.discard(v)
                
                for nb in affected:
                    # Determine how nb's gain changes
                    # For each of u and v, if nb is a neighbor:
                    #   Check if nb's relationship with u/v changed
                    change = 0
                    if adj_int[u, nb] == 1:
                        # u and nb are connected
                        # If labels[nb] == labels[u] (new), they are same: nb's gain += 2
                        # If labels[nb] != labels[u] (new), they are opp: nb's gain -= 2
                        if labels[nb] == labels[u]:
                            change += 2
                        else:
                            change -= 2
                    if adj_int[v, nb] == 1:
                        # v and nb are connected
                        if labels[nb] == labels[v]:
                            change += 2
                        else:
                            change -= 2
                    gains[nb] += change
                
                # Recompute gains for u and v
                new_same_u = adj_int[u][labels == labels[u]].sum()
                new_opp_u = adj_int[u][labels != labels[u]].sum()
                gains[u] = int(new_same_u - new_opp_u)
                
                new_same_v = adj_int[v][labels == labels[v]].sum()
                new_opp_v = adj_int[v][labels != labels[v]].sum()
                gains[v] = int(new_same_v - new_opp_v)
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
        
        # Adaptive cooling
        if total_count % window == 0:
            accept_rate = accept_count / total_count
            if accept_rate > 0.7:
                temp *= (1 - adapt_rate)  # Cool faster
            elif accept_rate < 0.2:
                temp *= (1 + adapt_rate)  # Heat up slightly
            accept_count = 0
            total_count = 0
        
        temp *= cooling
    
    return best_labels, best_cut


def _tabu_search(adj, labels, deadline, rng, tabu_length=12, max_iters=8000):
    """Tabu search to escape local optima."""
    n = len(labels)
    labels = labels.copy().astype(np.int8)
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    tabu_until = np.full(n, -1, dtype=np.int32)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        gains = _all_flip_gains(adj, labels)
        
        # Find best non-tabu move
        non_tabu = it >= tabu_until
        masked_gains = gains.copy()
        masked_gains[~non_tabu] = -999999
        
        best_v = int(np.argmax(masked_gains))
        best_gain = gains[best_v]
        
        if best_gain <= 0 and rng.random() < 0.4:
            # Diversify: random move
            best_v = int(rng.integers(0, n))
            best_gain = gains[best_v]
        
        labels[best_v] = 1 - labels[best_v]
        tabu_until[best_v] = it + tabu_length
        current_cut += best_gain
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adj = np.asarray(adjacency, dtype=bool)
    rng = np.random.default_rng(42)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    # Spectral
    initializations.append(_spectral_partition(adj, n))
    
    # Greedy
    initializations.append(_greedy_partition(adj, n, deadline))
    
    # Alternating
    initializations.append((np.arange(n) % 2).astype(np.int8))
    
    # Degree-based
    degrees = adj.sum(axis=1)
    initializations.append((degrees > degrees.mean()).astype(np.int8))
    
    # Random initializations
    for _ in range(6):
        initializations.append(rng.integers(0, 2, n).astype(np.int8))
    
    # Phase 2: Local search from each initialization
    for init_labels in initializations:
        if time.time() > deadline - 25.0:
            break
        labels = _local_search(adj, init_labels, deadline - 25.0)
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = (np.arange(n) % 2).astype(np.int8)
        best_cut = _cut_size(adj, best_labels)
    
    # Phase 3: SA with 2-flip moves from best local search result
    if time.time() < deadline - 20.0:
        sa_deadline = deadline - 20.0
        sa_labels, sa_cut = _simulated_annealing_2flip(adj, best_labels, sa_deadline, rng)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
        
        # Polish after SA
        polished = _local_search(adj, best_labels, deadline - 20.0)
        pol_cut = _cut_size(adj, polished)
        if pol_cut > best_cut:
            best_cut = pol_cut
            best_labels = polished
    
    # Phase 4: Tabu search from best result
    if time.time() < deadline - 10.0:
        tabu_deadline = deadline - 10.0
        tabu_labels, tabu_cut = _tabu_search(adj, best_labels, tabu_deadline, rng)
        if tabu_cut > best_cut:
            best_cut = tabu_cut
            best_labels = tabu_labels.copy()
        
        # Polish after tabu
        polished = _local_search(adj, best_labels, deadline - 10.0)
        pol_cut = _cut_size(adj, polished)
        if pol_cut > best_cut:
            best_cut = pol_cut
            best_labels = polished
    
    # Phase 5: Additional SA restarts from perturbed best if time remains
    if time.time() < deadline - 5.0:
        remaining = deadline - time.time()
        if remaining > 8.0:
            for extra in range(3):
                if time.time() > deadline - 5.0:
                    break
                perturbed = best_labels.copy().astype(np.int8)
                num_perturb = max(1, n // 10)
                perturbation_indices = rng.choice(n, size=num_perturb, replace=False)
                perturbed[perturbation_indices] = 1 - perturbed[perturbation_indices]
                
                sa_deadline = deadline - 5.0
                sa_labels, sa_cut = _simulated_annealing_2flip(adj, perturbed, sa_deadline, rng)
                if sa_cut > best_cut:
                    best_cut = sa_cut
                    best_labels = sa_labels.copy()
                
                # Polish
                polished = _local_search(adj, best_labels, deadline - 5.0)
                pol_cut = _cut_size(adj, polished)
                if pol_cut > best_cut:
                    best_cut = pol_cut
                    best_labels = polished
    
    # Final polish
    if time.time() < deadline - 1.0:
        final_labels = _local_search(adj, best_labels, deadline - 1.0)
        final_cut = _cut_size(adj, final_labels)
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    return best_labels
# EVOLVE-BLOCK-END
