# EVOLVE-BLOCK-START
"""Rewritten program for Maximum Cut using a direct structural approach."""
import numpy as np
import time
import random


def partition(adjacency):
    """
    Assigns 0/1 labels to vertices to maximize the number of edges crossing the partition.
    Uses a combination of:
      1. Spectral initialization (Laplacian eigenvector)
      2. Fast local search (greedy hill-climbing) with vectorized gains
      3. Quantum-inspired annealing for escape from local optima
      4. Simulated annealing with vectorized gains
      5. Multi-start with random restarts
      6. Final high-precision polish pass
    Deadline guard ensures return well before 150s.
    """
    try:
        adj = np.asarray(adjacency, dtype=np.int32)
        n = adj.shape[0]
        if n == 0:
            return []

        # Ensure symmetric and no self loops for safety
        adj = (adj + adj.T) // 2
        np.fill_diagonal(adj, 0)

        start_time = time.time()
        DEADLINE = start_time + 140.0  # Safety margin before 150s kill

        def time_left():
            return DEADLINE - time.time()

        # Precompute
        degrees = adj.sum(axis=1)
        E = int(degrees.sum()) // 2
        adj_float = adj.astype(np.float64)
        degrees_float = degrees.astype(np.float64)

        # Precompute neighbor lists for fast access in inner loops
        neighbors = [np.nonzero(adj[i])[0] for i in range(n)]

        # Cut size: number of edges with endpoints in different partitions
        def cut_size(labels):
            """Fast cut size: count edges crossing the cut."""
            diff = labels[:, None] != labels[None, :]
            return int(np.sum(adj * diff)) // 2

        def compute_gains(labels):
            """Vectorized computation of gain for flipping each vertex.
            gain[v] = 2 * (sum of adj[v,u] where labels[u] == labels[v]) - degrees[v]
            If gain > 0, flipping v increases the cut."""
            l0_mask = (labels == 0).astype(np.int8)
            l1_mask = (labels == 1).astype(np.int8)
            
            # For vertices with label 0: same_count = adj @ l0_mask
            # For vertices with label 1: same_count = adj @ l1_mask
            adj_l0 = adj_float @ l0_mask
            adj_l1 = adj_float @ l1_mask
            
            same_count = np.where(labels == 0, adj_l0, adj_l1)
            gains = 2.0 * same_count - degrees_float
            return gains

        def local_search(labels, max_rounds=500):
            """Greedy hill-climbing with incremental gain tracking."""
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            
            # Precompute gains for initial state
            gains = compute_gains(best_labels)
            
            for rnd in range(max_rounds):
                if time_left() < 2.0:
                    break
                improved = False
                # Try flipping each vertex once in random order
                order = np.random.permutation(n)
                for v in order:
                    if time_left() < 1.0:
                        break
                    
                    delta = int(gains[v])
                    
                    if delta > 0:
                        # Flip vertex v
                        new_label = 1 - best_labels[v]
                        best_labels[v] = new_label
                        best_cut += delta
                        
                        # Update gains incrementally
                        nbrs = neighbors[v]
                        for u in nbrs:
                            if best_labels[u] == new_label:
                                gains[u] += 2
                            else:
                                gains[u] -= 2
                        
                        # Update gain for v itself
                        gains[v] = -gains[v]
                        
                        improved = True
                
                if not improved:
                    break
            return best_labels, best_cut

        def simulated_annealing(labels, max_iters=50000, T_start=4.0, T_end=0.05, cooling=0.99997):
            """Simulated annealing to escape local optima with incremental gain tracking."""
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            current_labels = labels.copy().astype(np.int8)
            current_cut = best_cut
            T = T_start
            
            # Precompute gains for initial state
            gains = compute_gains(current_labels)

            for it in range(max_iters):
                if time_left() < 3.0:
                    break
                
                v = np.random.randint(0, n)
                delta = int(gains[v])

                if delta >= 0:
                    # Accept
                    new_label = 1 - current_labels[v]
                    current_labels[v] = new_label
                    current_cut += delta
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = current_labels.copy()
                    
                    # Update gains incrementally
                    nbrs = neighbors[v]
                    for u in nbrs:
                        if current_labels[u] == new_label:
                            gains[u] += 2
                        else:
                            gains[u] -= 2
                    gains[v] = -gains[v]
                else:
                    # Accept with probability exp(delta / T)
                    if T > 0.001:
                        prob = np.exp(delta / T)
                        if np.random.random() < prob:
                            new_label = 1 - current_labels[v]
                            current_labels[v] = new_label
                            current_cut += delta
                            
                            # Update gains incrementally
                            nbrs = neighbors[v]
                            for u in nbrs:
                                if current_labels[u] == new_label:
                                    gains[u] += 2
                                else:
                                    gains[u] -= 2
                            gains[v] = -gains[v]

                T = max(T * cooling, T_end)

            return best_labels, best_cut

        def quantum_annealing(labels, max_iters=80000, s_start=1.0, s_end=0.0, cooling=0.99998):
            """Quantum-inspired annealing with incremental gain tracking."""
            best_labels = labels.copy().astype(np.int8)
            best_cut = cut_size(best_labels)
            current_labels = labels.copy().astype(np.int8)
            current_cut = best_cut
            s = s_start  # transverse field strength (1.0 = pure quantum, 0.0 = pure classical)
            
            # Precompute gains for initial state
            gains = compute_gains(current_labels)
            
            for it in range(max_iters):
                if time_left() < 3.0:
                    break
                
                # Compute gains for the selected vertex
                v = np.random.randint(0, n)
                delta = int(gains[v])

                # Quantum tunneling: effective temperature scales with transverse field
                T_eff = s * 3.0 + 0.05  # minimum temperature to allow some exploration
                
                if delta >= 0:
                    # Accept
                    new_label = 1 - current_labels[v]
                    current_labels[v] = new_label
                    current_cut += delta
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = current_labels.copy()
                    
                    # Update gains incrementally
                    nbrs = neighbors[v]
                    for u in nbrs:
                        if current_labels[u] == new_label:
                            gains[u] += 2
                        else:
                            gains[u] -= 2
                    gains[v] = -gains[v]
                else:
                    # Accept with probability exp(delta / T_eff)
                    if T_eff > 0.01:
                        prob = np.exp(delta / T_eff)
                        if np.random.random() < prob:
                            new_label = 1 - current_labels[v]
                            current_labels[v] = new_label
                            current_cut += delta
                            
                            # Update gains incrementally
                            nbrs = neighbors[v]
                            for u in nbrs:
                                if current_labels[u] == new_label:
                                    gains[u] += 2
                                else:
                                    gains[u] -= 2
                            gains[v] = -gains[v]

                # Update transverse field
                s = max(s * cooling, s_end)

            return best_labels, best_cut

        def spectral_init():
            """Use the Laplacian eigenvector to initialize the partition."""
            try:
                D = np.diag(degrees)
                L = D - adj_float
                from scipy.linalg import eigh
                evals, evecs = eigh(L, subset_by_index=[0, 1])
                v = evecs[:, 1]
                labels = (v > 0).astype(int)
                if labels.sum() == 0 or labels.sum() == n:
                    labels = np.random.randint(0, 2, size=n)
                return labels
            except Exception:
                return np.random.randint(0, 2, size=n)

        def goemans_williamson_relaxation(max_power_iters=500, num_hyperplanes=200):
            """
            Goemans-Williamson style continuous relaxation:
            1. Initialize vertex vectors via spectral method
            2. Refine via power iteration on adjacency (minimize trace term)
            3. Random hyperplane rounding with multiple trials
            Returns best labels found.
            """
            if n < 2:
                return np.random.randint(0, 2, size=n)

            # Step 1: Initialize vertex vectors using spectral decomposition
            try:
                D = np.diag(degrees)
                L = D - adj_float
                from scipy.linalg import eigh
                evals, evecs = eigh(L, subset_by_index=[0, min(2, n-1)])
                # Use the Fiedler vector (second eigenvector) as initial direction
                v_init = evecs[:, 1]
            except Exception:
                v_init = np.random.randn(n)

            # Create initial vertex vectors in R^d (use 2D for simplicity, or embed in higher dim)
            # Each vertex gets a unit vector; we'll use the spectral vector as one component
            # and add random components for diversity
            d = 8  # embedding dimension
            V = np.zeros((n, d))
            V[:, 0] = v_init
            if d > 1:
                V[:, 1:] = np.random.randn(n, d-1)
            # Normalize to unit vectors
            norms = np.linalg.norm(V, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            V = V / norms

            # Step 2: Power iteration to refine vectors
            # We want to minimize Tr(V^T L V) = sum_{i,j} L_{ij} V_i^T V_j
            # Equivalently maximize Tr(V^T A V) = sum_{i,j} A_{ij} V_i^T V_j
            # Gradient ascent: V_new = A @ V, then project to sphere
            for it in range(max_power_iters):
                if time_left() < 5.0:
                    break
                # V_new[i] = sum_j A_{ij} V[j]
                V_new = adj_float @ V
                # Project each row to unit sphere
                norms = np.linalg.norm(V_new, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                V = V_new / norms

            # Step 3: Random hyperplane rounding with multiple trials
            best_labels_gw = None
            best_cut_gw = -1

            for trial in range(num_hyperplanes):
                if time_left() < 3.0:
                    break
                # Random hyperplane normal
                r = np.random.randn(d)
                # Assign label based on sign of V_i^T r
                projections = V @ r
                labels = (projections > 0).astype(int)

                # Ensure not all same label
                if labels.sum() == 0 or labels.sum() == n:
                    labels = np.random.randint(0, 2, size=n)

                cut = cut_size(labels)
                if cut > best_cut_gw:
                    best_cut_gw = cut
                    best_labels_gw = labels.copy()

            if best_labels_gw is None:
                best_labels_gw = np.random.randint(0, 2, size=n)

            return best_labels_gw, best_cut_gw

        # Strategy:
        # 1. Spectral initialization + local search
        # 2. Goemans-Williamson continuous relaxation + hyperplane rounding
        # 3. Multiple random restarts with local search
        # 4. Quantum-inspired annealing from best solutions
        # 5. Simulated annealing from best solutions
        # 6. Final high-precision polish

        best_labels = None
        best_cut = -1

        # Phase 1: Spectral init + local search
        labels_spec = spectral_init()
        labels_ls, cut_ls = local_search(labels_spec, max_rounds=200)
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls.copy()

        # Phase 1.5: Goemans-Williamson continuous relaxation
        if time_left() > 10.0:
            labels_gw, cut_gw = goemans_williamson_relaxation(max_power_iters=300, num_hyperplanes=100)
            if cut_gw > best_cut:
                best_cut = cut_gw
                best_labels = labels_gw.copy()
            # Polish the GW result with local search
            if time_left() > 5.0:
                labels_polish_gw, cut_polish_gw = local_search(labels_gw, max_rounds=150)
                if cut_polish_gw > best_cut:
                    best_cut = cut_polish_gw
                    best_labels = labels_polish_gw.copy()

        # Phase 2: Random restarts with local search
        num_restarts = 25
        for r in range(num_restarts):
            if time_left() < 5.0:
                break
            labels_rand = np.random.randint(0, 2, size=n)
            labels_rs, cut_rs = local_search(labels_rand, max_rounds=100)
            if cut_rs > best_cut:
                best_cut = cut_rs
                best_labels = labels_rs.copy()

        # Phase 3: Quantum-inspired annealing from best
        if time_left() > 15.0 and best_labels is not None:
            labels_qa, cut_qa = quantum_annealing(best_labels, max_iters=100000)
            if cut_qa > best_cut:
                best_cut = cut_qa
                best_labels = labels_qa.copy()
            # Polish after quantum annealing
            if time_left() > 5.0:
                labels_polish, cut_polish = local_search(best_labels, max_rounds=100)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()

        # Phase 4: Simulated annealing from best
        if time_left() > 15.0 and best_labels is not None:
            labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=80000)
            if cut_sa > best_cut:
                best_cut = cut_sa
                best_labels = labels_sa.copy()
            # Polish after simulated annealing
            if time_left() > 5.0:
                labels_polish, cut_polish = local_search(best_labels, max_rounds=100)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()

        # Phase 5: More random restarts with quantum annealing if time permits
        if time_left() > 25.0:
            for r in range(5):
                if time_left() < 5.0:
                    break
                labels_rand = np.random.randint(0, 2, size=n)
                labels_qa2, cut_qa2 = quantum_annealing(labels_rand, max_iters=50000)
                if cut_qa2 > best_cut:
                    best_cut = cut_qa2
                    best_labels = labels_qa2.copy()
                # Polish
                if time_left() > 3.0:
                    labels_polish, cut_polish = local_search(best_labels, max_rounds=50)
                    if cut_polish > best_cut:
                        best_cut = cut_polish
                        best_labels = labels_polish.copy()

        # Phase 6: Final high-precision polish with more iterations
        if time_left() > 3.0 and best_labels is not None:
            for _ in range(20):
                if time_left() < 1.5:
                    break
                labels_polish, cut_polish = local_search(best_labels, max_rounds=200)
                if cut_polish > best_cut:
                    best_cut = cut_polish
                    best_labels = labels_polish.copy()
                else:
                    break

        # Phase 7: Additional GW restarts if time permits
        if time_left() > 15.0:
            for _ in range(3):
                if time_left() < 5.0:
                    break
                labels_gw2, cut_gw2 = goemans_williamson_relaxation(max_power_iters=200, num_hyperplanes=80)
                if cut_gw2 > best_cut:
                    best_cut = cut_gw2
                    best_labels = labels_gw2.copy()
                # Polish
                if time_left() > 3.0:
                    labels_polish2, cut_polish2 = local_search(labels_gw2, max_rounds=100)
                    if cut_polish2 > best_cut:
                        best_cut = cut_polish2
                        best_labels = labels_polish2.copy()

        if best_labels is None:
            best_labels = np.random.randint(0, 2, size=n)

        return best_labels.tolist()

    except Exception as e:
        # Fallback to random partition if anything fails
        try:
            n = np.asarray(adjacency).shape[0]
            return np.random.randint(0, 2, size=n).tolist()
        except:
            return []
# EVOLVE-BLOCK-END
