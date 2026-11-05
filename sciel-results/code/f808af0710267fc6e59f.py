"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Convert to numpy int matrix for fast vectorized ops
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    # Deadline guard: kill after ~110s to be safe under 120s limit
    start_time = time.time()
    DEADLINE = start_time + 110.0

    def time_left():
        return DEADLINE - time.time()

    E = int(np.sum(adj)) // 2

    def cut_size_v2(labels):
        """Compute cut size using x^T A x formulation."""
        x = 2 * labels.astype(np.float64) - 1
        xTadjx = x @ adj @ x
        return 0.5 * E - 0.25 * xTadjx

    def local_search(labels, max_rounds=100):
        """Hill-climbing local search. Returns best labels found."""
        best_labels = labels.copy()
        best_cut = cut_size_v2(best_labels)
        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False
            for v in range(n):
                if time_left() < 1.0:
                    break
                old_contrib = adj[v][labels != labels[v]].sum()
                new_contrib = adj[v][labels == labels[v]].sum()
                if new_contrib > old_contrib:
                    labels[v] = 1 - labels[v]
                    improved = True
            if not improved:
                break
        final_cut = cut_size_v2(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        return best_labels, best_cut

    def simulated_annealing(labels, max_iters=50000, T_start=10.0, T_end=0.01, alpha=0.9999):
        """Simulated annealing for maximum cut. Returns best labels found."""
        best_labels = labels.copy()
        best_cut = cut_size_v2(best_labels)
        current_cut = best_cut
        T = T_start

        # Precompute degrees for efficiency
        degrees = adj.sum(axis=1)

        for it in range(max_iters):
            if time_left() < 3.0:
                break
            # Pick random vertex
            v = np.random.randint(0, n)
            # Compute gain from flipping v
            # Current contribution to cut from edges incident to v:
            # edges from v to opposite side
            old_contrib = adj[v][labels != labels[v]].sum()
            new_contrib = adj[v][labels == labels[v]].sum()
            delta = new_contrib - old_contrib  # change in cut if we flip v

            if delta >= 0:
                # Accept if it improves or is neutral
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                # Accept with probability exp(delta / T)
                if T > 0 and np.random.random() < np.exp(delta / T):
                    labels[v] = 1 - labels[v]
                    current_cut += delta

            # Cool down
            T = max(T * alpha, T_end)

        return best_labels, best_cut

    def multi_start(num_starts, labels_init=None, use_sa=False):
        """Try multiple random starts and return the best result."""
        best_labels = None
        best_cut = -1
        for s in range(num_starts):
            if time_left() < 3.0:
                break
            if s == 0 and labels_init is not None:
                labels = labels_init.copy()
            else:
                labels = np.random.randint(0, 2, size=n)

            if use_sa:
                labels, cut = simulated_annealing(labels)
            else:
                labels, cut = local_search(labels)

            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        return best_labels, best_cut

    # Strategy:
    # 1. Start with alternating, run local search
    # 2. Try several random restarts with local search
    # 3. If time permits, try simulated annealing

    # Initial: alternating
    labels_alt = np.arange(n) % 2
    best_labels, best_cut = local_search(labels_alt, max_rounds=100)

    # Random restarts with local search
    num_restarts = 20
    labels_rs, cut_rs = multi_start(num_restarts)
    if cut_rs > best_cut:
        best_labels = labels_rs
        best_cut = cut_rs

    # If time permits, try simulated annealing
    if time_left() > 15.0:
        # Try SA from best found so far
        labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=50000, T_start=10.0, T_end=0.01)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa

        # More random restarts with SA
        if time_left() > 10.0:
            labels_sa2, cut_sa2 = multi_start(10, use_sa=True)
            if cut_sa2 > best_cut:
                best_cut = cut_sa2
                best_labels = labels_sa2

    # Final polish: run local search a few more times on the best
    if time_left() > 3.0:
        for _ in range(5):
            if time_left() < 1.0:
                break
            labels_polish, cut_polish = local_search(best_labels, max_rounds=50)
            if cut_polish > best_cut:
                best_cut = cut_polish
                best_labels = labels_polish
            else:
                break

    return best_labels.astype(int)
# EVOLVE-BLOCK-END
