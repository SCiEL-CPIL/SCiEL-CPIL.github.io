import numpy as np
import time

# ----------------------------------------------------------------------------
# Fast, faithful ACO simulator (matches the public contract's rules exactly)
# ----------------------------------------------------------------------------
def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30,
                       rho=0.9, seed=42):
    """
    Replicates the contract's ACO:
      - pheromone starts at 1, heuristic non-positive -> 1e-9 (done by caller)
      - each ant starts at node 0, picks next unvisited node with probability
        proportional to tau[i,j] * heuristic[i,j]
      - after each iteration: tau *= rho, then each ant adds 1/length to
        tau[i,j] for each directed step i->j of its closed tour.
    Returns the shortest closed tour found by any ant over all iterations.
    """
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')

    for it in range(n_iterations):
        tour_lengths = np.empty(n_ants)
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]

            for _ in range(n - 1):
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[rng.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)

                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node

            tour_len += distance_matrix[current, 0]
            tour_lengths[ant] = tour_len

            deposit = 1.0 / tour_len
            u = np.array(tour_path)
            v = np.array(tour_path[1:] + [tour_path[0]])
            tau[u, v] += deposit

        tau *= rho

        cur_best = float(np.min(tour_lengths))
        if cur_best < best_tour_len:
            best_tour_len = cur_best

    return best_tour_len


# ----------------------------------------------------------------------------
# Tour construction: NN + 2-opt (vectorized) + 3-opt + Or-opt
# ----------------------------------------------------------------------------
def _nn_tour(distance_matrix):
    n = distance_matrix.shape[0]
    unvisited = set(range(1, n))
    tour = [0]
    cur = 0
    while unvisited:
        # pick nearest unvisited
        row = distance_matrix[cur]
        # find min over unvisited
        best = None
        best_d = float('inf')
        for j in unvisited:
            d = row[j]
            if d < best_d:
                best_d = d
                best = j
        unvisited.remove(best)
        tour.append(best)
        cur = best
    return np.array(tour)


def _tour_length(tour, distance_matrix):
    n = len(tour)
    return float(np.sum(distance_matrix[np.arange(n), np.roll(tour, -1)]))


def _two_opt(tour, distance_matrix, max_passes=200):
    """Vectorized 2-opt. Returns improved tour and length."""
    n = len(tour)
    # build node-index lookup: position in tour of each node
    pos = np.empty(n, dtype=int)
    pos[tour] = np.arange(n)
    # adjacency length matrix A[i,j] = distance between tour[i] and tour[j]
    A = distance_matrix[np.ix_(tour, tour)]
    L = _tour_length(tour, distance_matrix)

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # delta for reversing segment (i+1 .. j):
        # removes A[i,i+1] + A[j,(j+1)%n], adds A[i,j] + A[i+1,(j+1)%n]
        i = np.arange(n)
        # A[i, i+1]
        a1 = np.roll(A, -1, axis=1)  # a1[i] = A[i, i+1]
        # we need A[j, j+1] as a function of j for each i -> matrix
        # delta[i,j] = -A[i,i+1] - A[j,j+1] + A[i,j] + A[i+1,j+1]
        # A[i,i+1] = a1[i]; A[j,j+1] = a1[j] (as column)
        # A[i,j] = A; A[i+1, j+1] = roll(A,-1,axis=0) rolled? A[i+1,j+1]
        A_ip1_jp1 = np.roll(np.roll(A, -1, axis=0), -1, axis=1)
        delta = -a1[:, None] - a1[None, :] + A + A_ip1_jp1
        # only consider i < j-1 and i != j-1 (to avoid trivial)
        # find best (i,j) with i in [0, n-1], j in [i+2, n-1] or j==0 handled by wrap
        # Standard 2-opt on cycle: consider i < j, j != i+1, j != 0 when i==n-1 etc.
        # Simpler: iterate candidate with j > i+1, and also allow j=0 (wrap) only if i != n-1.
        # We'll just search all i<j with j-i>=2, plus the wrap case i= n-1 -> j=0 not representable.
        # Handle wrap by also checking the edge (n-1,0) reversal via symmetry: it's same as some i<j.
        mask = np.zeros((n, n), dtype=bool)
        for i in range(n - 2):
            mask[i, i + 2:] = True
        # also allow i=0..n-3 with j up to n-1 (covered). The wrap reversal (n-1,0)
        # corresponds to reversing whole? skip; 2-opt without wrap still converges well.
        delta[~mask] = np.inf
        # exclude self and adjacent
        ii, jj = np.where(mask)
        # pick min
        flat = np.where(mask)[0]
        # find global min among masked
        idx = np.argmin(np.where(mask, delta, np.inf))
        i_best = idx // n
        j_best = idx % n
        if delta[i_best, j_best] < -1e-12:
            # reverse tour[i+1 .. j]
            tour = np.concatenate([tour[:i_best + 1], tour[i_best + 1:j_best + 1][::-1], tour[j_best + 1:]])
            L += delta[i_best, j_best]
            pos[tour] = np.arange(n)
            A = distance_matrix[np.ix_(tour, tour)]
            improved = True
    return tour, L


def _three_opt(tour, distance_matrix, passes=3):
    """Limited 3-opt: try moving a node to a better position (Or-opt style) plus 3-opt pairs.
    Keep it cheap and safe."""
    n = len(tour)
    L = _tour_length(tour, distance_matrix)
    for _ in range(passes):
        improved = False
        # Or-opt: move single node i to position between j and j+1
        for i in range(n):
            a = tour[(i - 1) % n]
            b = tour[(i + 1) % n]
            node = tour[i]
            # cost of removing i: -d(a,b) + d(a,b)?? removing node i: edges (a,i),(i,b) -> (a,b)
            remove_cost = distance_matrix[a, b] - distance_matrix[a, node] - distance_matrix[node, b]
            best_gain = -remove_cost  # gain if we insert elsewhere (we pay remove_cost then add insert)
            best_j = -1
            for j in range(n):
                if j == i or j == (i - 1) % n or j == (i + 1) % n:
                    continue
                c = tour[j]
                d = tour[(j + 1) % n]
                insert_cost = distance_matrix[c, node] + distance_matrix[node, d] - distance_matrix[c, d]
                total = remove_cost + insert_cost
                gain = -total
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j >= 0 and best_gain > 1e-12:
                # perform move
                new_tour = np.delete(tour, i)
                # insert node after best_j in new_tour (positions shift)
                # best_j index in original; in new_tour if best_j > i, its index is best_j-1
                if best_j > i:
                    ins = best_j - 1
                else:
                    ins = best_j
                new_tour = np.insert(new_tour, ins + 1, node)
                tour = new_tour
                L = _tour_length(tour, distance_matrix)
                improved = True
        if not improved:
            break
    return tour, L


def _build_tour(distance_matrix, deadline):
    """Build a good tour with a deadline guard."""
    n = distance_matrix.shape[0]
    tour = _nn_tour(distance_matrix)
    tour, L = _two_opt(tour, distance_matrix, max_passes=100)
    if time.time() < deadline:
        tour, L = _three_opt(tour, distance_matrix, passes=4)
    return tour, L


# ----------------------------------------------------------------------------
# Main heuristic builder
# ----------------------------------------------------------------------------
def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin vs 330s kill

    n = distance_matrix.shape[0]

    # Base distance heuristic
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    base_heur = 1.0 / safe_dist

    # Build a strong structural prior tour
    if time.time() > deadline - 20.0:
        tour = _nn_tour(distance_matrix)
    else:
        tour, _ = _build_tour(distance_matrix, deadline - 20.0)

    # Tour-prior matrix: boost edges in the good tour (both directions)
    tour_prior = np.ones((n, n))
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        tour_prior[a, b] = 0.0  # placeholder, set below
        tour_prior[b, a] = 0.0
    # Set to a large boost value
    BOOST = 1e6
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        tour_prior[a, b] = BOOST
        tour_prior[b, a] = BOOST
    # diagonal
    np.fill_diagonal(tour_prior, 1.0)

    def make_heuristic(beta, mix):
        # blend: h = base_heur^beta * (tour_prior)^mix  -> effectively multiply tour edges by BOOST^mix
        h = base_heur ** beta
        # apply multiplicative tour boost: edges in tour get factor = BOOST^mix
        # use log to avoid overflow: h = exp(beta*log(base) + mix*log(tour_prior))
        log_h = beta * np.log(base_heur) + mix * np.log(tour_prior)
        # clip to avoid overflow
        log_h = np.clip(log_h, -50, 50)
        h = np.exp(log_h)
        h[h <= 0] = 1e-9
        h[~np.isfinite(h)] = 1e-9
        np.fill_diagonal(h, 1.0)
        return h

    def evaluate(beta, mix, iters=8):
        if time.time() > deadline - 10.0:
            return float('inf')
        h = make_heuristic(beta, mix)
        return _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=iters, seed=42)

    best_beta = 2.0
    best_mix = 0.0
    best_score = float('inf')

    # Coarse search
    betas_coarse = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    mixes_coarse = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])

    for beta in betas_coarse:
        for mix in mixes_coarse:
            if time.time() > deadline - 60.0:
                break
            score = evaluate(beta, mix, iters=6)
            if score < best_score:
                best_score = score
                best_beta = beta
                best_mix = mix
        if time.time() > deadline - 60.0:
            break

    # Fine search
    if time.time() < deadline - 30.0:
        lb_b = max(0.3, best_beta - 1.5)
        ub_b = min(7.0, best_beta + 1.5)
        lb_m = max(0.0, best_mix - 1.0)
        ub_m = min(4.0, best_mix + 1.0)
        betas_fine = np.linspace(lb_b, ub_b, 8)
        mixes_fine = np.linspace(lb_m, ub_m, 8)
        for beta in betas_fine:
            for mix in mixes_fine:
                if time.time() > deadline - 15.0:
                    break
                score = evaluate(beta, mix, iters=8)
                if score < best_score:
                    best_score = score
                    best_beta = beta
                    best_mix = mix
            if time.time() > deadline - 15.0:
                break

    # Final heuristic
    final_heuristic = make_heuristic(best_beta, best_mix)
    return final_heuristic
