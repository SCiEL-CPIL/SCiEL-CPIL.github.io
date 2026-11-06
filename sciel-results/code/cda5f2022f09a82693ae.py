import random
import time
import numpy as np

def construct():
    n = 20
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)

    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c

    def get_neighbors(mask, n):
        result = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            result.append(idx)
            temp ^= lsb
        return result

    def is_k33_free_add(u, v, M):
        A_mask = M[v] & ~(1 << u)
        B_mask = M[u] & ~(1 << v)
        a_count = popcount(A_mask)
        b_count = popcount(B_mask)
        if a_count < 2 or b_count < 2:
            return True
        # Iterate over pairs from the smaller set
        if a_count <= b_count:
            a_list = get_neighbors(A_mask, n)
            for i in range(len(a_list)):
                for j in range(i + 1, len(a_list)):
                    common = M[a_list[i]] & M[a_list[j]] & B_mask
                    if common & (common - 1):
                        return False
        else:
            b_list = get_neighbors(B_mask, n)
            for i in range(len(b_list)):
                for j in range(i + 1, len(b_list)):
                    common = M[b_list[i]] & M[b_list[j]] & A_mask
                    if common & (common - 1):
                        return False
        return True

    def count_edges(M):
        return sum(popcount(M[i]) for i in range(n)) // 2

    def spectral_init():
        """Initialize using spectral method: create dense random-like graph."""
        # Generate random symmetric matrix
        A = np.random.rand(n, n)
        A = (A + A.T) / 2.0
        np.fill_diagonal(A, 0)

        # Get top eigenvector
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        top_vec = eigenvectors[:, -1]

        # Threshold at mean to create dense graph
        threshold = np.mean(top_vec)
        M = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if top_vec[i] * top_vec[j] > threshold:
                    if is_k33_free_add(i, j, M):
                        M[i] |= (1 << j)
                        M[j] |= (1 << i)
        return M

    def greedy_fill(M):
        """Try to add edges in random order."""
        shuffled = edges[:]
        random.shuffle(shuffled)
        for u, v in shuffled:
            if not (M[u] & (1 << v)):
                if is_k33_free_add(u, v, M):
                    M[u] |= (1 << v)
                    M[v] |= (1 << u)
        return M

    def edge_swap_search(M, deadline, max_swaps_per_round=200):
        """Local search using edge swaps: remove (u,v), add (a,b) where (a,b) != (u,v)."""
        current = M[:]
        best_count = count_edges(current)
        no_improve_count = 0

        while time.time() < deadline:
            improved = False
            # Get current edge list
            edge_list = []
            for i in range(n):
                temp = current[i]
                while temp:
                    lsb = temp & -temp
                    j = lsb.bit_length() - 1
                    if i < j:
                        edge_list.append((i, j))
                    temp ^= lsb

            # Get non-edge list
            non_edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                         if not (current[i] & (1 << j))]

            random.shuffle(edge_list)
            random.shuffle(non_edges)

            swaps_tried = 0
            for u, v in edge_list:
                if time.time() >= deadline or swaps_tried >= max_swaps_per_round:
                    break
                for a, b in non_edges:
                    if swaps_tried >= max_swaps_per_round:
                        break
                    if (a, b) == (u, v) or (a, b) == (v, u):
                        continue
                    # Skip if a or b is u or v (would be same as just adding)
                    # Actually we can allow any swap
                    trial = current[:]
                    trial[u] &= ~(1 << v)
                    trial[v] &= ~(1 << u)
                    if is_k33_free_add(a, b, trial):
                        trial[a] |= (1 << b)
                        trial[b] |= (1 << a)
                        new_count = count_edges(trial)
                        if new_count > best_count:
                            current = trial
                            best_count = new_count
                            improved = True
                            swaps_tried += 1
                            # After improvement, restart shuffle
                            break
                    swaps_tried += 1

            if not improved:
                no_improve_count += 1
                if no_improve_count >= 5:
                    break
            else:
                no_improve_count = 0

        return current, best_count

    def local_search(M, deadline):
        """Local search: edge swaps + occasional greedy fill after removal."""
        current = M[:]
        best_count = count_edges(current)
        no_improve_rounds = 0

        while time.time() < deadline:
            improved = False

            # Phase 1: Edge swaps
            current, best_count = edge_swap_search(current, min(time.time() + 2.0, deadline))
            if best_count > count_edges(M):
                improved = True

            # Phase 2: Try removing one edge and doing a quick greedy fill
            edge_list = []
            for i in range(n):
                temp = current[i]
                while temp:
                    lsb = temp & -temp
                    j = lsb.bit_length() - 1
                    if i < j:
                        edge_list.append((i, j))
                    temp ^= lsb

            random.shuffle(edge_list)
            for u, v in edge_list[:30]:  # Limit to 30 removals per round
                if time.time() >= deadline:
                    break
                trial = current[:]
                trial[u] &= ~(1 << v)
                trial[v] &= ~(1 << u)
                trial = greedy_fill(trial)
                new_count = count_edges(trial)
                if new_count > best_count:
                    current = trial
                    best_count = new_count
                    improved = True

            if not improved:
                no_improve_rounds += 1
                if no_improve_rounds >= 3:
                    break
            else:
                no_improve_rounds = 0

        return current, best_count

    best_M = [0] * n
    best_count = 0

    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin from 150s limit

    # Multiple restarts with different strategies
    while time.time() < deadline:
        # Strategy 1: Spectral init
        M = spectral_init()
        count = count_edges(M)
        if count > best_count:
            best_count = count
            best_M = M[:]

        M = greedy_fill(M)
        count = count_edges(M)
        if count > best_count:
            best_count = count
            best_M = M[:]

        # Strategy 2: Pure random greedy
        M2 = [0] * n
        shuffled = edges[:]
        random.shuffle(shuffled)
        for u, v in shuffled:
            if is_k33_free_add(u, v, M2):
                M2[u] |= (1 << v)
                M2[v] |= (1 << u)
        count2 = count_edges(M2)
        if count2 > best_count:
            best_count = count2
            best_M = M2[:]

        # Use the better of the two for local search
        M_ls = M if count >= count2 else M2

        # Local search with edge swaps
        if time.time() < deadline - 5.0:
            M_ls, count_ls = local_search(M_ls, deadline - 3.0)
            if count_ls > best_count:
                best_count = count_ls
                best_M = M_ls[:]

    # Convert best_M to list of edges
    result = []
    for i in range(n):
        temp = best_M[i]
        while temp:
            lsb = temp & -temp
            j = lsb.bit_length() - 1
            if i < j:
                result.append([i, j])
            temp ^= lsb
    return result
