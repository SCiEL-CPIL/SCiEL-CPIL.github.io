import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def exact_check(n, adj):
        """Exact check: no K4 and no independent 6-set."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    m = common
                    while m:
                        lsb = m & -m
                        a = lsb.bit_length() - 1
                        m ^= lsb
                        rest = common & ~(1 << a)
                        if adj[a] & rest:
                            return False
        all_mask = (1 << n) - 1
        def find_indep_6(remaining, size):
            if size == 6:
                return True
            if remaining.bit_count() < size:
                return False
            m = remaining
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                non_n = remaining & ~(adj[u] | (1 << u))
                if non_n.bit_count() >= size - 1:
                    if find_indep_6(non_n, size - 1):
                        return True
            return False
        if find_indep_6(all_mask, 6):
            return False
        return True

    def spectral_construction(n, time_budget):
        """
        Spectral sign-flipping construction:
        1. Start with a random symmetric ±1 matrix (zero diagonal).
        2. Iteratively flip entries to reduce the max absolute eigenvalue.
        3. Threshold to get a graph: edge where value > 0.
        4. Verify and return if valid.
        """
        deadline_local = start_time + time_budget
        rng = np.random.RandomState(42)
        
        # Start with random ±1 matrix
        M = rng.choice([-1, 1], size=(n, n))
        # Make symmetric
        M = (M + M.T) // 2
        # Zero diagonal
        np.fill_diagonal(M, 0)
        
        best_spectral_radius = float('inf')
        best_M = M.copy()
        
        # Iterative sign-flipping to reduce spectral radius
        max_iterations = 50000
        for it in range(max_iterations):
            if time.time() > deadline_local:
                break
            
            # Compute eigenvalues
            try:
                eigvals = np.linalg.eigvalsh(M)
            except np.linalg.LinAlgError:
                break
            
            max_abs_eig = np.max(np.abs(eigvals))
            
            if max_abs_eig < best_spectral_radius:
                best_spectral_radius = max_abs_eig
                best_M = M.copy()
            
            # If spectral radius is very small, we're in a good region
            if max_abs_eig < 0.5:
                break
            
            # Try to flip a random entry to reduce spectral radius
            # Focus on off-diagonal entries
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip M[u,v] and M[v,u]
            M[u, v] = -M[u, v]
            M[v, u] = -M[v, u]
            
            # Check new spectral radius
            try:
                new_eigvals = np.linalg.eigvalsh(M)
            except np.linalg.LinAlgError:
                M[u, v] = -M[u, v]
                M[v, u] = -M[v, u]
                continue
            
            new_max_abs_eig = np.max(np.abs(new_eigvals))
            
            # Accept if it improved or with some probability (simulated annealing style)
            if new_max_abs_eig < max_abs_eig:
                max_abs_eig = new_max_abs_eig
                # Keep the flip
            else:
                # Revert
                M[u, v] = -M[u, v]
                M[v, u] = -M[v, u]
        
        # Use the best matrix found
        M = best_M
        
        # Threshold: edge where M[i,j] > 0
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if M[i, j] > 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        return adj

    def spectral_construction_v2(n, time_budget):
        """
        Improved spectral construction with:
        - Multiple restarts with different random seeds
        - Adaptive thresholding
        - Local refinement after thresholding
        """
        deadline_local = start_time + time_budget
        best_adj = None
        best_metric = float('inf')
        
        for restart in range(10):
            if time.time() > deadline_local:
                break
            
            rng = np.random.RandomState(restart * 7919 + 12345)
            
            # Start with random ±1 matrix
            M = rng.choice([-1, 1], size=(n, n))
            M = (M + M.T) // 2
            np.fill_diagonal(M, 0)
            
            # Iterative sign-flipping
            max_iterations = 30000
            current_radius = float('inf')
            
            for it in range(max_iterations):
                if time.time() > deadline_local:
                    break
                
                # Compute eigenvalues (use smaller matrix for speed)
                try:
                    eigvals = np.linalg.eigvalsh(M)
                except np.linalg.LinAlgError:
                    break
                
                max_abs_eig = np.max(np.abs(eigvals))
                
                if max_abs_eig < current_radius:
                    current_radius = max_abs_eig
                
                # If spectral radius is very small, we're in a good region
                if max_abs_eig < 0.3:
                    break
                
                # Try to flip a random entry
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue
                
                M[u, v] = -M[u, v]
                M[v, u] = -M[v, u]
                
                try:
                    new_eigvals = np.linalg.eigvalsh(M)
                    new_max_abs_eig = np.max(np.abs(new_eigvals))
                except np.linalg.LinAlgError:
                    M[u, v] = -M[u, v]
                    M[v, u] = -M[v, u]
                    continue
                
                if new_max_abs_eig < max_abs_eig:
                    current_radius = new_max_abs_eig
                else:
                    M[u, v] = -M[u, v]
                    M[v, u] = -M[v, u]
            
            # Threshold with different biases
            for bias in [0, 0.1, -0.1, 0.2, -0.2]:
                if time.time() > deadline_local:
                    break
                
                adj = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if M[i, j] + bias > 0:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                
                # Quick check
                if exact_check(n, adj):
                    return adj
                
                # Compute a metric: count of K4s + count of I6s
                k4_count = 0
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj[u] >> v) & 1:
                            common = adj[u] & adj[v]
                            common &= ~(1 << u)
                            common &= ~(1 << v)
                            m = common
                            while m:
                                lsb = m & -m
                                a = lsb.bit_length() - 1
                                m ^= lsb
                                rest = common & ~(1 << a)
                                k4_count += (adj[a] & rest).bit_count()
                k4_count //= 3
                
                if k4_count < best_metric:
                    best_metric = k4_count
                    best_adj = adj[:]
        
        return best_adj

    def local_refinement(n, adj, time_budget):
        """
        Local refinement: try to flip individual edges to reduce violations.
        """
        deadline_local = start_time + time_budget
        
        def count_violations(n, adj):
            k4_count = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        m = common
                        while m:
                            lsb = m & -m
                            a = lsb.bit_length() - 1
                            m ^= lsb
                            rest = common & ~(1 << a)
                            k4_count += (adj[a] & rest).bit_count()
            k4_count //= 3
            
            i6_count = 0
            all_mask = (1 << n) - 1
            def count_indep_sets(remaining, size):
                if size == 0:
                    return 1
                if remaining.bit_count() < size:
                    return 0
                lsb = remaining & -remaining
                u2 = lsb.bit_length() - 1
                rest = remaining ^ lsb
                count = count_indep_sets(rest, size)
                non_n2 = rest & ~adj[u2]
                count += count_indep_sets(non_n2, size - 1)
                return count
            i6_count = count_indep_sets(all_mask, 6)
            return k4_count + i6_count
        
        current_violations = count_violations(n, adj)
        best_violations = current_violations
        best_adj = adj[:]
        
        for it in range(50000):
            if time.time() > deadline_local:
                break
            
            if current_violations == 0:
                return best_adj
            
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Flip the edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
            new_violations = count_violations(n, adj)
            
            if new_violations < current_violations:
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_adj = adj[:]
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
        
        return best_adj

    def try_target_n(target_n, time_budget):
        """Try to construct a valid graph on target_n vertices."""
        deadline_local = start_time + time_budget
        
        # Strategy 1: Spectral construction v2
        if time.time() < deadline_local - 10:
            adj = spectral_construction_v2(target_n, deadline_local - time.time() - 10)
            if adj is not None and exact_check(target_n, adj):
                return adj
        
        # Strategy 2: Local refinement from random init
        if time.time() < deadline_local - 5:
            for restart in range(5):
                if time.time() > deadline_local:
                    break
                
                adj = [0] * target_n
                for u in range(target_n):
                    for v in range(u + 1, target_n):
                        if random.random() < 0.4:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                adj = local_refinement(target_n, adj, deadline_local - time.time() - 5)
                if exact_check(target_n, adj):
                    return adj
        
        return None

    best_graph = None
    best_n = 0

    # Try to construct graphs of increasing size
    for target_n in [35, 34, 33, 32, 31, 30, 28, 25, 20]:
        if time.time() > deadline - 10:
            break
        
        time_budget = min(40, deadline - time.time() - 5)
        if time_budget < 10:
            break
        
        result = try_target_n(target_n, time_budget)
        if result is not None:
            best_graph = result
            best_n = target_n
            break

    if best_graph is None:
        # Final fallback: Paley graph of order 17
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
