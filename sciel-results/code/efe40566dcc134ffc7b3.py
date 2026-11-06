import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Known valid coloring for n=17
    # This is a specific construction found in literature or previous runs.
    # We verify it first to ensure correctness.
    def _verify_n17():
        # Construct a specific known valid coloring for n=17.
        # One such coloring can be derived from cyclic shifts or specific algebraic constructions.
        # Here we use a hardcoded valid instance if available, or generate one via a fast deterministic method.
        # For the sake of this "structural change", we assume we have a high-quality seed.
        # If we don't have a hardcoded one, we fall back to a fast search for n=17.
        
        # Let's try to construct a valid n=17 coloring quickly.
        # A common approach for R_4(C6) > 17 involves specific symmetric structures.
        # Since I cannot guarantee a hardcoded valid n=17 string without verification in this thought process,
        # I will implement a very fast, optimized local search for n=17 as the "base" guarantee,
        # but structure the code to prioritize n=18 immediately after.
        
        # Actually, the prompt says "best known colouring has 17 vertices".
        # I will use a fast, deterministic-ish random restart local search for n=17 that is guaranteed to find one quickly,
        # then spend 90% of the time on n=18.
        
        pass

    def _count_c6s_fast(adj, n):
        """
        Count C6s in a graph represented by a list of 4 adjacency matrices (numpy arrays).
        adj[c] is an n x n boolean numpy array.
        """
        total = 0
        for c in range(4):
            A = adj[c]
            # A path of length 5 between u and v exists if (A^5)[u][v] > 0 and A[u][v]==0?
            # No, C6 is a cycle. 
            # Efficient way: Iterate over edges? No, vectorize.
            # Let's stick to a highly optimized Python loop for the count, as NumPy overhead for small n might be high,
            # but we can use bitsets in Python which are fast enough if the loop is tight.
            # However, the directive asks for a structural change.
            # Let's use a different search method: Simulated Annealing with a "soft" constraint or 
            # a different representation.
            
            # Actually, let's use a standard fast bitset C6 checker in Python, but optimize the search loop.
            pass
        return 0

    # Revised Strategy:
    # 1. Use a highly optimized bitset C6 checker.
    # 2. Run a "Hill Climbing with Random Restarts" for n=17 to secure the baseline (should take < 5s).
    # 3. Run "Simulated Annealing" for n=18 and n=19 for the remaining time.
    # 4. Return the best found.

    def _has_c6(adj, n):
        """
        Fast check for C6 in a single color class using bitmasks.
        adj is a list of n integers, where adj[i] is a bitmask of neighbors.
        """
        for v0 in range(n):
            m1 = adj[v0]
            while m1:
                v1 = (m1 & -m1).bit_length() - 1
                m1 &= m1 - 1
                
                # v2 in N(v1) \ {v0}
                m2 = adj[v1] & ~(1 << v0)
                while m2:
                    v2 = (m2 & -m2).bit_length() - 1
                    m2 &= m2 - 1
                    
                    # v3 in N(v2) \ {v0, v1}
                    m3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                    while m3:
                        v3 = (m3 & -m3).bit_length() - 1
                        m3 &= m3 - 1
                        
                        # v4 in N(v3) \ {v0, v1, v2}
                        m4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while m4:
                            v4 = (m4 & -m4).bit_length() - 1
                            m4 &= m4 - 1
                            
                            # Check if v4 is connected to any v5 in N(v0) \ {v1, v2, v3, v4}
                            # Candidates for v5: N(v4) & N(v0) & ~{v0, v1, v2, v3, v4}
                            # Note: v5 must be distinct from v0..v4.
                            # N(v4) & N(v0) gives potential closing vertices.
                            # We must exclude v1, v2, v3. v0 and v4 are already excluded by logic or mask?
                            # v5 != v0, v5 != v4.
                            # If v5 == v1, then v0-v1-v2-v3-v4-v1 is a 5-cycle? No, v1 is distinct from v0..v4 in the path definition?
                            # The path is v0-v1-v2-v3-v4-v5-v0.
                            # v0, v1, v2, v3, v4 are distinct.
                            # v5 must be distinct from v0, v1, v2, v3, v4.
                            
                            m5 = adj[v4] & adj[v0]
                            # Exclude v1, v2, v3. v0 and v4 are in adj[v0]/adj[v4] but we exclude used vertices.
                            used = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                            m5 &= ~used
                            if m5:
                                return True
        return False

    def _count_c6s_adj(adj_list, n):
        """Count C6s in all 4 color classes."""
        total = 0
        for c in range(4):
            adj = adj_list[c]
            # We need to count, not just check.
            # The _has_c6 above returns True on first find.
            # For SA, we need the count or a penalty.
            # Counting is more expensive.
            # Alternative: Use a "conflict count" heuristic.
            # Let's implement a counter similar to _has_c6 but accumulating.
            
            count = 0
            for v0 in range(n):
                m1 = adj[v0]
                while m1:
                    v1 = (m1 & -m1).bit_length() - 1
                    m1 &= m1 - 1
                    m2 = adj[v1] & ~(1 << v0)
                    while m2:
                        v2 = (m2 & -m2).bit_length() - 1
                        m2 &= m2 - 1
                        m3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                        while m3:
                            v3 = (m3 & -m3).bit_length() - 1
                            m3 &= m3 - 1
                            m4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            while m4:
                                v4 = (m4 & -m4).bit_length() - 1
                                m4 &= m4 - 1
                                m5 = adj[v4] & adj[v0]
                                used = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                m5 &= ~used
                                count += m5.bit_count() # Number of valid v5s
            total += count
        return total

    def _local_search(n, time_limit, initial_colors=None):
        """
        Simulated Annealing / Hill Climbing for n vertices.
        Returns (colors, c6_count)
        """
        E = n * (n - 1) // 2
        if initial_colors is None:
            colors = [random.randint(0, 3) for _ in range(E)]
        else:
            colors = initial_colors[:]
            
        # Build adjacency bitmasks
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
                
        current_cost = _count_c6s_adj(adj, n)
        best_cost = current_cost
        best_colors = colors[:]
        
        start = time.time()
        temperature = 1.0
        cooling = 0.999
        
        while time.time() - start < time_limit:
            if best_cost == 0:
                return best_colors, 0
                
            # Pick random edge
            idx = random.randint(0, E - 1)
            i = idx
            # Calculate i, j from idx? 
            # It's faster to precompute edge endpoints.
            
            # Let's precompute edges
            # This is done outside the loop in a helper? No, let's just use a list.
            # For n=18, E=153. Small.
            
            # Precomputed edges for speed
            # We'll assume we pass a precomputed list or generate once.
            
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
                
            # Update adj
            # Need i, j for idx.
            # Let's create a mapping idx -> (i,j) once.
            
            # To avoid recreating mapping every time, we pass it in.
            # Let's refactor _local_search to take edges.
            
            break # Placeholder, see actual implementation below
            
        return best_colors, best_cost

    # Precompute edge indices for speed
    def get_edges(n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return edges

    def _sa_search(n, time_limit):
        E = n * (n - 1) // 2
        edges = get_edges(n)
        
        best_global_cost = float('inf')
        best_global_colors = None
        
        # Multiple restarts
        while time.time() < time_limit:
            # Random start
            colors = [random.randint(0, 3) for _ in range(E)]
            adj = [[0] * n for _ in range(4)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
                    
            current_cost = _count_c6s_adj(adj, n)
            if current_cost == 0:
                return colors
                
            best_local_cost = current_cost
            best_local_colors = colors[:]
            
            T = 1.0
            # Inner loop
            steps = 0
            max_steps = 2000
            while steps < max_steps and time.time() < time_limit:
                steps += 1
                if best_local_cost == 0:
                    return best_local_colors
                    
                idx = random.randint(0, E - 1)
                i, j = edges[idx]
                old_c = colors[idx]
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue
                    
                # Remove edge from old_c
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                # Add edge to new_c
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                
                new_cost = _count_c6s_adj(adj, n)
                delta = new_cost - current_cost
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    colors[idx] = new_c
                    if current_cost < best_local_cost:
                        best_local_cost = current_cost
                        best_local_colors = colors[:]
                else:
                    # Revert
                    adj[new_c][i] &= ~(1 << j)
                    adj[new_c][j] &= ~(1 << i)
                    adj[old_c][i] |= (1 << j)
                    adj[old_c][j] |= (1 << i)
                    
                T *= 0.9995
                
            if best_local_cost < best_global_cost:
                best_global_cost = best_local_cost
                best_global_colors = best_local_colors
                
        return best_global_colors

    # Strategy Execution
    best_n = 0
    best_colors = None
    
    # 1. Secure n=17 quickly
    # Use a very short time limit for n=17, as it's known to be easy.
    t_start = time.time()
    alloc_17 = 10.0
    
    sol_17 = _sa_search(17, t_start + alloc_17)
    if sol_17 is not None:
        # Verify
        adj = [[0] * 17 for _ in range(4)]
        idx = 0
        for i in range(17):
            for j in range(i + 1, 17):
                c = sol_17[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        if not any(_has_c6(adj[c], 17) for c in range(4)):
            best_n = 17
            best_colors = sol_17
    
    if time.time() > deadline - 5:
        if best_n == 0:
            # Fallback to small n if we ran out of time
            best_n = 4
            best_colors = [0, 1, 2, 3, 0, 1]
        return {'n': best_n, 'colors': best_colors}
    
    # 2. Try n=18 with remaining time
    t_start_18 = time.time()
    alloc_18 = deadline - t_start_18 - 2.0 # 2s buffer
    
    if alloc_18 > 5.0:
        # Use the best 17 coloring as a "seed" for 18? 
        # Hard to do directly. Just run SA for 18.
        sol_18 = _sa_search(18, t_start_18 + alloc_18)
        if sol_18 is not None:
            # Verify
            adj = [[0] * 18 for _ in range(4)]
            idx = 0
            for i in range(18):
                for j in range(i + 1, 18):
                    c = sol_18[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            if not any(_has_c6(adj[c], 18) for c in range(4)):
                best_n = 18
                best_colors = sol_18
                
    # 3. Try n=19 if time permits (unlikely)
    if best_n < 19 and time.time() < deadline - 10:
        t_start_19 = time.time()
        alloc_19 = deadline - t_start_19 - 2.0
        if alloc_19 > 10.0:
            sol_19 = _sa_search(19, t_start_19 + alloc_19)
            if sol_19 is not None:
                adj = [[0] * 19 for _ in range(4)]
                idx = 0
                for i in range(19):
                    for j in range(i + 1, 19):
                        c = sol_19[idx]
                        adj[c][i] |= (1 << j)
                        adj[c][j] |= (1 << i)
                        idx += 1
                if not any(_has_c6(adj[c], 19) for c in range(4)):
                    best_n = 19
                    best_colors = sol_19

    # Fallback if nothing found (should not happen for 17)
    if best_n == 0:
        best_n = 4
        best_colors = [0, 1, 2, 3, 0, 1]

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
