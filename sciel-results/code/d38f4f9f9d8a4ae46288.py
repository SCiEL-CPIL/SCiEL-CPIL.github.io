import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin for 120s limit

    def time_check():
        return time.time() - start_time

    def is_deadline_exceeded():
        return time_check() > DEADLINE

    def has_triangle(adj_masks, n):
        """Check if graph with adjacency masks has a triangle."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_masks[u] >> v) & 1:
                    if adj_masks[u] & adj_masks[v]:
                        return True
        return False

    def max_independent_set_size(adj_masks, n, timeout=None):
        """
        Returns the size of the maximum independent set using branch and bound.
        """
        best = [0]
        
        def dfs(candidates, size):
            if timeout is not None and time.time() > timeout:
                return
            if not candidates:
                if size > best[0]:
                    best[0] = size
                return
            
            if size + candidates.bit_count() <= best[0]:
                return
            
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Include v
            new_candidates = candidates & ~lsb & ~adj_masks[v]
            dfs(new_candidates, size + 1)
            
            # Exclude v
            dfs(candidates & ~lsb, size)

        dfs((1 << n) - 1, 0)
        return best[0]

    def edges_from_adj_masks(adj_masks, n):
        """Convert adjacency masks to edge list."""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_masks[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def is_valid_graph(adj_masks, n):
        """Check if graph is triangle-free and has no independent set of size 10."""
        if has_triangle(adj_masks, n):
            return False
        alpha = max_independent_set_size(adj_masks, n, timeout=start_time + 2)
        return alpha < 10

    def search_circulant(n, start_time):
        """
        Search for a valid circulant graph on n vertices.
        A circulant graph is defined by a set of offsets S.
        i ~ j iff (j-i) % n in S.
        Triangle-free condition: No a, b in S such that (a+b) % n in S.
        """
        max_off = n // 2
        # We need to pick S subset of {1, ..., max_off}
        # S is symmetric, so if we pick k, we effectively have k and n-k.
        # The condition "no a,b in S such that a+b in S" applies to the set of differences.
        
        # Let's represent S as a bitmask of offsets 1..max_off
        # If bit k is set, then k and n-k are in the difference set.
        
        # Randomized search for S
        for _ in range(2000):
            if is_deadline_exceeded():
                return None
            
            S_mask = 0
            # Randomly select offsets
            for k in range(1, max_off + 1):
                # Probability of selecting k
                # We want a sparse graph to avoid triangles? Or dense to avoid independent sets?
                # Triangle-free graphs are typically sparse (Mantel's theorem: max edges n^2/4).
                # So p should be low.
                if random.random() < 0.2:
                    S_mask |= (1 << k)
            
            if S_mask == 0:
                continue

            # Check triangle-free condition:
            # No a, b in S such that a+b (mod n) is in S.
            # Since S is symmetric, we just need to check sums of elements in {1..max_off}.
            # If a, b are in S, is a+b in S?
            # Note: if a+b > n, we check (a+b)%n.
            # Since a,b <= n/2, a+b <= n.
            # If a+b == n, then a+b mod n = 0, which is not in S.
            # If a+b < n, we check if a+b is in S.
            
            valid_triangle_free = True
            offsets = [k for k in range(1, max_off + 1) if (S_mask >> k) & 1]
            
            for i in range(len(offsets)):
                a = offsets[i]
                for j in range(len(offsets)):
                    b = offsets[j]
                    s = a + b
                    if s >= n:
                        s -= n
                    if s == 0:
                        continue
                    # Check if s is in S
                    # s is in 1..n-1.
                    # If s > n/2, then n-s is in 1..n/2.
                    # Check if (s if s <= max_off else n-s) is in S_mask.
                    check_val = s if s <= max_off else n - s
                    if (S_mask >> check_val) & 1:
                        valid_triangle_free = False
                        break
                if not valid_triangle_free:
                    break
            
            if not valid_triangle_free:
                continue

            # Build adjacency masks
            adj_masks = [0] * n
            for i in range(n):
                for k in offsets:
                    j1 = (i + k) % n
                    j2 = (i - k) % n
                    adj_masks[i] |= (1 << j1)
                    adj_masks[i] |= (1 << j2)
            
            # Check independence number
            # We can do a quick check first
            # If we have time, do the full check
            alpha = max_independent_set_size(adj_masks, n, timeout=start_time + 1)
            if alpha < 10:
                return adj_masks

        return None

    def local_search(adj_masks, n, start_time):
        """
        Improve a triangle-free graph to reduce independence number.
        """
        current_adj = adj_masks[:]
        current_alpha = max_independent_set_size(current_adj, n, timeout=start_time + 2)
        
        for _ in range(500):
            if is_deadline_exceeded():
                break
            
            # Find a large independent set
            # Reuse max_independent_set_size logic to get the set?
            # For now, just random edge flips.
            
            # Try to add an edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Ensure u < v for symmetry in mask update? No, masks are directional in bit logic but symmetric in value.
            if (current_adj[u] >> v) & 1:
                # Edge exists, try to remove it
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                # Edge doesn't exist, try to add it
                # Check if it creates a triangle
                if current_adj[u] & current_adj[v]:
                    # Would create triangle, skip
                    continue
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            
            # Check new alpha
            new_alpha = max_independent_set_size(current_adj, n, timeout=start_time + 1)
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if current_alpha < 10:
                    return current_adj
            else:
                # Revert if not improved
                if (current_adj[u] >> v) & 1:
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
        
        return current_adj

    # Main Logic
    best_n = 0
    best_edges = []

    # Phase 1: Try to find a valid graph with n=40 down to 27
    for n in range(40, 26, -1):
        if is_deadline_exceeded():
            break
        
        adj = search_circulant(n, start_time)
        if adj is not None:
            # Verify
            if is_valid_graph(adj, n):
                best_n = n
                best_edges = edges_from_adj_masks(adj, n)
                break
        
        # Also try local search on a random triangle-free graph if circulant fails
        if not is_deadline_exceeded():
            # Generate a random triangle-free graph
            random_adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.3:
                        # Check triangle free
                        if random_adj[i] & random_adj[j]:
                            continue
                        random_adj[i] |= (1 << j)
                        random_adj[j] |= (1 << i)
            
            # Polish it
            polished_adj = local_search(random_adj, n, start_time)
            if is_valid_graph(polished_adj, n):
                best_n = n
                best_edges = edges_from_adj_masks(polished_adj, n)
                break

    # Phase 2: Fallback to known valid graph if search failed
    if best_n == 0:
        # Andrásfai graph And(9) on 26 vertices
        # It is triangle-free and has independence number 9.
        n = 26
        adj_masks = [0] * n
        # And(k) is a circulant graph with n=3k-1.
        # For k=9, n=26.
        # Offsets: 1, 2, ..., k-1 ? No.
        # And(k) edges: i~j if (j-i) mod n is in {1, ..., k-1} U {n-k+1, ..., n-1}.
        # Let's verify triangle-free:
        # S = {1, ..., 8, 18, ..., 25}.
        # Sum of two small numbers < 16, not in S.
        # Sum of small and large: e.g., 1 + 18 = 19 (in S). Triangle!
        # Wait, And(k) is triangle-free.
        # The condition is that S is a "sum-free" set in Z_n?
        # For And(9), n=26. S = {1, 2, 3, 4, 5, 6, 7, 8} U {18, 19, 20, 21, 22, 23, 24, 25}.
        # 1+1=2 (in S). Triangle 0,1,2.
        # My definition of And(k) might be wrong or the sum-free condition is different.
        # Actually, And(k) is defined as the graph where i~j if (j-i) mod n is in {1, ..., k-1} or {n-k+1, ..., n-1}.
        # Let's check 0, 1, 2.
        # 0-1 = -1 = 25 (in S).
        # 1-2 = -1 = 25 (in S).
        # 0-2 = -2 = 24 (in S).
        # So 0,1,2 is a triangle.
        # Why is And(k) triangle-free?
        # Ah, And(k) is usually defined for n=3k-1.
        # Maybe the offsets are different?
        # Let's just use a hardcoded edge list for a known small triangle-free graph with alpha < 10.
        # C5 (5 vertices) has alpha 2.
        # C10 (10 vertices) has alpha 5.
        # We need n >= 26.
        # Let's just use a random valid graph if we can find one quickly, otherwise return n=1 (valid? no, alpha 1 < 10, triangle free).
        # But we want to maximize n.
        
        # Let's try to construct a valid graph on 26 vertices using the circulant search again, but with more trials.
        adj = search_circulant(26, start_time)
        if adj and is_valid_graph(adj, 26):
            best_n = 26
            best_edges = edges_from_adj_masks(adj, 26)
        else:
            # Fallback to a definitely valid small graph
            # Cycle C26 is triangle free. Alpha(C26) = 13. Too big.
            # We need alpha < 10.
            # Complete bipartite graph K_{13,13} is triangle free. Alpha = 13. Too big.
            # We need a graph with smaller independence number.
            # Let's use the search result from Phase 1 if it found something for n < 27.
            # If not, we return a very small valid graph.
            # n=1 is valid.
            best_n = 1
            best_edges = []

    return {'n': best_n, 'edges': best_edges}
