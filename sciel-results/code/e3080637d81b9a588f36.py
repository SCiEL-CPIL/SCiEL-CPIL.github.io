import random
import time
import sys
sys.setrecursionlimit(10000)

def construct():
    n = 24
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for safety

    # Generate the pinned edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))

    r = random.Random(614)
    r.shuffle(edges)

    # Map edge to index
    edge_to_idx = {}
    for idx, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx

    # Precompute adjacency lists sorted by edge arrival index
    adj_raw = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj_raw[u].append((v, idx))
        adj_raw[v].append((u, idx))

    for i in range(n):
        adj_raw[i].sort(key=lambda x: x[1])

    full_mask = (1 << n) - 1

    # Helper: Check if G_K has a Hamilton cycle and return it.
    def find_hamiltonian_cycle(K):
        # Build adjacency bitmasks
        adj_mask = [0] * n
        for i in range(K):
            u, v = edges[i]
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

        # Quick check: min degree
        for i in range(n):
            if adj_mask[i].bit_count() < 2:
                return None

        # Connectivity check
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            u = stack.pop()
            m = adj_mask[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    stack.append(v)
        if count < n:
            return None

        # Precompute for each vertex its neighbors in order of fewest remaining options
        # We'll use Warnsdorff's rule with dynamic degree calculation

        path = [0]
        visited_mask = 1  # vertex 0 is visited

        # Iterative DFS with explicit stack to avoid recursion depth issues
        # and to allow better pruning
        # We'll use a recursive DFS with strong pruning: 
        # 1. Warnsdorff's rule
        # 2. Check if any unvisited vertex becomes isolated (0 unvisited neighbors)
        
        def dfs(u, vmask, depth):
            nonlocal deadline
            if time.time() > deadline:
                return False

            if depth == n:
                return (adj_mask[u] & 1) != 0

            candidates = adj_mask[u] & (~vmask)
            if candidates == 0:
                return False

            # Pruning: Check for unvisited vertices with 0 unvisited neighbors
            # This is O(N) per step, which might be slow, but N=24 is small.
            # We only need to check if the count of unvisited vertices with deg 0 is > 0.
            # Note: The current vertex u is not unvisited, but its neighbors matter for others.
            # Actually, we check for any v not in vmask, if (adj_mask[v] & ~vmask).bit_count() == 0, then v is stranded.
            # Wait, v is unvisited. Its neighbors in the remaining graph are (adj_mask[v] & ~vmask).
            # If this is 0, v cannot be visited later because it has no connection to any other unvisited vertex.
            # It must be connected to u (current) or some previously visited vertex?
            # No, in a Hamiltonian path/cycle construction, we build the path.
            # If v is unvisited, it must be connected to the next vertex we pick, or if it's the last one, to 0.
            # If v has 0 unvisited neighbors, it can only be connected to visited vertices.
            # But the path is linear. v must have 2 neighbors in the final cycle.
            # One neighbor is the previous vertex in the path (which is u if v is next, or some other if not).
            # If v is not next, its previous neighbor in the path must be unvisited? No, the path is built sequentially.
            # The only unvisited vertices are those not yet in the path.
            # If v is unvisited, its neighbors in the final cycle must be:
            # 1. The vertex before it in the path.
            # 2. The vertex after it in the path.
            # If v is not the next vertex, the vertex before it is unvisited (except for the first vertex 0).
            # This logic is complex.
            # Simpler prune: If there is an unvisited vertex v such that (adj_mask[v] & ~vmask) == 0, 
            # then v has no unvisited neighbors. 
            # If v is not the last vertex to be visited, it must connect to an unvisited vertex (its predecessor in the path).
            # So if v is not the last one, it must have at least 1 unvisited neighbor.
            # If v is the last one, it connects to 0.
            # So, if there is any unvisited v with 0 unvisited neighbors, 
            # it MUST be the last vertex.
            # If there are TWO or more such vertices, fail.
            # If there is ONE such vertex v, it MUST be the last vertex.
            # This means we must eventually reach v, and then go to 0.
            # This implies that the current vertex u cannot be such that we can't reach v?
            # Actually, if there is exactly one such vertex v, we must ensure that v is the last one.
            # This doesn't restrict the next move directly unless v is a candidate?
            # If v is a candidate (adjacent to u), we can go to v only if depth+1 == n-1? No, if we go to v, v becomes visited.
            # If we go to v, and v was the only one with 0 unvisited neighbors, after visiting v, the condition changes.
            
            # Let's implement the "Isolated" check simply:
            # Count unvisited vertices with 0 unvisited neighbors.
            # If count > 1, return False.
            # If count == 1, let v be that vertex.
            # Then v must be the last vertex.
            # This means the path must end at v, then go to 0.
            # So, if we are at u, and u != v, we cannot visit v now unless depth+1 == n-1?
            # No, if we visit v now, v is visited. The "unvisited" set shrinks.
            # The constraint is that v must be the LAST unvisited vertex.
            # So, if depth+1 < n-1, we should NOT visit v?
            # If we visit v now, v is not the last one (unless depth+1 == n-1).
            # So if depth+1 < n-1, we must NOT visit v.
            # If depth+1 == n-1, we MUST visit v (if v is a candidate).
            # If v is not a candidate, we fail? No, we just can't visit it now.
            
            isolated_count = 0
            isolated_vertex = -1
            temp_vmask = vmask
            # Iterate through all unvisited vertices
            unvisited = (~vmask) & full_mask
            while unvisited:
                lsb = unvisited & -unvisited
                v = lsb.bit_length() - 1
                unvisited ^= lsb
                deg = (adj_mask[v] & ~vmask).bit_count()
                if deg == 0:
                    isolated_count += 1
                    isolated_vertex = v
                    if isolated_count > 1:
                        return False

            # Warnsdorff's rule: try neighbors with fewest options first
            cand_list = []
            temp = candidates
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                temp ^= lsb
                # Count unvisited neighbors of v
                deg = (adj_mask[v] & ~vmask).bit_count()
                
                # If v is the isolated one, and we are not at the penultimate step, we should avoid it?
                # Actually, if v is isolated (0 unvisited neighbors), it MUST be the last vertex.
                # So if depth + 1 < n - 1, we should NOT pick v.
                # If depth + 1 == n - 1, we MUST pick v (if it's a candidate).
                
                if isolated_vertex == v:
                    if depth + 1 < n - 1:
                        continue # Do not pick v yet
                    elif depth + 1 == n - 1:
                        # Must pick v
                        # We'll handle this by prioritizing it or forcing it
                        pass
                
                cand_list.append((deg, v))

            if not cand_list:
                return False
                
            # If there is an isolated vertex, and it is a candidate, and we are at the penultimate step,
            # we must pick it.
            if isolated_vertex != -1:
                if isolated_vertex in [v for d, v in cand_list]:
                    if depth + 1 == n - 1:
                        # Force pick isolated vertex
                        path.append(isolated_vertex)
                        if dfs(isolated_vertex, vmask | (1 << isolated_vertex), depth + 1):
                            return True
                        path.pop()
                        return False
                    else:
                        # Should not have picked it, so we skipped it in cand_list?
                        # If we skipped it, and it's the only candidate? Then fail.
                        # If there are other candidates, we pick them.
                        pass

            cand_list.sort()

            for deg, v in cand_list:
                path.append(v)
                if dfs(v, vmask | (1 << v), depth + 1):
                    return True
                path.pop()

            return False

        if dfs(0, 1, 1):
            return path
        return None

    # Find the smallest K such that min degree >= 2
    # This is a lower bound for any Hamiltonian cycle
    degs = [0] * n
    K_min_deg = len(edges)
    for idx, (u, v) in enumerate(edges):
        degs[u] += 1
        degs[v] += 1
        if all(d >= 2 for d in degs):
            K_min_deg = idx + 1
            break

    # Linear search from K_min_deg upwards
    # This is better if the minimum is close to the lower bound
    best_cycle = None
    best_K = len(edges)
    
    # Fallback for full graph
    fallback_cycle = find_hamiltonian_cycle(len(edges))
    if fallback_cycle:
        best_cycle = fallback_cycle
        best_K = len(edges)

    for K in range(K_min_deg, len(edges) + 1):
        if time.time() > deadline:
            break
        cycle = find_hamiltonian_cycle(K)
        if cycle:
            best_cycle = cycle
            best_K = K
            break # Found the smallest K, so we are done

    if best_cycle:
        return best_cycle
    return fallback_cycle if fallback_cycle else list(range(n))

if __name__ == "__main__":
    pass
