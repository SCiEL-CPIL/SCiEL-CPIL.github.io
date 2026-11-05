import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n = 40
    best_edges = []
    best_count = 0

    # Fast C6 check using bitsets
    def path5(u, v, adj):
        # Checks if there is a simple path of length 5 from u to v
        # u-a-b-c-d-v
        if u == v:
            return False
        a_mask = adj[u] & ~(1 << v)
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit

            b_mask = adj[a] & ~(1 << u) & ~(1 << v)
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit

                c_mask = adj[b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit

                    # d must be in N(c) and N(v), and not u, a, b, v
                    d_mask = adj[c] & adj[v] & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                    if d_mask:
                        return True
        return False

    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (1 << j) & adj[i]:
                    edges.append([i, j])
        return edges

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def has_c6(adj):
        # Check if the graph contains a C6
        # We can use the path5 check: if there is a path of length 5 between u and v,
        # and u and v are adjacent, then we have a C6.
        # But wait, path5(u,v) checks for u-a-b-c-d-v. If (u,v) is an edge, then u-a-b-c-d-v-u is a C6.
        # However, the vertices must be distinct.
        # In path5, we exclude u, v, a, b, c from the masks appropriately?
        # Let's re-verify path5 logic for distinctness.
        # a != u, v (a_mask excludes v, and a is from adj[u] so a!=u)
        # b != u, v, a (b_mask excludes u, v. b is from adj[a]. Could b be u? adj[a] contains u if (a,u) is edge. We excluded u. Could b be v? excluded. Could b be a? no, simple graph.)
        # c != u, v, a, b (c_mask excludes u, a, v. c is from adj[b]. Could c be b? no. Could c be u? excluded. Could c be a? excluded. Could c be v? excluded.)
        # d != u, v, a, b, c (d_mask excludes u, a, b, v. d is from adj[c] and adj[v]. Could d be c? no. Could d be u? excluded. Could d be a? excluded. Could d be b? excluded. Could d be v? excluded.)
        # So u, a, b, c, d, v are all distinct.
        # Thus, if (u,v) is an edge and path5(u,v) is True, we have a C6.
        
        for u in range(n):
            for v in range(u + 1, n):
                if (1 << v) & adj[u]:
                    if path5(u, v, adj):
                        return True
        return False

    for restart in range(200):
        if time.time() > deadline:
            break

        # Initialize with 8 disjoint K5s
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    add_edge(adj, base + a, base + b)

        # Phase 1: Greedy Add
        # Try adding all missing edges
        missing_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not ((1 << j) & adj[i]):
                    missing_edges.append((i, j))

        random.shuffle(missing_edges)

        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if not path5(u, v, adj):
                add_edge(adj, u, v)

        # Phase 2: Local Search (Swap-based)
        # We will try to remove one edge and add multiple edges.
        # To be efficient, we will track the current number of edges.
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break

            # Get list of current edges to try removing
            current_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (1 << j) & adj[i]:
                        current_edges.append((i, j))

            random.shuffle(current_edges)

            for u, v in current_edges:
                if time.time() > deadline:
                    break

                # Try removing (u, v)
                remove_edge(adj, u, v)
                old_count = count_edges(adj)

                # Try to add as many missing edges as possible
                # We iterate through all pairs. If it's not an edge, and path5 is false, add it.
                added_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not ((1 << j) & adj[i]):
                            if not path5(i, j, adj):
                                add_edge(adj, i, j)
                                added_edges.append((i, j))

                new_count = count_edges(adj)

                if new_count > old_count + 1:  # Net gain > 0
                    improved = True
                    # Keep the changes
                else:
                    # Revert
                    for i, j in added_edges:
                        remove_edge(adj, i, j)
                    add_edge(adj, u, v)

        # Phase 3: Additional Local Search with different randomization
        # Sometimes a single swap isn't enough. Let's try a few more rounds.
        for _ in range(3):
            if time.time() > deadline:
                break
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break

                current_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (1 << j) & adj[i]:
                            current_edges.append((i, j))

                random.shuffle(current_edges)

                for u, v in current_edges:
                    if time.time() > deadline:
                        break

                    remove_edge(adj, u, v)
                    old_count = count_edges(adj)

                    added_edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if not ((1 << j) & adj[i]):
                                if not path5(i, j, adj):
                                    add_edge(adj, i, j)
                                    added_edges.append((i, j))

                    new_count = count_edges(adj)

                    if new_count > old_count + 1:
                        improved = True
                    else:
                        for i, j in added_edges:
                            remove_edge(adj, i, j)
                        add_edge(adj, u, v)

        # Update best
        current_edges_list = get_edges(adj)
        if len(current_edges_list) > best_count:
            best_count = len(current_edges_list)
            best_edges = current_edges_list

    return best_edges
