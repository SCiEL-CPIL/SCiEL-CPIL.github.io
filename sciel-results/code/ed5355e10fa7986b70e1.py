import random
import time

def construct():
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)

    def is_k33_free_add(u, v, M):
        """
        Fast bitmask check: can we add edge (u,v) without creating a K3,3?
        K3,3 exists iff some 3 vertices have 3+ common neighbours.
        Adding (u,v) only creates new common-neighbor relationships involving u or v.
        """
        # If u or v already adjacent, nothing to do
        if M[u] & (1 << v):
            return False

        # After adding (u,v), v becomes a new common neighbor of any triple {u, a, b}
        # where a,b are both adjacent to v. We need |N(u)∩N(a)∩N(b)| <= 1 (excluding v).
        # Similarly, u becomes a new common neighbor of any triple {v, a, b} where a,b
        # are both adjacent to u. We need |N(v)∩N(a)∩N(b)| <= 1 (excluding u).

        # Case 1: v as new common neighbor
        # For each pair (a,b) both adjacent to v (a != u, b != u, a != v, b != v):
        #   count common neighbors of u, a, b in current graph. Must be <= 1.
        A = M[v] & ~(1 << u)  # neighbors of v excluding u
        # We need to check pairs from A
        # But also need a != u (already excluded) and b != u (already excluded)
        # And a != v, b != v (v is not in M[v] typically for simple graph, but let's be safe)
        A &= ~(1 << v)

        a_list = []
        temp = A
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            a_list.append(idx)
            temp ^= lsb

        for i in range(len(a_list)):
            for j in range(i + 1, len(a_list)):
                a = a_list[i]
                b = a_list[j]
                # Common neighbors of u, a, b
                common = M[u] & M[a] & M[b]
                # v is not in common (since u-v not yet added, and v not adjacent to u in current graph... wait)
                # Actually v IS adjacent to a and b (by construction), and we're checking if v would become a common neighbor.
                # The existing common neighbors of {u,a,b} are in M[u] & M[a] & M[b].
                # v is NOT in M[u] yet (we're about to add it), so v is not in common.
                # After adding u-v, v will be a common neighbor, so total = bit_count(common) + 1.
                # We need total <= 2, so bit_count(common) <= 1.
                if common.bit_count() > 1:
                    return False

        # Case 2: u as new common neighbor
        B = M[u] & ~(1 << v)  # neighbors of u excluding v
        B &= ~(1 << u)

        b_list = []
        temp = B
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            b_list.append(idx)
            temp ^= lsb

        for i in range(len(b_list)):
            for j in range(i + 1, len(b_list)):
                a = b_list[i]
                b = b_list[j]
                common = M[v] & M[a] & M[b]
                # u is not in common yet
                if common.bit_count() > 1:
                    return False

        return True

    def greedy_fill(M, order):
        for u, v in order:
            if not (M[u] & (1 << v)):
                if is_k33_free_add(u, v, M):
                    M[u] |= (1 << v)
                    M[v] |= (1 << u)
        return sum(M[i].bit_count() for i in range(n)) // 2

    def local_search(M, deadline):
        count = sum(M[i].bit_count() for i in range(n)) // 2
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Build current edge list
            cur_edges = []
            for i in range(n):
                temp = M[i]
                while temp:
                    lsb = temp & -temp
                    j = lsb.bit_length() - 1
                    if i < j:
                        cur_edges.append((i, j))
                    temp ^= lsb

            # Try removing each edge and see if we can add 2+ back
            for rem_edge in cur_edges:
                u, v = rem_edge
                M[u] &= ~(1 << v)
                M[v] &= ~(1 << u)
                added_edges = []
                for cu, cv in all_edges:
                    if (cu, cv) == rem_edge:
                        continue
                    if M[cu] & (1 << cv):
                        continue
                    if is_k33_free_add(cu, cv, M):
                        M[cu] |= (1 << cv)
                        M[cv] |= (1 << cu)
                        added_edges.append((cu, cv))
                        if len(added_edges) >= 2:
                            break
                new_count = sum(M[i].bit_count() for i in range(n)) // 2
                if new_count > count:
                    count = new_count
                    improved = True
                    break
                else:
                    for cu, cv in added_edges:
                        M[cu] &= ~(1 << cv)
                        M[cv] &= ~(1 << cu)
                    M[u] |= (1 << v)
                    M[v] |= (1 << u)
        return count

    def kick_and_fill(deadline):
        M = [0] * n
        kick_size = random.randint(5, 15)
        kick_edges = random.sample(all_edges, kick_size)
        for u, v in kick_edges:
            if is_k33_free_add(u, v, M):
                M[u] |= (1 << v)
                M[v] |= (1 << u)
        order = all_edges[:]
        random.shuffle(order)
        count = greedy_fill(M, order)
        count = local_search(M, deadline)
        return M, count

    def greedy_random(deadline):
        M = [0] * n
        order = all_edges[:]
        random.shuffle(order)
        count = greedy_fill(M, order)
        return M, count

    best_M = [0] * n
    best_count = 0

    start_time = time.time()
    deadline = start_time + 115.0

    iteration = 0
    while time.time() < deadline:
        iteration += 1
        if time.time() > deadline:
            break
        if iteration % 3 == 0:
            M, count = kick_and_fill(deadline)
        else:
            M, count = greedy_random(deadline)
        if count > best_count:
            best_count = count
            best_M = M[:]

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
