import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 285

    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)

    def bits_to_list(mask):
        result = []
        temp = mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            result.append(idx)
            temp ^= lsb
        return result

    def has_k3_in_mask(adj, mask):
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            neigh = adj[u] & mask & ~(1 << u)
            if neigh == 0:
                continue
            n_list = bits_to_list(neigh)
            for j in range(len(n_list)):
                v = n_list[j]
                if adj[v] & neigh & ~(1 << v) & ~(1 << u):
                    return True
        return False

    def has_i3_in_mask(adj, mask):
        verts = bits_to_list(mask)
        for i in range(len(verts)):
            u = verts[i]
            non_neigh = mask & ~adj[u] & ~(1 << u)
            if non_neigh == 0:
                continue
            n_list = bits_to_list(non_neigh)
            for j in range(len(n_list)):
                v = n_list[j]
                if (non_neigh & ~adj[v] & ~(1 << v) & ~(1 << u)):
                    return True
        return False

    def has_k4_in_mask(adj, mask):
        verts = bits_to_list(mask)
        for u in verts:
            neigh = adj[u] & mask & ~(1 << u)
            if has_k3_in_mask(adj, neigh):
                return True
        return False

    def has_i4_in_mask(adj, mask):
        verts = bits_to_list(mask)
        for u in verts:
            non_neigh = mask & ~adj[u] & ~(1 << u)
            if has_i3_in_mask(adj, non_neigh):
                return True
        return False

    def is_valid(adj, n):
        all_mask = (1 << n) - 1
        for u in range(n):
            if has_k4_in_mask(adj, adj[u]):
                return False
            non_neigh = all_mask & ~adj[u] & ~(1 << u)
            if has_i4_in_mask(adj, non_neigh):
                return False
        return True

    if not is_valid(adj_masks, n):
        return n, []

    best_n = n
    best_adj = adj_masks[:]
    current_n = n
    current_adj = adj_masks[:]
    all_mask = (1 << current_n) - 1

    def try_add_vertex(adj, n, neighbors_mask, all_mask):
        if has_k4_in_mask(adj, neighbors_mask):
            return None
        non_neighbors_mask = all_mask & ~neighbors_mask
        if has_i4_in_mask(adj, non_neighbors_mask):
            return None
        new_n = n + 1
        new_adj = adj + [0]
        for v in range(n):
            if neighbors_mask & (1 << v):
                new_adj[v] |= (1 << new_n)
                new_adj[new_n] |= (1 << v)
        return new_n, new_adj

    def try_flip_edge(adj, n, i, j, all_mask):
        if adj[i] & (1 << j):
            mask = (all_mask & ~adj[i] & ~adj[j]) & ~(1 << i) & ~(1 << j)
            if has_i3_in_mask(adj, mask):
                return None
            new_adj = adj[:]
            new_adj[i] &= ~(1 << j)
            new_adj[j] &= ~(1 << i)
            return new_adj
        else:
            mask = (adj[i] & adj[j]) & ~(1 << i) & ~(1 << j)
            if has_k3_in_mask(adj, mask):
                return None
            new_adj = adj[:]
            new_adj[i] |= (1 << j)
            new_adj[j] |= (1 << i)
            return new_adj

    def generate_candidates(adj, n, all_mask):
        candidates = set()
        for v in range(n):
            candidates.add(adj[v])
            candidates.add(all_mask & ~adj[v] & ~(1 << v))
            neigh = bits_to_list(adj[v])
            non_neigh = bits_to_list(all_mask & ~adj[v] & ~(1 << v))
            if len(neigh) >= 2:
                for a in range(len(neigh)):
                    for b in range(a + 1, len(neigh)):
                        m = 0
                        m |= (1 << neigh[a])
                        m |= (1 << neigh[b])
                        candidates.add(m)
            if len(non_neigh) >= 2:
                for a in range(len(non_neigh)):
                    for b in range(a + 1, len(non_neigh)):
                        m = 0
                        m |= (1 << non_neigh[a])
                        m |= (1 << non_neigh[b])
                        candidates.add(m)
        for _ in range(15):
            size = random.randint(max(1, n // 4), max(2, 3 * n // 4))
            neighbors = random.sample(range(n), min(size, n))
            mask = 0
            for v in neighbors:
                mask |= (1 << v)
            candidates.add(mask)
        return list(candidates)

    stuck_count = 0
    flip_attempts = 0
    while time.time() < deadline:
        added = False
        candidates = generate_candidates(current_adj, current_n, all_mask)
        random.shuffle(candidates)
        for mask in candidates:
            if time.time() > deadline:
                break
            result = try_add_vertex(current_adj, current_n, mask, all_mask)
            if result is not None:
                current_n, current_adj = result
                best_n = current_n
                best_adj = current_adj[:]
                all_mask = (1 << current_n) - 1
                added = True
                stuck_count = 0
                flip_attempts = 0
                break

        if added:
            continue

        stuck_count += 1
        if stuck_count > 50:
            current_n = best_n
            current_adj = best_adj[:]
            all_mask = (1 << current_n) - 1
            stuck_count = 0

        for _ in range(20):
            if time.time() > deadline:
                break
            i = random.randint(0, current_n - 1)
            j = random.randint(0, current_n - 1)
            if i == j:
                continue
            new_adj = try_flip_edge(current_adj, current_n, i, j, all_mask)
            if new_adj is not None:
                current_adj = new_adj
                flip_attempts += 1
                break

    final_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                final_edges.append((i, j))
    return best_n, final_edges
