import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill

    n = 18
    full_mask = (1 << n) - 1

    # ---------- graph helpers ----------
    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return True
        common = adj[u] & adj[v]
        c = common
        while c:
            lsb = c & -c
            w = lsb.bit_length() - 1
            if adj[w] & (common & ~lsb):
                return False
            c ^= lsb
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_triangles(adj):
        triangles = []
        for u in range(n):
            v_mask = adj[u] & ~((1 << (u + 1)) - 1)
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                while common:
                    wlsb = common & -common
                    w = wlsb.bit_length() - 1
                    common ^= wlsb
                    triangles.append((1 << u) | (1 << v) | (1 << w))
        return triangles

    def has_k4(adj):
        # Check all 4-subsets for a K4 (only ~3060 subsets, cheap)
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] & (1 << b)):
                    continue
                for c in range(b + 1, n):
                    if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                        continue
                    for d in range(c + 1, n):
                        if (adj[a] & (1 << d)) and (adj[b] & (1 << d)) and (adj[c] & (1 << d)):
                            return True
        return False

    def repair_k4(adj):
        """Remove edges until the graph is K4-free. Returns True if any edge was removed."""
        changed = False
        # Iteratively find and break K4s
        for _ in range(200):
            found = False
            for a in range(n):
                if found:
                    break
                for b in range(a + 1, n):
                    if found:
                        break
                    if not (adj[a] & (1 << b)):
                        continue
                    for c in range(b + 1, n):
                        if found:
                            break
                        if not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                            continue
                        for d in range(c + 1, n):
                            if (adj[a] & (1 << d)) and (adj[b] & (1 << d)) and (adj[c] & (1 << d)):
                                # Found K4 on {a,b,c,d}; remove a random edge from it
                                k4_edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
                                u, v = random.choice(k4_edges)
                                remove_edge(adj, u, v)
                                found = True
                                changed = True
                                break
            if not found:
                break
        return changed

    # ---------- evaluation: largest triangle-free induced subgraph ----------
    def evaluate(adj):
        """Return the size of the largest triangle-free induced subgraph."""
        triangles = get_triangles(adj)
        if not triangles:
            return n
        triangles = list(set(triangles))
        num_tris = len(triangles)
        all_tris_mask = (1 << num_tris) - 1

        # Precompute for each vertex which triangles it hits
        vertex_hits = [0] * n
        for v in range(n):
            mask = 1 << v
            hit = 0
            for i, tri in enumerate(triangles):
                if tri & mask:
                    hit |= (1 << i)
            vertex_hits[v] = hit

        # Find minimum m such that some m-subset hits all triangles.
        # Answer = n - m.
        for m in range(1, n + 1):
            found = False

            def backtrack(start, chosen, covered):
                nonlocal found
                if found:
                    return
                if covered == all_tris_mask:
                    found = True
                    return
                if len(chosen) == m:
                    return
                # Pruning: if even picking all remaining vertices can't cover, stop
                # (simple bound: remaining vertices count)
                remaining_slots = m - len(chosen)
                remaining_vertices = n - start
                if remaining_vertices < remaining_slots:
                    return
                for v in range(start, n):
                    new_covered = covered | vertex_hits[v]
                    if new_covered == covered:
                        # v adds no new coverage; still might be needed to fill m,
                        # but if we already can cover all without it, skip for speed
                        # Only skip if we still have enough other vertices to reach m
                        # For safety, don't skip (could be needed to pad m)
                        # Actually if new_covered == covered, picking v is wasteful
                        # unless we need to fill m. But filling m with a non-covering
                        # vertex never helps cover more. So skip.
                        continue
                    backtrack(v + 1, chosen + [v], new_covered)
                    if found:
                        return

            backtrack(0, [], 0)
            if found:
                return n - m
        return 1

    # ---------- initializations ----------
    def init_k666():
        adj = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    def generate_random_k4_free():
        adj = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        return adj

    def generate_triangular_perturb(base_adj, strength=0.3):
        """Perturb a tripartite-like graph by adding/removing edges while keeping K4-free."""
        adj = base_adj[:]
        # Try to add some random edges
        for _ in range(20):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if not (adj[u] & (1 << v)):
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
        # Try to remove some random edges
        for _ in range(10):
            u = random.randint(0, n - 1)
            v_mask = adj[u]
            if v_mask:
                bits = []
                tmp = v_mask
                while tmp:
                    b = (tmp & -tmp).bit_length() - 1
                    bits.append(b)
                    tmp &= tmp - 1
                if bits:
                    v = random.choice(bits)
                    remove_edge(adj, u, v)
        return adj

    # ---------- main search ----------
    best_adj = None
    best_score = n + 1

    # Try several initializations
    initializations = []

    # 1. K6,6,6 complete tripartite
    initializations.append(init_k666())

    # 2. Several random K4-free graphs
    for _ in range(4):
        initializations.append(generate_random_k4_free())

    # 3. Perturbed tripartite
    initializations.append(generate_triangular_perturb(init_k666(), 0.4))
    initializations.append(generate_triangular_perturb(init_k666(), 0.7))

    for init_adj in initializations:
        if time.time() > deadline - 5:
            break
        # Ensure K4-free (should already be, but repair just in case)
        repair_k4(init_adj)
        score = evaluate(init_adj)
        if score < best_score:
            best_score = score
            best_adj = init_adj[:]

    if best_adj is None:
        best_adj = generate_random_k4_free()
        repair_k4(best_adj)
        best_score = evaluate(best_adj)

    current_adj = best_adj[:]
    current_score = best_score

    # Local search with simulated annealing
    t = 2.0
    cooling = 0.997
    min_t = 0.05
    no_improve_steps = 0
    max_no_improve = 200

    while time.time() < deadline and t > min_t:
        # Decide move type: mostly add edges (densifies, reduces triangle-free subgraphs)
        # occasionally remove to escape local minima
        move_type = random.random()

        if move_type < 0.65:
            # Add a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if current_adj[u] & (1 << v):
                continue
            if not can_add_edge(current_adj, u, v):
                continue

            old_u = current_adj[u]
            old_v = current_adj[v]
            add_edge(current_adj, u, v)
            new_score = evaluate(current_adj)

            delta = new_score - current_score
            if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / t)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                current_adj[u] = old_u
                current_adj[v] = old_v
                no_improve_steps += 1
        else:
            # Remove a random edge
            u = random.randint(0, n - 1)
            v_mask = current_adj[u]
            if not v_mask:
                continue
            bits = []
            tmp = v_mask
            while tmp:
                b = (tmp & -tmp).bit_length() - 1
                bits.append(b)
                tmp &= tmp - 1
            if not bits:
                continue
            v = random.choice(bits)

            old_u = current_adj[u]
            old_v = current_adj[v]
            remove_edge(current_adj, u, v)
            new_score = evaluate(current_adj)

            delta = new_score - current_score
            if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / t)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                current_adj[u] = old_u
                current_adj[v] = old_v
                no_improve_steps += 1

        t *= cooling

        # If stuck for too long, restart from best with perturbation
        if no_improve_steps > max_no_improve:
            current_adj = generate_triangular_perturb(best_adj, 0.5)
            repair_k4(current_adj)
            current_score = evaluate(current_adj)
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj[:]
            no_improve_steps = 0
            t = max(t, 1.0)  # reheat slightly

    # Final safety: ensure best_adj is K4-free
    if has_k4(best_adj):
        repair_k4(best_adj)
        # Re-evaluate after repair
        final_score = evaluate(best_adj)
        if final_score < best_score:
            best_score = final_score

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])

    return edges
