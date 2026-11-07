import time
import random

def construct():
    n = 15
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill

    # All possible edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    total_edges = len(all_edges)  # 105

    # Precompute edge index
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx

    def check_validity(adj_masks, deg):
        """Check no simple 3-edge path s-u-v-t has deg[s]==deg[t]."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_masks[u] >> v) & 1:
                    continue
                # For edge (u,v), check all s in N(u)\{v}, t in N(v)\{u,s}
                su_mask = adj_masks[u] & ~(1 << v)  # neighbors of u except v
                tv_mask = adj_masks[v] & ~(1 << u)  # neighbors of v except u
                # For each s in su_mask, t in tv_mask, s != t
                s = su_mask
                while s:
                    s_bit = s & (-s)
                    s_idx = s_bit.bit_length() - 1
                    s = s - s_bit
                    # t in tv_mask, t != s
                    t_mask = tv_mask & ~(1 << s_idx)
                    t = t_mask
                    while t:
                        t_bit = t & (-t)
                        t_idx = t_bit.bit_length() - 1
                        t = t - t_bit
                        if deg[s_idx] == deg[t_idx]:
                            return False
        return True

    def check_validity_fast(adj_masks, deg):
        """Optimized: for each edge (u,v), check if any s in N(u)\\{v} and t in N(v)\\{u} with s!=t and deg[s]==deg[t]."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_masks[u] >> v) & 1:
                    continue
                # Collect degree groups
                deg_s = {}  # degree -> bitmask of vertices
                su = adj_masks[u] & ~(1 << v)
                s = su
                while s:
                    sb = s & (-s)
                    si = sb.bit_length() - 1
                    s -= sb
                    d = deg[si]
                    deg_s[d] = deg_s.get(d, 0) | (1 << si)
                
                tv = adj_masks[v] & ~(1 << u)
                t = tv
                while t:
                    tb = t & (-t)
                    ti = tb.bit_length() - 1
                    t -= tb
                    d = deg[ti]
                    if d in deg_s:
                        # There exists s with deg[s]==d. Need s != t.
                        mask = deg_s[d]
                        if mask & ~(1 << ti):
                            return False
        return True

    def greedy_build(edge_order, adj_masks, deg, deadline):
        """Add edges in given order if valid."""
        edges_set = set()
        for idx in edge_order:
            if time.time() > deadline:
                break
            u, v = all_edges[idx]
            if (u, v) in edges_set or (v, u) in edges_set:
                continue
            # Try adding
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            if check_validity_fast(adj_masks, deg):
                edges_set.add((u, v))
            else:
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                deg[u] -= 1
                deg[v] -= 1
        return edges_set

    def local_search(adj_masks, deg, edges_set, deadline, max_iters=2000):
        """Try edge swaps: remove one, add up to two."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iters += 1
            # Try removing each edge and adding two others
            edges_list = list(edges_set)
            random.shuffle(edges_list)
            for rem_e in edges_list:
                if time.time() > deadline:
                    break
                ru, rv = rem_e
                # Remove
                adj_masks[ru] &= ~(1 << rv)
                adj_masks[rv] &= ~(1 << ru)
                deg[ru] -= 1
                deg[rv] -= 1
                edges_set.discard(rem_e)
                edges_set.discard((rv, ru))
                
                # Try adding two edges
                best_pair = None
                best_gain = 0
                # Find candidate edges to add
                candidates = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if i == ru and j == rv:
                            continue
                        if i == rv and j == ru:
                            continue
                        if (i, j) in edges_set or (j, i) in edges_set:
                            continue
                        candidates.append((i, j))
                
                # Limit candidates for speed
                if len(candidates) > 30:
                    random.shuffle(candidates)
                    candidates = candidates[:30]
                
                # Try pairs
                for a in range(len(candidates)):
                    if time.time() > deadline:
                        break
                    ca = candidates[a]
                    # Try adding ca
                    adj_masks[ca[0]] |= (1 << ca[1])
                    adj_masks[ca[1]] |= (1 << ca[0])
                    deg[ca[0]] += 1
                    deg[ca[1]] += 1
                    
                    if check_validity_fast(adj_masks, deg):
                        # Try adding one more
                        for b in range(a + 1, len(candidates)):
                            cb = candidates[b]
                            # Check if cb is still addable
                            if (cb[0], cb[1]) in edges_set or (cb[1], cb[0]) in edges_set:
                                continue
                            adj_masks[cb[0]] |= (1 << cb[1])
                            adj_masks[cb[1]] |= (1 << cb[0])
                            deg[cb[0]] += 1
                            deg[cb[1]] += 1
                            if check_validity_fast(adj_masks, deg):
                                # Valid with both added: net gain +1
                                if best_pair is None or 2 > best_gain:
                                    best_pair = (ca, cb)
                                    best_gain = 2
                                # Undo cb
                                adj_masks[cb[0]] &= ~(1 << cb[1])
                                adj_masks[cb[1]] &= ~(1 << cb[0])
                                deg[cb[0]] -= 1
                                deg[cb[1]] -= 1
                            else:
                                # Undo cb
                                adj_masks[cb[0]] &= ~(1 << cb[1])
                                adj_masks[cb[1]] &= ~(1 << cb[0])
                                deg[cb[0]] -= 1
                                deg[cb[1]] -= 1
                        # If only ca added is valid, that's gain 0 (remove 1, add 1)
                        # Not useful, but keep ca for now
                        if best_gain < 1:
                            best_gain = 1  # at least ca is valid
                    else:
                        # Undo ca
                        adj_masks[ca[0]] &= ~(1 << ca[1])
                        adj_masks[ca[1]] &= ~(1 << ca[0])
                        deg[ca[0]] -= 1
                        deg[ca[1]] -= 1
                        continue
                    
                    # Undo ca (will re-add if we accept)
                    adj_masks[ca[0]] &= ~(1 << ca[1])
                    adj_masks[ca[1]] &= ~(1 << ca[0])
                    deg[ca[0]] -= 1
                    deg[ca[1]] -= 1
                
                # Decide
                if best_pair and best_gain >= 2:
                    # Re-add rem_e? No, we removed it. Add the pair.
                    ca, cb = best_pair
                    adj_masks[ca[0]] |= (1 << ca[1])
                    adj_masks[ca[1]] |= (1 << ca[0])
                    deg[ca[0]] += 1
                    deg[ca[1]] += 1
                    adj_masks[cb[0]] |= (1 << cb[1])
                    adj_masks[cb[1]] |= (1 << cb[0])
                    deg[cb[0]] += 1
                    deg[cb[1]] += 1
                    edges_set.add(ca)
                    edges_set.add(cb)
                    improved = True
                else:
                    # Re-add rem_e
                    adj_masks[ru] |= (1 << rv)
                    adj_masks[rv] |= (1 << ru)
                    deg[ru] += 1
                    deg[rv] += 1
                    edges_set.add(rem_e)
        
        return edges_set

    best_edges = set()
    best_count = 0

    # Strategy 1: Greedy with different orderings
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Create edge ordering: prioritize edges between low-degree and high-degree potential
        # Initially all degree 0, so use random or structural ordering
        order = list(range(total_edges))
        
        if restart == 0:
            # Bipartite K_{7,8} first
            bipartite_edges = set()
            for i in range(7):
                for j in range(7, 15):
                    bipartite_edges.add(edge_index[(i, j)])
            order = list(bipartite_edges) + [e for e in order if e not in bipartite_edges]
        elif restart == 1:
            # Random
            random.shuffle(order)
        else:
            # Weighted random: bias toward edges that connect vertices far apart in index
            weights = []
            for idx in order:
                u, v = all_edges[idx]
                w = abs(u - v)  # prefer long edges
                weights.append(w)
            # Sample without replacement weighted
            remaining = list(order)
            new_order = []
            rem_weights = weights[:]
            while remaining:
                total_w = sum(rem_weights)
                if total_w == 0:
                    new_order.append(remaining[0])
                    remaining.pop(0)
                    rem_weights.pop(0)
                else:
                    r = random.random() * total_w
                    cum = 0
                    for i, w in enumerate(rem_weights):
                        cum += w
                        if r <= cum:
                            new_order.append(remaining[i])
                            remaining.pop(i)
                            rem_weights.pop(i)
                            break
            order = new_order
        
        adj_masks = [0] * n
        deg = [0] * n
        edges_set = greedy_build(order, adj_masks, deg, deadline)
        
        if len(edges_set) > best_count:
            best_count = len(edges_set)
            best_edges = set(edges_set)
        
        # Local search
        if time.time() < deadline - 15 and len(edges_set) > best_count - 3:
            adj_masks2 = adj_masks[:]
            deg2 = deg[:]
            edges_set2 = set(edges_set)
            edges_set2 = local_search(adj_masks2, deg2, edges_set2, deadline, max_iters=500)
            if len(edges_set2) > best_count:
                best_count = len(edges_set2)
                best_edges = set(edges_set2)

    # Strategy 2: Start from dense random and prune
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        # Start with all edges, remove invalid ones
        adj_masks = [0] * n
        deg = [0] * n
        edges_set = set()
        
        # Add all edges
        for idx in range(total_edges):
            u, v = all_edges[idx]
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            edges_set.add((u, v))
        
        # Now remove edges that cause conflicts
        # Iteratively find and remove offending edges
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            if not check_validity_fast(adj_masks, deg):
                # Find a conflict
                for u in range(n):
                    if time.time() > deadline - 2:
                        break
                    for v in range(u + 1, n):
                        if not (adj_masks[u] >> v) & 1:
                            continue
                        su = adj_masks[u] & ~(1 << v)
                        tv = adj_masks[v] & ~(1 << u)
                        s = su
                        found = False
                        while s and not found:
                            sb = s & (-s)
                            si = sb.bit_length() - 1
                            s -= sb
                            t_mask = tv & ~(1 << si)
                            t = t_mask
                            while t and not found:
                                tb = t & (-t)
                                ti = tb.bit_length() - 1
                                t -= tb
                                if deg[si] == deg[ti]:
                                    # Conflict on edge (u,v) with s=si, t=ti
                                    # Remove edge (u,v)
                                    adj_masks[u] &= ~(1 << v)
                                    adj_masks[v] &= ~(1 << u)
                                    deg[u] -= 1
                                    deg[v] -= 1
                                    edges_set.discard((u, v))
                                    edges_set.discard((v, u))
                                    changed = True
                                    found = True
                                    break
                        if found:
                            break
                    if changed:
                        break
        
        if len(edges_set) > best_count:
            best_count = len(edges_set)
            best_edges = set(edges_set)
        
        # Try to add back edges
        if time.time() < deadline - 5:
            order = list(range(total_edges))
            random.shuffle(order)
            edges_set = greedy_build(order, adj_masks, deg, deadline)
            if len(edges_set) > best_count:
                best_count = len(edges_set)
                best_edges = set(edges_set)

    # Final local search on best
    if time.time() < deadline - 3 and best_count > 0:
        adj_masks = [0] * n
        deg = [0] * n
        for (u, v) in best_edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        edges_set = local_search(adj_masks, deg, set(best_edges), deadline, max_iters=1000)
        if len(edges_set) > best_count:
            best_count = len(edges_set)
            best_edges = set(edges_set)

    # Convert to result
    result = []
    for (u, v) in best_edges:
        result.append([int(u), int(v)])
    
    return result
