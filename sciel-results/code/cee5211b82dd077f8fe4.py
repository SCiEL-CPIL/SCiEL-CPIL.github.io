"""Improved program: B&B maximum-clique on comparability graph + local search."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute bit counts
    bit_counts = [0] * total_masks
    for mask in range(1, total_masks):
        bit_counts[mask] = bit_counts[mask & (mask - 1)] + 1

    # Precompute comparability as bytearray
    comp = bytearray(total_masks * total_masks)
    for a in range(total_masks):
        base = a * total_masks
        for b in range(total_masks):
            if (a & b) == a or (a & b) == b:
                comp[base + b] = 1

    # Build adjacency lists (only comparable pairs, excluding self)
    adj = [[] for _ in range(total_masks)]
    for a in range(total_masks):
        base = a * total_masks
        for b in range(total_masks):
            if a != b and comp[base + b]:
                adj[a].append(b)

    # Sort adjacency lists by descending degree (for better B&B ordering)
    for a in range(total_masks):
        adj[a].sort(key=lambda x: -len(adj[x]))

    def calc_score(family):
        score = 0
        L = len(family)
        for i in range(L):
            fi = family[i]
            base = fi * total_masks
            for j in range(i + 1, L):
                if comp[base + family[j]]:
                    score += 1
        return score

    # Build initial solution: chain + greedy fill
    def build_greedy_initial():
        family = []
        used = set()
        # Start with a full chain (13 nested sets)
        chain_masks = [(1 << s) - 1 for s in range(n + 1)]
        for mask in chain_masks:
            family.append(mask)
            used.add(mask)

        remaining_slots = m - len(family)
        for _ in range(remaining_slots):
            best_mask = None
            best_comp_count = -1
            for size in [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 5, 6, 7]:
                candidates = [mask for mask in range(total_masks) if bit_counts[mask] == size and mask not in used]
                random.shuffle(candidates)
                for mask in candidates[:80]:
                    base = mask * total_masks
                    comp_count = 0
                    for f in family:
                        if comp[base + f]:
                            comp_count += 1
                    if comp_count > best_comp_count:
                        best_comp_count = comp_count
                        best_mask = mask
                if best_mask is not None and best_comp_count >= len(family) * 0.6:
                    break

            if best_mask is not None:
                family.append(best_mask)
                used.add(best_mask)
            else:
                all_remaining = [mask for mask in range(total_masks) if mask not in used]
                if all_remaining:
                    random.shuffle(all_remaining)
                    family.append(all_remaining[0])
                    used.add(all_remaining[0])

        return family[:m]

    # Branch-and-bound maximum clique with greedy coloring
    # Returns the largest clique found (as list of node indices)
    def max_clique_bnb(target_size):
        """Find a clique of size >= target_size, or the largest found."""
        best_clique = []
        best_size = 0

        # We use the Bron-Kerbosch variant with coloring bound
        # P = candidate set, X = excluded set, clique = current clique
        # We want to find a clique of size target_size

        # Start with all nodes as candidates
        # Order nodes by descending degree for better pruning

        # Greedy coloring of candidate set for upper bound
        def color_sort(candidates):
            """Greedy coloring to get upper bound on clique size in candidates."""
            # Returns list of (node, color) sorted by color descending
            colored = []
            color_map = {}
            uncolored = list(candidates)
            color_num = 0

            while uncolored:
                color_num += 1
                # Find maximal independent set (greedy)
                independent = []
                remaining = []
                for node in uncolored:
                    if all(node not in adj[i] for i in independent):
                        independent.append(node)
                    else:
                        remaining.append(node)
                for node in independent:
                    color_map[node] = color_num
                    colored.append((node, color_num))
                uncolored = remaining

            # Sort by color descending (higher color = later in ordering)
            colored.sort(key=lambda x: -x[1])
            return colored, color_map

        # B&B search
        # We'll use a recursive approach with iterative deepening
        # Actually, let's use a simpler approach: try to find clique of size target_size

        # Use a stack-based approach to avoid recursion depth issues
        # State: (P_list, X_list, current_clique)
        # But this can be expensive. Let's use a different strategy.

        # Alternative: Use the fact that we want a 40-clique in a 4096-node graph.
        # The comparability graph has special structure. Let's use a heuristic B&B.

        # Greedy approach with backtracking:
        # 1. Order nodes by degree
        # 2. Try to build a clique greedily, backtracking when stuck

        # Actually, let's use a proper B&B with bitsets for speed
        # Represent sets as Python ints (bitsets)

        # Precompute adjacency as bitsets
        adj_bits = [0] * total_masks
        for a in range(total_masks):
            bits = 0
            for b in adj[a]:
                bits |= (1 << b)
            adj_bits[a] = bits

        # B&B with bitset operations
        best_clique_found = []

        def bnb_search(P_bits, X_bits, clique, depth, max_depth):
            """
            P_bits: bitset of candidate vertices
            X_bits: bitset of excluded vertices
            clique: current clique list
            depth: current depth
            max_depth: target clique size
            """
            nonlocal best_clique_found

            if len(clique) >= max_depth:
                if len(clique) > len(best_clique_found):
                    best_clique_found = list(clique)
                return True

            if P_bits == 0:
                return False

            # Upper bound: len(clique) + popcount(P_bits)
            if len(clique) + bin(P_bits).count('1') <= len(best_clique_found):
                return False

            # Greedy coloring bound on P
            # For speed, use a simple bound: popcount(P)
            # Better: greedy coloring

            # Extract nodes from P_bits
            P_nodes = []
            temp = P_bits
            while temp:
                lsb = temp & (-temp)
                node = lsb.bit_length() - 1
                P_nodes.append(node)
                temp ^= lsb

            # Order by descending degree within P
            P_nodes.sort(key=lambda x: -bin(adj_bits[x] & P_bits).count('1'))

            # Try each node in order
            for node in P_nodes:
                if not (P_bits & (1 << node)):
                    continue

                # Add node to clique
                clique.append(node)

                # New candidates: neighbors of node in P
                new_P = P_bits & adj_bits[node]
                # Remove node from P
                P_bits_without = P_bits & ~(1 << node)

                # Check if we can still reach target
                if len(clique) + bin(new_P).count('1') >= max_depth:
                    # Recursive call
                    # To avoid deep recursion, use iterative approach
                    # For now, recursive (Python recursion limit should be ok for depth 40)
                    result = bnb_search(new_P, X_bits | (1 << node), clique, depth + 1, max_depth)
                    if result:
                        # Found a clique of target size
                        if len(clique) > len(best_clique_found):
                            best_clique_found = list(clique)
                        clique.pop()
                        return True

                clique.pop()

                # Move node to X
                X_bits |= (1 << node)
                P_bits = P_bits_without

                # Pruning: if remaining P can't help
                if len(clique) + bin(P_bits).count('1') <= len(best_clique_found):
                    break

            return False

        # Start B&B
        # Initial P: all nodes
        # To make it faster, we can start with a greedy initial clique and use it as lower bound

        # First, build a greedy initial clique
        greedy_clique = []
        remaining = (1 << total_masks) - 1
        while remaining and len(greedy_clique) < m:
            # Pick node with max degree in remaining
            best_node = -1
            best_deg = -1
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                node = lsb.bit_length() - 1
                deg = bin(adj_bits[node] & remaining).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_node = node
                temp ^= lsb

            if best_node == -1:
                break

            greedy_clique.append(best_node)
            # New remaining: intersection of neighbors
            new_remaining = 0
            for node in greedy_clique:
                new_remaining &= adj_bits[node]
            remaining = new_remaining & ~(1 << best_node)

        if len(greedy_clique) >= m:
            best_clique_found = list(greedy_clique)
            return best_clique_found

        # Now run B&B to find a larger clique
        # Use the greedy clique as initial lower bound
        # This B&B might be slow for 4096 nodes. Let's use a time-limited approach.

        # Actually, the B&B above may be too slow. Let's use a simpler heuristic:
        # Start from the greedy clique and try to expand/modify it.

        return best_clique_found

    # Since B&B on 4096 nodes might be too slow, let's use a hybrid approach:
    # 1. Build a good initial solution with greedy
    # 2. Use local search (single swaps, double swaps)
    # 3. Use a limited B&B to try to find a better solution

    random.seed(42)
    family = build_greedy_initial()
    current_score = calc_score(family)
    best_family = list(family)
    best_score = current_score

    # Local search with single and double swaps
    no_improve_count = 0
    max_no_improve = 3000
    restarts = 0
    max_restarts = 12

    family_set = set(family)

    def try_single_swap(fam, fam_set, score):
        """Try all single swaps, return (new_fam, new_fam_set, new_score, improved)."""
        m_local = len(fam)
        best_delta = 0
        best_idx = -1
        best_new_mask = -1

        for idx in range(m_local):
            old_mask = fam[idx]
            old_base = old_mask * total_masks
            for new_mask in range(total_masks):
                if new_mask == old_mask or new_mask in fam_set:
                    continue
                new_base = new_mask * total_masks
                delta = 0
                for j in range(m_local):
                    if j == idx:
                        continue
                    other = fam[j]
                    if comp[old_base + other]:
                        delta -= 1
                    if comp[new_base + other]:
                        delta += 1
                if delta > best_delta:
                    best_delta = delta
                    best_idx = idx
                    best_new_mask = new_mask

        if best_delta > 0 and best_idx >= 0:
            old_mask = fam[best_idx]
            fam_set.remove(old_mask)
            fam_set.add(best_new_mask)
            fam[best_idx] = best_new_mask
            return fam, fam_set, score + best_delta, True
        return fam, fam_set, score, False

    def try_double_swap(fam, fam_set, score):
        """Try double swaps (replace 2 elements with 2 new elements)."""
        m_local = len(fam)
        best_delta = 0
        best_i = -1
        best_j = -1
        best_new_i = -1
        best_new_j = -1

        # Limit the search: only try promising pairs
        # First, find the 2 least-connected members
        comp_counts = [0] * m_local
        for i in range(m_local):
            base = fam[i] * total_masks
            c = 0
            for j in range(m_local):
                if i != j and comp[base + fam[j]]:
                    c += 1
            comp_counts[i] = c

        # Sort indices by comp_count ascending
        sorted_idx = sorted(range(m_local), key=lambda x: comp_counts[x])
        # Try the worst 5 pairs
        candidate_pairs = []
        for a in range(min(5, m_local)):
            for b in range(a + 1, min(5, m_local)):
                candidate_pairs.append((sorted_idx[a], sorted_idx[b]))

        for i, j in candidate_pairs:
            old_mask_i = fam[i]
            old_mask_j = fam[j]
            old_base_i = old_mask_i * total_masks
            old_base_j = old_mask_j * total_masks

            # Try to find 2 new masks that together give better score
            # For speed, try a limited set of new masks
            # The new masks should be comparable with each other and with the rest

            # Get the rest of the family
            rest = [fam[k] for k in range(m_local) if k != i and k != j]

            # For each pair of candidate new masks (comparable with each other),
            # compute the delta
            # To limit, try masks that are comparable with many rest members

            # Precompute for each rest member which masks are comparable
            # This is expensive. Let's just try a random sample.

            # Actually, let's try: for each new_mask_i in a sample, find best new_mask_j
            sample_size = 200
            candidates = list(range(total_masks))
            random.shuffle(candidates)
            candidates = candidates[:sample_size]

            for new_mask_i in candidates:
                if new_mask_i in fam_set or new_mask_i == old_mask_i or new_mask_i == old_mask_j:
                    continue
                new_base_i = new_mask_i * total_masks

                # Check if new_mask_i is comparable with old_mask_j (to allow swap)
                # Actually, we're replacing both, so no constraint

                # Compute contribution of new_mask_i to rest
                delta_i = 0
                for other in rest:
                    if comp[new_base_i + other]:
                        delta_i += 1
                    if comp[old_base_i + other]:
                        delta_i -= 1
                    if comp[old_base_j + other]:
                        delta_i -= 1

                # Find best new_mask_j
                for new_mask_j in candidates:
                    if new_mask_j == new_mask_i or new_mask_j in fam_set or new_mask_j == old_mask_i or new_mask_j == old_mask_j:
                        continue
                    # Check comparability between new_mask_i and new_mask_j
                    if not comp[new_base_i + new_mask_j]:
                        continue

                    new_base_j = new_mask_j * total_masks
                    delta_j = 0
                    for other in rest:
                        if comp[new_base_j + other]:
                            delta_j += 1

                    total_delta = delta_i + delta_j
                    # Add the comparability between new_mask_i and new_mask_j
                    # Already counted? No, we need to add it once
                    # Actually, in the original, old_mask_i and old_mask_j might be comparable
                    if comp[old_base_i + old_mask_j]:
                        total_delta -= 1
                    # And we add the new comparability
                    total_delta += 1  # comp[new_base_i + new_mask_j] is True

                    if total_delta > best_delta:
                        best_delta = total_delta
                        best_i = i
                        best_j = j
                        best_new_i = new_mask_i
                        best_new_j = new_mask_j

        if best_delta > 0 and best_i >= 0:
            fam_set.remove(fam[best_i])
            fam_set.remove(fam[best_j])
            fam_set.add(best_new_i)
            fam_set.add(best_new_j)
            fam[best_i] = best_new_i
            fam[best_j] = best_new_j
            return fam, fam_set, score + best_delta, True
        return fam, fam_set, score, False

    while time.time() < deadline:
        if no_improve_count >= max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break
            random.seed(42 + restarts * 1000)
            family = build_greedy_initial()
            family_set = set(family)
            current_score = calc_score(family)
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
            no_improve_count = 0
            continue

        # Alternate between single and double swaps (bias toward single for speed)
        if random.random() < 0.8:
            family, family_set, current_score, improved = try_single_swap(family, family_set, current_score)
        else:
            family, family_set, current_score, improved = try_double_swap(family, family_set, current_score)

        if improved:
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
        else:
            no_improve_count += 1

    # Final polish: exhaustive single-swap pass (iterate only unassigned masks)
    if time.time() < deadline - 3:
        family_set = set(best_family)
        # Precompute list of all masks not in family_set for faster iteration
        unassigned = [mask for mask in range(total_masks) if mask not in family_set]
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            best_delta = 0
            best_idx = -1
            best_new_mask = -1
            for idx in range(m):
                old_mask = best_family[idx]
                old_base = old_mask * total_masks
                for new_mask in unassigned:
                    if new_mask == old_mask:
                        continue
                    new_base = new_mask * total_masks
                    delta = 0
                    for j in range(m):
                        if j == idx:
                            continue
                        other = best_family[j]
                        if comp[old_base + other]:
                            delta -= 1
                        if comp[new_base + other]:
                            delta += 1
                    if delta > best_delta:
                        best_delta = delta
                        best_idx = idx
                        best_new_mask = new_mask
            if best_delta > 0 and best_idx >= 0:
                old = best_family[best_idx]
                family_set.remove(old)
                family_set.add(best_new_mask)
                best_family[best_idx] = best_new_mask
                # Update unassigned list
                unassigned.remove(old)
                unassigned.append(best_new_mask)
                best_score += best_delta
                improved = True

    # Convert to lists of integers
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)

    return result
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
