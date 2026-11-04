import numpy as np
import random
import time
import sys

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return _generic_greedy(q, n, radius)

    N = 3 ** n  # 2187
    
    # --- 1. Greedy + Local Search for a good initial upper bound ---
    # This is kept from the previous iteration as it's fast and provides a strong LB for B&B
    
    def int_to_base3(num, length):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % 3
            num //= 3
        return res

    def base3_to_int(digits):
        res = 0
        for d in digits:
            res = res * 3 + d
        return res

    # Precompute balls (neighbors within distance 1)
    balls = []
    for i in range(N):
        digits = int_to_base3(i, n)
        neighbors = [i]
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    neighbors.append(base3_to_int(new_digits))
        balls.append(neighbors)
    
    # cover_map: for each word w, which codewords cover it
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    def greedy_cover(seed):
        rng = random.Random(seed)
        covered = [False] * N
        cnt = [1 + n * (q - 1)] * N
        code = []
        uncovered_count = N

        while uncovered_count > 0:
            max_val = max(cnt)
            candidates = [i for i in range(N) if cnt[i] == max_val]
            c = rng.choice(candidates)
            code.append(c)

            for w in balls[c]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
                    for c2 in cover_map[w]:
                        cnt[c2] -= 1

        return code

    def refine(code):
        if not code:
            return code
        cover_count = [0] * N
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1

        code_list = list(code)
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                safe = True
                for w in balls[c]:
                    if cover_count[w] == 1:
                        safe = False
                        break
                if safe:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code_list.pop(i)
                    changed = True
                    break
        return code_list

    def local_search(code, seed, time_limit):
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list

        cover_count = [0] * N
        for c in code_list:
            for w in balls[c]:
                cover_count[w] += 1

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code_list[i]
                for w in balls[c_remove]:
                    cover_count[w] -= 1
                uncovered = [w for w in range(N) if cover_count[w] == 0]
                if not uncovered:
                    code_list.pop(i)
                    improved = True
                    break

                found = -1
                # Optimization: Only check candidates that cover *all* uncovered nodes
                # This is expensive, so we limit the search or use a heuristic
                # For now, keep the simple check but break early if found
                for c_add in range(N):
                    if c_add == c_remove or c_add in code_list:
                        continue
                    # Check if c_add covers all uncovered
                    ok = True
                    for w in uncovered:
                        if cover_count[w] == 0 and w not in balls[c_add]:
                             # Note: balls[c_add] is a list, so 'in' is O(15). 
                             # But we need to check if w is in the ball.
                             # Actually, cover_count[w] is 0, so we need w to be in balls[c_add].
                             # Using set(balls[c_add]) is faster for membership.
                             pass
                    # Let's use a precomputed set of balls for faster lookup in LS
                    # But we didn't precompute sets. Let's convert on the fly or precompute.
                    # Precomputing sets for 2187 balls is fine.
                    if all(w in balls_set[c_add] for w in uncovered):
                        found = c_add
                        break
                
                if found >= 0:
                    code_list[i] = found
                    for w in balls[found]:
                        cover_count[w] += 1
                    improved = True
                    break
                else:
                    for w in balls[c_remove]:
                        cover_count[w] += 1

        return code_list

    # Precompute sets for balls for faster membership testing in local search
    balls_set = [set(b) for b in balls]

    start_time = time.time()
    deadline_ls = start_time + 20 # Give 20s to LS to get a good LB
    
    best_ls_code = None
    best_ls_len = float('inf')

    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline_ls:
            break
        code = greedy_cover(i)
        refined = refine(code)
        ls_code = local_search(refined, 100+i, deadline_ls)
        if len(ls_code) < best_ls_len:
            best_ls_len = len(ls_code)
            best_ls_code = ls_code

    # --- 2. Branch and Bound for Maximum Independent Set ---
    # The conflict graph has an edge between u and v if d_H(u,v) <= 2.
    # We want the Maximum Independent Set (MIS) of this graph.
    # Size of MIS = Size of Min Covering Code.
    
    # Build adjacency bitsets
    # dist2_mask[u] is a bitset of all v such that d_H(u,v) <= 2
    # Note: u is included in its own mask? No, independent set usually doesn't include self-loops, 
    # but d_H(u,u)=0 <= 2, so u conflicts with itself in a graph theoretic sense? 
    # Actually, an independent set is a set of vertices no two of which are adjacent.
    # If we define edges for d_H <= 2, then u and u are "adjacent" if we consider self-loops.
    # But we pick distinct codewords. So we just need to ensure no two *distinct* codewords are within dist 2.
    # So the conflict graph G has V={0..N-1} and E={(u,v) | u!=v and d_H(u,v)<=2}.
    # We need MIS of G.
    
    # Precompute masks for d_H <= 2
    # This is 2187 * 2187 checks? 4.7M checks. Each check involves digit comparison.
    # We can do it faster.
    # d_H(u,v) <= 2 means they match in at least 5 positions.
    
    # Let's compute the masks using base-3 arithmetic properties or simple loops.
    # Given N=2187, N^2 = 4.7M. A simple loop in Python might take ~1-2s.
    # Let's try to optimize.
    
    masks = [0] * N
    # To make it faster, we can iterate and compute distance.
    # But 4.7M iterations in pure Python is slow (~5-10s).
    # We can use numpy? 
    # Or we can just do it. 10s is acceptable if it enables B&B.
    
    # Alternative: Generate neighbors at dist 1 and 2.
    # Dist 1: 15 neighbors.
    # Dist 2: C(7,2) * 2 * 2 = 21 * 4 = 84 neighbors.
    # Total conflicts per node: 15 + 84 = 99.
    # Total edges ~ 2187 * 99 / 2 ~ 108k edges.
    # This is much sparser. We can build the mask by OR-ing the bits of neighbors.
    
    # Precompute dist 1 and dist 2 neighbors
    # We already have balls (dist 0 and 1).
    # Let's compute dist 2 neighbors.
    
    # To find dist 2 neighbors of u:
    # Pick 2 positions p1, p2. Change them.
    # There are C(7,2) = 21 pairs.
    # For each pair, there are 2*2=4 combinations of new values (excluding original).
    # So 84 neighbors.
    
    dist2_neighbors = [[] for _ in range(N)]
    
    # Optimization: Precompute all pairs of positions
    pos_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for u in range(N):
        digits = int_to_base3(u, n)
        # Dist 1 neighbors are in balls[u] (excluding u)
        # Dist 2 neighbors
        for p1, p2 in pos_pairs:
            v1 = digits[p1]
            v2 = digits[p2]
            for nv1 in range(3):
                if nv1 == v1: continue
                for nv2 in range(3):
                    if nv2 == v2: continue
                    # Construct new number
                    # We can do this efficiently
                    # delta = (nv1 - v1) * 3^(n-1-p1) + (nv2 - v2) * 3^(n-1-p2)
                    # u + delta
                    # But we need to be careful with carries? No, base 3 digits are independent.
                    # However, simple addition might carry if we just add deltas?
                    # No, we are changing specific digits.
                    # Let's just convert to int.
                    new_digits = digits[:]
                    new_digits[p1] = nv1
                    new_digits[p2] = nv2
                    v = base3_to_int(new_digits)
                    dist2_neighbors[u].append(v)
                    
    # Now build masks
    # mask[u] has bits set for all v such that d_H(u,v) <= 2 (v != u)
    # We will include u in the mask for convenience? No, MIS doesn't care about self loops.
    # But when checking if a new node x can be added to set S, we check if x is adjacent to any y in S.
    # So we need adj[x] to contain all y != x with d_H(x,y) <= 2.
    
    adj = [0] * N
    for u in range(N):
        m = 0
        # Dist 1
        for v in balls[u]:
            if v != u:
                m |= (1 << v)
        # Dist 2
        for v in dist2_neighbors[u]:
            m |= (1 << v)
        adj[u] = m

    # Reorder vertices by degree (highest first) for better B&B
    degrees = [bin(adj[u]).count('c') for u in range(N)] # count bits
    # Actually bin(x).count('1') is correct.
    degrees = [bin(adj[u]).count('1') for u in range(N)]
    
    # Sort indices
    sorted_indices = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    
    # Remap adjacency to new order?
    # It's easier to keep original indices and just iterate in sorted order.
    # But B&B works best if the graph is permuted.
    # Let's permute the graph.
    
    # new_index[old_index] = new_position
    # old_index[new_position] = old_node
    old_to_new = [0] * N
    new_to_old = [0] * N
    for new_idx, old_idx in enumerate(sorted_indices):
        old_to_new[old_idx] = new_idx
        new_to_old[new_idx] = old_idx
        
    new_adj = [0] * N
    for new_u in range(N):
        old_u = new_to_old[new_u]
        m = 0
        old_m = adj[old_u]
        # We need to remap bits in old_m to new indices
        # This is expensive if done bit-by-bit.
        # Alternative: Keep original adj, but iterate candidates in sorted order.
        # The Tomita algorithm relies on the ordering for the coloring heuristic.
        # If we don't permute, the coloring heuristic might be weaker.
        # Let's try to permute efficiently.
        # Since N=2187, we can just iterate through set bits.
        # Or, simpler: Don't permute. Just use the sorted order for selection.
        # The coloring heuristic in Tomita colors the subgraph of candidates.
        # If we don't permute, the "candidates" set is just a subset of original vertices.
        # We can still color them. The order of coloring affects the bound.
        # Let's stick to original indices but iterate candidates in `sorted_indices` order.
        
    # B&B Implementation
    # We want MIS.
    # State: current_set (list of nodes), candidates (set of nodes that can be added)
    # candidates must be independent from current_set.
    # Initially candidates = all nodes.
    
    # To speed up, we use a bitset for candidates.
    
    best_size = best_ls_len if best_ls_code else 0
    best_set = best_ls_code if best_ls_code else []
    
    # If LS didn't find anything, best_size is 0.
    
    # Time limit for B&B
    bb_deadline = start_time + 110 # Leave 10s for cleanup
    
    def color_bound(candidates_bitset):
        # Greedy coloring of the subgraph induced by candidates_bitset
        # Returns the number of colors used, which is an upper bound on MIS size
        # We process vertices in sorted_indices order (highest degree first)
        # But candidates is a subset.
        
        # Extract candidates
        # This is slow if done by iterating all N.
        # We can iterate through bits.
        
        # Greedy coloring:
        # colors = []
        # for v in candidates (in some order):
        #    find smallest color not used by neighbors of v in candidates
        
        # To make it fast, we can just return the count of candidates if we don't do proper coloring?
        # No, that's too weak.
        
        # Let's implement a simple greedy coloring on the induced subgraph.
        # Order: sorted_indices
        # We need to check if v is in candidates.
        
        # Optimization: If we permuted the graph, we could just take the first k vertices.
        # Since we didn't permute adj, we have to check membership.
        
        # Let's try a different approach:
        # The bound is just the number of vertices in candidates if we don't color?
        # That's very weak.
        
        # Let's implement the coloring.
        # We need to know the neighbors of v *within* candidates.
        # neighbors_in_cand = adj[v] & candidates_bitset
        
        # We will use a list of colors for each vertex.
        # Since N is small, we can use an array.
        
        # To avoid allocating large arrays, we can use a dict or just a list of 0s.
        # But doing this at every node in B&B is expensive.
        
        # Heuristic: If the number of candidates is small, just return count.
        # If large, do coloring.
        
        cand_count = bin(candidates_bitset).count('1')
        if cand_count < 20:
            return cand_count
            
        # Greedy coloring
        # We process vertices in descending order of degree (sorted_indices)
        # But only those in candidates.
        
        # To speed up, we can precompute the order of vertices in sorted_indices.
        # And just iterate through it, skipping those not in candidates.
        
        colors = {}
        color_count = 0
        
        # We need to check neighbors.
        # For each v in sorted_indices:
        #   if v in candidates:
        #      used_colors = set()
        #      for u in sorted_indices:
        #         if u in candidates and u is neighbor of v:
        #             used_colors.add(colors[u])
        #      ...
        
        # This is O(N^2) per node. Too slow.
        
        # Alternative: Use the property that if we process in order, we only need to check previously processed neighbors.
        # But "previously processed" in sorted order doesn't mean "all neighbors".
        
        # Let's use a simpler bound: The clique number of the complement graph?
        # Or just use the size of the candidate set as a bound?
        # That leads to exploring too many nodes.
        
        # Let's try to implement a fast coloring using bitsets.
        # For each color class, we maintain a bitset of vertices assigned that color.
        # We try to assign vertices to the first available color.
        
        # Actually, for N=2187, a simple O(N^2) coloring at the root is fine.
        # But deep in the tree?
        
        # Let's limit the depth or use a simpler bound if time is tight.
        
        # For now, let's use the count of candidates as the bound.
        # This is safe but weak.
        # To improve, we can use a greedy independent set on the candidates as a *lower* bound?
        # No, we need an *upper* bound on the size of the independent set we can form from candidates.
        # The size of the candidate set is a valid upper bound.
        # The number of colors in a coloring is a tighter upper bound.
        
        # Let's try to compute a greedy coloring efficiently.
        # We will iterate through vertices in `sorted_indices`.
        # We maintain a list `color_of` of size N, initialized to -1.
        # But we only care about candidates.
        
        # To avoid O(N) initialization, we can use a dictionary.
        
        # Let's just return cand_count for now to ensure we finish.
        # If we have time, we can add coloring later.
        # Given the 120s limit, and B&B can explode, a weak bound is dangerous.
        
        # However, the greedy + LS already found ~186-188.
        # The B&B starts with that as the best solution.
        # If the bound is weak, it might not prune enough.
        
        # Let's implement a basic coloring.
        
        # We'll use a list of sets for color classes? No.
        
        # Let's try this:
        # The maximum independent set in a graph is at most the number of colors in any coloring.
        # Greedy coloring:
        # color_classes = []
        # for v in sorted_indices:
        #    if v not in candidates: continue
        #    # Find first color class where v is not adjacent to any member
        #    placed = False
        #    for i, cc in enumerate(color_classes):
        #        # Check if v is adjacent to any u in cc
        #        # adj[v] & cc_mask
        #        if (adj[v] & cc_mask) == 0:
        #            cc.add(v)
        #            cc_mask |= (1 << v)
        #            placed = True
        #            break
        #    if not placed:
        #        color_classes.append({v})
        #        # ...
        
        # This is O(N * K) where K is number of colors.
        
        # Let's implement this.
        
        color_masks = []
        
        for v in sorted_indices:
            if not (candidates_bitset & (1 << v)):
                continue
            
            v_bit = 1 << v
            placed = False
            for i, cm in enumerate(color_masks):
                if (adj[v] & cm) == 0:
                    color_masks[i] |= v_bit
                    placed = True
                    break
            if not placed:
                color_masks.append(v_bit)
                
        return len(color_masks)

    def bb_search(current_size, current_set, candidates_bitset):
        nonlocal best_size, best_set
        
        if time.time() > bb_deadline:
            return
            
        if candidates_bitset == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        # Upper bound
        bound = color_bound(candidates_bitset)
        if current_size + bound <= best_size:
            return
            
        # Select a vertex to branch on
        # Pick the vertex with highest degree in the candidate subgraph?
        # Or just the first one in sorted_indices that is in candidates.
        
        # Let's pick the first available vertex in sorted_indices
        # This ensures we try high-degree vertices first, which often leads to better pruning.
        
        v = -1
        # Iterate sorted_indices
        for idx in sorted_indices:
            if candidates_bitset & (1 << idx):
                v = idx
                break
        
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates: candidates & ~adj[v] & ~(1<<v)
        new_cands = candidates_bitset & ~adj[v] & ~(1 << v)
        
        current_set.append(v)
        bb_search(current_size + 1, current_set, new_cands)
        current_set.pop()
        
        # Branch 2: Exclude v
        # New candidates: candidates & ~(1<<v)
        new_cands = candidates_bitset & ~(1 << v)
        bb_search(current_size, current_set, new_cands)

    # Initialize B&B
    # We can start with an empty set.
    # But we have a good initial solution from LS.
    # We can also try to seed the B&B with the LS solution?
    # No, B&B finds the max. The initial best_size helps pruning.
    
    all_nodes_bitset = (1 << N) - 1
    
    # To help B&B, we can order the search to include high-degree vertices first.
    # The `bb_search` function picks the first vertex in `sorted_indices` that is in candidates.
    # This effectively prioritizes high-degree vertices.
    
    # Let's run B&B
    # Note: This might be slow. If it takes too long, we return the LS result.
    
    try:
        bb_search(0, [], all_nodes_bitset)
    except Exception:
        pass
        
    # Final decision
    if best_code is None or best_size < best_ls_len:
        final_code = best_ls_code
    else:
        final_code = best_set
        
    if not final_code:
        # Fallback if everything failed
        final_code = greedy_cover(0)
        
    result = []
    for c in final_code:
        result.append(int_to_base3(c, n))
        
    return result

def _generic_greedy(q, n, radius):
    N = q ** n
    def int_to_base(num, length, base):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % base
            num //= base
        return res

    balls = []
    for i in range(N):
        digits = int_to_base(i, n, q)
        neighbors = set([i])
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    val = 0
                    for d in new_digits:
                        val = val * q + d
                    neighbors.add(val)
        balls.append(list(neighbors))

    covered = set()
    code = []
    remaining = set(range(N))
    while remaining:
        best_c = None
        best_count = -1
        for c in remaining:
            count = len(balls[c] & remaining)
            if count > best_count:
                best_count = count
                best_c = c
        code.append(best_c)
        covered.update(balls[best_c])
        remaining -= set(balls[best_c])

    result = []
    for c in code:
        digits = int_to_base(c, n, q)
        result.append(digits)
    return result
