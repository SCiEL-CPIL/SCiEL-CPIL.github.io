"""Cochromatic partition solver for pinned random graph on 14 vertices."""
import time
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 80  # safety margin before 90s kill

    # Build adjacency as bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    full_mask = (1 << n) - 1

    # Precompute all maximal cliques
    def find_max_cliques():
        """Bron-Kerbosch to find all maximal cliques."""
        max_cliques = []
        
        def bronk(R, P_set, X):
            if P_set == 0 and X == 0:
                if R != 0:
                    max_cliques.append(R)
                return
            # Choose pivot u from P_set | X
            PX = P_set | X
            u_bit = 1
            while u_bit & PX == 0:
                u_bit <<= 1
            u = (u_bit.bit_length() - 1)
            # For each v in P_set \ N(u)
            candidates = P_set & ~adj[u]
            while candidates:
                v_bit = candidates & (-candidates)
                v = v_bit.bit_length() - 1
                candidates ^= v_bit
                bronk(R | v_bit, P_set & adj[v], X & adj[v])
                P_set ^= v_bit
                X |= v_bit
        
        bronk(0, full_mask, 0)
        return max_cliques

    # Precompute all maximal independent sets (cliques in complement)
    def find_max_independent():
        """Find all maximal independent sets = maximal cliques in complement."""
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = full_mask ^ (1 << i) ^ adj[i]
        
        max_indep = []
        
        def bronk(R, P_set, X):
            if P_set == 0 and X == 0:
                if R != 0:
                    max_indep.append(R)
                return
            PX = P_set | X
            u_bit = 1
            while u_bit & PX == 0:
                u_bit <<= 1
            u = (u_bit.bit_length() - 1)
            candidates = P_set & ~comp_adj[u]
            while candidates:
                v_bit = candidates & (-candidates)
                v = v_bit.bit_length() - 1
                candidates ^= v_bit
                bronk(R | v_bit, P_set & comp_adj[v], X & comp_adj[v])
                P_set ^= v_bit
                X |= v_bit
        
        bronk(0, full_mask, 0)
        return max_indep

    max_cliques = find_max_cliques()
    max_indep = find_max_independent()
    
    # All valid parts: maximal cliques + maximal independent sets
    # A valid part must be a clique or independent set. Any subset of a maximal clique is a clique,
    # any subset of a maximal independent set is independent. But for the DP, we only need maximal ones
    # because if a non-maximal S is used, the remaining vertices could have been covered with S' superset.
    # Actually that's not quite right - we need to be careful. Let's use all maximal cliques and max indep sets.
    
    valid_parts = list(set(max_cliques + max_indep))
    # Sort by size descending for better pruning
    valid_parts.sort(key=lambda m: m.bit_count(), reverse=True)
    
    # For each valid part, precompute its size
    part_sizes = [m.bit_count() for m in valid_parts]
    
    # Lower bound: greedy packing
    # For a given mask, a simple lower bound is: ceil(|mask| / max_part_size_in_mask)
    # Better: use a greedy approach to pack and count
    
    # Memoized DP: f(mask) = min parts to cover mask
    memo = {}
    
    def lower_bound(mask):
        """Lower bound on number of parts needed to cover mask."""
        if mask == 0:
            return 0
        # Simple bound: at least 1
        # Better: try to greedily pack and see how many we need
        # Actually for lower bound, we want a minimum.
        # A vertex must be in some part. Each part has size at most the max clique or max indep in mask.
        # Simple: 1 if mask is a clique or independent set, else 2, etc.
        # Use: ceil(popcount(mask) / max_possible_part_size)
        # But we don't know max possible part size easily.
        # Simple lower bound: 1
        # Better: check if mask itself is valid
        if mask in valid_parts_set:
            return 1
        # Check if mask can be covered by 2 parts
        # For now, use a simple bound
        return max(1, (mask.bit_count() + max_part_size - 1) // max_part_size)
    
    # Precompute set of valid parts for O(1) lookup
    valid_parts_set = set(valid_parts)
    
    # Max part size for lower bound
    max_part_size = max(part_sizes) if part_sizes else 1
    
    # For the DP, we need to check if a subset S of mask is a valid part.
    # S is valid if S is a clique or S is an independent set.
    # S is a clique: for all u,v in S, edge(u,v) exists.
    # S is independent: for all u,v in S, no edge(u,v).
    
    def is_clique(mask):
        if mask == 0 or mask == (mask & (mask - 1)):
            return True
        # Check all pairs
        m = mask
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            m ^= v_bit
            if (mask & ~v_bit & ~adj[v]) != 0:
                # Some vertex in mask\{v} is not adjacent to v
                return False
        return True
    
    def is_independent(mask):
        if mask == 0 or mask == (mask & (mask - 1)):
            return True
        m = mask
        while m:
            v_bit = m & (-m)
            v = v_bit.bit_length() - 1
            m ^= v_bit
            if (mask & ~v_bit & adj[v]) != 0:
                # Some vertex in mask\{v} is adjacent to v
                return False
        return True
    
    def is_valid_part(mask):
        return is_clique(mask) or is_independent(mask)
    
    # DP with memoization
    # f(mask) = min over all valid S subset of mask of (1 + f(mask \ S))
    # To make this efficient, we iterate over maximal valid subsets.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def solve_dp(mask):
        """Return minimum number of parts to partition mask, or None if too slow."""
        if time.time() > deadline:
            return None
        if mask == 0:
            return 0
        if mask in memo:
            return memo[mask]
        
        # Lower bound
        lb = 1
        if mask not in valid_parts_set:
            # Try to see if 2 parts suffice (for pruning)
            lb = 2
        
        best = None
        # Iterate over all maximal valid subsets of mask
        # We can iterate over subsets of mask that are valid parts
        # But that's exponential. Instead, iterate over valid_parts and check if subset of mask.
        
        # Better: enumerate subsets of mask that are maximal cliques or independent sets
        # For each maximal clique C, if C & mask != 0, then C & mask is a clique subset of mask
        # Similarly for independent sets
        
        # Actually, let's iterate over all valid parts and check containment
        # This is O(|valid_parts|) per state, which is fine for 14 vertices
        
        for part in valid_parts:
            if part & mask == part:  # part is subset of mask
                remaining = mask ^ part
                sub = solve_dp(remaining)
                if sub is not None:
                    candidate = 1 + sub
                    if best is None or candidate < best:
                        best = candidate
                    if best == lb:
                        break
        # Also need to consider that the optimal part might not be maximal
        # But any valid part S is contained in some maximal clique or maximal independent set.
        # If S is a clique, it's contained in a maximal clique C. Then C & mask is a clique 
        # that contains S. Using C & mask instead of S covers at least as many vertices,
        # so it's never worse. Thus we only need to consider maximal parts.
        
        if best is not None:
            memo[mask] = best
        
        return best
    
    result_count = solve_dp(full_mask)
    
    if result_count is not None and time.time() < deadline - 10:
        # Now reconstruct the actual partition
        def reconstruct(mask, k):
            """Find a partition of mask into k valid parts."""
            if mask == 0:
                return []
            if k == 1:
                if is_valid_part(mask):
                    return [list(range(n))]
                return None
                # Actually need to return the actual vertices
                return [sorted([i for i in range(n) if mask & (1 << i)])]
            
            for part in valid_parts:
                if part & mask == part:
                    remaining = mask ^ part
                    sub = solve_dp(remaining)
                    if sub is not None and sub == k - 1:
                        sub_parts = reconstruct(remaining, k - 1)
                        if sub_parts is not None:
                            part_vertices = sorted([i for i in range(n) if part & (1 << i)])
                            return [part_vertices] + sub_parts
            return None
        
        partition = reconstruct(full_mask, result_count)
        if partition is not None:
            # Validate
            for part in partition:
                if not part:
                    continue
                mask = 0
                for v in part:
                    mask |= (1 << v)
                if not is_valid_part(mask):
                    raise ValueError("Invalid part found")
            return [list(p) for p in partition]
    
    # Fallback: backtracking search
    # Use the same approach as before but with the precomputed info
    deg = [adj[v].bit_count() for v in range(n)]
    order = sorted(range(n), key=lambda v: -deg[v])
    
    def backtrack_search(k):
        """Try to find a valid cochromatic partition with exactly k parts."""
        assignment = [-1] * n
        slot_type = [-1] * k  # -1 undetermined, 0 clique, 1 independent
        slot_members = [0] * k  # bitmask
        
        def is_edge(u, v):
            return bool(adj[u] & (1 << v))
        
        def can_add(v, s):
            """Check if vertex v can be added to slot s."""
            st = slot_type[s]
            members = slot_members[s]
            if members == 0:
                return True, st
            v_bit = 1 << v
            if st == 0:  # clique
                # v must be adjacent to all members
                if (members & ~adj[v]) != 0:
                    return False, st
                return True, st
            elif st == 1:  # independent
                # v must not be adjacent to any member
                if (members & adj[v]) != 0:
                    return False, st
                else:
                    return True, st
            else:  # undetermined
                # Check if adding v forces a type
                has_edge = bool(members & adj[v])
                has_nonedge = bool(members & ~adj[v] & full_mask)
                # More precisely: nonedge means not adjacent
                nonadj = members & ~adj[v] & ~(1 << v)
                has_nonedge = bool(nonadj)
                if has_edge and has_nonedge:
                    return False, st
                if has_edge:
                    return True, 0
                elif has_nonedge:
                    return True, 1
                else:
                    return True, st
        
        def bt(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                parts = []
                for s in range(k):
                    if slot_members[s]:
                        verts = sorted([i for i in range(n) if slot_members[s] & (1 << i)])
                        parts.append(verts)
                return parts
            
            v = order[idx]
            v_bit = 1 << v
            
            if idx == 0:
                slots_to_try = [0]
            elif idx == 1 and k >= 2:
                slots_to_try = [0, 1]
            else:
                slots_to_try = list(range(k))
            
            for s in slots_to_try:
                st_before = slot_type[s]
                ok, new_type = can_add(v, s)
                if not ok:
                    continue
                
                slot_type[s] = new_type
                slot_members[s] |= v_bit
                
                result = bt(idx + 1)
                if result is not None:
                    return result
                
                slot_members[s] ^= v_bit
                slot_type[s] = st_before
            
            return None
        
        return bt(0)
    
    for k in range(2, n + 1):
        if time.time() > deadline - 5:
            break
        result = backtrack_search(k)
        if result is not None:
            return [list(p) for p in result]
    
    # Ultimate fallback: greedy
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial_mask = 0
            for u in part:
                trial_mask |= (1 << u)
            trial_mask |= (1 << v)
            if is_valid_part(trial_mask):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return [list(p) for p in parts]

# EVOLVE-BLOCK-END
