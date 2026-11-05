import time
import random

_START = time.time()
_DEADLINE = 55.0  # Safety margin before 60s limit

MOD = 101
N = 10
_FULL = 1 << N  # 1024


def _max_matching(left_nodes, adj):
    """Kuhn's algorithm for maximum bipartite matching.
    left_nodes: list of left-side node indices
    adj: dict mapping left node -> list of right nodes it can match to
    Returns (matched_pairs, match_left, match_right)
    """
    match_right = {}  # right -> left
    match_left = {}   # left -> right

    def try_kuhn(v, visited):
        for u in adj.get(v, []):
            if u in visited:
                continue
            visited.add(u)
            if u not in match_right or try_kuhn(match_right[u], visited):
                match_right[u] = v
                match_left[v] = u
                return True
        return False

    # Sort left nodes by degree (ascending) for better performance
    sorted_left = sorted(left_nodes, key=lambda v: len(adj.get(v, [])))

    # Try to match each left node
    for v in sorted_left:
        visited = set()
        try_kuhn(v, visited)

    pairs = list(match_left.items())
    return pairs, match_left, match_right


def construct():
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        # Fallback: return a trivial valid answer (all residues get [1])
        # This is not valid since [1] sums to inv(1)=1, not to arbitrary r.
        # We need a valid fallback. Use the greedy approach as emergency.
        pass

    # 1. Compute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # 2. Precompute sum and popcount for all 1024 masks
    sums = [0] * _FULL
    pop = [0] * _FULL
    for mask in range(1, _FULL):
        lsb = mask & (-mask)
        idx = lsb.bit_length()  # 1-based index
        prev = mask ^ lsb
        sums[mask] = (sums[prev] + inv[idx]) % MOD
        pop[mask] = pop[prev] + 1

    # 3. Build adjacency: for each residue r, which masks (of size <= K) achieve it?
    # We'll do binary search on K.

    # First, check feasibility for a given K
    def feasible(K):
        """Check if all 101 residues can be covered using subsets of size <= K.
        Uses bipartite matching: residues on left, masks on right.
        Each residue can match to any mask that achieves it with popcount <= K.
        """
        # Build adjacency: residue -> list of masks
        adj = {r: [] for r in range(MOD)}
        for mask in range(1, _FULL):
            if pop[mask] <= K:
                r = sums[mask]
                adj[r].append(mask)

        # Check if every residue has at least one mask
        for r in range(MOD):
            if not adj[r]:
                return False, None

        # Maximum bipartite matching
        pairs, match_left, match_right = _max_matching(list(range(MOD)), adj)

        if len(pairs) == MOD:
            return True, match_left
        else:
            return False, None

    # Binary search on K
    lo, hi = 1, N
    best_K = N
    best_match = None

    for K in range(1, N + 1):
        ok, match = feasible(K)
        if ok:
            best_K = K
            best_match = match
            break
        if time.time() - _START > _DEADLINE - 5:
            # Time pressure: fall back to greedy
            break

    if best_match is None:
        # Fallback: greedy approach (same as original)
        min_pop_for_residue = [100] * MOD
        best_mask_for_residue = [0] * MOD
        for mask in range(1, _FULL):
            r = sums[mask]
            p = pop[mask]
            if p < min_pop_for_residue[r]:
                min_pop_for_residue[r] = p
                best_mask_for_residue[r] = mask

        rows = []
        for r in range(MOD):
            mask = best_mask_for_residue[r]
            denoms = []
            for i in range(N):
                if mask & (1 << i):
                    denoms.append(i + 1)
            rows.append(denoms)
        return rows

    # 4. Construct rows from the matching
    rows = []
    for r in range(MOD):
        mask = best_match[r]
        denoms = []
        for i in range(N):
            if mask & (1 << i):
                denoms.append(i + 1)
        rows.append(denoms)

    # 5. Local search refinement: try to reduce row sizes
    # Build a map: residue -> list of all masks that achieve it (with popcount <= best_K)
    residue_to_masks = {r: [] for r in range(MOD)}
    for mask in range(1, _FULL):
        if pop[mask] <= best_K:
            r = sums[mask]
            residue_to_masks[r].append(mask)
    
    # Sort masks in each residue list by popcount (ascending)
    for r in residue_to_masks:
        residue_to_masks[r].sort(key=lambda m: pop[m])
    
    # Track which masks are currently used
    used_masks = set()
    for r in range(MOD):
        mask = best_match[r]
        used_masks.add(mask)
    
    # Try to improve: for each row, try to find a smaller mask that's not used
    improved = True
    max_iter = 100
    while improved and max_iter > 0:
        improved = False
        max_iter -= 1
        for r in range(MOD):
            current_mask = best_match[r]
            current_pop = pop[current_mask]
            # Try to find a mask with smaller popcount
            for candidate in residue_to_masks[r]:
                cand_pop = pop[candidate]
                if cand_pop >= current_pop:
                    break  # Since sorted, no need to continue
                # Check if candidate is not used by another row
                if candidate not in used_masks:
                    # Swap: remove current, add candidate
                    used_masks.discard(current_mask)
                    used_masks.add(candidate)
                    best_match[r] = candidate
                    improved = True
                    break
                # Check if we can swap with the row that uses candidate
                else:
                    # Find which row uses candidate
                    for r2 in range(MOD):
                        if r2 != r and best_match[r2] == candidate:
                            # Check if r2 has an alternative
                            for alt in residue_to_masks[r2]:
                                if alt != current_mask and alt not in used_masks and pop[alt] <= pop[candidate]:
                                    # Do the swap
                                    used_masks.discard(candidate)
                                    used_masks.discard(current_mask)
                                    used_masks.add(alt)
                                    used_masks.add(candidate)
                                    best_match[r] = candidate
                                    best_match[r2] = alt
                                    improved = True
                                    break
                            if improved:
                                break
                            break
            if time.time() - _START > _DEADLINE - 2:
                break
    
    # 6. Construct final rows
    rows = []
    for r in range(MOD):
        mask = best_match[r]
        denoms = []
        for i in range(N):
            if mask & (1 << i):
                denoms.append(i + 1)
        rows.append(denoms)

    return rows
