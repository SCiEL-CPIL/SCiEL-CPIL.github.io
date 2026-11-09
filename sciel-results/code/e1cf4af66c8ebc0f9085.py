import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 256
    nums = list(range(1, N + 1))

    # Build conflict adjacency matrix: A[i][j] = 1 if (nums[i]*nums[j]) % (nums[i]+nums[j]) == 0
    # i, j are 0-indexed; nums[i] = i+1
    A = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                A[i, j] = 1
                A[j, i] = 1

    # Compute degrees
    degrees = A.sum(axis=1).astype(np.int32)

    # Spectral ordering: use the principal eigenvector of A to get a "natural" ordering
    # Nodes with larger eigenvector components are "more central" in the conflict graph
    # We'll use this to create a few different orderings for randomized restarts
    try:
        # Use power iteration for speed (A is sparse-ish and small)
        v = np.random.rand(N)
        v /= np.linalg.norm(v)
        for _ in range(100):
            v_new = A.astype(np.float64) @ v
            norm = np.linalg.norm(v_new)
            if norm == 0:
                break
            v = v_new / norm
        # Ordering: sort by eigenvector component (descending = most central first)
        spectral_order = np.argsort(-v)
    except:
        spectral_order = np.arange(N)

    # Bitset representation for fast conflict checking
    # For each node i, conflict_mask[i] is a 256-bit integer where bit j is 1 if i conflicts with j
    # Using Python integers for arbitrary precision bitsets
    conflict_masks = []
    for i in range(N):
        mask = 0
        for j in range(N):
            if A[i, j]:
                mask |= (1 << j)
        conflict_masks.append(mask)

    def can_add_fast(node_idx, current_mask):
        """Check if node_idx can be added given current_mask (bitset of selected nodes).
        Returns True if no conflict."""
        # Check if any selected node conflicts with node_idx
        # conflict_masks[node_idx] has bits set for all nodes conflicting with node_idx
        # current_mask has bits set for all selected nodes
        # If their intersection is non-zero, there's a conflict
        return (conflict_masks[node_idx] & current_mask) == 0

    def greedy_order(order, current_mask=None, current_set=None):
        """Greedy independent set construction following given order."""
        if current_mask is None:
            current_mask = 0
            current_set = []
        for idx in order:
            if can_add_fast(idx, current_mask):
                current_mask |= (1 << idx)
                current_set.append(idx)
        return current_mask, current_set

    def local_search_1flip(mask, node_list, max_iters=200):
        """Try to improve by removing one node and adding others (1-flip local search)."""
        improved = True
        while improved:
            improved = False
            for rem_idx in node_list:
                if not (mask & (1 << rem_idx)):
                    continue
                # Remove rem_idx
                new_mask = mask & ~(1 << rem_idx)
                new_list = [x for x in node_list if x != rem_idx]
                # Try adding all unselected nodes
                for add_idx in range(N):
                    if new_mask & (1 << add_idx):
                        continue
                    if can_add_fast(add_idx, new_mask):
                        new_mask |= (1 << add_idx)
                        new_list.append(add_idx)
                if len(new_list) > len(node_list):
                    mask = new_mask
                    node_list = new_list
                    improved = True
                    break
        return mask, node_list

    # Multiple strategies for ordering
    best_mask = 0
    best_list = []
    best_size = 0

    # Strategy 1: Spectral order (most central first)
    order1 = list(spectral_order)
    m1, l1 = greedy_order(order1)
    if len(l1) > best_size:
        best_size = len(l1)
        best_mask = m1
        best_list = l1

    # Strategy 2: Reverse spectral order (least central first)
    order2 = list(reversed(spectral_order))
    m2, l2 = greedy_order(order2)
    if len(l2) > best_size:
        best_size = len(l2)
        best_mask = m2
        best_list = l2

    # Strategy 3: Degree-ascending order
    order3 = list(np.argsort(degrees))
    m3, l3 = greedy_order(order3)
    if len(l3) > best_size:
        best_size = len(l3)
        best_mask = m3
        best_list = l3

    # Strategy 4: Degree-descending order
    order4 = list(np.argsort(-degrees))
    m4, l4 = greedy_order(order4)
    if len(l4) > best_size:
        best_size = len(l4)
        best_mask = m4
        best_list = l4

    # Local search on best
    best_mask, best_list = local_search_1flip(best_mask, best_list)
    best_size = len(best_list)

    # Randomized restarts: perturb the spectral order
    n_restarts = 2000  # will be limited by time
    for r in range(n_restarts):
        if time.time() > deadline:
            break

        # Create a perturbed ordering
        # Mix of spectral order and random
        base_order = list(spectral_order)
        # Swap random pairs
        n_swaps = random.randint(5, 30)
        for _ in range(n_swaps):
            i, j = random.randint(0, N - 1), random.randint(0, N - 1)
            base_order[i], base_order[j] = base_order[j], base_order[i]

        m, l = greedy_order(base_order)
        if len(l) > best_size:
            best_size = len(l)
            best_mask = m
            best_list = l

        # Also try pure random order occasionally
        if r % 10 == 0:
            rand_order = list(range(N))
            random.shuffle(rand_order)
            mr, lr = greedy_order(rand_order)
            if len(lr) > best_size:
                best_size = len(lr)
                best_mask = mr
                best_list = lr

    # Final local search on best
    if time.time() < deadline:
        best_mask, best_list = local_search_1flip(best_mask, best_list)
        best_size = len(best_list)

    # Convert to actual values (1-indexed)
    result = sorted([x + 1 for x in best_list])

    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break

    if not valid or len(result) == 0:
        # Fallback: all odds
        return list(range(1, N + 1, 2))

    return result
