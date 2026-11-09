"""Rewrite for long intervals with no modular inverse pair, kept within the sliding-window family."""
# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    p = 1009
    deadline = t0 + 55

    # Inverses modulo 1009
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)

    # Forbidden elements because x*x % p == 1: 1 and 1008
    forbidden_self = {1, 1008}

    # For each position, the closest earlier position that conflicts with it (if any).
    # conflict[i] is the largest j < i such that inv[i] == j, or -1 if none.
    # Then a window [s, i] is valid iff for every k in [s,i], conflict[k] < s and k not forbidden_self.
    # We can maintain, for each i, the minimum allowed start as max(forbidden positions' lower bound, conflict[k]+1)
    # Simpler: use a two-pointer with a data structure that counts conflicts inside window.
    # But since conflict pairs are unique, a window is invalid iff it contains both x and inv[x] or contains a forbidden self.
    # We'll maintain counts of pairs present.

    # Build a map from value to its inverse partner (same) and pair id
    pair_id = {}
    pair_count = [0] * p  # count of elements from each pair present in window (0,1,2)
    next_pair_id = 0
    for i in range(2, p):
        if i in forbidden_self:
            continue
        j = inv[i]
        if j == i:
            continue
        if i < j:
            pair_id[i] = next_pair_id
            pair_id[j] = next_pair_id
            next_pair_id += 1

    best_len = 0
    best_start = 1

    s = 1
    for e in range(1, p):
        if time.time() > deadline:
            break

        # Try to add e to window
        # If e is forbidden self, we cannot include it; shrink from left until s > e? Actually we just skip e.
        if e in forbidden_self:
            # We cannot include e. Move s past e? No, window [s, e-1] is current. We just don't extend.
            # Actually our window is [s, e-1] before this iteration. Since e is forbidden, we cannot extend to e.
            # We just continue; s stays same.
            continue

        # Add e: if its partner is already in window, we have a conflict.
        pid = pair_id.get(e, None)
        if pid is not None:
            # Check if partner already present
            partner = inv[e]
            # We need to know if partner is in [s, e-1]
            # Since we're adding in order, we can track counts
            # Let's use an array to track if each value is in window
            # But to avoid overhead, we can just check: if partner >= s and partner < e, then conflict
            if s <= partner < e:
                # Conflict: need to move s past the earlier of e and partner
                # Since partner < e, we need s > partner
                new_s = partner + 1
                # Remove elements from old s to new_s - 1
                for x in range(s, new_s):
                    if x in forbidden_self:
                        continue
                    p_id = pair_id.get(x, None)
                    if p_id is not None:
                        pair_count[p_id] -= 1
                s = new_s

        # Now window [s, e] should be valid
        # Add e to pair counts
        if pid is not None:
            pair_count[pid] += 1

        current_len = e - s + 1
        if current_len > best_len:
            best_len = current_len
            best_start = s

    if best_len == 0:
        best_start = 2
        best_len = 1

    return [best_start, best_len]
# EVOLVE-BLOCK-END
