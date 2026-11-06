"""Improved: bitmask-accelerated local search with smarter neighborhood and more restarts."""
import time
import random

N = 256
DEADLINE = 55.0  # safety margin under 90s kill


def construct():
    start = time.time()
    random.seed(12345)

    # Precompute 4-AP conflicts: for each x, list of masks of 3 elements that with x form a 4-AP
    # Actually, precompute for each x, the set of pairs (y,z) such that {x,y,z,w} is a 4-AP
    # Better: precompute for each x, list of (d, role) where adding x creates a 4-AP if certain elements present

    # Precompute all 4-APs in [1,256]
    # A 4-AP is (a, a+d, a+2d, a+3d) with d > 0
    # For each element x, precompute which other elements would form a 4-AP with x
    # conf[x] = list of (y, z, w) such that {x,y,z,w} is a 4-AP

    # Actually, let's precompute a conflict structure:
    # For each x, and for each possible d, the 3 other elements in a 4-AP containing x
    # When checking if adding x creates a 4-AP, we need to check if any 3 of those are in the mask

    # Precompute for each x: list of triples (i, j, k) where {x, i, j, k} forms a 4-AP
    # This is expensive to store, so let's use a different approach

    # Precompute: for each x, a list of (y, z, w) tuples
    # But this could be large. Let's estimate: for x in [1,256], the number of 4-APs containing x
    # x as 1st: d from 1 to (256-x)//3, so ~85 max
    # x as 2nd: d from 1 to min(x-1, (256-x)//2), so ~64 max
    # x as 3rd: d from 1 to min((x-1)//2, 256-x), so ~64 max
    # x as 4th: d from 1 to (x-1)//3, so ~85 max
    # Total per x: ~200-300, times 256 = ~50k-77k triples. That's manageable.

    # Let's precompute: for each x, list of (y, z, w) where {x,y,z,w} is a 4-AP, y < z < w
    # And for each such triple, we need to check if all three are in the mask

    # Actually, for efficiency, let's precompute bitmasks for each x:
    # For each x, a list of 256-bit masks, where each mask has bits set at the 3 other positions
    # Then checking is: (mask & conflict_mask) == conflict_mask

    # Let's precompute this
    conflict_masks = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        seen = set()
        # x as 1st term: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            other = (x + d, x + 2*d, x + 3*d)
            if other not in seen:
                seen.add(other)
                cm = (1 << (other[0] - 1)) | (1 << (other[1] - 1)) | (1 << (other[2] - 1))
                conflict_masks[x].append(cm)
        # x as 2nd term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            other = (x - d, x + d, x + 2*d)
            if other not in seen:
                seen.add(other)
                cm = (1 << (other[0] - 1)) | (1 << (other[1] - 1)) | (1 << (other[2] - 1))
                conflict_masks[x].append(cm)
        # x as 3rd term: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            other = (x - 2*d, x - d, x + d)
            if other not in seen:
                seen.add(other)
                cm = (1 << (other[0] - 1)) | (1 << (other[1] - 1)) | (1 << (other[2] - 1))
                conflict_masks[x].append(cm)
        # x as 4th term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            other = (x - 3*d, x - 2*d, x - d)
            if other not in seen:
                seen.add(other)
                cm = (1 << (other[0] - 1)) | (1 << (other[1] - 1)) | (1 << (other[2] - 1))
                conflict_masks[x].append(cm)

    def mask_has_4ap(mask, x):
        """Return True if mask ∪ {x} contains a nonconstant 4-AP involving x."""
        for cm in conflict_masks[x]:
            if (mask & cm) == cm:
                return True
        return False

    def delta_add(mask, x):
        """Number of new 3-APs created by adding x to mask (x not in mask)."""
        cnt = 0
        # x as first: x, x+d, x+2d
        max_d = (N - x) // 2
        for d in range(1, max_d + 1):
            if (mask >> (x + d - 1)) & 1 and (mask >> (x + 2*d - 1)) & 1:
                cnt += 1
        # x as middle: x-d, x, x+d
        max_d = min(x - 1, N - x)
        for d in range(1, max_d + 1):
            if (mask >> (x - d - 1)) & 1 and (mask >> (x + d - 1)) & 1:
                cnt += 1
        # x as last: x-2d, x-d, x
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if (mask >> (x - 2*d - 1)) & 1 and (mask >> (x - d - 1)) & 1:
                cnt += 1
        return cnt

    def count_3aps(mask):
        """Total number of distinct 3-term APs in mask."""
        total = 0
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                m = mask & ~(1 << (x - 1))
                total += delta_add(m, x)
        return total

    def greedy_seed():
        """Greedy construction: add elements in order if no 4-AP is formed."""
        m = 0
        for x in range(1, N + 1):
            if not mask_has_4ap(m, x):
                m |= (1 << (x - 1))
        return m

    def local_search(mask, current_3ap, best_mask, best_3ap, time_budget):
        """Run local search until no improvement or time budget exhausted."""
        while time.time() - start < time_budget:
            improved = False
            elems = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            missing = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]

            if not elems or not missing:
                break

            random.shuffle(elems)
            random.shuffle(missing)

            # Try swaps: remove one, add another
            for rem in elems:
                if time.time() - start > time_budget:
                    break
                rem_contrib = delta_add(mask & ~(1 << (rem - 1)), rem)
                for add in missing:
                    if time.time() - start > time_budget:
                        break
                    # Check 4-AP for add in context of mask without rem
                    if mask_has_4ap(mask & ~(1 << (rem - 1)), add):
                        continue
                    add_contrib = delta_add(mask & ~(1 << (rem - 1)), add)
                    delta = add_contrib - rem_contrib
                    if delta > 0:
                        mask = (mask & ~(1 << (rem - 1))) | (1 << (add - 1))
                        current_3ap += delta
                        if current_3ap > best_3ap:
                            best_3ap = current_3ap
                            best_mask = mask
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                # Try single additions
                found = False
                for add in missing:
                    if time.time() - start > time_budget:
                        break
                    if mask_has_4ap(mask, add):
                        continue
                    delta = delta_add(mask, add)
                    if delta > 0:
                        mask |= (1 << (add - 1))
                        current_3ap += delta
                        if current_3ap > best_3ap:
                            best_3ap = current_3ap
                            best_mask = mask
                        found = True
                        break
                if not found:
                    # Perturbation: remove 2-3 random elements to escape local optimum
                    if len(elems) >= 2:
                        num_remove = random.randint(2, 3)
                        to_remove = random.sample(elems, min(num_remove, len(elems)))
                        for x in to_remove:
                            rem_contrib = delta_add(mask & ~(1 << (x - 1)), x)
                            mask &= ~(1 << (x - 1))
                            current_3ap -= rem_contrib

        return mask, current_3ap, best_mask, best_3ap

    # Multi-start local search
    best_mask = 0
    best_3ap = 0

    # Start 1: greedy seed
    mask = greedy_seed()
    current_3ap = count_3aps(mask)
    if current_3ap > best_3ap:
        best_3ap = current_3ap
        best_mask = mask

    # Start 2: greedy in reverse order
    mask = 0
    for x in range(N, 0, -1):
        if not mask_has_4ap(mask, x):
            mask |= (1 << (x - 1))
    current_3ap = count_3aps(mask)
    if current_3ap > best_3ap:
        best_3ap = current_3ap
        best_mask = mask

    # Start 3: greedy with random order
    random.seed(42)
    order = list(range(1, N + 1))
    random.shuffle(order)
    mask = 0
    for x in order:
        if not mask_has_4ap(mask, x):
            mask |= (1 << (x - 1))
    current_3ap = count_3aps(mask)
    if current_3ap > best_3ap:
        best_3ap = current_3ap
        best_mask = mask

    # Run local search on best seed
    remaining = DEADLINE - (time.time() - start)
    if remaining > 3:
        time_per_restart = max(4.0, remaining / 5)
        mask, current_3ap, best_mask, best_3ap = local_search(
            best_mask, best_3ap, best_mask, best_3ap,
            min(time.time() - start + time_per_restart, DEADLINE)
        )

    # More restarts with perturbation
    for restart in range(2, 8):
        if time.time() - start > DEADLINE - 4:
            break
        remaining = DEADLINE - (time.time() - start)
        time_per_restart = max(3.0, remaining / (9 - restart))
        
        # Create a perturbed version of best_mask
        mask = best_mask
        elems = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        # Remove 4-6 random elements
        num_remove = random.randint(4, 6)
        if len(elems) >= num_remove:
            to_remove = random.sample(elems, num_remove)
            for x in to_remove:
                mask &= ~(1 << (x - 1))
        
        current_3ap = count_3aps(mask)
        mask, current_3ap, best_mask, best_3ap = local_search(
            mask, current_3ap, best_mask, best_3ap,
            min(time.time() - start + time_per_restart, DEADLINE)
        )

    # Final polish: run local search on best_mask with remaining time
    if time.time() - start < DEADLINE - 2:
        current_3ap = count_3aps(best_mask)
        mask, current_3ap, best_mask, best_3ap = local_search(
            best_mask, current_3ap, best_mask, best_3ap, DEADLINE
        )

    # Convert best_mask to sorted list
    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    return sorted(result)
