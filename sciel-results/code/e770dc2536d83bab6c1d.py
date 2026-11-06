"""Bitmask-accelerated local search for 4-AP-free sets maximizing 3-APs."""
import time
import random

N = 256
DEADLINE = 85.0  # safety margin under 90s kill


def construct():
    start = time.time()
    random.seed(12345)

    # Precompute 4-AP conflict masks for each x
    # For each x, a list of 256-bit masks where each mask has bits at the 3 other positions of a 4-AP containing x
    conflict_masks = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        seen = set()
        # x as 1st term: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            other = (x + d, x + 2 * d, x + 3 * d)
            key = (other[0], other[1], other[2])
            if key not in seen:
                seen.add(key)
                cm = (1 << (other[0] - 1)) | (1 << (other[1] - 1)) | (1 << (other[2] - 1))
                conflict_masks[x].append(cm)
        # x as 2nd term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            other = (x - d, x + d, x + 2 * d)
            key = (other[0], other[1], other[2])
            if key not in seen:
                seen.add(key)
                cm = (1 << (other[0] - 1)) | (1 << (other[1] - 1)) | (1 << (other[2] - 1))
                conflict_masks[x].append(cm)
        # x as 3rd term: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            other = (x - 2 * d, x - d, x + d)
            key = (other[0], other[1], other[2])
            if key not in seen:
                seen.add(key)
                cm = (1 << (other[0] - 1)) | (1 << (other[1] - 1)) | (1 << (other[2] - 1))
                conflict_masks[x].append(cm)
        # x as 4th term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            other = (x - 3 * d, x - 2 * d, x - d)
            key = (other[0], other[1], other[2])
            if key not in seen:
                seen.add(key)
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
            if (mask >> (x + d - 1)) & 1 and (mask >> (x + 2 * d - 1)) & 1:
                cnt += 1
        # x as middle: x-d, x, x+d
        max_d = min(x - 1, N - x)
        for d in range(1, max_d + 1):
            if (mask >> (x - d - 1)) & 1 and (mask >> (x + d - 1)) & 1:
                cnt += 1
        # x as last: x-2d, x-d, x
        max_d = (x - 1) // 2
        for d in range(1, max_d + 1):
            if (mask >> (x - 2 * d - 1)) & 1 and (mask >> (x - d - 1)) & 1:
                cnt += 1
        return cnt

    def delta_remove(mask, x):
        """Number of 3-APs destroyed by removing x from mask (x in mask)."""
        return delta_add(mask & ~(1 << (x - 1)), x)

    def count_3aps(mask):
        """Total number of distinct 3-term APs in mask."""
        total = 0
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                total += delta_remove(mask, x)
        return total

    def greedy_seed(order=None):
        """Greedy construction: add elements in given order if no 4-AP is formed."""
        if order is None:
            order = list(range(1, N + 1))
        m = 0
        for x in order:
            if not mask_has_4ap(m, x):
                m |= (1 << (x - 1))
        return m

    def local_search(mask, current_3ap, time_budget):
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
            found_improvement = False
            for rem in elems:
                if time.time() - start > time_budget:
                    break
                rem_contrib = delta_remove(mask, rem)
                for add in missing:
                    if time.time() - start > time_budget:
                        break
                    mask_without_rem = mask & ~(1 << (rem - 1))
                    if mask_has_4ap(mask_without_rem, add):
                        continue
                    add_contrib = delta_add(mask_without_rem, add)
                    delta = add_contrib - rem_contrib
                    if delta > 0:
                        mask = mask_without_rem | (1 << (add - 1))
                        current_3ap += delta
                        found_improvement = True
                        break
                if found_improvement:
                    break

            if found_improvement:
                continue

            # Try single additions
            found_add = False
            for add in missing:
                if time.time() - start > time_budget:
                    break
                if mask_has_4ap(mask, add):
                    continue
                delta = delta_add(mask, add)
                if delta > 0:
                    mask |= (1 << (add - 1))
                    current_3ap += delta
                    found_add = True
                    break

            if found_add:
                continue

            # Try single removals that allow subsequent additions
            # (handled by perturbation below)

            # Perturbation: remove 2-4 random elements to escape local optimum
            if len(elems) >= 2:
                num_remove = random.randint(2, 4)
                to_remove = random.sample(elems, min(num_remove, len(elems)))
                for x in to_remove:
                    rem_contrib = delta_remove(mask, x)
                    mask &= ~(1 << (x - 1))
                    current_3ap -= rem_contrib

        return mask, current_3ap

    # Multi-start local search
    best_mask = 0
    best_3ap = 0

    # Start 1: greedy in natural order
    mask = greedy_seed()
    current_3ap = count_3aps(mask)
    if current_3ap > best_3ap:
        best_3ap = current_3ap
        best_mask = mask

    # Start 2: greedy in reverse order
    mask = greedy_seed(list(range(N, 0, -1)))
    current_3ap = count_3aps(mask)
    if current_3ap > best_3ap:
        best_3ap = current_3ap
        best_mask = mask

    # Start 3: greedy with random order
    random.seed(42)
    order = list(range(1, N + 1))
    random.shuffle(order)
    mask = greedy_seed(order)
    current_3ap = count_3aps(mask)
    if current_3ap > best_3ap:
        best_3ap = current_3ap
        best_mask = mask

    # Start 4: another random order
    random.seed(99)
    order = list(range(1, N + 1))
    random.shuffle(order)
    mask = greedy_seed(order)
    current_3ap = count_3aps(mask)
    if current_3ap > best_3ap:
        best_3ap = current_3ap
        best_mask = mask

    # Run local search on best seed
    remaining = DEADLINE - (time.time() - start)
    if remaining > 3:
        time_per_restart = max(5.0, remaining / 6)
        mask, current_3ap = local_search(
            best_mask, best_3ap,
            min(time.time() - start + time_per_restart, DEADLINE)
        )
        if current_3ap > best_3ap:
            best_3ap = current_3ap
            best_mask = mask

    # More restarts with perturbation
    for restart in range(2, 10):
        if time.time() - start > DEADLINE - 5:
            break
        remaining = DEADLINE - (time.time() - start)
        time_per_restart = max(4.0, remaining / (11 - restart))

        # Create a perturbed version of best_mask
        mask = best_mask
        elems = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        # Remove 3-7 random elements
        num_remove = random.randint(3, 7)
        if len(elems) >= num_remove:
            to_remove = random.sample(elems, num_remove)
            for x in to_remove:
                mask &= ~(1 << (x - 1))

        current_3ap = count_3aps(mask)
        mask, current_3ap = local_search(
            mask, current_3ap,
            min(time.time() - start + time_per_restart, DEADLINE)
        )
        if current_3ap > best_3ap:
            best_3ap = current_3ap
            best_mask = mask

    # Final polish: run local search on best_mask with remaining time
    if time.time() - start < DEADLINE - 2:
        current_3ap = count_3aps(best_mask)
        mask, current_3ap = local_search(
            best_mask, current_3ap, DEADLINE
        )
        if current_3ap > best_3ap:
            best_3ap = current_3ap
            best_mask = mask

    # Convert best_mask to sorted list
    result = [x for x in range(1, N + 1) if (best_mask >> (x - 1)) & 1]
    return sorted(result)
