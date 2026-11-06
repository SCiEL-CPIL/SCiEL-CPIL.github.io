"""Combined: bitmask-accelerated local search with incremental 3-AP deltas and restarts."""
import time
import random

N = 256
DEADLINE = 52.0  # safety margin under 90s kill


def construct():
    start = time.time()
    random.seed(12345)

    # Bitmask: bit (x-1) set means element x is in the set
    # Python ints are arbitrary precision, so 256-bit masks are fine

    def mask_has_4ap(mask, x):
        """Return True if mask ∪ {x} contains a nonconstant 4-AP involving x."""
        # x as 1st term: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (mask >> (x + d - 1)) & 1 and (mask >> (x + 2*d - 1)) & 1 and (mask >> (x + 3*d - 1)) & 1:
                return True
        # x as 2nd term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            if (mask >> (x - d - 1)) & 1 and (mask >> (x + d - 1)) & 1 and (mask >> (x + 2*d - 1)) & 1:
                return True
        # x as 3rd term: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, N - x)
        for d in range(1, max_d + 1):
            if (mask >> (x - 2*d - 1)) & 1 and (mask >> (x - d - 1)) & 1 and (mask >> (x + d - 1)) & 1:
                return True
        # x as 4th term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (mask >> (x - 3*d - 1)) & 1 and (mask >> (x - 2*d - 1)) & 1 and (mask >> (x - d - 1)) & 1:
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
                    # Perturbation: remove a random element
                    if elems:
                        rem = random.choice(elems)
                        rem_contrib = delta_add(mask & ~(1 << (rem - 1)), rem)
                        mask &= ~(1 << (rem - 1))
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

    time_per_restart = max(5.0, (DEADLINE - (time.time() - start)) / 6)
    mask, current_3ap, best_mask, best_3ap = local_search(
        mask, current_3ap, best_mask, best_3ap,
        min(time.time() - start + time_per_restart, DEADLINE)
    )

    # Start 2-6: random restarts with perturbation
    for restart in range(2, 7):
        if time.time() - start > DEADLINE - 5:
            break
        time_per_restart = max(3.0, (DEADLINE - (time.time() - start)) / (7 - restart))
        
        # Create a perturbed version of best_mask
        mask = best_mask
        elems = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        # Remove 3-5 random elements
        num_remove = random.randint(3, 5)
        to_remove = random.sample(elems, min(num_remove, len(elems)))
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
