import time
import random
import sys

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    # Precompute all 4-APs
    all_aps = []
    for d in range(1, N):
        max_a = N - 3 * d
        for a in range(max_a):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    n_aps = len(all_aps)

    # For each element, list of AP indices containing it
    elem_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(all_aps):
        for x in ap:
            elem_aps[x].append(idx)

    # For each AP, the other 3 elements (for fast counting)
    # ap_others[ap_idx][pos] = the other 3 elements of that AP
    # Actually, for each element x in AP i, we need the other 3
    # Let's precompute: for each (x, ap_idx) pair, store the other 3 elements
    # Better: for each x, for each ap containing x, store the other 3 as a tuple
    x_ap_others = [[] for _ in range(N)]  # x_ap_others[x] = list of (ap_idx, (o1,o2,o3))
    for idx, (a, b, c, e) in enumerate(all_aps):
        x_ap_others[a].append((idx, (b, c, e)))
        x_ap_others[b].append((idx, (a, c, e)))
        x_ap_others[c].append((idx, (a, b, e)))
        x_ap_others[e].append((idx, (a, b, c)))

    # For each AP, store its 4 elements as a tuple for quick access
    ap_elems = all_aps  # already tuples of 4

    def count_completed_aps_for_x(x, in_set):
        """Count how many 4-APs containing x are fully in in_set."""
        cnt = 0
        for ap_idx, others in x_ap_others[x]:
            o1, o2, o3 = others
            if in_set[o1] and in_set[o2] and in_set[o3]:
                cnt += 1
        return cnt

    def can_add_fast(x, in_set):
        """Check if x can be added without creating a 4-AP."""
        for ap_idx, others in x_ap_others[x]:
            o1, o2, o3 = others
            if in_set[o1] and in_set[o2] and in_set[o3]:
                return False
        return True

    def add_element(x, in_set, conflict_count):
        """Add x to in_set, update conflict_count for all elements in APs with x."""
        in_set[x] = True
        for ap_idx, others in x_ap_others[x]:
            o1, o2, o3 = others
            # For each other element, check if this AP is now completed
            # The AP is completed if all 4 elements are in set
            # We just added x, so check if the other 3 are in set
            if in_set[o1] and in_set[o2] and in_set[o3]:
                # This AP is now completed - increment conflict_count for all 4
                conflict_count[x] += 1
                conflict_count[o1] += 1
                conflict_count[o2] += 1
                conflict_count[o3] += 1

    def remove_element(x, in_set, conflict_count):
        """Remove x from in_set, update conflict_count for all elements in APs with x."""
        for ap_idx, others in x_ap_others[x]:
            o1, o2, o3 = others
            # Check if this AP was completed before removal
            if in_set[o1] and in_set[o2] and in_set[o3]:
                # This AP was completed - decrement conflict_count for all 4
                conflict_count[x] -= 1
                conflict_count[o1] -= 1
                conflict_count[o2] -= 1
                conflict_count[o3] -= 1
        in_set[x] = False

    # Simulated annealing
    def simulated_annealing(initial_in_set, initial_size, initial_conflict_count, time_budget):
        """Run SA starting from initial state. Returns (best_in_set, best_size, best_conflict_count)."""
        in_set = initial_in_set.copy()
        conflict_count = initial_conflict_count.copy()
        current_size = initial_size

        best_in_set = initial_in_set.copy()
        best_size = initial_size
        best_conflict_count = initial_conflict_count.copy()

        T_start = 1.0
        T_end = 0.001
        alpha = 0.9995  # cooling rate

        # Estimate number of steps we can do
        # Each step involves: pick random element, check can_add (O(deg)), maybe add/remove
        # deg is roughly n_aps * 4 / N ≈ 2000 * 4 / 512 ≈ 15-20
        # So each step is ~20-50 operations
        # Let's just run until time budget is up

        T = T_start
        step = 0
        time_limit = time.time() + time_budget

        while time.time() < time_limit:
            step += 1
            # Pick a random element
            x = random.randrange(N)

            if in_set[x]:
                # Try to remove x
                # After removal, we can try to add other elements
                # But for simplicity, just remove
                remove_element(x, in_set, conflict_count)
                current_size -= 1

                # Try to add some random elements that can now be added
                for _ in range(3):
                    y = random.randrange(N)
                    if not in_set[y] and can_add_fast(y, in_set):
                        add_element(y, in_set, conflict_count)
                        current_size += 1

                if current_size > best_size:
                    best_size = current_size
                    best_in_set = in_set.copy()
                    best_conflict_count = conflict_count.copy()
            else:
                # Try to add x
                if can_add_fast(x, in_set):
                    add_element(x, in_set, conflict_count)
                    current_size += 1

                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                        best_conflict_count = conflict_count.copy()
                else:
                    # Can't add x directly. Try a swap: remove a random element that blocks x, then add x
                    # Find a blocker
                    blocker = None
                    for ap_idx, others in x_ap_others[x]:
                        o1, o2, o3 = others
                        if in_set[o1] and in_set[o2] and in_set[o3]:
                            # This AP blocks x. Pick a random one of the others to remove
                            blocker = random.choice([o1, o2, o3])
                            break

                    if blocker is not None and blocker != x:
                        # Remove blocker, then try to add x
                        remove_element(blocker, in_set, conflict_count)
                        current_size -= 1

                        if can_add_fast(x, in_set):
                            add_element(x, in_set, conflict_count)
                            current_size += 1

                            # Try to add the blocker back or other elements
                            if can_add_fast(blocker, in_set):
                                add_element(blocker, in_set, conflict_count)
                                current_size += 1

                            # Try to add a few more random elements
                            for _ in range(2):
                                y = random.randrange(N)
                                if not in_set[y] and can_add_fast(y, in_set):
                                    add_element(y, in_set, conflict_count)
                                    current_size += 1

                            if current_size > best_size:
                                best_size = current_size
                                best_in_set = in_set.copy()
                                best_conflict_count = conflict_count.copy()

            # Cool down
            T *= alpha

        return best_in_set, best_size, best_conflict_count

    # Phase 1: Generate diverse initial solutions using random greedy
    best_in_set = None
    best_size = 0
    best_conflict_count = None

    num_initial = 50
    time_per_initial = (deadline - start_time - 5) / (num_initial + 1)  # reserve time for SA

    for i in range(num_initial):
        if time.time() > deadline - 10:
            break

        in_set = [False] * N
        conflict_count = [0] * N
        size = 0

        # Random greedy: add elements in random order if possible
        order = list(range(N))
        random.shuffle(order)
        for x in order:
            if can_add_fast(x, in_set):
                add_element(x, in_set, conflict_count)
                size += 1

        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            best_conflict_count = conflict_count.copy()

    # Phase 2: Simulated annealing from the best initial solution
    if best_in_set is not None:
        remaining_time = deadline - start_time - 2
        if remaining_time > 5:
            best_in_set, best_size, best_conflict_count = simulated_annealing(
                best_in_set, best_size, best_conflict_count, remaining_time
            )

    # Phase 3: Final greedy fill
    if best_in_set is not None:
        in_set = best_in_set.copy()
        conflict_count = best_conflict_count.copy()
        size = best_size

        # Try to add all remaining elements in order
        for x in range(N):
            if not in_set[x] and can_add_fast(x, in_set):
                add_element(x, in_set, conflict_count)
                size += 1

        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            best_conflict_count = conflict_count.copy()

    # Convert to 1-indexed list
    if best_in_set is None:
        # Fallback: just return a small valid set
        return [1]

    result = [i + 1 for i in range(N) if best_in_set[i]]
    return sorted(result)


# Quick self-test
if __name__ == "__main__":
    import time
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Size: {len(result)}, Time: {t1-t0:.2f}s")
    # Verify no 4-AP
    s = set(result)
    found = False
    for a in range(1, 512):
        for d in range(1, 512):
            if a in s and (a+d) in s and (a+2*d) in s and (a+3*d) in s:
                print(f"Found 4-AP: {a}, {a+d}, {a+2*d}, {a+3*d}")
                found = True
                break
        if found:
            break
    if not found:
        print("Valid: no 4-AP found")
