import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    N = 1000
    MAX_SUM = 2 * N  # 2000
    best_set = None
    best_size = 0

    def can_add(x, chosen_arr, counts):
        if counts[2 * x] >= 2:
            return False
        for y in range(1, N + 1):
            if not chosen_arr[y]:
                continue
            s = x + y
            if counts[s] >= 2:
                return False
        return True

    def add_element(x, chosen_arr, counts):
        chosen_arr[x] = 1
        counts[2 * x] += 1
        for y in range(1, N + 1):
            if chosen_arr[y]:
                counts[x + y] += 1

    def remove_element(x, chosen_arr, counts):
        chosen_arr[x] = 0
        counts[2 * x] -= 1
        for y in range(1, N + 1):
            if chosen_arr[y]:
                counts[x + y] -= 1

    def greedy_construction():
        chosen_arr = [0] * (N + 1)
        counts = [0] * (MAX_SUM + 1)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)

        for x in candidates:
            if time.time() > deadline:
                break
            # Fast check: only look at sums that would exceed limit
            if counts[2 * x] >= 2:
                continue
            ok = True
            for y in range(1, N + 1):
                if chosen_arr[y]:
                    if counts[x + y] >= 2:
                        ok = False
                        break
            if ok:
                add_element(x, chosen_arr, counts)

        return chosen_arr, counts

    def local_search(chosen_arr, counts):
        chosen_arr = list(chosen_arr)
        counts = list(counts)
        chosen_set = set(i for i in range(1, N + 1) if chosen_arr[i])
        unchosen = set(range(1, N + 1)) - chosen_set

        improved = True
        passes = 0
        max_passes = 8

        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1

            # Phase 1: Try adding any unchosen element
            unchosen_list = list(unchosen)
            random.shuffle(unchosen_list)

            for x in unchosen_list:
                if time.time() > deadline:
                    break
                if x not in unchosen:
                    continue
                if can_add(x, chosen_arr, counts):
                    add_element(x, chosen_arr, counts)
                    unchosen.remove(x)
                    chosen_set.add(x)
                    improved = True

            if time.time() > deadline:
                break

            # Phase 2: Remove-one-add-two (net +1 growth)
            unchosen_list2 = list(unchosen)
            random.shuffle(unchosen_list2)

            for x1 in unchosen_list2:
                if time.time() > deadline:
                    break
                if x1 not in unchosen:
                    continue

                # Try to find a y to remove such that we can add x1 and another element
                if not chosen_set:
                    continue

                chosen_list = list(chosen_set)
                random.shuffle(chosen_list)

                for y in chosen_list:
                    if time.time() > deadline:
                        break

                    # Temporarily remove y
                    remove_element(y, chosen_arr, counts)

                    # Check if x1 can be added
                    if can_add(x1, chosen_arr, counts):
                        add_element(x1, chosen_arr, counts)
                        unchosen.remove(x1)
                        chosen_set.add(x1)

                        # Now try to add another element z
                        improved2 = False
                        remaining = list(unchosen)
                        random.shuffle(remaining)
                        for z in remaining:
                            if time.time() > deadline:
                                break
                            if can_add(z, chosen_arr, counts):
                                add_element(z, chosen_arr, counts)
                                unchosen.remove(z)
                                chosen_set.add(z)
                                improved2 = True
                                break

                        if improved2:
                            # Net +1: removed y, added x1 and z
                            unchosen.add(y)
                            chosen_set.discard(y)
                            improved = True
                        else:
                            # Net 0: removed y, added x1, put y back
                            remove_element(x1, chosen_arr, counts)
                            unchosen.add(x1)
                            chosen_set.discard(x1)
                            add_element(y, chosen_arr, counts)
                            chosen_set.add(y)
                    else:
                        # x1 can't be added, restore y
                        add_element(y, chosen_arr, counts)
                    break  # Only try first y for this x1 to save time

            # Phase 3: 1-for-1 swaps
            if time.time() > deadline:
                break

            unchosen_list3 = list(unchosen)
            random.shuffle(unchosen_list3)

            for x in unchosen_list3:
                if time.time() > deadline:
                    break
                if x not in unchosen:
                    continue

                if not chosen_set:
                    continue

                chosen_list = list(chosen_set)
                random.shuffle(chosen_list)

                for y in chosen_list:
                    if time.time() > deadline:
                        break

                    remove_element(y, chosen_arr, counts)

                    if can_add(x, chosen_arr, counts):
                        add_element(x, chosen_arr, counts)
                        unchosen.remove(x)
                        unchosen.add(y)
                        chosen_set.add(x)
                        chosen_set.discard(y)
                        improved = True
                        break
                    else:
                        add_element(y, chosen_arr, counts)
                    break

        return chosen_arr, counts

    for restart in range(200):
        if time.time() > deadline:
            break

        chosen_arr, counts = greedy_construction()
        if time.time() < deadline:
            chosen_arr, counts = local_search(chosen_arr, counts)

        current_size = sum(chosen_arr[1:])
        if current_size > best_size:
            best_size = current_size
            best_set = [i for i in range(1, N + 1) if chosen_arr[i]]

    if best_set is None:
        best_set = []

    return best_set

random.seed(42)
