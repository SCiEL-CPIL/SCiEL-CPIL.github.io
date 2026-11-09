"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Key mathematical insight:
    # For a set S, the condition is: for all a<b<=c in S, a does NOT divide b+c.
    # 
    # Consider the tail T_t = {t, t+1, ..., n}. For a<b<=c all in T_t:
    #   b+c is in [2t, 2n]
    #   a is in [t, n]
    #   a divides b+c means b+c = k*a for some integer k >= 1
    #   Since b+c >= 2t > t >= a (when t > a, but a >= t so b+c > a always for t>=1 and b>=t)
    #   Actually b+c >= 2t and a <= n, so k*a >= 2t means k >= 2t/a.
    #   Since a >= t, 2t/a <= 2. So k >= 2.
    #   k=2: b+c = 2a. Since a < b <= c, we have b >= a+1, c >= b >= a+1.
    #     b+c >= 2a+2 > 2a. Contradiction. So k=2 is impossible.
    #   k=3: b+c = 3a. Need 3a <= 2n, i.e., a <= 2n/3 = 170.67.
    #     Also need b+c >= 2t, so 3a >= 2t, i.e., a >= 2t/3.
    #     For t >= 171: 3a >= 513 > 512 = 2n. Impossible.
    #   k>=4: 4a >= 4t >= 4*171 = 684 > 512. Impossible.
    #   So T_t is valid for t >= 171.
    #
    # For t < 171, we need to check if k=3 conflicts exist.
    #
    # Strategy: Start with a valid large tail, then greedily add smaller elements.
    # Use a fast check function.

    def check_add(S_set, S_sorted, x):
        """
        Check if x can be added to S_set while maintaining the property.
        S_sorted is the sorted list of elements in S_set.
        """
        if not S_sorted:
            return True

        size = len(S_sorted)

        # Role 1: x is 'a' (smallest in triple)
        # Need: for all b,c in S with x < b <= c, x does NOT divide b+c
        b_start = 0
        while b_start < size and S_sorted[b_start] <= x:
            b_start += 1

        for i in range(b_start, size):
            b = S_sorted[i]
            target_residue = (-b) % x
            for j in range(i, size):
                c = S_sorted[j]
                if c % x == target_residue:
                    return False

        # Role 2: x is 'b' (middle)
        # Need: for all a,c in S with a < x <= c, a does NOT divide x+c
        for i in range(size):
            a = S_sorted[i]
            if a >= x:
                break
            for j in range(size):
                c = S_sorted[j]
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False

        # Role 3: x is 'c' (largest)
        # Need: for all a,b in S with a < b <= x, a does NOT divide b+x
        for i in range(size):
            a = S_sorted[i]
            for j in range(i + 1, size):
                b = S_sorted[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False

        return True

    def is_valid_set(S):
        """Check if set S satisfies the property."""
        S_list = sorted(S)
        size = len(S_list)
        for i in range(size):
            a = S_list[i]
            for j in range(i + 1, size):
                b = S_list[j]
                for k in range(j, size):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    best_set = set()
    best_size = 0

    # Strategy 1: Start with valid tails and greedily add smaller elements
    # Tails starting at t >= 171 are automatically valid
    for start_threshold in range(171, 80, -1):
        if time.time() > deadline - 5:
            break

        current_set = set(range(start_threshold, n + 1))
        if not current_set:
            continue

        # Verify initial set is valid (should be for t >= 171, but check anyway)
        if not is_valid_set(current_set):
            continue

        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)

        # Try to add smaller elements in descending order
        candidates = list(range(start_threshold - 1, 0, -1))
        S_sorted = sorted(current_set)
        for x in candidates:
            if time.time() > deadline - 3:
                break
            if check_add(current_set, S_sorted, x):
                current_set.add(x)
                S_sorted = sorted(current_set)
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)

    # Strategy 2: Greedy from scratch - add elements in increasing order if valid
    if time.time() < deadline - 10:
        current_set = set()
        S_sorted = []
        for x in range(1, n + 1):
            if time.time() > deadline - 3:
                break
            if check_add(current_set, S_sorted, x):
                current_set.add(x)
                S_sorted = sorted(current_set)
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)

    # Strategy 3: Multiple random restarts with different starting points
    import random
    if time.time() < deadline - 15:
        for restart in range(50):
            if time.time() > deadline - 3:
                break

            # Random starting threshold
            start_threshold = random.randint(130, 180)
            current_set = set(range(start_threshold, n + 1))
            if not current_set:
                continue

            if not is_valid_set(current_set):
                continue

            S_sorted = sorted(current_set)

            # Add smaller elements in random order
            candidates = list(range(1, start_threshold))
            random.shuffle(candidates)

            for x in candidates:
                if time.time() > deadline - 3:
                    break
                if check_add(current_set, S_sorted, x):
                    current_set.add(x)
                    S_sorted = sorted(current_set)

            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)

    # Strategy 4: Local search - try to improve by swapping elements
    if time.time() < deadline - 10 and best_set:
        best_list = sorted(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_list)):
                if time.time() > deadline - 5:
                    break
                # Try removing best_list[i] and adding a different element
                removed = best_list[i]
                temp_set = set(best_list[:i] + best_list[i+1:])
                S_sorted = sorted(temp_set)

                # Try to add a better element
                for x in range(1, n + 1):
                    if x in temp_set:
                        continue
                    if check_add(temp_set, S_sorted, x):
                        new_set = temp_set | {x}
                        if len(new_set) > len(best_set):
                            best_set = new_set
                            best_list = sorted(best_set)
                            best_size = len(best_set)
                            improved = True
                            break

    # Strategy 5: Try to add more elements to the best set found
    if time.time() < deadline - 5 and best_set:
        S_sorted = sorted(best_set)
        for x in range(1, n + 1):
            if x in best_set:
                continue
            if check_add(best_set, S_sorted, x):
                best_set.add(x)
                S_sorted = sorted(best_set)
                best_size = len(best_set)

    # Final verification
    if best_set and not is_valid_set(best_set):
        # Fallback: use a simple valid set
        fallback = set(range(171, n + 1))
        if is_valid_set(fallback):
            best_set = fallback
        else:
            for t in range(172, n + 1):
                fallback = set(range(t, n + 1))
                if is_valid_set(fallback):
                    best_set = fallback
                    break
            if not best_set:
                best_set = {256}

    return sorted(list(best_set))
# EVOLVE-BLOCK-END
