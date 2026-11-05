import random
import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin for 30s limit

    def build_used_sums(A):
        used_sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                used_sums.add(A[i] + A[j])
        return used_sums

    def is_sidon_compatible(x, A, used_sums):
        if 2 * x in used_sums:
            return False
        for y in A:
            if x + y in used_sums:
                return False
        return True

    def get_newly_blocked(x, A, used_sums, unblocked):
        new_used_sums = set(used_sums)
        new_used_sums.add(2 * x)
        for y in A:
            new_used_sums.add(x + y)

        newly_blocked = []
        for z in unblocked:
            if z == x:
                continue
            if 2 * z in new_used_sums:
                newly_blocked.append(z)
                continue
            blocked = False
            for y in A:
                if z + y in new_used_sums:
                    blocked = True
                    break
            if not blocked and z + x in new_used_sums:
                blocked = True
            if blocked:
                newly_blocked.append(z)
        return newly_blocked

    def is_maximal(A):
        """Check if A is inclusion-maximal Sidon in [1,N]."""
        used_sums = build_used_sums(A)
        A_set = set(A)
        for x in range(1, N + 1):
            if x in A_set:
                continue
            # Check if x can be added
            if 2 * x in used_sums:
                continue
            compatible = True
            for y in A:
                if x + y in used_sums:
                    compatible = False
                    break
            if compatible:
                return False
        return True

    def greedy_max_coverage(init_A=None, randomize_ties=False, bias=0.0):
        if init_A is None:
            A = []
        else:
            A = list(init_A)

        used_sums = build_used_sums(A)
        A_set = set(A)

        unblocked = set(range(1, N + 1)) - A_set

        while unblocked:
            if time.time() > deadline:
                break

            best_x = None
            best_score = -1
            best_newly_blocked = []

            for x in range(1, N + 1):
                if x in A_set:
                    continue
                if not is_sidon_compatible(x, A, used_sums):
                    continue

                newly_blocked = get_newly_blocked(x, A, used_sums, unblocked)
                score = len(newly_blocked)

                if bias != 0.0:
                    if bias > 0:
                        score -= x * bias
                    else:
                        score += x * abs(bias)

                if randomize_ties:
                    score += random.uniform(0, 0.5)

                if score > best_score:
                    best_score = score
                    best_x = x
                    best_newly_blocked = newly_blocked

            if best_x is None:
                break

            A.append(best_x)
            A_set.add(best_x)
            used_sums.add(2 * best_x)
            for y in A:
                if y != best_x:
                    used_sums.add(best_x + y)

            for z in best_newly_blocked:
                if z in unblocked:
                    unblocked.remove(z)

            if best_x in unblocked:
                unblocked.remove(best_x)

        return A

    def local_search_remove_only(A, max_iters=50):
        """Try to reduce |A| by removing elements while preserving maximality."""
        A = list(A)
        best_A = list(A)
        best_size = len(A)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            A_set = set(A)

            # Try removing each element
            for i in range(len(A)):
                candidate = A[:i] + A[i+1:]
                if len(candidate) == 0:
                    continue
                # Check if candidate is still maximal
                if is_maximal(candidate):
                    A = candidate
                    best_A = list(A)
                    best_size = len(A)
                    improved = True
                    break

            if not improved:
                break

        return best_A

    # Initial deterministic run
    best_set = greedy_max_coverage()
    best_size = len(best_set)

    # Local search on initial
    improved = local_search_remove_only(best_set, max_iters=30)
    if len(improved) < best_size:
        best_size = len(improved)
        best_set = improved

    # Randomized restarts with various strategies
    while time.time() < deadline:
        r = random.random()
        if r < 0.2:
            current_set = greedy_max_coverage(randomize_ties=True)
        elif r < 0.4:
            current_set = greedy_max_coverage(randomize_ties=True, bias=0.02)
        elif r < 0.6:
            current_set = greedy_max_coverage(randomize_ties=True, bias=-0.02)
        elif r < 0.8:
            init_x = random.randint(1, N)
            current_set = greedy_max_coverage(init_A=[init_x], randomize_ties=True)
        else:
            # Start with 2 random elements
            init_x = random.randint(1, N)
            init_y = random.randint(1, N)
            while init_y == init_x:
                init_y = random.randint(1, N)
            current_set = greedy_max_coverage(init_A=[init_x, init_y], randomize_ties=True)

        # Local search
        improved = local_search_remove_only(current_set, max_iters=20)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved

    return best_set
