    def is_valid(in_set):
        counts = [0] * (max_sum + 1)
        for i in range(1, N + 1):
            if in_set[i]:
                counts[2*i] += 1
                for a in range(1, i):
                    if in_set[a]:
                        counts[a+i] += 1
        for s in range(2, max_sum + 1):
            if counts[s] > 2:
                return False
        return True

    # Before return
    if not is_valid(best_in_set):
        return [i for i in range(1, N + 1) if in_set[i]] # Greedy
    return [i for i in range(1, N + 1) if best_in_set[i]]
