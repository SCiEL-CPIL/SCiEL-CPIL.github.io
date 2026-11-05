    def _greedy(order, n=512):
        in_set = [False] * (n + 1)
        count = 0
        for x in order:
            if _can_add(in_set, x, n):
                in_set[x] = True
                count += 1
        return [i for i in range(1, n+1) if in_set[i]]
    