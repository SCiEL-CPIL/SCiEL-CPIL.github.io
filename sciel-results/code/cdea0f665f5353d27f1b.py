def _score(S):
    S_list = list(S)
    S_set = S
    diffs = set()
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            b = S_list[j]
            if b > a:
                d = b - a
                c = a + 2 * d
                if c in S_set:
                    diffs.add(d)
    return len(diffs)
