def construct():
    n = 256
    S = []
    for a in range(n, 0, -1):
        can_add = True
        for i in range(len(S)):
            for j in range(i, len(S)):
                if (S[i] + S[j]) % a == 0:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            S.append(a)
    return sorted(S)
