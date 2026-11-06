    C4 = [0]*(n+1)
    C3 = [0]*(n+1)
    C2 = [0]*(n+1)
    C1 = [0]*(n+1)
    for m in range(n+1):
        C1[m] = m
        C2[m] = m*(m-1)//2 if m>=2 else 0
        C3[m] = m*(m-1)*(m-2)//6 if m>=3 else 0
        C4[m] = m*(m-1)*(m-2)*(m-3)//24 if m>=4 else 0

    C_n_4 = C4[n]

    def get_rank(a, b, c, d):
        rank = 0
        rank += C_n_4 - C4[n - a]
        rank += C3[n - a - 1] - C3[n - b]
        rank += C2[n - b - 1] - C2[n - c]
        rank += C1[n - c - 1] - C1[n - d]
        return rank
