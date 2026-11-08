for (i, a, sA) in AF[m]:
    for (a2, k, sB) in BF[m]:
        if a == a2 and (i, k) == (ci, ck):
            simple(i, a, k, s * sA * sB)
