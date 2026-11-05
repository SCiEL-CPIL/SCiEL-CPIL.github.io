    for k in range(2, N + 2):
        p = int(spf[k])
        m = k
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        
        if e >= 2:
            powerful[k] = powerful[m] * (p ** e)
        else:
            powerful[k] = powerful[m]
