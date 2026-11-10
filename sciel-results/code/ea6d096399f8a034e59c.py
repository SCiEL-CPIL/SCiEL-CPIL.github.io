    for i in range(2, N + 1):
        if sp[i] == 0:
            sp[i] = i
            primes.append(i)
            sigma[i] = i + 1
            pp[i] = i
            p_sigma[i] = i + 1
        
        for p in primes:
            if p > sp[i] or i * p > N:
                break
            sp[i * p] = p
            if p == sp[i]:
                pp[i * p] = pp[i] * p
                p_sigma[i * p] = p_sigma[i] + pp[i * p]
                sigma[i * p] = sigma[i] // p_sigma[i] * p_sigma[i * p]
                break
            else:
                pp[i * p] = p
                p_sigma[i * p] = p + 1
                sigma[i * p] = sigma[i] * (p + 1)
