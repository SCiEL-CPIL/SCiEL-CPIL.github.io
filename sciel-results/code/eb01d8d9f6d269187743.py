    for k in range(max_k, 0, -1):
        seq.append(primes[idx])
        if k > 1:
            idx = prev_idx[k][idx]
