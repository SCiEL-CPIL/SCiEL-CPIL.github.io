# Precompute
pf_cache = {s: get_primes(s, spf) for s in range(2, 20001)}

def eval(arr):
    union = set()
    for i in range(n-1):
        ai = arr[i]
        for j in range(i+1, n):
            union |= pf_cache[ai + arr[j]]
    return len(union)
