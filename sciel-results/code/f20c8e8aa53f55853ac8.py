def generate_bitonic_asc(n):
    if n == 1:
        return []
    half = n // 2
    net1 = generate_bitonic_asc(half)
    net2 = generate_bitonic_desc(half)
    net2 = [(i + half, j + half) for i, j in net2]
    merge = generate_bitonic_merge(n)
    return net1 + net2 + merge

def generate_bitonic_desc(n):
    if n == 1:
        return []
    half = n // 2
    net1 = generate_bitonic_desc(half)
    net2 = generate_bitonic_asc(half)
    net2 = [(i + half, j + half) for i, j in net2]
    merge = generate_bitonic_merge(n)
    return net1 + net2 + merge

def generate_bitonic_merge(n):
    # n is power of 2
    merge_net = []
    k = n // 2
    while k > 0:
        for i in range(0, n, 2 * k):
            for j in range(i, i + k):
                merge_net.append((j, j + k))
        k //= 2
    return merge_net
