def construct():
    bits = []
    for _ in range(N):
        # Greedy choice: 1 maximizes the objective
        bits.append('1')
    return "".join(bits)
