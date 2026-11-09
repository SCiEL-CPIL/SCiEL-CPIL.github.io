N = 16

# EVOLVE-BLOCK-START
def construct():
    colors = [0] * (N*(N-1)//2)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            # Base coloring: min(i, j)
            c = i
            # Exception for the last vertex to add one more color
            if j == N - 1 and i == N - 2:
                c = N - 1
            colors[idx] = c
            idx += 1
    return colors
# EVOLVE-BLOCK-END
