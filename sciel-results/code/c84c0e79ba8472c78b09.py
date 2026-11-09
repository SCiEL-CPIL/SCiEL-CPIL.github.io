N = 16

# EVOLVE-BLOCK-START
def construct():
    colors = [0] * (N*(N-1)//2)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            # Base coloring: min(i, j)
            c = i
            # Use color 15 for the last edge to ensure 16 distinct colors
            if i == N - 2 and j == N - 1:
                c = N - 1
            colors[idx] = c
            idx += 1
    return colors
# EVOLVE-BLOCK-END
