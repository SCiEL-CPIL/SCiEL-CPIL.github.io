return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
