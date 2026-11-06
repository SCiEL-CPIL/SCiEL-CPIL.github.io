def construct():
    N = 128
    colors = [bin(i).count('1') for i in range(N)]
    return colors
