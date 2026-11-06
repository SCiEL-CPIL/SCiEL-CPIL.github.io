N = 16

# EVOLVE-BLOCK-START
def construct():
    colors = [0] * (N*(N-1)//2)
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            if i < 8 and j < 8:
                # Block 1: color by j
                colors[idx] = j
            elif i >= 8 and j >= 8:
                # Block 2: color by j
                # j ranges 9..15
                colors[idx] = j
            else:
                # Inter-block: default 0
                # Exception: (0,8) gets 8 to use color 8
                if i == 0 and j == 8:
                    colors[idx] = 8
                else:
                    colors[idx] = 0
            idx += 1
    return colors
# EVOLVE-BLOCK-END
