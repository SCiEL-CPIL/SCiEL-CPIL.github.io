def construct():
    n = 128
    colors = [-1] * (n + 1)
    
    for x in range(1, n + 1):
        for c in range(128):
            valid = True
            for d in range(1, (x // 3) + 1):
                a = x - 3 * d
                if a < 1:
                    break
                if colors[a] != -1 and colors[a+d] != -1 and colors[a+2*d] != -1:
                    if len({colors[a], colors[a+d], colors[a+2*d], c}) < 3:
                        valid = False
                        break
            if valid:
                colors[x] = c
                break
    
    return [colors[i] for i in range(1, n + 1)]
