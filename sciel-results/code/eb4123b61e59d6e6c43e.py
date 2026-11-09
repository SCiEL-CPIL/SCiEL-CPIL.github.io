def construct():
    n = 128
    
    for num_colors in [3, 4, 5]:
        for seed in range(50):
            import random
            random.seed(seed)
            color_order = list(range(num_colors))
            random.shuffle(color_order)
            
            colors = [None] * (n + 1)
            success = True
            for x in range(1, n + 1):
                placed = False
                for c in color_order:
                    valid = True
                    for d in range(1, (x - 1) // 3 + 1):
                        a = x - 3 * d
                        b = x - 2 * d
                        cc = x - d
                        s = {colors[a], colors[b], colors[cc], c}
                        if len(s) < 3:
                            valid = False
                            break
                    if valid:
                        colors[x] = c
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            if success:
                return [colors[i] for i in range(1, n + 1)]
    
    # Fallback
    colors = [None] * (n + 1)
    for x in range(1, n + 1):
        for c in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                b = x - 2 * d
                cc = x - d
                s = {colors[a], colors[b], colors[cc], c}
                if len(s) < 3:
                    valid = False
                    break
            if valid:
                colors[x] = c
                break
    return [colors[i] for i in range(1, n + 1)]
