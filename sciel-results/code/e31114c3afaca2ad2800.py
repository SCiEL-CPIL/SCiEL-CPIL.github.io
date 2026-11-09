import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    
    def check_validity(colors):
        for a in range(1, n + 1):
            for d in range(1, n + 1):
                a4 = a + 3 * d
                if a4 > n:
                    break
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                c4 = colors[a4 - 1]
                if len({c1, c2, c3, c4}) < 3:
                    return False
        return True
    
    # Try several patterns for 3-coloring
    patterns = [
        lambda i: (i * 3 + (i // 3) * 2) % 3,
        lambda i: (i % 3) ^ ((i // 3) % 3),
        lambda i: (i * 7) % 3,
        lambda i: (i * i + i) % 3,
        lambda i: (i.bit_count() + i) % 3,
    ]
    
    for pat in patterns:
        colors = [pat(i) for i in range(1, n + 1)]
        if check_validity(colors):
            return colors
    
    # Fallback: 4 colors
    colors = [(i - 1) % 4 for i in range(1, n + 1)]
    if check_validity(colors):
        return colors
    
    # Final fallback
    colors = [(i - 1) % 5 for i in range(1, n + 1)]
    return colors
