import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    colors = [-1] * n
    
    progressions_ending_at = [[] for _ in range(n)]
    for s in range(n):
        for d in range(1, (n - s) // 3 + 1):
            last = s + 3 * d
            if last < n:
                progressions_ending_at[last].append((s, s + d, s + 2 * d, last))
    
    def check_valid(pos, color):
        for (i, j, k, l) in progressions_ending_at[pos]:
            c_i = colors[i]
            c_j = colors[j]
            c_k = colors[k]
            if c_i < 0 or c_j < 0 or c_k < 0:
                continue
            if c_i == c_j == c_k:
                return False
            if c_i == c_j:
                if c_k == color:
                    return False
            elif c_i == c_k:
                if c_j == color:
                    return False
            elif c_j == c_k:
                if c_i == color:
                    return False
        return True
    
    def backtrack(pos, max_color):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        for c in range(max_color + 1):
            if check_valid(pos, c):
                colors[pos] = c
                if backtrack(pos + 1, max_color):
                    return True
                colors[pos] = -1
        return False
    
    colors[0] = 0
    found = False
    for num_colors in range(3, 10):
        for i in range(1, n):
            colors[i] = -1
        colors[0] = 0
        if backtrack(1, num_colors - 1):
            found = True
            break
    
    result = []
    for i in range(n):
        if colors[i] < 0:
            colors[i] = 0
        result.append(colors[i])
    
    return result
