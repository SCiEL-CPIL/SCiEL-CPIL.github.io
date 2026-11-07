def construct():
    import time
    start = time.time()
    deadline = start + 85
    
    S = set()
    for x in range(256, 1, -1):
        if time.time() > deadline:
            break
        S_list = list(S)
        can_add = True
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a, b = S_list[i], S_list[j]
                if (a * b) % x == 0:
                    can_add = False
                    break
                if (x * b) % a == 0:
                    can_add = False
                    break
                if (x * a) % b == 0:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            S.add(x)
    
    return [int(x) for x in sorted(S)]
