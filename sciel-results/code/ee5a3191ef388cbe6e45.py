def construct():
    selected = []
    for n in range(1, 513):
        x = n
        valid = True
        while x > 0:
            if x % 3 == 2:
                valid = False
                break
            x //= 3
        if valid:
            selected.append(n)
    return selected
