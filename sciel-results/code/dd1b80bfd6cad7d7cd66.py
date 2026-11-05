for m in range(2, n + 1):
    ...
    perms, seen = [], set()
    # Extract unique (m-1)-length windows with distinct symbols
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        # Find maximum overlap
        ...
        new += block[overlap:]
    word = new
