count = 0
for c in range(4):
    m = masks[c]
    for p in patterns:
        if (m & p) == p:
            count += 1
