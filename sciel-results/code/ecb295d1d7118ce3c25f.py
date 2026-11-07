# Refined inner loop for removal
for idx, x in enumerate(S):
    conflicts = 0
    for i in range(len(S)):
        if i == idx: continue
        b = S[i]
        for j in range(i + 1, len(S)):
            if j == idx: continue
            c = S[j]
            if (b * c) % x == 0:
                conflicts += 1
            if (x * c) % b == 0:
                conflicts += 1
            if (x * b) % c == 0:
                conflicts += 1
    if conflicts > 0:
        valid = False
        if conflicts > max_conflicts:
            max_conflicts = conflicts
            worst_idx = idx
            worst_x = x
