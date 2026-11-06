<<<<<<< SEARCH
    # Strategy 2: Random dense starts
    for restart in range(4):
        if not budget_left(15.0):
            break
        density = 0.62 + 0.07 * restart
        adj2 = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj2[i] |= (1 << j)
                    adj2[j] |= (1 << i)
        try_start(adj2, 12.0)

    # Strategy 3: K16 minus random edges
    for restart in range(3):
        if not budget_left(12.0):
            break
        adj3 = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                adj3[i] |= (1 << j)
                adj3[j] |= (1 << i)
        num_removals = 8 + random.randint(0, 6)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:num_removals]:
            adj3[u] ^= (1 << v)
            adj3[v] ^= (1 << u)
        try_start(adj3, 10.0)
=======
    # Strategy 2: Random dense starts
    for restart in range(5):
        if not budget_left(15.0):
            break
        density = 0.60 + 0.08 * restart
        adj2 = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj2[i] |= (1 << j)
                    adj2[j] |= (1 << i)
        try_start(adj2, 12.0)

    # Strategy 3: K16 minus random edges
    for restart in range(4):
        if not budget_left(12.0):
            break
        adj3 = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                adj3[i] |= (1 << j)
                adj3[j] |= (1 << i)
        num_removals = 7 + random.randint(0, 7)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:num_removals]:
            adj3[u] ^= (1 << v)
            adj3[v] ^= (1 << u)
        try_start(adj3, 10.0)
>>>>>>> REPLACE

