def construct():
    deadline = time.time() + 50
    best = [1,2]
    best_size = 2
    random.seed(0)
    universe = list(range(1,257))
    def try_grow(cur):
        # attempt adds until stuck
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in random.sample(universe, 20):
                if x in cur:
                    continue
                cand = cur + [x]
                if _is_valid(cand):
                    cur = cand
                    changed = True
                    break
        return cur
    cur = [1,2]
    cur = try_grow(cur)
    if len(cur) > best_size:
        best, best_size = cur, len(cur)
    # perturb loop
    while time.time() < deadline:
        # random swap to perturb
        if len(cur) >= 2:
            i = random.randrange(len(cur))
            x = random.choice(universe)
            if x not in cur:
                cand = cur[:i] + [x] + cur[i+1:]
                if _is_valid(cand):
                    cur = cand
        cur = try_grow(cur)
        if len(cur) > best_size:
            best, best_size = cur, len(cur)
        # occasionally restart
        if random.random() < 0.01:
            cur = [random.choice(universe), random.choice(universe)]
            cur = try_grow(cur)
            if len(cur) > best_size:
                best, best_size = cur, len(cur)
    # ensure best valid
    assert _is_valid(best)
    return [int(x) for x in best]
