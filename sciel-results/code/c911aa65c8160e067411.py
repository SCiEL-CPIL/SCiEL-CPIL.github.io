<<<<<<< SEARCH
    # ---------- SA local search ----------
    def sa_search(S, sc, max_iters=300, time_budget=None):
        S = int(S)
        T = 4.0
        T_min = 0.1
        cool = 0.98
        for _ in range(max_iters):
            if time_budget is not None and time.time() > time_budget:
                break
            r = random.random()
            if r < 0.45:
                # add
                cands = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1) and can_add(x, S)]
                if not cands:
                    continue
                x = random.choice(cands)
                S_new = S | (1 << (x-1))
                delta = gain_add(x, S)
            elif r < 0.75:
                # remove
                in_s = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                if not in_s:
                    continue
                y = random.choice(in_s)
                S_new = S & ~(1 << (y-1))
                delta = -loss_remove(y, S)
            else:
                # swap
                in_s = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                not_s = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1)]
                if not in_s or not not_s:
                    continue
                y = random.choice(in_s)
                S_wo = S & ~(1 << (y-1))
                valid = [x for x in not_s if can_add(x, S_wo)]
                if not valid:
                    continue
                x = random.choice(valid)
                S_new = S_wo | (1 << (x-1))
                delta = gain_add(x, S_wo) - loss_remove(y, S)

            if delta > 0 or random.random() < np.exp(delta / T):
                S = S_new
                sc += delta
            T = max(T_min, T * cool)

        # Final greedy polish
        for x in range(1, n+1):
            if time_budget is not None and time.time() > time_budget:
                break
            if (S >> (x-1)) & 1:
                continue
            if not can_add(x, S):
                continue
            g = gain_add(x, S)
            if g > 0:
                S |= (1 << (x-1))
                sc += g
        return S, sc
=======
    # ---------- SA local search ----------
    def sa_search(S, sc, max_iters=300, time_budget=None):
        S = int(S)
        T = 5.0
        T_min = 0.05
        cool = 0.985
        for _ in range(max_iters):
            if time_budget is not None and time.time() > time_budget:
                break
            r = random.random()
            if r < 0.40:
                # add
                cands = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1) and can_add(x, S)]
                if not cands:
                    continue
                x = random.choice(cands)
                S_new = S | (1 << (x-1))
                delta = gain_add(x, S)
            elif r < 0.65:
                # remove
                in_s = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                if not in_s:
                    continue
                y = random.choice(in_s)
                S_new = S & ~(1 << (y-1))
                delta = -loss_remove(y, S)
            elif r < 0.85:
                # swap
                in_s = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                not_s = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1)]
                if not in_s or not not_s:
                    continue
                y = random.choice(in_s)
                S_wo = S & ~(1 << (y-1))
                valid = [x for x in not_s if can_add(x, S_wo)]
                if not valid:
                    continue
                x = random.choice(valid)
                S_new = S_wo | (1 << (x-1))
                delta = gain_add(x, S_wo) - loss_remove(y, S)
            else:
                # 2-swap: remove two, add two
                in_s = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                not_s = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1)]
                if len(in_s) < 2 or len(not_s) < 2:
                    continue
                y1, y2 = random.sample(in_s, 2)
                S_wo = S & ~(1 << (y1-1)) & ~(1 << (y2-1))
                valid1 = [x for x in not_s if can_add(x, S_wo)]
                if not valid1:
                    continue
                x1 = random.choice(valid1)
                S_mid = S_wo | (1 << (x1-1))
                valid2 = [x for x in not_s if x != x1 and can_add(x, S_mid)]
                if not valid2:
                    continue
                x2 = random.choice(valid2)
                S_new = S_mid | (1 << (x2-1))
                delta = gain_add(x1, S_wo) + gain_add(x2, S_mid) - loss_remove(y1, S) - loss_remove(y2, S & ~(1<<(y1-1)))

            if delta > 0 or random.random() < np.exp(delta / T):
                S = S_new
                sc += delta
            T = max(T_min, T * cool)

        # Final greedy polish
        for x in range(1, n+1):
            if time_budget is not None and time.time() > time_budget:
                break
            if (S >> (x-1)) & 1:
                continue
            if not can_add(x, S):
                continue
            g = gain_add(x, S)
            if g > 0:
                S |= (1 << (x-1))
                sc += g
        return S, sc
>>>>>>> REPLACE

