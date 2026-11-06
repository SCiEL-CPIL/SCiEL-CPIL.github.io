<<<<<<< SEARCH
    def two_swap_search(edges, conflict, colour, used, rng, time_budget=5.0):
        """Try 2-swap moves to enable merges."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()

        classes = {}
        class_masks = {}
        for e in range(m):
            c = col_of[e]
            if c >= 0:
                if c not in classes:
                    classes[c] = []
                    class_masks[c] = 0
                classes[c].append(e)
                class_masks[c] |= 1 << e

        for _ in range(5000):
            if time.time() - start > time_budget:
                break
            maxc = max(classes.keys()) if classes else -1
            if maxc < 1:
                break

            a = rng.randrange(maxc + 1)
            b = rng.randrange(a) if a > 0 else 0
            if a == b or a not in classes or b not in classes:
                continue
            cls_a = classes[a]
            cls_b = classes[b]
            if not cls_a or not cls_b:
                continue
            e = rng.choice(cls_a)
            f = rng.choice(cls_b)

            mask_b_without_f = class_masks[b] & ~(1 << f)
            if conflict[e] & mask_b_without_f:
                continue
            mask_a_without_e = class_masks[a] & ~(1 << e)
            if conflict[f] & mask_a_without_e:
                continue

            col_of[e] = b
            col_of[f] = a
            classes[b].remove(f)
            classes[b].append(e)
            class_masks[b] = (class_masks[b] & ~(1 << f)) | (1 << e)
            classes[a].remove(e)
            classes[a].append(f)
            class_masks[a] = (class_masks[a] & ~(1 << e)) | (1 << f)

            # Try to merge after swap
            for c in range(maxc, -1, -1):
                if c not in classes:
                    continue
                cls_c = classes[c]
                mask_c = class_masks[c]
                for d in range(c):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]
                    ok = True
                    for edge_c in cls_c:
                        if conflict[edge_c] & mask_d:
                            ok = False
                            break
                    if ok:
                        for edge_c in cls_c:
                            col_of[edge_c] = d
                        classes[d].extend(cls_c)
                        class_masks[d] |= mask_c
                        del classes[c]
                        del class_masks[c]
                        break
                if c not in classes:
                    break

        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)
=======
    def two_swap_search(edges, conflict, colour, used, rng, time_budget=5.0):
        """Try 2-swap moves to enable merges."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()

        classes = {}
        class_masks = {}
        for e in range(m):
            c = col_of[e]
            if c >= 0:
                if c not in classes:
                    classes[c] = []
                    class_masks[c] = 0
                classes[c].append(e)
                class_masks[c] |= 1 << e

        def try_merge():
            maxc = max(classes.keys()) if classes else -1
            if maxc < 0:
                return False
            for c in range(maxc, -1, -1):
                if c not in classes:
                    continue
                cls_c = classes[c]
                mask_c = class_masks[c]
                for d in range(c):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]
                    ok = True
                    for e in cls_c:
                        if conflict[e] & mask_d:
                            ok = False
                            break
                    if ok:
                        for e in cls_c:
                            col_of[e] = d
                        classes[d].extend(cls_c)
                        class_masks[d] |= mask_c
                        del classes[c]
                        del class_masks[c]
                        return True
            return False

        for _ in range(8000):
            if time.time() - start > time_budget:
                break
            maxc = max(classes.keys()) if classes else -1
            if maxc < 1:
                break

            # Try all pairs of colors for swaps
            a = rng.randrange(maxc + 1)
            b = rng.randrange(a) if a > 0 else 0
            if a == b or a not in classes or b not in classes:
                continue
            cls_a = classes[a]
            cls_b = classes[b]
            if not cls_a or not cls_b:
                continue
            e = rng.choice(cls_a)
            f = rng.choice(cls_b)

            mask_b_without_f = class_masks[b] & ~(1 << f)
            if conflict[e] & mask_b_without_f:
                continue
            mask_a_without_e = class_masks[a] & ~(1 << e)
            if conflict[f] & mask_a_without_e:
                continue

            col_of[e] = b
            col_of[f] = a
            classes[b].remove(f)
            classes[b].append(e)
            class_masks[b] = (class_masks[b] & ~(1 << f)) | (1 << e)
            classes[a].remove(e)
            classes[a].append(f)
            class_masks[a] = (class_masks[a] & ~(1 << e)) | (1 << f)

            # Try to merge after swap
            try_merge()

        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)
>>>>>>> REPLACE

