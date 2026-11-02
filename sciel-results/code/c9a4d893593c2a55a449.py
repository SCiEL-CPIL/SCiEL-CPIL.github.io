import itertools
import random

def construct(q, n, radius):
    total = q ** n
    if total > 50000:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    def get_covers(cw):
        covers = [tuple(cw)]
        for pos in range(n):
            for sym in range(q):
                if sym == cw[pos]:
                    continue
                new_w = list(cw)
                new_w[pos] = sym
                covers.append(tuple(new_w))
        return covers
    
    def build_cover_data(code_list):
        cc = [0] * total
        ccov = []
        for cw in code_list:
            covers = get_covers(cw)
            ccov.append(covers)
            for w in covers:
                cc[word_to_idx[w]] += 1
        return cc, ccov
    
    def try_remove_redundant(code_list, cc, ccov):
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                can_remove = True
                for w in ccov[i]:
                    if cc[word_to_idx[w]] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for w in ccov[i]:
                        cc[word_to_idx[w]] -= 1
                    del ccov[i]
                    code_list = code_list[:i] + code_list[i+1:]
                    changed = True
                    break
        return code_list
    
    # Phase 1: Remove redundant
    cover_count, cw_covers = build_cover_data(code)
    code = try_remove_redundant(code, cover_count, cw_covers)
    
    # Phase 2: Randomized swaps
    random.seed(42)
    for _ in range(200):
        if len(code) < 10:
            break
        i = random.randint(0, len(code) - 1)
        non_codewords = [w for w in all_words if w not in code]
        if not non_codewords:
            break
        new_cw = random.choice(non_codewords)
        new_code = code[:i] + [list(new_cw)] + code[i+1:]
        cc, ccov = build_cover_data(new_code)
        new_code = try_remove_redundant(new_code, cc, ccov)
        if len(new_code) < len(code):
            code = new_code
    
    return [list(w) for w in code]
