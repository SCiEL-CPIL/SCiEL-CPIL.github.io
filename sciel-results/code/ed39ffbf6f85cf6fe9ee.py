import time
import random
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0

    if n <= 4:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                               if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    def generate_blocks_classic(n):
        word = [1]
        for m in range(2, n):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1)
                               if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        perms, seen = [], set()
        for i in range(len(word) - (n - 1) + 1):
            window = tuple(word[i:i + n - 1])
            if len(set(window)) == n - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        blocks = []
        for p in perms:
            blocks.append(list(p) + [n] + list(p))
        return blocks

    def build_word(order, blocks):
        word = list(blocks[order[0]])
        for i in range(1, len(order)):
            b = blocks[order[i]]
            a = blocks[order[i-1]]
            max_o = min(len(a), len(b))
            ov = 0
            for o in range(max_o, -1, -1):
                if o == 0:
                    break
                if a[-o:] == b[:o]:
                    ov = o
                    break
            word += b[ov:]
        return word

    def is_valid(word, n):
        if len(word) < n:
            return False
        seen = set()
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i+n])
            if len(set(w)) == n:
                seen.add(w)
        return len(seen) == math.factorial(n)

    blocks = generate_blocks_classic(n)
    num_blocks = len(blocks)
    block_len = len(blocks[0])
    total_block_len = num_blocks * block_len

    overlap_cache = {}
    def get_ov(i, j):
        key = i * num_blocks + j
        if key in overlap_cache:
            return overlap_cache[key]
        a = blocks[i]
        b = blocks[j]
        max_o = min(len(a), len(b))
        ov = 0
        for o in range(max_o, -1, -1):
            if o == 0:
                break
            if a[-o:] == b[:o]:
                ov = o
                break
        overlap_cache[key] = ov
        return ov

    def total_len(order):
        overlaps = 0
        for i in range(num_blocks - 1):
            overlaps += get_ov(order[i], order[i+1])
        return total_block_len - overlaps

    order = list(range(num_blocks))
    best_len = total_len(order)
    best_order = order[:]

    random.seed(42)
    T = 10.0
    cooling = 0.9995
    while time.time() < deadline - 5.0:
        if random.random() < 0.5:
            i, j = random.sample(range(num_blocks), 2)
            oi, oj = order[i], order[j]
            order[i], order[j] = oj, oi
            new_len = total_len(order)
            delta = new_len - best_len
            if delta < 0 or random.random() < math.exp(-delta / T):
                if new_len < best_len:
                    best_len = new_len
                    best_order = order[:]
            else:
                order[i], order[j] = oi, oj
        else:
            i = random.randrange(num_blocks)
            j = random.randrange(num_blocks)
            if i == j:
                continue
            b = order.pop(i)
            order.insert(j, b)
            new_len = total_len(order)
            delta = new_len - best_len
            if delta < 0 or random.random() < math.exp(-delta / T):
                if new_len < best_len:
                    best_len = new_len
                    best_order = order[:]
            else:
                order.pop(j)
                order.insert(i, b)
        T *= cooling

    word = build_word(best_order, blocks)
    if is_valid(word, n):
        return "".join(map(str, word))
    classic_word = build_word(list(range(num_blocks)), blocks)
    return "".join(map(str, classic_word))
