import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s margin

    best_seq = []
    best_len = 0

    def dfs(seq, used_sums, last_val):
        nonlocal best_seq, best_len

        if time.time() > deadline:
            return

        current_len = len(seq)
        if current_len > best_len:
            best_len = current_len
            best_seq = list(seq)

        # Upper bound: at most (256 - last_val) more elements
        if current_len + (256 - last_val) <= best_len:
            return

        # Try extending, in order of smallest x first (to keep values small)
        for x in range(last_val + 1, 257):
            if x in used_sums:
                continue

            # Check all new block sums
            new_sums_list = [x]
            valid = True
            s = 0
            for v in reversed(seq):
                s += v
                ns = x + s
                if ns in used_sums:
                    valid = False
                    break
                new_sums_list.append(ns)

            if not valid:
                continue

            # Recursive call
            seq.append(x)
            used_sums.update(new_sums_list)
            dfs(seq, used_sums, x)
            seq.pop()
            used_sums.difference_update(new_sums_list)

    dfs([], set(), 0)
    return best_seq
