import numpy as np

def construct():
    N = 1000

    def count_sums_fast(arr):
        n = len(arr)
        size = 1
        while size < 2 * n:
            size *= 2
        f = np.fft.rfft(arr.astype(np.float64), size)
        conv = np.fft.irfft(f * f, size)[:2 * n]
        conv = np.round(conv).astype(np.int32)
        counts = np.zeros(2 * n, dtype=np.int32)
        s_range = np.arange(2, 2 * n)
        ordered = conv[s_range]
        diag = ((s_range % 2 == 0) & (s_range // 2 < n) & arr[s_range // 2]).astype(np.int32)
        counts[s_range] = (ordered + diag) // 2
        return counts

    def greedy():
        chosen = []
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        for x in range(1, N + 1):
            ok = True
            if counts[2 * x] >= 2:
                ok = False
            else:
                for y in chosen:
                    if y >= x and counts[x + y] >= 2:
                        ok = False
                        break
            if ok:
                chosen.append(x)
                counts[2 * x] += 1
                for y in chosen:
                    if y >= x:
                        counts[x + y] += 1
        return chosen

    def local_search():
        np.random.seed(123)
        best_size = 0
        best_arr = None

        for trial in range(30):
            density = np.random.uniform(0.45, 0.70)
            size = int(N * density)
            init = np.random.choice(range(1, N + 1), size=size, replace=False)
            arr = np.zeros(N + 1, dtype=np.int32)
            for x in init:
                arr[x] = 1

            for _ in range(100):
                counts = count_sums_fast(arr)
                viol = np.where(counts > 2)[0]
                if len(viol) == 0:
                    break
                s = viol[np.random.randint(len(viol))]
                removed = False
                for a in range(1, s // 2 + 1):
                    b = s - a
                    if b > N:
                        break
                    if arr[a] and arr[b]:
                        if np.random.rand() < 0.5:
                            arr[a] = 0
                        else:
                            arr[b] = 0
                        removed = True
                        break
                if not removed:
                    break

            counts = count_sums_fast(arr)
            current_size = int(arr.sum())

            candidates = [x for x in range(1, N + 1) if not arr[x]]
            np.random.shuffle(candidates)

            for x in candidates:
                ok = True
                if counts[2 * x] >= 2:
                    ok = False
                else:
                    y_vals = np.where(arr[x:])[0] + x
                    if len(y_vals) > 0 and np.any(counts[x + y_vals] >= 2):
                        ok = False
                if ok:
                    arr[x] = 1
                    y_vals = np.where(arr[x:])[0] + x
                    counts[x + y_vals] += 1
                    counts[2 * x] += 1
                    current_size += 1

            if current_size > best_size:
                best_size = current_size
                best_arr = arr.copy()

        return [int(x) for x in range(1, N + 1) if best_arr[x]]

    greedy_result = greedy()
    greedy_size = len(greedy_result)

    ls_result = local_search()
    ls_size = len(ls_result)

    if ls_size >= greedy_size:
        return ls_result
    else:
        return greedy_result
