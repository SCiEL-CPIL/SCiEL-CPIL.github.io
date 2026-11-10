    def do_add(x):
        nonlocal S, counts, S_arr
        S.add(x)
        if len(S_arr) > 0:
            sums = x + S_arr
            counts[sums] += 1
        counts[2*x] += 1
        S_arr = np.append(S_arr, x)
