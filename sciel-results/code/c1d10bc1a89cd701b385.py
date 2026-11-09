def count_violations(colors):
    c_aps = colors[aps]
    s = np.sort(c_aps, axis=1)
    eq1 = s[:, 0] == s[:, 1]
    eq2 = s[:, 1] == s[:, 2]
    eq3 = s[:, 2] == s[:, 3]
    true_count = eq1.astype(np.int8) + eq2.astype(np.int8) + eq3.astype(np.int8)
    return np.sum(true_count >= 2)
