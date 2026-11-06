while time < deadline - 20:
    s = random sample
    k = check_window(s, 200)
    if k > best_k:
        best_k = k; best_s = s
        # local refinement
        refine(best_s)
        # extend
        extend(best_s, best_k)
