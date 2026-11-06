    scores = pp[min_n:max_n+1] * pp[min_n+1:max_n+2]
    idx = int(np.argmax(scores))
    best_score = int(scores[idx])
    best_n = min_n + idx
    ...
    # CRT search
    if actual_score > best_score:
        best_score = actual_score
        best_n = n_val
