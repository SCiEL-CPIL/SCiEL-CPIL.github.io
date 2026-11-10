    current_size = np.sum(s)
    best_size = current_size
    # ...
                if valid:
                    s[i] = 1
                    current_size += 1
                    # ...
                    if current_size > best_size:
                        best_size = current_size
                        best_s = s.copy()
    # ...
        s[j] = 0
        current_size -= 1
