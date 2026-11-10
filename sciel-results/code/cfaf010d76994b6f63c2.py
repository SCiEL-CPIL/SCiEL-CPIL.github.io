    for right in range(2, 1008):
        if left <= inv[right] < right:
            left = inv[right] + 1
        
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
            max_start = left
