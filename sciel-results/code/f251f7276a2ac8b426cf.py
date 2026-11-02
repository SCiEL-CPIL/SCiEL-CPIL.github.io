def greedy_select(candidates, K):
    max_dot = K // 2
    selected = []
    selected_arr = np.array([]).reshape(0, D)
    
    random.shuffle(candidates)
    
    for v in candidates:
        v_np = np.array(v)
        if len(selected) == 0:
            selected.append(v)
            selected_arr = v_np.reshape(1, D)
            continue
            
        dots = selected_arr @ v_np
        if np.any(dots > max_dot):
            continue
        else:
            selected.append(v)
            selected_arr = np.vstack((selected_arr, v_np))
            
    return selected
