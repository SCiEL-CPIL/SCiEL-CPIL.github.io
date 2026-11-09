<<<<<<< SEARCH
    np.random.seed(42)
    num_starts = 20
    for start_idx in range(num_starts):
        if time.time() > deadline - 80:
            break
        if start_idx == 0:
            x_start = x0.copy()
        else:
            x_start = x0.copy()
            noise_scale = 0.4 * (start_idx + 1)
            for i in range(11):
                x_start[3*i] += np.random.randn() * noise_scale
                x_start[3*i+1] += np.random.randn() * noise_scale
                x_start[3*i+2] += np.random.randn() * 25
            x_start[33] = np.random.randn() * 0.4
            x_start[34] = np.random.randn() * 0.4
            x_start[35] = np.random.randn() * 25
=======
    np.random.seed(42)
    num_starts = 20
    for start_idx in range(num_starts):
        if time.time() > deadline - 80:
            break
        if start_idx == 0:
            x_start = x0.copy()
        else:
            x_start = x0.copy()
            noise_scale = 0.3 * (start_idx + 1)
            for i in range(11):
                x_start[3*i] += np.random.randn() * noise_scale
                x_start[3*i+1] += np.random.randn() * noise_scale
                x_start[3*i+2] += np.random.randn() * 20
            x_start[33] = np.random.randn() * 0.3
            x_start[34] = np.random.randn() * 0.3
            x_start[35] = np.random.randn() * 20
>>>>>>> REPLACE

