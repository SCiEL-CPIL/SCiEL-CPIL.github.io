    # If we have time left, polish with Riemannian Gradient Descent
    if time.time() < deadline - 2:
        time_left = deadline - 2 - time.time()
        if time_left > 1.0:
            try:
                # Ensure best_x is on sphere
                best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
                best_x, e_polish = _riemannian_polish(best_x, max_time=time_left)
                if e_polish < best_energy:
                    best_energy = e_polish
            except Exception:
                pass
