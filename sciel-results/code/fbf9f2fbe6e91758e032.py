    def run_greedy(seed_val):
        rng = np.random.default_rng(seed_val)
        # ...
        while np.any(uncovered):
            # ...
            masked_gain = gain.copy()
            masked_gain[selected_mask] = -1.0
            # Add noise to break ties
            noise = rng.random(num_k) * 0.01
            masked_gain += noise
            best_idx = int(np.argmax(masked_gain))
            # ...
