    if isinstance(distribution_spec, dict):
        # Handle keys as tuples or lists
        probs = np.zeros(n_types)
        for k, v in distribution_spec.items():
            t = tuple(k)
            if t in types:
                idx = types.index(t)
                probs[idx] = v
        # If some probs are missing, fill with 0 or uniform?
        # Assuming full support provided.
