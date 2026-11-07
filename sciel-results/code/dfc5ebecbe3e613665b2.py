            if sigma < 1e-8 and calls_used < budget - 10:
                # Restart
                m = np.random.uniform(-5, 5, dim)
                sigma = 2.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                continue
