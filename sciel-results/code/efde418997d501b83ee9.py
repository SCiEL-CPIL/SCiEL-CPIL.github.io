<<<<<<< SEARCH
        steps = 20000
        for step in range(steps):
            if time.time() > deadline:
                break
=======
        # Adaptive steps: use more steps for later restarts
        base_steps = 15000
        steps = base_steps + (restart // 5) * 5000
        for step in range(steps):
            if time.time() > deadline:
                break
>>>>>>> REPLACE

