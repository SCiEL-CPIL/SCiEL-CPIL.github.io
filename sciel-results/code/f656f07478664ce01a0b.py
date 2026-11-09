    def local_search(K, max_time):
        end_time = time.time() + max_time
        while time.time() < end_time:
            # Single run of SA
            colors = [random.randint(0, K - 1) for _ in range(120)]
            # ... SA loop with a shorter time limit, e.g., 5s or until T < min_T ...
            # If found, return
            # If not, loop to restart
        return None
