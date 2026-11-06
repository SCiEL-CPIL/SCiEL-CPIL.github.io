    for n in [30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]:
        if time.time() > deadline:
            break
            
        # For n=30, use spectral init + local search
        if n == 30:
            if time.time() > deadline - 20: break
            # Spectral init
            # Generate random graph, compute Laplacian eigenvectors, use them to bias colors
            # This is a heuristic.
            colors = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
            colors = local_search(n, colors, max_time=15.0)
            if check_coloring(n, colors):
                return {'n': n, 'colors': colors}
            continue

        # For n=29, try Paley
        if n == 29:
            colors = paley(n)
            if colors and check_coloring(n, colors):
                return {'n': n, 'colors': colors}
            # Fallback to local search for 29
            colors = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
            colors = local_search(n, colors, max_time=10.0)
            if check_coloring(n, colors):
                return {'n': n, 'colors': colors}
            continue

        # ... rest
