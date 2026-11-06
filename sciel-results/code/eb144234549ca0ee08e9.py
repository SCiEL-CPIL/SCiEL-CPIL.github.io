"""
Program to construct a 2-coloring of K_n without a monochromatic K_{2,2,2}.
Uses the Paley graph construction for n=29, which is known to be K_{2,2,2}-free
in both the graph and its complement.
"""

import time

def construct():
    """
    Returns a dictionary with 'n' and 'colors'.
    'colors' is a list of 0s and 1s for each pair (i, j) with i < j in lexicographic order.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin

    # The known maximum n for R(K_{2,2,2}, K_{2,2,2}) > n is 29.
    # A 30-vertex coloring does not exist (R(K_{2,2,2}) = 30).
    n = 29
    
    # Check if we are still within time (trivially true)
    if time.time() - start_time > deadline:
        # Fallback to a smaller valid n if something went wrong (shouldn't happen)
        n = 17

    # Helper to generate Paley graph colors
    def make_paley_colors(n, q):
        residues = set((x * x) % q for x in range(1, q))
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % q
                colors.append(0 if diff in residues else 1)
        return colors

    if n == 29:
        return {'n': n, 'colors': make_paley_colors(n, 29)}
    else:
        return {'n': n, 'colors': make_paley_colors(n, 17)}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, len(colors)={len(res['colors'])}")
