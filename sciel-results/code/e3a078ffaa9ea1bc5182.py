<<<<<<< SEARCH
if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print(f"Rank: {rank}")
    
    # Quick verification
    U_arr = np.array(result['U'])
    V_arr = np.array(result['V'])
    W_arr = np.array(result['W'])
    n, m, p = 3, 4, 8
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0
                for rr in range(rank):
                    t_val += result['U'][i*m+j][rr] * result['V'][j*p+k][rr] * result['W'][k*n+i][rr]
                if abs(t_val - 1) > 1e-9:
                    max_err = max(max_err, abs(t_val - 1))
    print(f"Max tensor error: {max_err}")
=======
if __name__ == "__main__":
    start = time.time()
    result = construct()
    rank = len(result['U'][0])
    elapsed = time.time() - start
    print(f"Rank: {rank} (time: {elapsed:.1f}s)")
    
    # Quick verification
    U_arr = np.array(result['U'])
    V_arr = np.array(result['V'])
    W_arr = np.array(result['W'])
    n, m, p = 3, 4, 8
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0
                for rr in range(rank):
                    t_val += result['U'][i*m+j][rr] * result['V'][j*p+k][rr] * result['W'][k*n+i][rr]
                if abs(t_val - 1) > 1e-9:
                    max_err = max(max_err, abs(t_val - 1))
    print(f"Max tensor error: {max_err}")
>>>>>>> REPLACE
