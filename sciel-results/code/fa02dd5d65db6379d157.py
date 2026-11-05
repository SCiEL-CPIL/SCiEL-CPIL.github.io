def construct():
    return {'edges': [(0,i) for i in range(1,N)] + [(i,i+1) for i in range(1,N-1)],
            'cycles': [list(range(k)) for k in range(3,N+1)]}
