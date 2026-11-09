class SimpleMLP:
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros((1, dims[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
    def forward(self, X):
        self.activations = [X]
        current = X
        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            current = np.maximum(0, z)
            self.activations.append(current)
        z = current @ self.weights[-1] + self.biases[-1]
        self.activations.append(z)
        return z

    def backward(self, y_true, y_pred, weight_decay=0.001):
        m = y_true.shape[0]
        dz = (2 * (y_pred - y_true)) / m
        dw = self.activations[-2].T @ dz + weight_decay * self.weights[-1]
        db = np.sum(dz, axis=0, keepdims=True)
        
        dws = [dw]
        dbs = [db]
        
        for i in range(len(self.weights) - 2, -1, -1):
            dz = (dz @ self.weights[i+1].T) * (self.activations[i+1] > 0)
            dw = self.activations[i].T @ dz + weight_decay * self.weights[i]
            db = np.sum(dz, axis=0, keepdims=True)
            dws.insert(0, dw)
            dbs.insert(0, db)
            
        return dws, dbs
