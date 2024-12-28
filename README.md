## Residual blocks

### A residual block adds a "skip connection" that lets information bypass some layers in a neural network:

Key components:
1. Input splits into two paths
2. Main path: Goes through conv layers/activations
3. Skip path: Direct connection bypassing those layers
4. Paths merge via addition at the end

Benefits:
1. Prevents vanishing gradients in deep networks
2. Easier optimization since network can learn residual function
3. Better information flow through network
