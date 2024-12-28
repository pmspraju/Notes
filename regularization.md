## Gradient Centralization

### Gradient Centralization (GC) is a training technique that centers the gradients of neural network weights by subtracting their mean. The operation is:

```
grad = grad - grad.mean(dim=list(range(1,len(grad.shape))), keepdim=True)
```

Key benefits:
1. Improves model convergence and generalization
2. Acts as implicit regularization
3. Computationally efficient - only requires mean subtraction
4. Can be easily integrated into existing optimizers

#### is this a kind of gradient normalization ?

No - while both techniques optimize gradients, they work differently:
Gradient Normalization divides gradients by their norm (magnitude)
Gradient Centralization subtracts the mean to center gradients around zero

GC is more similar to batch normalization but operates on gradients rather than activations.

#### if we subtract mean, is not there a chance that gradients become negative ?

Yes, gradients can become negative after subtracting the mean - this is normal and expected. 
Neural networks can learn effectively with both positive and negative gradients. 
The key is that the gradients are centered around zero, which helps stabilize training by removing any inherent bias in the gradient distribution.
Think of it like normalizing data around zero - having both positive and negative values helps capture the full range of directional updates needed for optimization.


