# gpr_aniso_trials

Different strategies to implement noiseless regression in Python.
The problem is a 3-D Euclidean grid, split into three case sizes to select from if you can't run big cases:
  full: 141×66×5.
  mid: 36×66×5
  small: 36×33×5

The regression options are:
* Gaussian Processes: (all with ARD kernels, apparently this problem won't fit otherwise)
  * scikit
  * gpflow (based on tensorflow)
  * GPYTorch Cbased on PyTorch)
* Neural networks
  * classical fully connected (keras)
  * multi-head self-attention (keras)
