# Minimal pure-Python torch stub to avoid heavy native imports (won't hang)
class Tensor:
    def __init__(self, data):
        self.data = data
    @property
    def shape(self):
        if isinstance(self.data, list):
            if len(self.data) and isinstance(self.data[0], list):
                return (len(self.data), len(self.data[0]))
            return (len(self.data),)
        return ()
    def __getitem__(self, idx):
        item = self.data[idx]
        return Tensor(item) if isinstance(item, list) else item
    def __repr__(self):
        return f"Tensor({self.data})"

def tensor(x):
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
        return Tensor([list(row) for row in x])
    if isinstance(x, (list, tuple)):
        return Tensor(list(x))
    return Tensor(x)

def stack(seq, dim=0):
    # only supports stacking identical lists into 2D
    return Tensor([s.data if isinstance(s, Tensor) else s for s in seq])

def ones(*shape):
    # simple ones for 2D/1D
    if len(shape) == 1:
        return Tensor([1]*shape[0])
    return Tensor([[1]*shape[1] for _ in range(shape[0])])

def tril(x):
    return x  # no-op for stub

def triu(x, diagonal=0):
    return x  # no-op for stub

def softmax(x, dim=-1):
    return x  # no-op stub; suitable only to avoid import hangs

class _nn:
    class Module: pass
    class Linear:
        def __init__(self, in_features, out_features, bias=True):
            self.in_features = in_features
            self.out_features = out_features
        def __call__(self, x):
            return x
nn = _nn()

# simple helpers used by tutorials
inf = float('inf')