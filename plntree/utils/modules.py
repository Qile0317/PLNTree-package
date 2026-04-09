import torch
import torch.nn as nn
from plntree.utils.utils import batch_matrix_product


class BoundLayer(nn.Module):
    def __init__(self, min_value, max_value, smoothing_factor=0.2):
        """
        Bounds the input using a soft thresholding function (differentiable).
        The smoothing factor controls divergence regarding the identify function on the unbounded domain.
        Parameters
        ----------
        min_value: float
        max_value: float
        smoothing_factor: float
        """
        super(BoundLayer, self).__init__()
        self.min_value = min(min_value, max_value)
        self.max_value = max(max_value, min_value)
        self.smoothing_factor = smoothing_factor

    def forward(self, x):
        return self.min_value + (self.max_value - self.min_value) * torch.sigmoid(
            self.smoothing_factor * (x - (self.min_value + self.max_value) / 2))


class DenseNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers):
        """
        Dense Neural Network module.
        Parameters
        ----------
        input_size: int
        hidden_size: int
        output_size: int
        n_layers: int
        """
        super(DenseNeuralNetwork, self).__init__()
        self.network = nn.Sequential()
        if n_layers == 0:
            self.network.append(nn.Linear(input_size, output_size))
        else:
            last_input_size = input_size
            for layer in range(n_layers):
                self.network.append(nn.Linear(last_input_size, hidden_size))
                self.network.append(nn.ReLU())
                last_input_size = hidden_size
            self.network.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.network(x)


class CombinedNeuralNetworks(nn.Module):
    def __init__(self, nn1, nn2, n_layers, output_size, output_sequential=None):
        """
        Concatenate two neural network outputs and feed them into a new neural network.
        If no second neural network is provided, only the first one is used without combination layer.
        If an output sequential is provided, it is applied to the output of the combination layer.
        Parameters
        ----------
        nn1: nn.Module
        nn2: None or DenseNeuralNetwork
        n_layers: int or None
        output_size: int or None
        output_sequential: nn.Module or None
        """
        super(CombinedNeuralNetworks, self).__init__()
        self.nn1 = nn1
        self.nn2 = nn2
        self.output_sequential = output_sequential
        # If the second neural network is not provided, we only use the first one
        if self.nn2 is not None:
            self.combined_layers = nn.Sequential()
            if type(nn1) == DenseNeuralNetwork:
                counts_size = nn1.network[-1].out_features
            elif type(nn1) == nn.Sequential:
                counts_size = nn1[-1].network[-1].out_features
            input_size = counts_size + nn2.network[-1].out_features
            if n_layers > 0:
                for layer in range(n_layers):
                    self.combined_layers.append(nn.Linear(input_size, input_size))
                    self.combined_layers.append(nn.ReLU())
                self.combined_layers.append(nn.Linear(input_size, output_size))
            else:
                self.combined_layers.append(nn.Linear(input_size, output_size))

    def forward(self, x, v):
        if self.nn2 is None:
            w = self.nn1(x)
        else:
            w = torch.cat((self.nn1(x), self.nn2(v)), dim=-1)
            w = self.combined_layers(w)
        if self.output_sequential is not None:
            w = self.output_sequential(w)
        return w


class BatchMatrixProduct(nn.Module):

    def __init__(self, matrix):
        """
        Matrix product with a batched tensor (n, p) and a matrix of size (m, p).
        Parameters
        ----------
        matrix: torch.Tensor
        """
        super(BatchMatrixProduct, self).__init__()
        self.A = matrix

    def forward(self, x):
        return batch_matrix_product(self.A, x)


class ClosedFormParameter(nn.Module):

    def __init__(self, size=None, data=None, shift=0.):
        """
        Parameter for which we have a closed-form optimization procedure (unaffected by gradient descent)
        Parameters
        ----------
        size: int
        data: torch.Tensor
        shift: float
        """
        super(ClosedFormParameter, self).__init__()
        if data is None:
            data = torch.randn(size) + shift
        self.data = data

    def forward(self, x):
        batch_size = x.size(0)
        repeat_dim = [1] * len(self.data.shape)
        return self.data.unsqueeze(0).repeat(batch_size, *repeat_dim)


class Cholesky(nn.Module):

    def __init__(self, diagonal=False):
        """
        Turn a vector into a lower triangular matrix with positive diagonal which corresponds to a Cholesky decomposition.
        Parameters
        ----------
        diagonal: bool
        """
        super(Cholesky, self).__init__()
        self.diagonal = diagonal
        self.positive = nn.Softplus()

    def forward(self, L_vec):
        # PATCH: when diagonal=True, accept a vector of length K (diagonal entries only)
        # instead of K*(K+1)/2 (full Cholesky), so the upstream MLP can output K values
        # instead of the full triangular vector. This fixes the bug where diagonal=True
        # still required ~2B params because the MLP output dim was unchanged.
        if self.diagonal:
            # L_vec is (batch, K) — just the diagonal entries
            L_diag = self.positive(L_vec)
            return torch.diag_embed(L_diag, dim1=-2, dim2=-1)
        size = -0.5 + (1 + 8 * L_vec.size(1)) ** .5 / 2
        size = int(size)
        batch = L_vec.size(0)
        # Tril to get the lower diagonal part and the diagonal
        i, j = torch.tril_indices(size, size)
        L = torch.zeros((batch, size, size), dtype=L_vec.dtype, device=L_vec.device)
        L[:, i, j] += L_vec
        L_below = torch.tril(L, diagonal=-1)
        L_diag = torch.diagonal(L, dim1=-2, dim2=-1)
        # Make the diagonal positive to ensure a unique decomposition
        L_diag = self.positive(L_diag)
        # Embed the diagonal in a matrix
        L_diag = torch.diag_embed(L_diag, dim1=-2, dim2=-1)
        L = L_diag + L_below
        # Triu to get the upper diagonal part out of the diagonal and set it to 0
        u, v = torch.triu_indices(size, size, offset=1)
        L[:, u, v] = 0.
        return L


class PositiveDefiniteMatrix(nn.Module):

    def __init__(self, min_diag=1e-8, diagonal=False, projector=None):
        """
        Module to turn an input vector into a positive definite matrix,
        with identifiability regarding a PLN-Tree model if a hierarchical projector is provided.
        Parameters
        ----------
        min_diag: float
            Regularizer of the positive definite matrix to ensure the invertibility.
        diagonal: bool
        projector: torch.Tensor
        """
        super(PositiveDefiniteMatrix, self).__init__()
        self.min_diag = min_diag
        self.diagonal = diagonal

    def forward(self, L_vec):
        cholesky = Cholesky(diagonal=self.diagonal)
        L = cholesky(L_vec)
        Omega = L @ L.mT
        # Regularize the diagonal to ensure invertibility
        i = torch.arange(Omega.size(-1))
        Omega[:, i, i] = Omega[:, i, i] + self.min_diag
        return Omega


class Vect1OrthogonalProjectorHierarchical(nn.Module):

    def __init__(self, tree, layer, K_eff):
        super(Vect1OrthogonalProjectorHierarchical, self).__init__()
        self.tree = tree
        # Create the tensors of projection of shape [L-1, K_l, K_l]
        self.P = self.projector(layer, K_eff)

    def projector(self, layer, K_eff):
        P = torch.eye(K_eff)
        mapping_index = 0
        for parent in self.tree.getNodesAtDepth(layer - 1):
            d = len(parent.children)
            # If it's a lonely child, then it's not accounted for in the modelisation
            if d == 1:
                continue
            Q = torch.eye(d) - torch.ones(d, d) / d
            span = torch.arange(mapping_index, mapping_index + d)
            P[span, span.unsqueeze(1)] = Q.clone()
            mapping_index += d
        return P

    def forward(self, X):
        P = self.P.expand(X.size(0), -1, -1).to(dtype=X.dtype, device=X.device)
        # If it's a vector, project it on Vect(1_d)^orthogonal
        if len(X.shape) == 2:
            return (P @ X.unsqueeze(-1)).squeeze(-1)
        # If it's a matrix, project it on Vect(1_{dxd})^orthogonal
        elif len(X.shape) == 3:
            return P @ X @ P
        return None


class CountsPreprocessing(nn.Module):

    def __init__(self, preprocessing):
        """
        Preprocessing of count data. If no preprocessing type is provided, the data is left as is.
        Parameters
        ----------
        preprocessing: str or list[str]
        """
        super(CountsPreprocessing, self).__init__()
        if preprocessing is None:
            preprocessing = []
        self.log_transform = 'log' in preprocessing
        self.standardize = 'standardize' in preprocessing
        self.normalize = 'normalize' in preprocessing
        self.proportion = 'proportion' in preprocessing
        self.clr = 'clr' in preprocessing

    def forward(self, X):
        x = X.clone()
        if self.clr:
            log_x = torch.log(x + 1e-32)
            return log_x - log_x.mean(dim=-1, keepdim=True)
        if self.proportion:
            x = x / (torch.sum(x, dim=-1, keepdim=True) + 1e-32)
        if self.log_transform is True and not self.proportion:
            x = torch.log(X + 1e-32)
        if self.standardize and not self.proportion:
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-32)
        if self.normalize and not self.proportion:
            x_max = X.max()
            x_min = X.min()
            x = (x - x_min) / (x_max - x_min)
        return x


class PartialCountsPreprocessing(nn.Module):
    def __init__(self, n_features, preprocessing=None):
        """
        Partial preprocessing of counts data, only the first n_features are preprocessed.
        Parameters
        ----------
        n_features: int
        preprocessing: str or None
        """
        super(PartialCountsPreprocessing, self).__init__()
        self.n_features = n_features
        self.preprocessing = CountsPreprocessing(preprocessing)

    def forward(self, input):
        X = input[:, :self.n_features]
        return torch.concat((self.preprocessing(X), input[:, self.n_features:]), dim=-1)


class AttentionHead(nn.Module):

    def __init__(self, counts_size, covariates_size, query_size):
        """
        Attention head module.
        Parameters
        ----------
        counts_size
        covariates_size
        query_size
        """
        super(AttentionHead, self).__init__()
        self.Q = nn.Linear(counts_size, query_size)
        self.K = nn.Linear(covariates_size, query_size)
        self.V = nn.Linear(covariates_size, query_size)

    def forward(self, X, C):
        q = self.Q(X)
        k = self.K(C)
        v = self.V(C)
        a = q @ k.mT / (q.size(-1) ** 0.5)
        a = torch.softmax(a, dim=1) @ v
        return a

class MultiHeadAttention(nn.Module):

    def __init__(self, counts_size, covariates_size, query_size, n_heads):
        """
        Multi-head attention module.
        Parameters
        ----------
        counts_size
        covariates_size
        query_size
        n_heads
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.attention_heads = nn.ModuleList(
            [AttentionHead(counts_size, covariates_size, query_size) for _ in range(n_heads)]
        )
        self.output_size = n_heads * query_size

    def forward(self, X, C):
        return torch.cat([head(X, C) for head in self.attention_heads], dim=-1)

class CovariatesAttentionLayer(nn.Module):

    def __init__(self, counts_size, covariates_size, attention_size, n_heads, output_size, counts_preprocessing=None):
        """
        Attention layer with a feedforward neural network.
        Parameters
        ----------
        counts_size
        covariates_size
        attention_size
        n_heads
        output_size
        counts_preprocessing
        """
        super(CovariatesAttentionLayer, self).__init__()
        self.attention = MultiHeadAttention(counts_size, covariates_size, attention_size, n_heads)
        if n_heads * attention_size != output_size:
            self.linear = nn.Linear(self.attention.output_size, output_size)
        else:
            self.linear = nn.Identity()
        self.counts_preprocessing = CountsPreprocessing(counts_preprocessing)

    def forward(self, X, C):
        X = self.counts_preprocessing(X)
        a = self.attention(X, C)
        return self.linear(a)

class AdditiveNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers):
        """
        Additive Neural Network module.
        Parameters
        ----------
        input_size
        hidden_size
        output_size
        n_layers
        """
        super(AdditiveNeuralNetwork, self).__init__()
        self.network = DenseNeuralNetwork(input_size, hidden_size, output_size, n_layers)
        self.norm = nn.BatchNorm1d(input_size)
    def forward(self, Z, A):
        S = Z + A
        S = self.norm(S)
        return self.network(S)

class AttentionBasedCombiningLayer(nn.Module):

    def __init__(self, counts_network, input_dim, output_dim, covariates_dim, n_heads, n_output_layers, output_module=None):
        """
        Attention-based combining layer between the counts and their associated covariates.
        Parameters
        ----------
        counts_network: nn.Module
        input_dim: int
        output_dim: int
        covariates_dim: int
        n_heads: int
        n_output_layers: int
        output_module: nn.Module
        """
        super(AttentionBasedCombiningLayer, self).__init__()
        self.counts_network = counts_network
        attention_dim = input_dim // n_heads
        self.cov_attention = CovariatesAttentionLayer(
            counts_size=input_dim,
            covariates_size=covariates_dim,
            attention_size=attention_dim,
            n_heads=n_heads,
            output_size=input_dim
        )
        self.additive_layer = AdditiveNeuralNetwork(
            input_size=input_dim, # Input size must be the output size of the attention layer, and that of the counts network
            hidden_size=output_dim,
            output_size=output_dim,
            n_layers=n_output_layers
        )
        self.output_module = output_module

        self.output_dim = output_dim
        self.input_dim = input_dim

    def forward(self, X, C):
        Z = self.counts_network(X)
        A = self.cov_attention(Z, C)
        output = self.additive_layer(Z, A)
        if self.output_module is not None:
            output = self.output_module(output)
        return output

class FiLM(nn.Module):

    def __init__(self, covariates_size, features_size, n_layers):
        """
        FiLM module for feature-wise linear modulation using covariates.

        Parameters
        ----------
        covariates_size: int
        features_size: int
        n_layers: int
        """
        super(FiLM, self).__init__()
        self.network = DenseNeuralNetwork(
            input_size=covariates_size,
            hidden_size=features_size * 2,
            output_size=features_size * 2,
            n_layers=n_layers
        )
        self.features_size = features_size

    def forward(self, X, C):
        output = self.network(C)
        alpha, beta = output[:, :self.features_size], output[:, self.features_size:]
        return alpha * X + beta

class AttentionFiLM(nn.Module):

    def __init__(self, counts_network, counts_preprocessing, covariates_dim, counts_dim, n_film_layers, output_module=None):
        """
        Attention FiLM module to inject covariates in counts modeling
        Parameters
        ----------
        counts_network: nn.Module
        covariates_dim: int
        output_module: nn.Module or None
        """
        super(AttentionFiLM, self).__init__()
        self.counts_network = counts_network
        self.film = FiLM(covariates_dim, counts_dim, n_layers=n_film_layers)
        self.output_module = output_module
        self.counts_preprocessing = counts_preprocessing


    def forward(self, X, C):
        if self.counts_preprocessing is not None:
            X_proc = self.counts_preprocessing(X)
        else:
            X_proc = X
        A = self.film(X_proc, C)
        Z = self.counts_network(A)
        if self.output_module is not None:
            return self.output_module(Z)
        return Z

