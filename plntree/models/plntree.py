import time
from tqdm.auto import tqdm

import torch.nn as nn
import torch.optim as optim
import torch.random
from torch.distributions import Poisson

import numpy as np
import pandas as pd

from plntree.utils.utils import (Tree, format_time_HH_MM_SS, dataloader_counts_covariates, batch_matrix_product, trace,
                                    lowrank_multivariate_normal_samples)
from plntree.utils.modules import (DenseNeuralNetwork, ClosedFormParameter, PositiveDefiniteMatrix, AttentionBasedCombiningLayer,
                                   CombinedNeuralNetworks, BatchMatrixProduct, AttentionFiLM)
from plntree.utils.variational_approximations import (mean_field, weak_backward_markov, residual_backward_markov)

from plntree.utils.utils import load_model_weights, save_model_weights

class PLNTree(nn.Module):

    def __init__(
            self,
            counts,
            covariates=None,
            offsets='zeros',
            latent_dynamic=None,
            variational_approx=None,
            covariates_params=None,
            level_regex='|',
            clade_regex='__',
            smart_init=True,
            device=None,
            seed=None,
    ):
        """
        PLN-Tree PyTorch module builder.

        The hierarchy indicates the tree structure of the data.
        Example:
            ['1__Entity|2__Mammal|3__Human',
            '1__Entity|2__Mammal|3__Dolphin',
            '1__Entity|2__Reptile|3__Lizard',
            '1__Entity|2__Reptile|3__Snake']

        Parameters
        ----------
        counts: pd.DataFrame
            Count dataframe with samples as rows and features as columns. Columns head should match the hierarchy regex format.
        covariates: pd.DataFrame
            Covariates dataframe with samples as rows and features as columns. If None, no covariates are used.
        offsets: str
            Offset type for the counts. Can be 'zeros' (default) or 'logsum'.
        latent_dynamic: dict
            Parameters for the latent dynamic model. If None, defaults to a Markov linear model with 1 layer.
        variational_approx: dict
            Parameters for the variational approximation method. If None, defaults to a residual backward Markov approximation.
        covariates_params: dict
            Parameters for the covariates processing. If None, defaults to FiLM with 1 layer preprocessing.
        level_regex: str
            Hierarchy regex to split the features into levels. Default is '|'.
        clade_regex: str
            Clade regex to split the features into clades. Default is '__'.
        smart_init: bool
            Initialize parameters at the first level based on the input data. If True, uses a smart initialization method.
        device: str
            Use 'cpu' or 'cuda' to specify the device for the model. If None, defaults to 'cpu'.
        seed: int
            Seed for the random number generator to ensure reproducibility. If None, no seed is set.
        """
        super(PLNTree, self).__init__()
        if seed is not None:
            torch.random.manual_seed(seed)
        self.seed = seed
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device
        self.use_smart_init = smart_init
        self.hierarchy = list(counts.columns)
        if offsets is None:
            offsets = 'zeros'
        self.offsets = offsets
        self.latent_dynamic = latent_dynamic
        self.variational_approx = variational_approx

        # Default latent dynamic is Markov linear
        if self.latent_dynamic is None:
            self.latent_dynamic = {
                'n_layers': 1,
                'diagonal': False,
                'covariates_layers': 1,
                'combining_layers': 1,
                'markov_covariance': True,
                'markov_means': True,
            }
        if 'n_layers' not in self.latent_dynamic:
            self.latent_dynamic['n_layers'] = 1
        if 'diagonal' not in self.latent_dynamic:
            self.latent_dynamic['diagonal'] = False
        if 'covariates_layers' not in self.latent_dynamic:
            self.latent_dynamic['covariates_layers'] = 1
        if 'combining_layers' not in self.latent_dynamic:
            self.latent_dynamic['combining_layers'] = 1
        if 'markov_covariance' not in self.latent_dynamic:
            self.latent_dynamic['markov_covariance'] = True
        if 'markov_means' not in self.latent_dynamic:
            self.latent_dynamic['markov_means'] = True
        if self.latent_dynamic['n_layers'] <= 0:
            self.latent_dynamic['markov_covariance'] = False
            self.latent_dynamic['markov_means'] = False
        # Default variational approximation is residual backward with proportion Embedder
        if self.variational_approx is None:
            self.variational_approx = {
                'method': 'residual',
                'embedder_type': 'GRU',
                'embedding_size': 32,
                'n_embedding_layers': 2,
                'n_embedding_neurons': 32,
                'n_after_layers': 1,
                'counts_preprocessing': 'proportion',
                'covariates_layers': 1,
                'combining_layers': 1,
            }
        # Default covariates parameters
        if covariates_params is None:
            covariates_params = {}
        if 'type' not in covariates_params:
            covariates_params['type'] = 'film'
        if 'n_heads' not in covariates_params:
            covariates_params['n_heads'] = 4
        if 'n_layers' not in covariates_params:
            covariates_params['n_layers'] = 1
        self.covariates_params = covariates_params
        self.covariates_params['size'] = 0

        # Build the tree structure from the hierarchy provided as a string
        self.tree = Tree(self.hierarchy, level_regex=level_regex, clade_regex=clade_regex, device=self.device)
        # Reorder the counts according to the tree structure
        self.counts = counts[self.tree.full_sorted_names]
        # Pre-compute the hierarchical counts associated with the provided counts (used in ELBO)
        self.hierarchical_counts = self.tree.hierarchical_counts(self.counts)
        # Define the covariates as tensors.
        self.covariates = covariates
        if self.covariates is not None:
            # Reorder the covariates lines to match the counts
            # Shape should be (n_samples, n_features)
            self.covariates = covariates.loc[self.counts.index].to_numpy()
            self.covariates = torch.tensor(self.covariates)
            # Add a bias term to the covariates
            constant_covariates = torch.ones(self.hierarchical_counts.size(0), 1)
            self.covariates = torch.cat([constant_covariates, self.covariates], dim=1).to(device=self.device, dtype=self.hierarchical_counts.dtype)
            self.covariates_params['size'] = self.covariates.size(1)
        # Build the projector matrices for the identifiability if used
        self.projectors = []
        for level in range(self.tree.L):
            K_l = self.tree.K_eff[level]
            if level == 0:
                # The first level parameters are always identifiable, no need to project them
                self.projectors += [BatchMatrixProduct(torch.eye(K_l, K_l).to(device=self.device, dtype=self.hierarchical_counts.dtype))]
            else:
                # Build the block-wise span(1) orthogonal projector i.e. clade wise projector
                P_l = torch.zeros(K_l, K_l).to(self.device)
                clades_l = self.tree.clades[level]
                clades, n_members = np.unique(clades_l, return_counts=True)
                assert np.sort(clades).all() == clades.all(), f"Clades are not sorted. Projector at level {level} could not be built."
                d_prev = 0
                for clade_index, d in enumerate(n_members):
                    # Zero is not a clade type but a padding, so we skip it
                    if clades[clade_index] == 0:
                        continue
                    # Skip the only child clade as it is not meant to be projected
                    if d == 1:
                        continue
                    span = torch.arange(d_prev, d_prev + d).to(self.device)
                    P_l[span, span.unsqueeze(1)] = torch.eye(d).to(self.device) - torch.ones(d, d).to(self.device) / d
                    d_prev += d
                self.projectors += [BatchMatrixProduct(P_l)]

        # Define the neural networks to parameterize the forward latent process
        self.mu_fun = nn.ModuleList()
        self.omega_fun = nn.ModuleList()
        for level in range(self.tree.L):
            # Level 0 is basically PLN, and we have closed-form expressions relatively to variational parameters
            # So we create parameters but they are not subject to the gradient optimization
            if level == 0:
                K_l = self.tree.K_eff[level]
                if covariates is not None:
                    # If we have covariates, the mean is given by a linear regression on the covariates
                    # So we only store B the regression coefficients and mu will be computed as C @ B
                    covariates_size = self.covariates.size(-1)
                    B_l = torch.zeros(covariates_size, K_l)
                    mu_l = ClosedFormParameter(data=B_l.to(device=self.device, dtype=self.hierarchical_counts.dtype))
                else:
                    # If we have no covariates, we just leave a bias to adjust the mean
                    mu_l = ClosedFormParameter(data=torch.zeros(K_l).to(device=self.device, dtype=self.hierarchical_counts.dtype))
                omega_l = ClosedFormParameter(data=torch.eye(K_l).to(device=self.device, dtype=self.hierarchical_counts.dtype))
            else:
                K_l, K_l_prev = self.tree.K_eff[level], self.tree.K_eff[level-1]
                # Positive definite module for Omega
                pdm = PositiveDefiniteMatrix(min_diag=1e-4, diagonal=self.latent_dynamic['diagonal'])

                # Define the propagation networks of the parameters, either with or without covariates
                if covariates is None:
                    if self.latent_dynamic['markov_means']:
                        mu_l = nn.Sequential(
                            DenseNeuralNetwork(K_l_prev, K_l_prev, K_l, self.latent_dynamic['n_layers'])
                        )
                        mu_l = CombinedNeuralNetworks(mu_l, None, None, None, None)
                    else:
                        mu_l = ClosedFormParameter(
                            data=torch.zeros(K_l).to(device=self.device, dtype=self.hierarchical_counts.dtype))

                    if self.latent_dynamic['markov_covariance']:
                        # PATCH: when diagonal=True, output K_l values (diagonal only)
                        # instead of K_l*(K_l+1)//2 (full Cholesky), avoiding ~2B wasted params
                        omega_out_dim = K_l if self.latent_dynamic['diagonal'] else K_l * (K_l + 1) // 2
                        omega_l = nn.Sequential(
                            DenseNeuralNetwork(K_l_prev, K_l_prev, omega_out_dim,
                                               self.latent_dynamic['n_layers'])
                        )
                        omega_l = CombinedNeuralNetworks(omega_l, None, None, None, pdm)
                    else:
                        omega_l = ClosedFormParameter(
                            data=torch.eye(K_l).to(device=self.device, dtype=self.hierarchical_counts.dtype))
                else:
                    covariates_size = self.covariates.size(-1)
                    if self.latent_dynamic['markov_means']:
                        mu_l_X = nn.Sequential(
                            DenseNeuralNetwork(K_l_prev, K_l_prev, K_l, self.latent_dynamic['n_layers'])
                        )
                        if covariates_params['type'] == 'film':
                            mu_l = AttentionFiLM(
                                counts_network=mu_l_X,
                                counts_preprocessing=None,
                                covariates_dim=covariates_size,
                                counts_dim=K_l_prev,
                                n_film_layers=self.covariates_params['n_layers'],
                                output_module=None
                            )
                        else:
                            mu_l = AttentionBasedCombiningLayer(
                                mu_l_X, K_l, K_l, covariates_size,
                                self.covariates_params['n_heads'], self.covariates_params['n_layers'], None,
                            )
                    else:
                        B_l = torch.zeros(covariates_size, K_l)
                        mu_l = ClosedFormParameter(
                            data=B_l.to(device=self.device, dtype=self.hierarchical_counts.dtype))

                    if self.latent_dynamic['markov_covariance']:
                        # PATCH: when diagonal=True, output K_l values (diagonal only)
                        # instead of K_l*(K_l+1)//2 (full Cholesky), avoiding wasted params
                        cholesky_size = K_l if self.latent_dynamic['diagonal'] else K_l * (K_l + 1) // 2
                        omega_l_X = nn.Sequential(
                            DenseNeuralNetwork(K_l_prev, K_l_prev, cholesky_size, self.latent_dynamic['n_layers'])
                        )
                        if covariates_params['type'] == 'film':
                            omega_l = AttentionFiLM(
                                counts_network=omega_l_X,
                                counts_preprocessing=None,
                                covariates_dim=covariates_size,
                                counts_dim=K_l_prev,
                                n_film_layers=self.covariates_params['n_layers'],
                                output_module=pdm
                            )
                        else:
                            omega_l = AttentionBasedCombiningLayer(
                                omega_l_X, cholesky_size, cholesky_size, covariates_size,
                                self.covariates_params['n_heads'], self.covariates_params['n_layers'],
                                pdm,
                            )
                    else:
                        omega_l = ClosedFormParameter(
                            data=torch.eye(K_l).to(device=self.device, dtype=self.hierarchical_counts.dtype))
            self.mu_fun.append(mu_l.to(self.device))
            self.omega_fun.append(omega_l.to(self.device))

        # We define the variational approximation parameters
        if self.variational_approx['method'] == 'mean_field':
            self.m_fun, self.log_S_fun = mean_field(
                tree=self.tree,
                covariates_params=self.covariates_params,
                n_layers=self.variational_approx['n_layers'],
                preprocessing=self.variational_approx['counts_preprocessing']
            )
        elif self.variational_approx['method'] == 'weak':
            self.m_fun, self.log_S_fun = weak_backward_markov(
                tree=self.tree,
                covariates_params=self.covariates_params,
                n_layers=self.variational_approx['n_layers'],
                preprocessing=self.variational_approx['counts_preprocessing']
            )
        elif self.variational_approx['method'] == 'residual':
            self.m_fun, self.log_S_fun, self.embedder = residual_backward_markov(
                tree=self.tree,
                covariates_params=self.covariates_params,
                embedder_type=self.variational_approx['embedder_type'],
                embedding_size=self.variational_approx['embedding_size'],
                n_embedding_layers=self.variational_approx['n_embedding_layers'],
                n_embedding_neurons=self.variational_approx['n_embedding_neurons'],
                n_after_layers=self.variational_approx['n_after_layers'],
                preprocessing=self.variational_approx['counts_preprocessing']
            )
        else:
            raise AssertionError(f'Variational approximation \'' + self.variational_approx['method'] + '\' was not recognized.')

    def fit(
            self,
            max_epoch=10_000,
            learning_rate=1e-3, grad_clip=5.,
            tolerance=1e-4, tolerance_smoothing=1000,
            batch_size=512, shuffle=True,
            monitor=None,
            verbose=1000,
            seed=None
    ):
        """
        Learn the PLN-Tree model on the instance provided counts with the given covariates using Adam optimizer.
        Parameters
        ----------
        max_epoch: int
            Maximum number of epochs to train the model.
        learning_rate: float
            Adam optimizer learning rate.
        grad_clip: float
            Gradient clipping value to avoid exploding gradients.
        tolerance: float
            Tolerance criterion to ELBO convergence, set to infinity to disable.
        tolerance_smoothing: int
            Tolerance smoothing window.
        batch_size: int
            Batch size for gradient descent.
        shuffle: bool
            Shuffle the training data.
        monitor: dict or None
            Monitoring parameters. If None, no monitoring is performed. If provided, monitor affects the training process through an offline early stopping.
        verbose: int
            Iteration to update the progress bar.
        seed: int
            Seed for the random number generator to ensure reproducibility.

        Returns
        -------
        list[float]
        """
        # Move the model to the selected device
        self.to(self.device)
        # Get ready for training in PyTorch framework
        self.train()
        if verbose is None:
            verbose = -1
        if monitor is not None:
            if type(monitor) != dict:
                monitor = {}
            if 'step' not in monitor:
                monitor['step'] = 200
            if 'repeat' not in monitor:
                monitor['repeat'] = 25
            if 'vamp' not in monitor:
                monitor['vamp'] = False
            if 'file' not in monitor:
                monitor['file'] = 'plntree_monitor'
            if 'function' not in monitor or monitor['function'] is None:
                print('[WARNING]: No monitoring function has been provided. The monitor will be disabled.')
                monitor = None
            if monitor['vamp']:
                print('[MONITOR]: Using VAMP sampling for monitoring.')
            print('[MONITOR]: Monitoring every ' + str(monitor['step']) + ' epochs, repeating ' + str(monitor['repeat']) + ' times.')
            monitor_values = []
            best_monitor = np.inf
        if seed is not None:
            torch.random.manual_seed(seed)
        # We use the Adam optimizer in PLN-Tree (other optimizers have not shown quite as good results)
        optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
        )
        # Build the dataloader from the input counts and covariates
        dataloader = dataloader_counts_covariates(self.hierarchical_counts, self.covariates, batch_size=batch_size, shuffle=shuffle)
        # Save the values of the loss over the epochs
        losses = []
        # Initialize the parameters of the first level based on the input data
        if self.use_smart_init:
            if verbose > 0:
                print('Initializing the first level parameters based on the input data...')
            self.smart_init()
        # Compute tolerance on the smoothed ELBO over numerous evaluations
        tol = np.inf
        # Compute the current epoch
        epoch = 0
        # Compute the total time
        total_timer = time.time()
        # Progress bar if verbose is on
        if verbose > 0:
            print('Learning PLN-Tree model on ' + str(self.device) + ' device.')
            pbar = tqdm(
                total=max_epoch, desc=f'Training PLN-Tree ({self.device}) over {max_epoch} epochs (update every {verbose})',
                unit='epoch', miniters=verbose
            )
        while tol > tolerance and epoch <= max_epoch:
            # Averaged loss over the full dataset
            loss = 0.
            batch_idx = 0
            for batch_idx, (X, C) in enumerate(dataloader):
                # If no covariates are accepted in the model, w
                # we make sure to remove them as it is not possible to provide None in torch dataloaders
                if self.covariates is None:
                    C = None
                optimizer.zero_grad()
                batch_loss = self.objective(X, self.forward(X, C))
                batch_loss.backward()
                loss += batch_loss.item()
                # Clip the gradient to avoid exploding steps
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                optimizer.step()
                # Closed-form optimization does not require gradient computations
                with torch.no_grad():
                    self.update_closed_forms(C, self.forward(X, C))
            loss = loss / (batch_idx+1)
            if np.isnan(loss):
                print(f'[ERROR] The loss is NaN (epoch {epoch+1}). Please adjust the training parameters (learning rate, gradient clipping) and verify the integrity of the data (non-zero sum).')
                break
            losses += [loss]
            epoch += 1

            if monitor is not None:
                if epoch % monitor['step'] == 0:
                    n_samples = self.hierarchical_counts.shape[0] * int(monitor['repeat'])
                    if monitor['vamp']:
                        X_monitor = self.vamp_sample(n_samples)[0]
                    else:
                        if self.covariates is None:
                            X_monitor = self.sample(n_samples)[0]
                        else:
                            # Repeat the covariances to match the number of samples
                            cov_monitor = torch.cat([self.covariates]*int(monitor['repeat']), dim=0)
                            X_monitor = self.sample(n_samples, covariates=cov_monitor)[0]
                    monitor_value = monitor['function'](self.hierarchical_counts, X_monitor)
                    if monitor_value < best_monitor:
                        monitor_epoch = epoch
                        best_monitor = monitor_value
                        save_model_weights(self, monitor['file'])
                    monitor_values.append(monitor_value)

            if len(losses) > 2*tolerance_smoothing and epoch % tolerance_smoothing == 0:
                smooth_loss = np.mean(losses[-tolerance_smoothing:])
                smooth_loss_prev = np.mean(losses[-2*tolerance_smoothing:-tolerance_smoothing])
                tol = np.abs(smooth_loss - smooth_loss_prev) / np.abs(smooth_loss_prev)
            if verbose > 0 and epoch % verbose == 0:
                if tol != np.inf:
                    pbar.set_description(
                        f"PLN-Tree ({self.device}) | " + 'Tolerance: {:.4f} | '.format(tol) + 'ELBO: {:.1f}'.format(np.mean(losses[-verbose:]))
                    )
                else:
                    pbar.set_description(
                        f"PLN-Tree ({self.device}) | " + 'ELBO: {:.1f}'.format(
                            np.mean(losses[-verbose:]))
                    )
                pbar.update(verbose)

        if verbose > 0:
            pbar.set_description(
                f"PLN-Tree ({self.device}) | " + 'Tolerance: {:.4f} | '.format(tol) + 'ELBO: {:.1f}'.format(
                    np.mean(losses[-verbose:]))
            )
            pbar.update(verbose)
        if verbose > 0 and tol <= tolerance:
            print('[-] Tolerance threshold reached in ' + format_time_HH_MM_SS(time.time() - total_timer) + ' at epoch ' + str(epoch) + '.')
        elif tol > tolerance and 2*tolerance_smoothing < max_epoch:
            print('[WARNING] The tolerance threshold was not reached, the model may have not converged yet. Tolerance: {:.4f}'.format(tol))
        # We are in evaluation mode after training
        self.eval()
        if monitor is not None:
            try:
                load_model_weights(self, monitor['file'])
                if verbose > 0:
                    print('[MONITOR]: Selected model parameters at epoch ' + str(monitor_epoch) + '.')
                return losses, monitor_values
            except:
                print('[WARNING] The monitor could not load the best model. The last model will be returned.')
                return losses, monitor_values
        return losses

    def smart_init(self):
        """
        Initialize the first parameters of the first level of the hierarchy based on estimators of the
        mean and covariance at this level.
        Returns
        -------
        PLNTree
        """
        X = self.hierarchical_counts[:, 0, :self.tree.K[0]]
        log_X_centered = torch.log(X + 1e-8) - torch.log(X + 1e-8).mean(axis=0)
        n_samples = X.size(0)
        Sigma_hat = log_X_centered.T @ log_X_centered / (n_samples - 1) + 1e-4 * torch.eye(self.tree.K[0]).to(device=self.device)
        self.omega_fun[0].data = torch.linalg.inv(Sigma_hat)
        if self.covariates is None:
            self.mu_fun[0].data = torch.log(X + 1e-8).mean(axis=0)
        else:
            try:
                C_TCinv = torch.linalg.inv(self.covariates.mT @ self.covariates)
            except:
                C_TCinv = torch.linalg.pinv(self.covariates.mT @ self.covariates)
            self.mu_fun[0].data = C_TCinv @ self.covariates.mT @ torch.log(X + 1e-8)
        return self

    def objective(self, X, output):
        """
        Compute minus the ELBO of the PLN-Tree model
        Parameters
        ----------
        X: torch.Tensor
        output: tuple

        Returns
        -------
        torch.Tensor
        """
        Z, O, m, log_S, mu, Omega = output
        # Initialize the ELBO value
        elbo = 0.
        # Normalization factor to stabilize the computation
        batch_size = X.size(0)
        norm_factor = float(batch_size * len(self.tree.K))
        # We loop over the layer, but we skip the only-child which do not participate the ELBO value
        for level, mask in enumerate(self.tree.K_mask):
            # Fetch the parameters at the current level
            m_l, log_S_l = m[level], log_S[level]
            mu_l, Omega_l = mu[level], Omega[level]

            # If we are at the first level, we should consider the offset
            if level == 0:
                mu_l = mu_l + O.view(-1, 1)
                m_l = m_l + O.view(-1, 1)

            # S_l is diagonal so the determinant is fast to compute: det_S_l = prod(diagonal(S_l, dim1=-2, dim2=-1), dim=1)
            log_det_S_l = torch.sum(log_S_l, dim=1)
            S_l = torch.exp(log_S_l)

            M = (mu_l - m_l).unsqueeze(-1)
            Sigma_hat = M @ M.mT + torch.diag_embed(S_l, dim1=-2, dim2=-1)
            # The log det of Omega can be computationally unstable, and it is not necessary to compyte it
            # if we are using a closed-form expression (no gradient propagation)
            # To avoid unnecessary NaN issues possible during the training, we skip it if it contains a NaN and
            # if it has no registered gradients
            if Omega_l.requires_grad:
                try:
                    # That is generally more stable than logdet, and since Omega is positive definite it's sign is +1
                    log_det_Omega_l = torch.linalg.slogdet(Omega_l)[1]
                except:
                    log_det_Omega_l = torch.logdet(Omega_l)
            else:
                log_det_Omega_l = 0.
            trace_SigmaOmega = trace(Sigma_hat @ Omega_l)

            elbo += 0.5 * torch.sum(log_det_Omega_l - trace_SigmaOmega + log_det_S_l) / norm_factor
            elbo += torch.sum(X[:, level, mask] * m_l) / norm_factor
            if level == 0:
                elbo += -torch.sum(torch.exp(m_l + S_l / 2)) / norm_factor
            else:
                X_l_prev = X[:, level-1, :self.tree.K[level-1]]
                sumexp_Z_l = batch_matrix_product(
                    self.tree.affiliation_matrix[level],
                    torch.exp(Z[:, level, :self.tree.K[level]])
                )
                elbo += -torch.sum(X_l_prev * torch.log(sumexp_Z_l)) / norm_factor

            elbo += -batch_size * self.tree.K_eff[level] / (2 * norm_factor)

        # The ELBO accounts for a log factorial term: sum log(X_L!) = sum log(gamma(X_L + 1))
        X_L = X[:, -1, :self.tree.K[-1]]
        elbo += -torch.lgamma(X_L + 1).sum() / norm_factor
        # Since we use gradient descent algorithms, we output minus the ELBO (since we wanted to maximize the ELBO)
        return -elbo

    def update_closed_forms(self, C, output):
        """
        Updates the parameters of the first level conditionally to the variational parameters as explicit estimates can be computed
        rather than performing iterative optimization.
        Parameters
        ----------
        C: torch.Tensor
            Covariates associated with the input counts of shape (n, d).
        output: tuple
            Model forward outputs encoding X containing the variational parameters and the latents.

        Returns
        -------
        PLNTree
        """
        Z, _, m, log_S, _, _ = output
        for level in range(self.tree.L):
            if level == 0 or not self.latent_dynamic['markov_means']:
                # Closed-form optimization is similar to that of the PLN models when available
                variational_means = m[level]
                if self.covariates is not None:
                    try:
                        C_TCinv = torch.linalg.inv(C.T @ C)
                    except:
                        C_TCinv = torch.linalg.pinv(C.T @ C)
                    # mu is given by the linear regression on the covariates
                    self.mu_fun[level].data = C_TCinv @ C.T @ variational_means
                else:
                    # mu is given by the mean of the latents mean
                    self.mu_fun[level].data = variational_means.mean(axis=0)
            if level == 0 or not self.latent_dynamic['markov_covariance']:
                variational_means = m[level]
                if self.covariates is not None:
                    mu = self.mu(level, None, C)
                else:
                    # Without covariates, we provide a placeholder value to obtain mu values duplicated along the indices
                    mu = self.mu(level, Z[:, level-1, self.tree.K_mask[level-1]], torch.zeros(Z.shape[0]).to(device=self.device))
                # Omega is given by the inverse of the mean of the empirical covariances
                S = torch.exp(log_S[level])
                # stacking mu new values based on Z batch size (no other impact)
                M = (mu - variational_means).unsqueeze(-1)
                Sigma_hat = M @ M.mT + torch.diag_embed(S, dim1=-2, dim2=-1).to(device=self.device)
                Sigma = Sigma_hat.mean(dim=0)
                self.omega_fun[level].data = torch.linalg.pinv(Sigma)
        return self

    def variational_forward(self, X, C, seed=None):
        """
        Compute and sample the proxy to the posterior distribution p(Z|X, C).
        Parameters
        ----------
        X: torch.Tensor
            Hierarchical counts in tensor form of shape (n_samples, n_levels, n_features).
        C: torch.Tensor
        seed: int

        Returns
        -------
        tuple
        """
        # Apply the seed
        if seed is not None:
            torch.random.manual_seed(seed)
        # Outputs list initializations
        Z, m, log_S = [], [], []

        batch_size = X.size(0)

        # We perform a reparameterization trick to propagate the gradient during parameter's inference
        def reparametrization_trick(mean, log_sigma_vec):
            # log_S_l is the log-variance
            std = torch.exp(0.5 * log_sigma_vec)
            eps = torch.randn_like(std).to(device=self.device)
            Z_ = mean + eps * std
            return Z_

        # We start by generating Z, then we will compute the parameters of the posterior
        if self.variational_approx['method'] == 'mean field':
            for level, mask in enumerate(self.tree.K_mask):
                # Compute the parameters of Z ~ N(mu, Sigma²)
                X_l = X[:, level, :self.tree.K[level]]
                m_fun, log_S_fun = self.m_fun[level], self.log_S_fun[level]
                m_l, log_S_l_vec = m_fun(X_l, C), log_S_fun(X_l, C)
                Z_l = reparametrization_trick(m_l, log_S_l_vec)
                # Embed Z_l in an identifiable form with the hierarchy
                Z_l_embed = torch.zeros((batch_size, X.size(2))).to(device=self.device)
                Z_l_embed[:, mask] += Z_l

                m += [m_l]
                log_S += [log_S_l_vec]
                Z += [Z_l_embed]
        # If it's not a mean-field variational approximation it's a backward model
        else:
            for index in range(self.tree.L):
                # Compute the parameters of Z ~ N(mu, Sigma²) in a backward fashion
                level = self.tree.L - index - 1
                if self.variational_approx['method'] in ['amortized', 'residual']:
                    # Embedded the chain X^{l+1:L} using a recurrent network (amortized framework)
                    X_1tol = X[:, :level + 1, :]
                    X_embed = self.embedder(X_1tol)
                    if self.variational_approx['method'] == 'residual':
                        # If we use the residual amortized backward, we concatenate the amortized chain with the last counts
                        X_l_flat = X[:, level, :self.tree.K[level]]
                        X_embed = torch.concat([X_l_flat, X_embed], dim=1)
                elif self.variational_approx['method'] == 'weak':
                    # In the weak amortized framework, we only consider the last count observed at the current level
                    X_embed = X[:, level, :self.tree.K[level]]
                else:
                    raise AssertionError(f"Variational approximation {self.variational_approx['method']} does not exist.")
                # We fetch the variational functions of interest at that layer
                m_fun, log_S_fun = self.m_fun[level], self.log_S_fun[level]
                # The last layer is only getting the amortized full hierarchical counts as X_embed
                # (level = 0 is actually level = L, we're going backward)
                if level == self.tree.L-1:
                    m_l, log_S_l_vec = m_fun(X_embed, C), log_S_fun(X_embed, C)
                else:
                    # Z_l is computed in a backward fashion, so Z^{l + 1} is the previous "Z" element
                    Z_l_next = Z[-1][:, self.tree.K_mask[level + 1]]
                    # We project Z^{l+1} according to the identifiability result to facilitate the inference
                    PZ_l_next = self.projectors[level + 1](Z_l_next)
                    data_input = torch.cat([X_embed, PZ_l_next], dim=-1)
                    m_l, log_S_l_vec = m_fun(data_input, C), log_S_fun(data_input, C)
                Z_l = reparametrization_trick(m_l, log_S_l_vec)
                # Embed Z_l in an identifiable form with the tree
                Z_l_embed = torch.zeros((batch_size, X.size(2))).to(device=self.device)
                Z_l_embed[:, self.tree.K_mask[level]] += Z_l

                m += [m_l]
                log_S += [log_S_l_vec]
                Z += [Z_l_embed]

            # We need to reverse the list of vectors in Z to get the right order since it was generated backward
            # Same for all the components related to q_phi
            Z = list(reversed(Z))
            m = list(reversed(m))
            log_S = list(reversed(log_S))

        return torch.stack(Z, dim=1), m, log_S

    def conditional_offsets(self, X):
        """
        Compute the offset from the observed counts.
        Parameters
        ----------
        X: torch.Tensor
            Hierarchical counts in tensor form (n_samples, n_levels, n_features).

        Returns
        -------
        torch.Tensor
        """
        if self.offsets == 'zeros':
            return torch.zeros(X.size(0)).to(self.device)
        elif self.offsets == 'logsum':
            return torch.tensor(torch.log(X.sum(dim=2)[:, 0])).to(self.device)
        else:
            raise f'The offset method {self.offsets} is not recognized.'

    def forward(self, X, C, seed=None):
        """
        Forward pass of the PLN-Tree model using the current batch of observed counts and covariates.
        Parameters
        ----------
        X: torch.Tensor
            Hierarchical counts in tensor form (n_samples, n_levels, n_features).
        C: torch.Tensor
            Covariates in tensor form.
        seed: int
            Seed for reproducibility.

        Returns
        -------
        tuple
        """
        mu, Omega = [], []

        # Sample Z conditionally on the counts X through the variational approximation
        Z, m, log_S = self.variational_forward(X, C, seed=seed)

        # We compute the parameters of observed forward process (mu, Omega) using the sampled latents Z
        for level, mask in enumerate(self.tree.K_mask):
            if level == 0:
                if self.covariates is None:
                    # If there are no covariates, we create a placeholder C_ to still get the information of the size
                    C_ = torch.zeros(X.size(0)).to(self.device)
                else:
                    C_ = C
                mu_l = self.mu(level, None, C_)
                Omega_l = self.omega(level, None, C_)
            else:
                Z_l_prev = Z[:, level - 1, self.tree.K_mask[level - 1]]
                mu_l, Omega_l = self.mu(level, Z_l_prev, C), self.omega(level, Z_l_prev, C)

            mu += [mu_l]
            Omega += [Omega_l]

        # Compute the offsets from the observed counts
        O = self.conditional_offsets(X)

        return Z, O, m, log_S, mu, Omega

    def encode(self, counts, covariates=None, return_all=False, seed=None):
        """
        Encode the counts into the latent space Z.
        Can be used for classification, data anonymization, network inference, ...
        Parameters
        ----------
        counts: np.ndarray
            Counts at the last layer of hierarchy. Must be following the same hierarchy as the instance's one.
        covariates: np.ndarray
            Covariates associates with the counts in a matrix format.
        seed: int
            Seed for reproducibility of the sampling.

        Returns
        -------

        """
        X = self.tree.hierarchical_counts(counts).to(self.device)
        if covariates is None:
            C = None
        else:
            C = torch.tensor(covariates)
            # Add a bias term to the covariates if it appears that none is present
            if C.size(1) == self.covariates.size(1) - 1:
                constant_covariates = torch.ones(C.size(0), 1)
                C = torch.cat([constant_covariates, C], dim=1)
            C = C.to(self.device, dtype=self.hierarchical_counts.data.dtype)
        if not return_all:
            if self.offsets == 'zeros':
                return self.forward(X, C, seed=seed)[0]
            return self.forward(X, C, seed=seed)[:2]
        else:
            return self.forward(X, C, seed=seed)

    def decode(self, Z, O=None, sparsity=1e-8, seed=None):
        """
        Turn a latent representation Z into a count observation X
        Parameters
        ----------
        Z: torch.Tensor
            Latent representation of the counts, shape (n_samples, L, K).
        O: torch.Tensor
            Offset values for the counts, shape (n_samples,).
        sparsity: float
            Numerical threshold to sparsify the multinomial sampling probabilities.
        seed: int
            Seed for reproducibility of the sampling.

        Returns
        -------
        torch.Tensor
        """
        if seed is not None:
            torch.random.manual_seed(seed)
        if O is None:
            if self.offsets == 'logsum':
                raise f'Can not decode the latents into counts as the offset is set on {self.offsets}. Please provide offset values.'
            else:
                O = torch.zeros(len(Z)).to(self.device)
        # Initialize the output counts
        X = torch.zeros_like(Z).to(self.device)

        batch_size = Z.size(0)
        for level, K_l in enumerate(self.tree.K):
            # Fetch the parameters at the current level
            Z_l = Z[:, level, :K_l]
            # If we are at the root level, X_1 is sampled of P(exp(Z_1 + O))
            if level == 0:
                Z_shift = Z_l + O.view(batch_size, 1)
                exp_Z_l = torch.exp(Z_shift)
                X[:, level, :K_l] = Poisson(rate=exp_Z_l).sample().to(device=self.device)
                assert (X[:, level, :K_l] >= 0).all(), f"X should be positive. Associated latents: {Z_shift[X[:, level, :K_l] < 0]}"
            else:
                # If we're not at the root, based on Z_l, we sample each C(X_k^(l-1))
                X_prev = X[:, level-1, :self.tree.K[level-1]]
                parent_cursor = 0
                current_clade = 1
                children_index = []
                # We add a final zero to the clades to always have a stopping criterion that is satisfied, like a padding
                # PATCH: build both tensors on self.device before concatenating. Previously torch.zeros(1) was created on
                # CPU and torch.cat was called before .to(self.device), which raised a device-mismatch RuntimeError when
                # self.tree.clades[level] was already on CUDA (e.g. when running benchmark.py with --device cuda).
                clades_l = self.tree.clades[level].to(self.device)
                padded_clades = torch.cat((clades_l, torch.zeros(1, device=self.device)), dim=0)
                for i, clade in enumerate(padded_clades):
                    if clade != current_clade:
                        # If we have changed clade, then it's time to compute previous clade's counts
                        # We start by fetching the parent
                        X_parent = X_prev[:, parent_cursor]
                        # If we have an only child, then it's taking the parent's value
                        if len(children_index) == 1:
                            X[:, level, children_index[0]] = X_parent
                        # If it's a clade, we compute the PLN-Tree latent parameters and sample the counts
                        else:
                            Z_l_child = Z_l[:, children_index]
                            Z_l_child_max = torch.max(Z_l_child, dim=1)[0].unsqueeze(-1).repeat(1, len(children_index))
                            probabilities = torch.softmax(Z_l_child - Z_l_child_max, dim=1)
                            for batch_index in range(batch_size):
                                if X_parent[batch_index] > 0:
                                    # Torch multinomial does not support well high values, we use numpy instead
                                    # Casting to float64 is required for numpy multinomial or we risk casting issues
                                    pvals = probabilities[batch_index].to('cpu').detach().numpy().astype('float64')
                                    # Sparsify the probabilities to avoid spurious counts
                                    pvals[pvals < sparsity] = 0
                                    # We normalize the probabilities to sum to 1
                                    pvals = pvals / pvals.sum(axis=-1, keepdims=True)
                                    X[batch_index, level, children_index] = torch.tensor(np.random.multinomial(
                                        n=int(X_parent[batch_index]),
                                        pvals=pvals
                                    )).to(device=self.device, dtype=X.dtype)
                        # Then we reset the clade's information to go for another clade
                        children_index = []
                        current_clade += 1
                        parent_cursor += 1
                    # If the observed clade is a padding, we it means we have reached the end of the nodes
                    if clade == 0:
                        break
                    # If we are in the same clade, we add the index of the clade members here (the children regarding the parent's node)
                    children_index += [i]
        return X

    def sample(self, n_samples, covariates=None, offsets=None, hierarchy_level=None, seed=None):
        """
        Generate data from the PLN-Tree model prior.
        Parameters
        ----------
        n_samples: int
            Number of samples to generate.
        covariates: np.ndarray
            Covariates associated with the samples in a matrix format. Only mandatory if used during training.
        offsets: np.ndarray
            Maximum total count for a sample to prevent overflow in multinomial sampling.
        hierarchy_level: str or None
            If specified, the samples will be returned only at that level of the hierarchy.
        seed: int
            Seed for reproducibility of the sampling.

        Returns
        -------
        (pd.DataFrame, torch.Tensor) or (torch.Tensor, torch.Tensor) or (pd.DataFrame, torch.Tensor, torch.Tensor) or (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        if seed is not None:
            torch.random.manual_seed(seed)
        if covariates is None and self.covariates is not None:
            covariates = pd.DataFrame(self.covariates).sample(n_samples, replace=True)
        if covariates is not None:
            if type(covariates) == pd.DataFrame:
                covariates = covariates.to_numpy()
            C = torch.tensor(covariates)
            # Add a bias term to the covariates if it appears that none is present
            if C.size(1) == self.covariates.size(1)-1:
                constant_covariates = torch.ones(C.size(0), 1)
                C = torch.cat([constant_covariates, C], dim=1)
            C = C.to(self.device, dtype=self.mu_fun[0].data.dtype)
        else:
            C = None

        if offsets is None and self.offsets == 'logsum':
            raise AssertionError(f"The PLN-Tree model was trained with {self.offsets} offsets. You need to provide offsets to the model to sample new data.")
        elif offsets is not None:
            O = torch.tensor(offsets).view(-1, 1).to(self.device)
        else:
            O = torch.zeros(n_samples).view(-1, 1).to(self.device)
        # We sample Z, then we will decode it into X and we use the same dtype as that of the model
        Z = torch.zeros((n_samples, self.tree.L, self.tree.K_max)).to(device=self.device, dtype=self.hierarchical_counts.dtype)
        O = O.to(dtype=self.hierarchical_counts.dtype)
        for level, mask in enumerate(self.tree.K_mask):
            if level == 0:
                if C is not None:
                    mu_l = self.mu(level, None, C.to(dtype=self.mu_fun[0].data.dtype)).detach()
                    Omega_l = self.omega(level, None, C.to(dtype=self.mu_fun[0].data.dtype)).detach()
                else:
                    # If we have no covariates, we just leave a bias to obtain the right batch size even though it has no impact.
                    mu_l = self.mu(level, None, torch.zeros(n_samples).to(self.device)).detach()
                    Omega_l = self.omega(level, None, torch.zeros(n_samples).to(self.device)).detach()
            else:
                Z_l_prev = Z[:, level - 1, self.tree.K_mask[level - 1]]
                mu_l, Omega_l = self.mu(level, Z_l_prev, C).detach(), self.omega(level, Z_l_prev, C).detach()

            try:
                # We try using the cholesky decomposition to perform the sampling as this approach is faster,
                # but if it fails, we fallback to the slower matrix inversion approach
                Omega_l_cholesky = torch.linalg.cholesky(Omega_l, upper=False)
                Omega_l_cholesky_T = torch.transpose(Omega_l_cholesky, dim0=-2, dim1=-1)
                # If we want Z ~ N(mu, Omega^-1) and that we have Omega = LL.T
                # Then Z = mu + L.-T eps works
                # Hence we sample eps ~ N(0, 1)
                # Then we solve L.T eps_tild = eps so that eps_tild ~ N(0, Omega^-1)
                # Then Z = mu + eps_tild
                eps = torch.randn_like(mu_l).unsqueeze(-1).to(device=self.device)
                eps = torch.linalg.solve_triangular(Omega_l_cholesky_T, eps, upper=True).squeeze(-1)
            except:
                # If the cholesky decomposition fails, it probably means Omega is low-rank
                # So we attempt a low-rank sampling
                eps = lowrank_multivariate_normal_samples(
                    # Increase the numerical stability by casting to float64
                    Omega_l.to(dtype=torch.float64)
                ).to(device=self.device)

            # Ensuring the dtype is the same as the samples
            Z[:, level, mask] = mu_l.to(dtype=Z.dtype) + eps.to(dtype=Z.dtype)

        # We can not generate samples that have a seed parameter over exp(threshold_multinomial) numerically using Multinomial
        threshold_multinomial = 20.
        denied_samples_mask = torch.any(Z[:, 0, :self.tree.K[0]] + O.view(Z.shape[0], 1) > threshold_multinomial, dim=1)
        O[denied_samples_mask] = threshold_multinomial - Z[denied_samples_mask, 0, :self.tree.K[0]].squeeze().max(dim=-1, keepdim=True)[0]

        # Simply unpack Z as X
        X = self.decode(Z, O, seed=seed)

        # If a level has been specified by the user, we output only the counts at that level into the precised dataframe
        # Otherwise we output the full hierarchical counts as a tensor
        if hierarchy_level is not None:
            level_index = np.where(self.tree.level_names == hierarchy_level)[0][0]
            X = X[:, level_index, :self.tree.K[level_index]]
            X = pd.DataFrame(data=X.cpu().numpy(), columns=self.tree.intermediate_clade_names(hierarchy_level))

        if self.offsets == 'zeros':
            return X, Z
        return X, Z, O

    def vamp_sample(
            self,
            n_samples=10,
            X_vamp=None,
            covariates=None,
            offsets=None,
            mean=False,
            threshold_offset=20.,
            sparsity=1e-4,
            seed=None
    ):
        """
        Sample from the post-hoc VAMP approximation.

        Parameters
        ----------
        n_samples: int
            Number of samples. Optional of covariates are precised.
        X_vamp: pd.DataFrame or None
            Samples to use for the VAMP approximation. If None, the model's counts are used.
        covariates: pd.DataFrame or None
            Covariates associated with the samples. If None, the model's covariates are used.
        offsets: pd.DataFrame or None
            Offsets associated with the samples. If None, the model's offsets are used.
        mean: bool
            Whether sampling is performed in the latent space or if the mean of the latents is used.
        threshold_offset: float
            Maximum total count for a given sample to avoid numerical issues with the multinomial sampling.
        sparsity: float
            Numerical sparsity of the softmax-transformed counts.
        seed: int or None
            Seed for reproducibility.

        Returns
        -------
        (torch.Tensor, torch.Tensor) or (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        if seed is not None:
            torch.random.manual_seed(seed)
        if X_vamp is None:
            X_vamp = self.counts
        # Reorder columns according the taxonomic tree
        X_vamp = X_vamp[self.counts.columns]
        # If covariates are provided, we sample the same number of samples as the covariates
        if covariates is not None:
            n_samples = covariates.shape[0]
        components = torch.randint(0, X_vamp.shape[0], (n_samples,))
        # If no covariates are provided, we use the covariates of the model if it's a conditional model
        if covariates is None and self.covariates is not None:
            covariates = self.covariates[components]
        if mean:
            # We fetch the mean of the latents
            Z_, _, m, _, _, _ = self.encode(X_vamp.iloc[components], covariates=covariates, seed=seed, return_all=mean)
            Z = torch.zeros_like(Z_)
            for level in range(self.tree.L):
                Z[:, level, self.tree.K_mask[level]] = m[level]
        else:
            # We sample the latents
            Z = self.encode(X_vamp.iloc[components], covariates=covariates, seed=seed)
        # We can not generate samples that have a seed parameter over exp(threshold_offset) numerically using Multinomial
        denied_samples_mask = torch.any(Z[:, 0, :self.tree.K[0]] > threshold_offset, dim=1)
        Z[denied_samples_mask, 0, :self.tree.K[0]] = threshold_offset - Z[denied_samples_mask, 0, :self.tree.K[0]].squeeze().max(dim=-1, keepdim=True)[0]
        # Decode the latents into the counts
        X = self.decode(Z, offsets, sparsity=sparsity, seed=seed)
        if covariates is not None:
            return X, Z, covariates
        return X, Z



    def mu(self, level, Z_l_prev, C):
        """
        Fetch the latent mean of the forward process at a given level conditionally on a set of covariates.
        Parameters
        ----------
        level: int
            Level of interest in the hierarchy.
        Z_l_prev: torch.Tensor or None
            Previous level latent. If level == 0, this is None.
        C: torch.Tensor
            Covariates.

        Returns
        -------
        torch.Tensor
        """
        if self.latent_dynamic['markov_means']:
            if level == 0:
                if self.covariates is None:
                    return self.mu_fun[0](C)
                else:
                    # If we have covariates, the means is computed as C @ B
                    return C @ self.mu_fun[0].data
            else:
                # We always provide projected latents according to the identifiability results as input of the networks
                PZ_l_prev = self.projectors[level - 1](Z_l_prev)
                return self.mu_fun[level](PZ_l_prev, C)
        else:
            if self.covariates is None:
                if C is None:
                    return self.mu_fun[level](Z_l_prev)
                return self.mu_fun[level](C)
            else:
                # If we have covariates, the means is computed as C @ B
                return C @ self.mu_fun[level].data

    def omega(self, level, Z_l_prev, C):
        """
        Fetch the latent precision matrices of the forward process at a given level conditionally on a set of covariates.
        Parameters
        ----------
        level: int
            Level of interest in the hierarchy.
        Z_l_prev: torch.Tensor or None
            Previous level latent. If level == 0, this is None.
        C: torch.Tensor
            Covariates.

        Returns
        -------
        torch.Tensor
        """
        if self.latent_dynamic['markov_covariance']:
            if level == 0:
                return self.omega_fun[0](C)
            else:
                # We always provide projected latents according to the identifiability results as input of the networks
                PZ_l_prev = self.projectors[level - 1](Z_l_prev)
                return self.omega_fun[level](PZ_l_prev, C)
        else:
            if C is None:
                return self.omega_fun[level](Z_l_prev)
            return self.omega_fun[level](C)


    def latent_tree_proportions(self, Z, clr=False, seed=None):
        """
        Turn a latent representation Z into its latent tree proportions.
        Parameters
        ----------
        Z: torch.Tensor
            Latent representation of the counts.
        clr: bool
            Apply the CLR transform upon the latent tree proportions.
        seed: int
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
        """
        if seed is not None:
            torch.random.manual_seed(seed)
        # Initialize the output counts
        probabilities = torch.zeros_like(Z).to(self.device)

        batch_size = Z.size(0)
        for level, K_l in enumerate(self.tree.K):
            # Fetch the parameters at the current level
            Z_l = Z[:, level, :K_l]
            # If we are at the root level, X_1 is sampled of P(exp(Z_1 + O))
            if level == 0:
                probabilities[:, level, :K_l] = torch.softmax(Z_l, dim=-1)
            else:
                parent_cursor = 0
                current_clade = 1
                children_index = []
                # We add a final zero to the clades to always have a stopping criterion that is satisfied, like a padding
                # PATCH: build both tensors on self.device before concatenating. Previously torch.zeros(1) was created on
                # CPU and torch.cat was called before .to(self.device), which raised a device-mismatch RuntimeError when
                # self.tree.clades[level] was already on CUDA (e.g. when running benchmark.py with --device cuda).
                clades_l = self.tree.clades[level].to(self.device)
                padded_clades = torch.cat((clades_l, torch.zeros(1, device=self.device)), dim=0)
                for i, clade in enumerate(padded_clades):
                    if clade != current_clade:
                        # If we have an only child, then it's taking the parent's value
                        if len(children_index) == 1:
                            probabilities[:, level, children_index[0]] = 1.
                        # If it's a clade, we compute the PLN-Tree latent parameters and sample the counts
                        else:
                            Z_l_child = Z_l[:, children_index]
                            Z_l_child_max = torch.max(Z_l_child, dim=1)[0].unsqueeze(-1).repeat(1, len(children_index))
                            probabilities[:, level, children_index] = torch.softmax(Z_l_child - Z_l_child_max, dim=-1)
                        # Then we reset the clade's information to go for another clade
                        children_index = []
                        current_clade += 1
                        parent_cursor += 1
                    # If the observed clade is a padding, we it means we have reached the end of the nodes
                    if clade == 0:
                        break
                    # If we are in the same clade, we add the index of the clade members here (the children regarding the parent's node)
                    children_index += [i]
        if clr:
            for level, mask in enumerate(self.tree.K_mask):
                # We apply the CLR transform to the latent probabilities
                probabilities[:, level, mask] = torch.log(probabilities[:, level, mask] + 1e-8) - torch.log(probabilities[:, level, mask] + 1e-8).mean(axis=1, keepdim=True)
        return probabilities


