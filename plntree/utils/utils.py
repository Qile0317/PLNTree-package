import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

import matplotlib.pyplot as plt
import matplotlib


class Tree:

    def __init__(self, hierarchy, level_regex='|', clade_regex='__', max_length=256, device='cpu'):
        """
        Hierarchy builder class used for training the model and incorporate the tree structure.
        Parameters
        ----------
        hierarchy: list[str]
        level_regex: str
        clade_regex: str
        max_length: int
        """
        # Device for CUDA computation if available
        self.device = device
        # Keep in memory the original hierarchy format
        self.hierarchy = hierarchy
        self.level_regex = level_regex
        self.clade_regex = clade_regex

        # Count the number of levels (has to be consistent across levels)
        self.n_levels = len(hierarchy[0].split(level_regex))

        # Build the list of levels in the hierarchy from the first entry
        self.level_names = []
        for clade_decomposition in hierarchy[0].split(level_regex):
            level_name, clade = clade_decomposition.split(clade_regex)
            self.level_names += [level_name]
        self.level_names = np.array(self.level_names)

        # Build the decomposition of each entity into the hierarchy
        self.decomposed_entities = []
        for entity in hierarchy:
            decomposition = entity.split(level_regex)
            assert len(
                decomposition) == self.n_levels, "All entities must have the same number of levels for the hierarchy to be valid."
            self.decomposed_entities += [decomposition]
            for clade in decomposition:
                assert clade.split(clade_regex)[
                           0] in self.level_names, f"All entities must follow the same hierarchy format: {self.level_names}."
        self.decomposed_entities = np.array(self.decomposed_entities, dtype=f'|S{max_length}').T
        # Sort the decomposition to have a consistent order of the clades
        sorted_decomposed_entities = self.decomposed_entities.copy()
        sorted_decomposed_entities = sorted_decomposed_entities[:, sorted_decomposed_entities[0].argsort()]
        for layer in range(1, len(sorted_decomposed_entities)):
            clades = np.unique(sorted_decomposed_entities[layer - 1])
            for clade in clades:
                indices = np.where(sorted_decomposed_entities[layer - 1, :] == clade)[0]
                sorted_indices = indices[sorted_decomposed_entities[layer, indices].argsort()]
                sorted_decomposed_entities[:, indices] = sorted_decomposed_entities[:, sorted_indices]
        self.decomposed_entities = sorted_decomposed_entities

        # We define the unique function because numpy does a sorting that we want to avoid
        def unique(a, return_counts=False, axis=None):
            if not return_counts:
                indexes = np.unique(a, return_index=True, axis=axis)[1]
                if axis == 1:
                    u = np.array([a[:, index] for index in sorted(indexes)])
                else:
                    u = np.array([a[index] for index in sorted(indexes)])
                return u
            u = []
            c = []
            for x in a:
                if x not in u:
                    u.append(x)
                    if return_counts:
                        c.append(1)
                else:
                    if return_counts:
                        c[u.index(x)] += 1
            return np.array(u), np.array(c)

        # Compute the depth of the hierarchy
        self.L = len(self.level_names)

        # TODO: check correctness - this was modified from original!
        # Compute the number of nodes at a given layer of the hierarchy
        # Use unique paths from root to each level (not just unique names),
        # because the same name can appear under different parents.
        self.K = torch.zeros(self.L, dtype=torch.int).to(self.device)
        for layer in range(self.L):
            paths = unique(self.decomposed_entities[:layer + 1], axis=1)
            self.K[layer] = len(paths)

        # Compute the maximum width of the hierarchy
        self.K_max = int(self.K[-1])
        assert max(self.K) == self.K_max, "The width of the hierarchy can not decrease over the levels."

        # Compute the clades mask over the matrix embedding of the hierarchy (used in the ELBO computation)
        self.clades = np.zeros((self.L, self.K_max))
        # Compute the name associated to each clade (match the clade mask with the names)
        self.clade_names = []
        # Define the root clades
        root_clades = unique(self.decomposed_entities[0])
        self.clades[0, :len(root_clades)] = np.arange(1, len(root_clades) + 1, 1)
        for layer in range(self.L - 1):
            # Get the sub-hierarchy composed of the current layer and its children
            # Prune the duplicates to obtain the minimal hierarchy and associate the clades
            sub_clades = unique(self.decomposed_entities[layer:layer + 2], axis=1).T
            # Get the existing clades at the layer and the amount of children they have to compute the clade matrix
            cursor = 0
            clades, n_children = unique(sub_clades[0], return_counts=True)
            self.clade_names += [clades]
            for clade_index in range(len(clades)):
                self.clades[layer + 1, cursor:cursor + n_children[clade_index]] = clade_index + 1
                cursor += n_children[clade_index]
        self.clade_names += [unique(self.decomposed_entities[-1])]
        self.clades = torch.tensor(self.clades)

        # Mask of not only-child (used in latents modelling and ELBO computation)
        self.K_mask = np.zeros((self.L, self.K_max), dtype=bool)
        self.K_mask[0, :self.K[0]] = 1
        for layer in range(self.L):
            clades, brotherhood = np.unique(self.clades[layer], return_counts=True)
            for clade, n_brothers in zip(clades, brotherhood):
                if clade == 0:
                    continue
                if n_brothers > 1:
                    self.K_mask[layer, np.where(self.clades[layer] == clade)] = 1
        self.K_eff = self.K_mask.sum(axis=1).astype(np.int32)

        # Build the affiliation matrices used for fast computation of the ELBO and avoid a loop through the hierarchy nodes
        # These matrix are also used to compute the hierarchical counts from the last layer X_L recursively
        # Basically, X_l = X_{l-1} @ A_l.T (for batch of matrices)
        self.affiliation_matrix = []
        for K_l, clade in zip(self.K, self.clades):
            K_l_prev = max(clade)
            A_l = torch.zeros(int(K_l_prev), int(K_l)).to(self.device)
            current_clade = 1
            for k, c in enumerate(clade):
                if c == 0:
                    break
                if c != current_clade:
                    current_clade += 1
                A_l[current_clade - 1, k] = 1
            self.affiliation_matrix += [A_l]

        # Compute the full sorted names of the entities in the tree used to reorder them in the model computations
        self.full_sorted_names = []
        for i in range(self.decomposed_entities.shape[1]):
            splitted = self.decomposed_entities[:, i]
            name = ''
            for s in splitted:
                name += '|' + s.decode('utf-8')
            self.full_sorted_names += [name[1:]]

    def intermediate_clade_names(self, level):
        """
        Get the names of the clades at a given level.
        Parameters
        ----------
        level: str

        Returns
        -------
        list[str]
        """
        level_index = np.where(self.level_names == level)[0][0]
        if level_index == len(self.level_names) - 1:
            return self.full_sorted_names
        clade_names_all_samples = self.decomposed_entities[level_index]
        clade_names = []
        for name in clade_names_all_samples:
            if name not in clade_names:
                clade_names += [name.decode('utf-8')]
        return clade_names


    def hierarchical_counts(self, counts):
        """
        Compute hierarchical counts from deepest hierarchical level counts.
        Parameters
        ----------
        counts: pd.DataFrame

        Returns
        -------
        torch.Tensor
        """
        df = counts[self.full_sorted_names]
        values = df.to_numpy()
        X = torch.zeros(len(values), self.L, self.K_max).to(self.device)
        # Fill the last layer with the count values
        X[:, -1] += torch.tensor(values).to(self.device)
        for i in range(self.L - 1):
            layer = self.L - i - 1
            # Recursively obtain the parent counts from the affiliated children counts
            X[:, layer - 1, :self.K[layer - 1]] += batch_matrix_product(self.affiliation_matrix[layer], X[:, layer, :self.K[layer]])
        return X

    def plot(self, figaxis=None, colormap='viridis_r', counts=None, visual='circle', text_args=None, legend_args=None):
        """
        Visualize the hierarchy or any hierarchical count data using the counts argument.
        The provided counts must be in the same ordering as the hierarchy of this instance.
        Parameters
        ----------
        figaxis: (matplotlib.figure.Figure, matplotlib.axes.Axes) or None
        colormap: str
        counts: pd.DataFrame or None
        visual: str
            Indicate if the tree is positioned on a circle, an hemicycle, a quartercycle, or a top-down plot.
        text_args: dic or None
        legend_args: dic or None

        Returns
        -------

        """
        if figaxis is None:
            fig, axis = plt.subplots()
        else:
            fig, axis = figaxis

        cmap = matplotlib.colormaps[colormap]
        if counts is None:
            proportions = np.ones(len(self.clades[-1]))
            color_highest_group = {}
            for k, highest_group in enumerate(self.clade_names[0]):
                color_highest_group[highest_group] = cmap((k + 1) / len(self.clade_names[0]) + 1e-8)

            def get_color(leaf_clade):
                highest_clade = self.decomposed_entities[:, np.where(self.decomposed_entities[-1] == leaf_clade)][0][0][0]
                return color_highest_group[highest_clade]

            if fig is not None:
                handles = [matplotlib.lines.Line2D([0], [0], color=color, lw=2) for color in color_highest_group.values()]
                if legend_args is None:
                    legend_args = {}
                if 'ncol' not in legend_args:
                    legend_args['ncol'] = 4
                if 'x' not in legend_args:
                    legend_args['x'] = .5
                if 'y' not in legend_args:
                    legend_args['y'] = -.44
                fig.legend(handles, [key.decode('utf-8') for key in color_highest_group.keys()], loc='lower center', ncol=legend_args['ncol'],
                               bbox_to_anchor=(legend_args['x'], legend_args['y']))

        else:
            df = counts[self.full_sorted_names].to_numpy()
            proportions = counts / sum(counts)

        if visual != 'top-down':
            # If we use a cycle plot of the tree, the tree is positioned on a ring in the polar space
            def polar_to_cartesian(r, theta):
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                return x, y

            if visual == 'circle':
                max_theta = 2 * np.pi
            elif visual == 'hemicycle':
                max_theta = np.pi
            elif visual == 'quartercycle':
                max_theta = 3 * np.pi / 2

            theta_L = np.linspace(0, max_theta, len(self.clades[-1]), endpoint=False)
            r_L = self.L * np.ones(len(theta_L))
            colors_L = []
            for k, (theta_k_L, r_k_L, prop_k_L) in enumerate(zip(theta_L, r_L, proportions)):
                if counts is not None:
                    color = cmap(prop_k_L)
                else:
                    color = get_color(self.clade_names[-1][k])
                colors_L += [color]

                if prop_k_L <= 1e-10 and counts is not None:
                    continue
                x_k_L, y_k_L = polar_to_cartesian(r_k_L, theta_k_L)
                axis.plot([x_k_L], [y_k_L], linestyle='', marker='.', color=color)
                if text_args is not None:
                    text = self.clade_names[-1][k].decode('utf-8')
                    x_text, y_text = polar_to_cartesian(r_k_L + text_args['scale'], theta_k_L)
                    axis.text(x_text, y_text,
                              text, ha='left', va='top', size=text_args['size'],
                              rotation=np.rad2deg(theta_k_L), rotation_mode='anchor',
                              color=get_color(self.clade_names[-1][k]))

            theta_l = theta_L
            r_l = r_L
            prop_l = np.array(proportions)
            colors_l = np.array(colors_L)
            assert(len(colors_l) == len(prop_l))
            for layer in range(1, self.L):
                theta_next_l = []
                r_next_l = []
                counts_next_l = []
                colors_next_l = []
                for clade in np.unique(self.clades[self.L - layer]):
                    if clade == 0:
                        continue
                    clade_indices = np.where(self.clades[self.L - layer] == clade)[0]
                    theta_children = theta_l[clade_indices]
                    theta_parent = theta_children.mean()
                    r_children = r_l[clade_indices].mean()
                    r_parent = r_children - 1
                    prop_parent = sum(prop_l[clade_indices])
                    color_parent = colors_l[clade_indices][0]
                    x_parent, y_parent = polar_to_cartesian(r_parent, theta_parent)

                    n_link = 1000
                    r_link = (r_children + r_parent)/2 * np.ones(n_link)
                    theta_link = np.linspace(theta_children.min(), theta_children.max(), n_link)
                    x_link, y_link = polar_to_cartesian(r_link, theta_link)
                    x_link_mean, y_link_mean = polar_to_cartesian(r_link[0], theta_parent)
                    if counts is None:
                        axis.plot(x_link, y_link, marker='', linestyle='-', color=color_parent, alpha=.8)
                        axis.plot([x_link_mean, x_parent],
                                  [y_link_mean, y_parent], marker='', linestyle='-', color=color_parent, alpha=.8)
                    for theta_l_k, prop_k_l in zip(theta_l[clade_indices], prop_l[clade_indices]):
                        if prop_k_l <= 1e-10 and counts is not None:
                            continue
                        x_l_k, y_l_k = polar_to_cartesian(r_children, theta_l_k)
                        c = color_parent if counts is None else cmap(prop_k_l)
                        if counts is not None:
                            axis.plot(
                                [x_l_k, x_parent], [y_l_k, y_parent],
                                marker='', linestyle='-', color=c
                            )
                        else:
                            x_child, y_child = polar_to_cartesian(r_children, theta_l_k)
                            x_link, y_link = polar_to_cartesian(r_link[0], theta_l_k)
                            axis.plot([x_child, x_link],
                                      [y_child, y_link], linestyle='-', marker='', color=c, alpha=.8)
                    if counts is None:
                        x_parent, y_parent = polar_to_cartesian(r_parent, theta_parent)
                        axis.plot([x_parent], [y_parent], linestyle='', marker='.', color=color_parent)
                    elif prop_parent > 1e-10:
                        x_parent, y_parent = polar_to_cartesian(r_parent, theta_parent)
                        axis.plot([x_parent], [y_parent], linestyle='', marker='.', color=cmap(prop_parent))
                    theta_next_l += [theta_parent]
                    r_next_l += [r_parent]
                    counts_next_l += [prop_parent]
                    colors_next_l += [color_parent]
                theta_l = np.array(theta_next_l)
                r_l = np.array(r_next_l)
                prop_l = np.array(counts_next_l)
                colors_l = np.array(colors_next_l)
            axis.set_aspect('equal', 'box')
            axis.axis('off')
        else:
            # If we don't use a cycle plot of the tree, it is a top-down plot in the cartesian space
            x_L = np.arange(len(self.clades[-1]))
            y_L = np.zeros(len(x_L))
            for x_k_L, y_k_L, prop_k_L in zip(x_L, y_L, proportions):
                if prop_k_L <= 1e-10 and counts is not None:
                    continue
                axis.plot([x_k_L], [y_k_L], linestyle='', marker='.', color=cmap(prop_k_L))

            x_l = x_L
            y_l = y_L
            prop_l = proportions
            for layer in range(1, self.L):
                x_next_l = []
                y_next_l = []
                counts_next_l = []
                for clade in np.unique(self.clades[self.L - layer]):
                    if clade == 0:
                        continue
                    clade_indices = np.where(self.clades[self.L - layer] == clade)[0]
                    x_parent = x_l[clade_indices].mean()
                    y_parent = y_l[clade_indices].mean() + 1
                    prop_parent = sum(prop_l[clade_indices])
                    for x_l_k, prop_k_l in zip(x_l[clade_indices], prop_l[clade_indices]):
                        if prop_k_l <= 1e-10 and counts is not None:
                            continue
                        axis.plot(
                            [x_l_k, x_parent], [y_parent - 1, y_parent],
                            marker='', linestyle='-', color=cmap(prop_k_l)
                        )
                    if counts is None:
                        axis.plot([x_parent], [y_parent], linestyle='', marker='.', color=cmap(prop_parent))
                    elif prop_parent > 1e-10:
                        axis.plot([x_parent], [y_parent], linestyle='', marker='.', color=cmap(prop_parent))
                    x_next_l += [x_parent]
                    y_next_l += [y_parent]
                    counts_next_l += [prop_parent]
                x_l = np.array(x_next_l)
                y_l = np.array(y_next_l)
                prop_l = np.array(counts_next_l)

            axis.set_yticks(np.arange(len(self.level_names)))
            axis.set_yticklabels(reversed(self.level_names))
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)
            axis.yaxis.set_ticks_position('left')
            axis.yaxis.set_tick_params(which='both', length=0)
            axis.get_xaxis().set_visible(False)
        return axis

def format_time_HH_MM_SS(seconds):
    """
    Format the time in seconds to a human-readable format.
    Parameters
    ----------
    seconds: float

    Returns
    -------
    str
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


def dataloader_counts_covariates(X, C, batch_size, shuffle=True, seed=None):
    """
    Create a DataLoader instance for the counts and covariates.
    Parameters
    ----------
    X: torch.Tensor
    C: torch.Tensor
    batch_size: int
    shuffle: bool
    seed: int or None

    Returns
    -------
    DataLoader
    """
    if C is None:
        C = torch.zeros(X.size(0))
    if seed is not None:
        torch.random.manual_seed(seed)
    dataset = TensorDataset(X, C)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def identity(batch_size, size):
    """
    Generate a batch of identity matrices.
    Parameters
    ----------
    batch_size: int
    size: int

    Returns
    -------
    torch.Tensor
    """
    return torch.eye(size).unsqueeze(0).expand(batch_size, -1, -1)


def batch_matrix_product(matrix, x):
    """
    Matrix product with a batched tensor (n, p) and a matrix of size (m, p).
    Parameters
    ----------
    matrix: torch.Tensor
    x: torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    X = x.unsqueeze(2)
    MX = (matrix.unsqueeze(0).expand(X.size(0), -1, -1) @ X).squeeze(-1)
    return MX


def trace(X):
    """
    Compute the trace of a batch of matrices.
    Parameters
    ----------
    X: torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    return X.diagonal(dim1=-2, dim2=-1).sum(-1)


def save_model_weights(model, path):
    """
    Save the model parameters (torch Module) into a file for later use.
    Parameters
    ----------
    model: plntree.PLNTree
    path: str

    Returns
    -------
    None
    """
    torch.save(model.state_dict(), path + '.pth')
    for level in range(model.tree.L):
        if level == 0:
            torch.save(model.mu_fun[0].data, path + '_mu_1.pth')
            torch.save(model.omega_fun[0].data, path + '_omega_1.pth')
        else:
            if not model.latent_dynamic['markov_means']:
                torch.save(model.mu_fun[level].data, path + f'_mu_{level+1}.pth')
            if not model.latent_dynamic['markov_covariance']:
                torch.save(model.omega_fun[level].data, path + f'_omega_{level+1}.pth')


def load_model_weights(model, path):
    """
    Load the model parameters (torch Module) from a file.
    Parameters
    ----------
    model: plntree.PLNTree
    path: str

    Returns
    -------
    Any
    """
    model.load_state_dict(torch.load(path + '.pth', map_location=model.device, weights_only=True))
    for level in range(model.tree.L):
        if level == 0:
            model.mu_fun[0].data = torch.load(path + '_mu_1.pth', map_location=model.device, weights_only=True)
            model.omega_fun[0].data = torch.load(path + '_omega_1.pth', map_location=model.device, weights_only=True)
        else:
            if not model.latent_dynamic['markov_means']:
                model.mu_fun[level].data = torch.load(path + f'_mu_{level+1}.pth', map_location=model.device, weights_only=True)
            if not model.latent_dynamic['markov_covariance']:
                model.omega_fun[level].data = torch.load(path + f'_omega_{level+1}.pth', map_location=model.device, weights_only=True)

def save_losses(losses, path):
    """
    Save the losses into a file for later visualisation.
    Parameters
    ----------
    losses: list[float]
    path: str
    """
    np.save(path + '.npy', np.array(losses))

def load_losses(path):
    """
    Load the losses from a file.
    Parameters
    ----------
    path: str

    Returns
    -------
    np.ndarray
    """
    return np.load(path + '.npy')


def lowrank_multivariate_normal_samples(omega):
    """
    Sample from a multivariate normal distribution with a low-rank precision matrix.
    Parameters
    ----------
    omega: torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    batch_size, d, _ = omega.shape

    # Eigenvalue decomposition for each precision matrix in the batch
    eigenvalues, eigenvectors = torch.linalg.eigh(omega)
    # Identify positive eigenvalues (numerical stability threshold)
    positive_mask = eigenvalues > 1e-8

    samples_batch = []
    for b in range(batch_size):
        # Extract the positive eigenvalues and corresponding eigenvectors for this batch
        positive_eigenvalues = eigenvalues[b, positive_mask[b]]
        Q_r = eigenvectors[b, :, positive_mask[b]].to(dtype=omega.dtype, device=omega.device)
        # Reduced dimensionality
        k = positive_eigenvalues.size(0)
        # Sample in the reduced space
        z = torch.randn(k, device=omega.device, dtype=omega.dtype)
        sample_reduced = z / torch.sqrt(positive_eigenvalues).to(dtype=omega.dtype, device=omega.device)
        # Map back to the original space
        sample = Q_r @ sample_reduced
        samples_batch.append(sample)

    # Stack the samples into a tensor
    return torch.stack(samples_batch, dim=0)
