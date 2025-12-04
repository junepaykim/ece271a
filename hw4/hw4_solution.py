import numpy as np
import scipy.io
import scipy.stats
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import os
from scipy.fftpack import dctn
from tqdm import tqdm

BLOCK_SIZE = 8
DATA_DIR = 'data/' if os.path.exists('data/') else '.'
TRAIN_FILE = os.path.join(DATA_DIR, 'TrainingSamplesDCT_8_new.mat')
IMAGE_FILE = os.path.join(DATA_DIR, 'cheetah.bmp')
MASK_FILE = os.path.join(DATA_DIR, 'cheetah_mask.bmp')
ZIGZAG_FILE = os.path.join(DATA_DIR, 'Zig-Zag Pattern.txt')
OUTPUT_DIR = 'hw4/output/'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_zigzag_pattern(path):
    """Loads the Zig-Zag pattern from a text file."""
    if not os.path.exists(path):
        return np.array([
            0, 1, 5, 6, 14, 15, 27, 28,
            2, 4, 7, 13, 16, 26, 29, 42,
            3, 8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        ])
    with open(path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.extend([int(x) for x in line.split()])
    return np.argsort(data)


def compute_dct_features(img, zigzag_order):
    """
    Computes DCT features for an image using an 8x8 sliding window.
    Returns an (N, 64) array of features and the image dimensions.
    """
    img = np.array(img, dtype=float) / 255.0

    if img.ndim == 3:
        img = np.mean(img, axis=2)

    h, w = img.shape

    pad_h = BLOCK_SIZE - 1
    pad_w = BLOCK_SIZE - 1
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), 'constant', constant_values=0)

    features = []

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(img_padded, (BLOCK_SIZE, BLOCK_SIZE))

    num_blocks = h * w
    flat_blocks = windows.reshape(num_blocks, BLOCK_SIZE, BLOCK_SIZE)

    print(f"Computing DCT for {num_blocks} blocks...")

    dct_blocks = dctn(flat_blocks, type=2, norm='ortho', axes=(1, 2))

    dct_flat = dct_blocks.reshape(num_blocks, 64)
    features = dct_flat[:, zigzag_order]

    return features, h, w


class GMMDiagonal:
    """
    Gaussian Mixture Model with Diagonal Covariance matrices trained via EM.
    """

    def __init__(self, n_components, n_iter=100, tol=1e-4, min_covar=1e-6):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.min_covar = min_covar
        self.weights = None
        self.means = None
        self.covariances = None
        self.converged_ = False

    def fit(self, X):
        """Trains the model using EM."""
        n_samples, n_features = X.shape

        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices] + np.random.rand(self.n_components, n_features) * 0.01

        global_var = np.var(X, axis=0)
        self.covariances = np.tile(global_var, (self.n_components, 1))

        self.weights = np.ones(self.n_components) / self.n_components

        log_likelihood_old = -np.inf

        for i in range(self.n_iter):
            log_resp, log_likelihood = self._e_step(X)

            if np.abs(log_likelihood - log_likelihood_old) < self.tol:
                self.converged_ = True
                break
            log_likelihood_old = log_likelihood

            self._m_step(X, log_resp)

    def _e_step(self, X):
        """Expectation step: calculate log responsibilities."""
        n_samples, n_features = X.shape
        weighted_log_prob = np.zeros((n_samples, self.n_components))

        const = -0.5 * n_features * np.log(2 * np.pi)

        for c in range(self.n_components):
            log_det = np.sum(np.log(self.covariances[c]))

            diff = X - self.means[c]
            mahalanobis = np.sum((diff ** 2) / self.covariances[c], axis=1)

            log_prob = const - 0.5 * (log_det + mahalanobis)
            weighted_log_prob[:, c] = np.log(self.weights[c] + 1e-300) + log_prob

        log_prob_norm = scipy.special.logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        return log_resp, np.mean(log_prob_norm)

    def _m_step(self, X, log_resp):
        """Maximization step: update parameters."""
        n_samples = X.shape[0]
        resp = np.exp(log_resp)

        Nk = np.sum(resp, axis=0) + 1e-10

        self.weights = Nk / n_samples

        self.means = (resp.T @ X) / Nk[:, np.newaxis]

        for c in range(self.n_components):
            diff = X - self.means[c]
            self.covariances[c] = np.sum(resp[:, c:c + 1] * (diff ** 2), axis=0) / Nk[c]

            self.covariances[c] += self.min_covar

    def score_samples(self, X):
        """Computes weighted log probability P(X|Model) for BDR."""
        n_samples, n_features = X.shape
        const = -0.5 * n_features * np.log(2 * np.pi)
        weighted_log_prob = np.zeros((n_samples, self.n_components))

        for c in range(self.n_components):
            log_det = np.sum(np.log(self.covariances[c]))
            diff = X - self.means[c]
            mahalanobis = np.sum((diff ** 2) / self.covariances[c], axis=1)
            log_prob = const - 0.5 * (log_det + mahalanobis)
            weighted_log_prob[:, c] = np.log(self.weights[c] + 1e-300) + log_prob

        return scipy.special.logsumexp(weighted_log_prob, axis=1)


def solve_problem():
    print("Loading data...")
    mat_data = scipy.io.loadmat(TRAIN_FILE)
    train_fg = mat_data['TrainsampleDCT_FG']
    train_bg = mat_data['TrainsampleDCT_BG']

    cheetah_img = imageio.imread(IMAGE_FILE)
    cheetah_mask = imageio.imread(MASK_FILE)

    cheetah_mask = (cheetah_mask > 127).astype(int)

    zigzag = load_zigzag_pattern(ZIGZAG_FILE)

    n_fg = train_fg.shape[0]
    n_bg = train_bg.shape[0]
    prior_fg = n_fg / (n_fg + n_bg)
    prior_bg = n_bg / (n_fg + n_bg)

    print(f"Priors: FG={prior_fg:.4f}, BG={prior_bg:.4f}")

    print("Extracting test image features...")
    test_features, h, w = compute_dct_features(cheetah_img, zigzag)

    dim_list = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]

    print("\n--- Part A: 5 Mixtures of 8 Components ---")

    n_runs = 5
    n_components = 8

    fg_models = []
    bg_models = []

    print(f"Training {n_runs} models for FG (C={n_components})...")
    for i in range(n_runs):
        gmm = GMMDiagonal(n_components=n_components, n_iter=200)
        gmm.fit(train_fg)
        fg_models.append(gmm)

    print(f"Training {n_runs} models for BG (C={n_components})...")
    for i in range(n_runs):
        gmm = GMMDiagonal(n_components=n_components, n_iter=200)
        gmm.fit(train_bg)
        bg_models.append(gmm)

    part_a_errors = []

    plt.figure(figsize=(12, 8))

    print("Evaluating 25 classifier pairs...")
    for i in range(n_runs):
        for j in range(n_runs):
            fg_model = fg_models[i]
            bg_model = bg_models[j]

            errors = []

            for dim in dim_list:
                X_test_dim = test_features[:, :dim]

                def get_log_prob(model, X_d, d):
                    temp_gmm = GMMDiagonal(model.n_components)
                    temp_gmm.weights = model.weights
                    temp_gmm.means = model.means[:, :d]
                    temp_gmm.covariances = model.covariances[:, :d]
                    return temp_gmm.score_samples(X_d)

                log_prob_fg = get_log_prob(fg_model, X_test_dim, dim)
                log_prob_bg = get_log_prob(bg_model, X_test_dim, dim)

                discriminant = (log_prob_fg + np.log(prior_fg)) - (log_prob_bg + np.log(prior_bg))
                pred_mask = (discriminant > 0).astype(int).reshape(h, w)

                mask_flat = cheetah_mask.flatten()
                pred_flat = pred_mask.flatten()

                idx_fg = (mask_flat == 1)
                idx_bg = (mask_flat == 0)

                err_fg = np.sum(pred_flat[idx_fg] == 0) / np.sum(idx_fg)
                err_bg = np.sum(pred_flat[idx_bg] == 1) / np.sum(idx_bg)

                total_error = err_fg * prior_fg + err_bg * prior_bg
                errors.append(total_error)

            part_a_errors.append(errors)
            plt.plot(dim_list, errors, color='blue', alpha=0.3)

    avg_errors = np.mean(part_a_errors, axis=0)
    plt.plot(dim_list, avg_errors, color='red', linewidth=2, label='Average PoE')

    plt.title('Part A: Probability of Error vs Dimension (25 Initialization Pairs)')
    plt.xlabel('Dimension')
    plt.ylabel('Probability of Error')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prob6_a_initialization.png'))
    plt.close()
    print(f"Part A Plot saved to {OUTPUT_DIR}")

    # --- Part B: Varying Components ---
    print("\n--- Part B: Varying Component Counts ---")
    components_list = [1, 2, 4, 8, 16, 32]

    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(components_list))]

    for idx, C in enumerate(components_list):
        print(f"Training mixture with C={C}...")

        gmm_fg = GMMDiagonal(n_components=C, n_iter=200)
        gmm_fg.fit(train_fg)

        gmm_bg = GMMDiagonal(n_components=C, n_iter=200)
        gmm_bg.fit(train_bg)

        errors_c = []

        for dim in dim_list:
            X_test_dim = test_features[:, :dim]

            def get_log_prob_c(model, X_d, d):
                temp_gmm = GMMDiagonal(model.n_components)
                temp_gmm.weights = model.weights
                temp_gmm.means = model.means[:, :d]
                temp_gmm.covariances = model.covariances[:, :d]
                return temp_gmm.score_samples(X_d)

            log_prob_fg = get_log_prob_c(gmm_fg, X_test_dim, dim)
            log_prob_bg = get_log_prob_c(gmm_bg, X_test_dim, dim)

            discriminant = (log_prob_fg + np.log(prior_fg)) - (log_prob_bg + np.log(prior_bg))
            pred_flat = (discriminant > 0).astype(int)

            mask_flat = cheetah_mask.flatten()
            idx_fg = (mask_flat == 1)
            idx_bg = (mask_flat == 0)

            err_fg = np.sum(pred_flat[idx_fg] == 0) / np.sum(idx_fg)
            err_bg = np.sum(pred_flat[idx_bg] == 1) / np.sum(idx_bg)

            total_error = err_fg * prior_fg + err_bg * prior_bg
            errors_c.append(total_error)

        plt.plot(dim_list, errors_c, marker='o', label=f'C={C}', color=colors[idx])
        print(f"  C={C} Errors: {['{:.4f}'.format(e) for e in errors_c]}")

    plt.title('Part B: Probability of Error vs Dimension (Varying Components)')
    plt.xlabel('Dimension')
    plt.ylabel('Probability of Error')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prob6_b_components.png'))
    plt.close()
    print(f"Part B Plot saved to {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    solve_problem()