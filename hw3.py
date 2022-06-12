from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    return x - np.mean(x, axis=0)


def get_covariance(dataset):
    return dataset.T @ dataset / (dataset.shape[0]-1)

def get_eig(S, m):
    w, v = eigh(S, subset_by_index=[S.shape[0]-m, S.shape[0]-1])
    w = np.diag(np.flip(w, axis=0))
    v = np.flip(v, axis=1)
    return w, v


def get_eig_prop(S, prop):
    w_sum = eigh(S, eigvals_only=True).sum()
    w, v = eigh(S, subset_by_value=[prop*w_sum, np.inf])
    w = np.diag(np.flip(w, axis=0))
    v = np.flip(v, axis=1)
    return w, v


def project_image(image, U):
    alpha = U.T @ image
    return U @ alpha

def display_image(orig, proj):
    orig = orig.reshape(32, -1).T
    proj = proj.reshape(32, -1).T

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))

    ax1.set_title('Original')
    ax2.set_title('Projection')

    left = ax1.imshow(orig)
    right = ax2.imshow(proj)
    fig.colorbar(left, ax=ax1)
    fig.colorbar(right, ax=ax2)

    plt.show()