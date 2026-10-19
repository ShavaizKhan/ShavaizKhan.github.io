import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load whitespace-separated data
data = pd.read_csv("2/assn2 material/assn2 material/housing.data",sep=r"\s+",header=None).to_numpy()

# First 13 columns features - last column is target
X = data[:, :13]
y = data[:, 13]

feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                 "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

# Required split: first 100 rows for training
X_train = X[:100]
y_train = y[:100]

# Remaining 406 rows for testing
X_test = X[100:]
y_test = y[100:]

print(X.shape)
print(X_train.shape)
print(X_test.shape)

# Include a bias term and powers of each feature up to degree d
# Not including cross-terms between different features

def polynomial_basis(X, degree):
    n_samples, n_features = X.shape

    columns = [np.ones(n_samples)]  # Bias term

    for j in range(n_features):
        for power in range(1, degree + 1):
            columns.append(X[:, j] ** power)

    return np.column_stack(columns)


# calculate RMS error
def rms_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Normalize every feature using TRAINING-set statistics only
# (the test set must be scaled with the training mean/std, never its own)
x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)

# CHAS is 0 for all of the first 100 rows, so its std is 0.
# Replace a zero std with 1 so we don't divide by zero (the column stays all 0).
x_std[x_std == 0] = 1.0

X_train_n = (X_train - x_mean) / x_std
X_test_n = (X_test - x_mean) / x_std


# Part A: Polynomial regression, degrees 1 to 7, all 13 features

# fit models for polynomial degrees 1 through 7
degrees = range(1, 8)

train_errors = []
test_errors = []

for degree in degrees:
    # construct polynomial basis matrices (from the NORMALIZED features)
    Phi_train = polynomial_basis(X_train_n, degree)
    Phi_test = polynomial_basis(X_test_n, degree)

    # ordinary least-squares solution using the pseudoinverse
    weights = np.linalg.pinv(Phi_train) @ y_train

    # Predict house values
    y_train_pred = Phi_train @ weights
    y_test_pred = Phi_test @ weights

    # Calculate RMS errors
    train_rms = rms_error(y_train, y_train_pred)
    test_rms = rms_error(y_test, y_test_pred)

    train_errors.append(train_rms)
    test_errors.append(test_rms)

    print(f"Degree {degree}: " f"Training RMS = {train_rms:.4f}, " f"Test RMS = {test_rms:.4e}")


# plot training and test RMS error against polynomial degree
# (test error grows by many orders of magnitude, so use a log y-axis)
plt.figure(figsize=(8, 5))

plt.semilogy(degrees, train_errors, "o-", label="Training RMS error")
plt.semilogy(degrees, test_errors, "s-", label="Test RMS error")

plt.xlabel("Polynomial Degree")
plt.ylabel("RMS Error (log scale)")
plt.title("Part A: Polynomial Regression, Training and Test Error")
plt.xticks(list(degrees))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("2/media/1a.png", dpi=150)
plt.show()


# Part B: Degree-1 polynomial (linear) regression, inspect the weights

Phi_train_b = polynomial_basis(X_train_n, 1)
weights_b = np.linalg.pinv(Phi_train_b) @ y_train

# weights_b[0] is the bias; weights_b[k] (k = 1..13) belongs to feature k
# (1-indexed), i.e. feature_names[k - 1]
print("\nPart B: degree-1 weights (features normalized)")
print(f"  bias        = {weights_b[0]:9.4f}")
for name, w in zip(feature_names, weights_b[1:]):
    print(f"  {name:<10}  = {w:9.4f}")

# RM (average rooms) and DIS (distance to employment centres)
rm_idx, dis_idx = 5, 7
print(f"\nWeight on RM  (column {rm_idx}, {feature_names[rm_idx]}):  "
      f"{weights_b[1 + rm_idx]:.4f}")
print(f"Weight on DIS (column {dis_idx}, {feature_names[dis_idx]}): "
      f"{weights_b[1 + dis_idx]:.4f}")

# Same weights converted back to the original units ($1000 per unit of feature)
# w_orig = w_normalized / std. Handy for interpreting the weights.
print("\nPart B: weights in original units (per 1 unit of each raw feature)")
for name, w, s in zip(feature_names, weights_b[1:], x_std):
    print(f"  {name:<10}  = {w / s:9.4f}")


# Part C: Visualize polynomial regression using one feature

# Use the second feature (ZN) and the target (MEDV)
X_c = data[:, 1]
y_c = data[:, 13]

# Use the first 100 samples for training
X_train_c = X_c[:100]
y_train_c = y_c[:100]

# Remaining samples for testing
X_test_c = X_c[100:]
y_test_c = y_c[100:]

# Normalize the feature using training-set statistics
x_mean_c = np.mean(X_train_c)
x_std_c = np.std(X_train_c)

X_train_cn = (X_train_c - x_mean_c) / x_std_c
X_test_cn = (X_test_c - x_mean_c) / x_std_c

# Plot low-, medium-, and high-degree polynomial fits
degrees_c = [1, 3, 8]

for degree in degrees_c:
    # Construct polynomial basis matrices
    Phi_train_c = polynomial_basis(
        X_train_cn.reshape(-1, 1), degree
    )

    # Fit the model using least squares
    weights_c = np.linalg.pinv(Phi_train_c) @ y_train_c

    # Generate a smooth curve for plotting
    x_plot = np.linspace(
        min(X_train_cn.min(), X_test_cn.min()),
        max(X_train_cn.max(), X_test_cn.max()),
        300
    )

    Phi_plot = polynomial_basis(x_plot.reshape(-1, 1), degree)
    y_plot = Phi_plot @ weights_c

    # Plot training data, fitted polynomial, and test data
    plt.figure(figsize=(8, 5))

    plt.scatter(
        X_train_cn, y_train_c,
        label="Training data", alpha=0.7
    )

    plt.plot(
        x_plot, y_plot,
        label=f"Degree {degree} polynomial"
    )

    plt.scatter(
        X_test_cn, y_test_c,
        label="Test data", alpha=0.7, marker="x"
    )

    plt.xlabel("Normalized ZN")
    plt.ylabel("Median House Value (MEDV)")
    plt.title(f"Part C: Polynomial Regression, Degree {degree}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"2/media/1c_degree_{degree}.png", dpi=150)
    plt.show()



# Part D: Ridge regression with 10-fold cross-validation

# Use the first 100 samples and the second feature (ZN)
X_d = X_train_cn.reshape(-1, 1)
y_d = y_train_c

degree_d = 8

lambdas_d = [0, 0.01, 0.1, 1, 10, 100, 1000]
n_folds = 10

# Create the degree-8 polynomial design matrix
Phi_d = polynomial_basis(X_d, degree_d)

# Randomly shuffle indices to create 10 folds
np.random.seed(42)
indices_d = np.random.permutation(len(y_d))
folds_d = np.array_split(indices_d, n_folds)

cv_errors_d = []

for lam in lambdas_d:
    fold_errors = []

    for fold in range(n_folds):
        # Current fold is validation data
        val_idx = folds_d[fold]

        # Remaining folds are training data
        train_idx = np.concatenate([
            folds_d[i]
            for i in range(n_folds)
            if i != fold
        ])

        Phi_train_d = Phi_d[train_idx]
        y_train_d = y_d[train_idx]

        Phi_val_d = Phi_d[val_idx]
        y_val_d = y_d[val_idx]

        # Ridge regression:
        # Do not penalize the bias/intercept weight
        penalty = np.eye(Phi_train_d.shape[1])
        penalty[0, 0] = 0

        weights_d = np.linalg.pinv(
            Phi_train_d.T @ Phi_train_d + lam * penalty
        ) @ Phi_train_d.T @ y_train_d

        # Validate the model
        y_val_pred_d = Phi_val_d @ weights_d
        fold_rms = rms_error(y_val_d, y_val_pred_d)

        fold_errors.append(fold_rms)

    # Average validation error over the 10 folds
    mean_error = np.mean(fold_errors)
    cv_errors_d.append(mean_error)

    print(
        f"Lambda = {lam:7g}: "
        f"Average validation RMS = {mean_error:.4f}"
    )

# Identify lambda with the lowest average validation error
best_index_d = np.argmin(cv_errors_d)
best_lambda_d = lambdas_d[best_index_d]

print(f"\nBest lambda: {best_lambda_d}")
print(f"Lowest average validation RMS: {cv_errors_d[best_index_d]:.4f}")

# Plot positive lambda values on a logarithmic x-axis
positive_indices = [
    i for i, lam in enumerate(lambdas_d) if lam > 0
]

positive_lambdas = [lambdas_d[i] for i in positive_indices]
positive_errors = [cv_errors_d[i] for i in positive_indices]

plt.figure(figsize=(8, 5))

plt.semilogx(
    positive_lambdas,
    positive_errors,
    "o-",
    label="10-fold validation RMS"
)

plt.xlabel("Regularization Constant (lambda)")
plt.ylabel("Average Validation RMS Error")
plt.title("Part D: Ridge Regression Cross-Validation")
plt.xticks(positive_lambdas, [str(x) for x in positive_lambdas])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("2/media/1d.png", dpi=150)
plt.show()

# Reporting unregularized result separately (lambda = 0)
print(
    f"Unregularized baseline (lambda = 0): "
    f"Average validation RMS = {cv_errors_d[0]:.4f}"
)


#  PART 2: Gaussian basis function regression (all 13 features)
 
# Gaussian basis function centred on a training point c_k:
#     phi_k(x) = exp( -||x - c_k||^2 / (2 * sigma^2) )
# A bias column of 1s is added as the first column of the design matrix.
def gaussian_basis(Z, centers, sigma):
    # squared distance from every row of Z to every centre -> (n_samples, K)
    sq_dist = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.column_stack([np.ones(len(Z)), np.exp(-sq_dist / (2 * sigma ** 2))])
 
 
# choose width sigma
# use the median distance between pairs of training points
pair_diffs = X_train_n[:, None, :] - X_train_n[None, :, :]
pair_dist = np.sqrt((pair_diffs ** 2).sum(axis=2))
sigma = np.median(pair_dist[np.triu_indices(len(X_train_n), k=1)])
print(f"\nPart 2: Gaussian width sigma (median pairwise distance) = {sigma:.3f}")
 
# Reference point: how well does just predicting the training mean do on test?
baseline_test = rms_error(y_test, np.full(len(y_test), y_train.mean()))
print(f"Reference: predicting the training mean gives test RMS = {baseline_test:.3f}")
 
 
# Part E: number of basis functions K = 5, ... no regularization
 
# One random permutation of the training points; the first K are the centres
# (so the centres for K = 5 are also included in K = 15, and so on)
perm = np.random.RandomState(0).permutation(len(X_train_n))
K_values = list(range(5, 100, 10))
 
def gaussian_errors(perm, K_values):
    train_e, test_e = [], []
    for K in K_values:
        centers = X_train_n[perm[:K]]
        Phi_tr = gaussian_basis(X_train_n, centers, sigma)
        Phi_te = gaussian_basis(X_test_n, centers, sigma)
 
        # least-squares weights (pseudoinverse), no regularization
        w = np.linalg.pinv(Phi_tr) @ y_train
 
        train_e.append(rms_error(y_train, Phi_tr @ w))
        test_e.append(rms_error(y_test, Phi_te @ w))
    return np.array(train_e), np.array(test_e)
 
train_errors_e, test_errors_e = gaussian_errors(perm, K_values)
 
for K, tr, te in zip(K_values, train_errors_e, test_errors_e):
    print(f"K = {K:2d}: Training RMS = {tr:.4f}, Test RMS = {te:.4f}")
 
plt.figure(figsize=(8, 5))
plt.plot(K_values, train_errors_e, "o-", label="Training RMS error")
plt.plot(K_values, test_errors_e, "s-", label="Test RMS error")
plt.xlabel("Number of Gaussian basis functions (K)")
plt.ylabel("RMS Error")
plt.title("Part E: Gaussian Basis Regression, Training and Test Error")
plt.xticks(K_values)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("2/media/2e.png", dpi=150)
plt.show()
 
# Robustness check
# repeat with 20 different random permutations
all_train_e, all_test_e = [], []
for s in range(20):
    tr_e, te_e = gaussian_errors(np.random.RandomState(s).permutation(len(X_train_n)), K_values)
    all_train_e.append(tr_e)
    all_test_e.append(te_e)
all_train_e, all_test_e = np.array(all_train_e), np.array(all_test_e)
 
plt.figure(figsize=(8, 5))
plt.plot(K_values, np.median(all_train_e, axis=0), "o-", label="Training RMS (median of 20 permutations)")
plt.plot(K_values, np.median(all_test_e, axis=0), "s-", label="Test RMS (median of 20 permutations)")
plt.fill_between(K_values, np.percentile(all_test_e, 25, axis=0),
                 np.percentile(all_test_e, 75, axis=0), alpha=0.2, label="Test RMS, 25th to 75th percentile")
plt.yscale("log")
plt.xlabel("Number of Gaussian basis functions (K)")
plt.ylabel("RMS Error (log scale)")
plt.title("Part E (robustness): 20 random choices of centres")
plt.xticks(K_values)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("2/media/2e_robust.png", dpi=150)
plt.show()
 
 
# Part F: ridge regression with K = 90 basis functions, 10-fold CV
 
K_f = 90
centers_f = X_train_n[perm[:K_f]]
Phi_f = gaussian_basis(X_train_n, centers_f, sigma)   # (100, 91) incl. bias
 
# Ridge solved as an augmented least-squares problem:
#     minimize ||Phi w - t||^2 + lam * ||w[1:]||^2
# which is the same as least squares on [Phi; sqrt(lam) * R] w = [t; 0]
# (R = identity without the bias entry). The bias is not penalized, as in Part D.
def ridge_fit(Phi, t, lam):
    n_cols = Phi.shape[1]
    R = np.eye(n_cols)
    R[0, 0] = 0
    A = np.vstack([Phi, np.sqrt(lam) * R])
    b = np.concatenate([t, np.zeros(n_cols)])
    return np.linalg.lstsq(A, b, rcond=None)[0]
 
def gaussian_cv_curve(seed):
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(y_train)), n_folds)
    curve = []
    for lam in lambdas_d:
        fold_errors = []
        for k in range(n_folds):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            w = ridge_fit(Phi_f[tr_idx], y_train[tr_idx], lam)
            fold_errors.append(rms_error(y_train[val_idx], Phi_f[val_idx] @ w))
        curve.append(np.mean(fold_errors))
    return np.array(curve)
 
cv_errors_f = gaussian_cv_curve(42)
 
for lam, err in zip(lambdas_d, cv_errors_f):
    print(f"Lambda = {lam:7g}: Average validation RMS = {err:.4f}")
 
best_lambda_f = lambdas_d[int(np.argmin(cv_errors_f))]
print(f"\nBest lambda: {best_lambda_f}")
print(f"Lowest average validation RMS: {cv_errors_f.min():.4f}")
 
plt.figure(figsize=(8, 5))
plt.semilogx(positive_lambdas, cv_errors_f[1:], "o-", label="10-fold validation RMS")
# lambda = 0 cannot be drawn on a log axis, so show it as a horizontal baseline
plt.axhline(cv_errors_f[0], color="red", linestyle="--",
            label=f"lambda = 0 (unregularized): {cv_errors_f[0]:.2f}")
plt.yscale("log")   # lambda = 0 is ~50x worse than the rest, so use a log y-axis
plt.xlabel("Regularization Constant (lambda)")
plt.ylabel("Average Validation RMS Error (log scale)")
plt.title("Part F: Gaussian Basis (K = 90) Ridge Regression Cross-Validation")
plt.xticks(positive_lambdas, [str(x) for x in positive_lambdas])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("2/media/2f.png", dpi=150)
plt.show()