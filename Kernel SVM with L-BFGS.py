import os, re, glob, json, time
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import __version__ as skver
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

"""
Kernel Soft SVM (L2 soft-margin, squared-hinge) trained with L-BFGS.

We only use the kernelized formulation with:
  - linear kernel
  - polynomial kernel
  - RBF kernel

Multi-class is composed via One-vs-Rest (OvR) or One-vs-One (OvO),
and compared against a Logistic Regression baseline.
"""

# ----------------------------
# Config
# ----------------------------
DATASETS = {
    "s": "train_data_s.csv",
    "m": "train_data_m.csv",
    "l": "train_data_l.csv"
}
TEST_FILE = "test_data.csv"  # unified validation file

C_GRID = [0.001,0.002, 0.003, 0.004, 0.005, 0.006, 0.007,0.008, 0.009, 0.01, 0.02,
0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0, 100.0]
MAXITER = 500
RANDOM_STATE = 42
TEST_SIZE = 0.2
STRATEGIES = ["ovr", "ovo"]
RUN_BASELINES = True

RESULTS_DIR = "artifacts"
PLOTS_DIR = "plots"

# three kernel strategies
KERNEL_CONFIG = {
    "linear": {},
    "rbf":    {"gamma": 0.1},
    "poly":   {"gamma": 0.1, "degree": 2, "coef0": 0.5}
}

REG_B = 1e-3  # slightly stronger bias regularization than before

USER_FEATURES = [
    'Age','Annual Income','Monthly Inhand Salary','Num Bank Accounts','Num Credit Card',
    'Interest Rate','Num of Loan','Delay from due date','Num of Delayed Payment',
    'changed credit Limit','Num Credit Inquiries','Credit Mix','0utstanding Debt',
    'credit Utilization Ratio','credit History Age','Payment of Min Amount',
    'Amount invested monthly','Monthly Balance','Payment Behaviour'
]
TARGET_NAMES = ["Credit_Score"]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ----------------------------
# Utils: IO & cleaning
# ----------------------------
def canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def safe_find(path):
    if os.path.exists(path):
        return path
    cands = glob.glob(path) + glob.glob(f"*/*{os.path.basename(path)}")
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find file: {path}")

def safe_read_table(filepath):
    try:
        df = pd.read_excel(filepath, engine=None)
        print(f"[INFO] Loaded as Excel: {filepath}")
        return df
    except Exception:
        pass
    for enc in ['utf-8','latin1','iso-8859-1','cp1252','gbk','gb2312']:
        try:
            df = pd.read_csv(filepath, encoding=enc, low_memory=False)
            print(f"[INFO] Loaded as CSV ({enc}): {filepath}")
            return df
        except Exception:
            continue
    return pd.read_csv(filepath, low_memory=False)

def clean_dataframe(df):
    df = df.copy()
    df.columns = [re.sub(r"[^\w]", "_", str(c)).strip("_") for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).apply(
            lambda x: re.sub(r"[^\w\s\.\-\+]", "", x) if pd.notna(x) else x
        )
    return df

def np_json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    return str(obj)

# ----------------------------
# Preprocessing factory
# ----------------------------
def make_onehot_dense():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(X, features_used):
    num_cols = [c for c in features_used if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in features_used if c not in num_cols]
    numeric_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_dense())
    ])
    transformers = [("num", numeric_tf, num_cols)]
    if len(cat_cols) > 0:
        transformers.append(("cat", categorical_tf, cat_cols))
    pre = ColumnTransformer(transformers)
    return pre, num_cols, cat_cols

# ----------------------------
# Label handling
# ----------------------------
def find_target_column(df):
    for t in TARGET_NAMES:
        ct = canon(t)
        for col in df.columns:
            if canon(col) == ct:
                return col
    for col in df.columns:
        cc = canon(col)
        if "credit" in cc and "score" in cc:
            return col
    raise ValueError("Target column not found")

def encode_y(y_raw):
    # Poor -> 0, Standard -> 1, Good -> 2
    if y_raw.dtype == object:
        order_pref = ['Poor', 'Standard', 'Good']
        uniq = list(pd.unique(y_raw))
        ordered = [c for c in order_pref if c in uniq] + \
                  [c for c in uniq if c not in order_pref]
        mapping = {c: i for i, c in enumerate(ordered[:3])}
        y = y_raw.map(mapping).fillna(0).astype(int)
        inv = {v: k for k, v in mapping.items()}
    else:
        y = y_raw.astype(int)
        uniq_vals = sorted(y.unique())
        if len(uniq_vals) > 3:
            y = pd.cut(y, bins=3, labels=[0, 1, 2]).astype(int)
        inv = {i: str(i) for i in sorted(y.unique())}
    assert y.nunique() == 3, f"Expect 3 classes, got {y.nunique()}"
    return y, inv

def encode_y_with_mapping(y_raw, forward_map):
    if y_raw.dtype == object:
        y = y_raw.map(forward_map).fillna(0).astype(int)
    else:
        y = y_raw.astype(int).clip(lower=0, upper=2)
    return y

def map_features(df, target_col):
    cmap = {canon(c): c for c in df.columns}
    out = []
    for feat in USER_FEATURES:
        key = canon(feat)
        if key in cmap:
            ac = cmap[key]
            if ac != target_col and ac not in out:
                out.append(ac)
        else:
            for col in df.columns:
                if key in canon(col) and col != target_col and col not in out:
                    out.append(col)
                    break
    if not out:
        out = [c for c in df.columns
               if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    return out

# ----------------------------
# Kernel utilities
# ----------------------------
def linear_kernel(X1, X2):
    return X1 @ X2.T

def rbf_kernel(X1, X2, gamma=1.0):
    X1_sq = np.sum(X1 ** 2, axis=1)[:, None]
    X2_sq = np.sum(X2 ** 2, axis=1)[None, :]
    K = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)
    return np.exp(-gamma * K)

def poly_kernel(X1, X2, degree=3, gamma=1.0, coef0=1.0):
    return (gamma * (X1 @ X2.T) + coef0) ** degree

def compute_kernel_matrix(X, kernel="linear", **kparams):
    if kernel == "linear":
        K = linear_kernel(X, X)
    elif kernel == "rbf":
        gamma = kparams.get("gamma", 1.0)
        K = rbf_kernel(X, X, gamma=gamma)
    elif kernel == "poly":
        gamma = kparams.get("gamma", 1.0)
        degree = kparams.get("degree", 3)
        coef0 = kparams.get("coef0", 1.0)
        K = poly_kernel(X, X, degree=degree, gamma=gamma, coef0=coef0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # simple normalization to improve conditioning
    dmean = np.mean(np.diag(K))
    if dmean > 0:
        K = K / dmean
    # small ridge to keep it well-conditioned
    n = K.shape[0]
    K = K + 1e-6 * np.eye(n)
    return K

# ----------------------------
# Kernel squared-hinge objective
# ----------------------------
def svm_binary_kernel_obj(theta, K, y_pm1, C=1.0, reg_b=REG_B):
    """
    F(alpha,b) = 0.5 a^T K a + C sum_i max(0, 1 - y_i f_i)^2 + 0.5 reg_b b^2
      a = alpha ⊙ y,   f = K a + b
    """
    n = K.shape[0]
    alpha = theta[:n]
    b = theta[n]
    y = y_pm1.astype(float)

    a = alpha * y
    f = K @ a + b
    m = 1.0 - y * f
    pos = m > 0

    w_norm2 = a @ (K @ a)
    loss = 0.5 * w_norm2 + 0.5 * reg_b * b * b
    if np.any(pos):
        loss += C * np.dot(m[pos], m[pos])

    # gradient wrt alpha
    Ka = K @ a
    grad_alpha_reg = Ka * y

    v = np.zeros_like(m)
    v[pos] = y[pos] * m[pos]              # v_i = y_i m_i if active
    Kv = K @ v
    grad_alpha_loss = -2.0 * C * y * Kv

    grad_alpha = grad_alpha_reg + grad_alpha_loss

    # gradient wrt b
    grad_b = reg_b * b - 2.0 * C * np.sum(v)

    grad = np.concatenate([grad_alpha, np.array([grad_b])])
    return float(loss), grad

def train_binary_svm_lbfgs_kernel(
        X, y_pm1, C=1.0, maxiter=MAXITER,
        kernel="rbf", reg_b=REG_B, **kparams):
    """
    L-BFGS-B on (alpha,b) with precomputed Gram matrix K.
    """
    n = X.shape[0]
    Kmat = compute_kernel_matrix(X, kernel=kernel, **kparams)
    theta0 = np.zeros(n + 1)
    history = {"f": [], "gnorm": [], "time": []}
    t0 = time.time()

    def cb(th):
        f, g = svm_binary_kernel_obj(th, Kmat, y_pm1, C=C, reg_b=reg_b)
        history["f"].append(float(f))
        history["gnorm"].append(float(np.linalg.norm(g)))
        history["time"].append(time.time() - t0)

    res = minimize(lambda th: svm_binary_kernel_obj(
                        th, Kmat, y_pm1, C=C, reg_b=reg_b),
                   theta0, method="L-BFGS-B", jac=True, callback=cb,
                   options={"maxiter": maxiter,
                            "ftol": 1e-6,
                            "gtol": 1e-4,
                            "disp": False})

    if len(history["f"]) == 0:
        f, g = svm_binary_kernel_obj(res.x, Kmat, y_pm1, C=C, reg_b=reg_b)
        history["f"] = [float(f)]
        history["gnorm"] = [float(np.linalg.norm(g))]
        history["time"] = [time.time() - t0]

    model = {
        "theta": res.x,
        "X_train": X,
        "y_pm1": y_pm1.astype(float),
        "kernel": kernel,
        "kparams": kparams
    }
    return model, res.nit, time.time() - t0, history

def predict_binary_kernel(model, X_test):
    X_train = model["X_train"]
    y = model["y_pm1"]
    theta = model["theta"]
    kernel = model["kernel"]
    kparams = model["kparams"]

    n = X_train.shape[0]
    alpha = theta[:n]
    b = theta[n]

    if kernel == "linear":
        Ktx = linear_kernel(X_test, X_train)
    elif kernel == "rbf":
        gamma = kparams.get("gamma", 1.0)
        Ktx = rbf_kernel(X_test, X_train, gamma=gamma)
    elif kernel == "poly":
        gamma = kparams.get("gamma", 1.0)
        degree = kparams.get("degree", 3)
        coef0 = kparams.get("coef0", 1.0)
        Ktx = poly_kernel(X_test, X_train, degree=degree, gamma=gamma, coef0=coef0)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    a = alpha * y
    f = Ktx @ a + b
    y_pred = np.where(f >= 0, 1, -1)
    return y_pred, f

# ----------------------------
# OvR / OvO wrappers (kernel only)
# ----------------------------
def train_ovr_lbfgs(X, y, C=1.0, maxiter=MAXITER, kernel="rbf", kparams=None):
    if kparams is None:
        kparams = {}
    K_classes = len(np.unique(y))
    models, traces, iters, wall = [], [], [], []
    for c in range(K_classes):
        y_pm1 = np.where(y == c, 1.0, -1.0).astype(float)
        model_c, nit, tsec, hist = train_binary_svm_lbfgs_kernel(
            X, y_pm1, C=C, maxiter=maxiter, kernel=kernel, **kparams
        )
        models.append(model_c)
        traces.append(hist)
        iters.append(nit)
        wall.append(tsec)
    return models, traces, sum(iters), sum(wall)

def predict_ovr(models, X):
    scores = []
    for m in models:
        _, f = predict_binary_kernel(m, X)
        scores.append(f)
    scores = np.column_stack(scores)
    y_pred = np.argmax(scores, axis=1)
    return y_pred, scores

def train_ovo_lbfgs(X, y, C=1.0, maxiter=MAXITER, kernel="rbf", kparams=None):
    if kparams is None:
        kparams = {}
    classes = sorted(np.unique(y))
    pairs = list(combinations(range(len(classes)), 2))
    models = {}
    traces = {}
    total_it = 0
    total_wall = 0.0
    for (i, j) in pairs:
        mask = (y == i) | (y == j)
        Xp, yp = X[mask], y[mask]
        y_pm1 = np.where(yp == i, 1.0, -1.0).astype(float)
        model_ij, nit, tsec, hist = train_binary_svm_lbfgs_kernel(
            Xp, y_pm1, C=C, maxiter=maxiter, kernel=kernel, **kparams
        )
        models[(i, j)] = model_ij
        traces[(i, j)] = hist
        total_it += nit
        total_wall += tsec
    return models, traces, total_it, total_wall

def predict_ovo(models, X):
    classes = sorted(set(k for pair in models.keys() for k in pair))
    Kc = len(classes)
    n = X.shape[0]
    votes = np.zeros((n, Kc), dtype=int)
    for (i, j), m in models.items():
        _, dec = predict_binary_kernel(m, X)
        votes[dec >= 0, i] += 1
        votes[dec < 0, j] += 1
    y_pred = np.argmax(votes, axis=1)
    return y_pred, votes

# ----------------------------
# Baselines
# ----------------------------
def run_baselines(X_tr, y_tr, X_va, y_va):
    out = {}
    try:
        t0 = time.time()
        logreg = LogisticRegression(max_iter=200, multi_class="auto", solver="lbfgs")
        logreg.fit(X_tr, y_tr)
        t1 = time.time() - t0
        yp = logreg.predict(X_va)
        out["logreg"] = {
            "acc": accuracy_score(y_va, yp),
            "f1": f1_score(y_va, yp, average="macro"),
            "precision": precision_score(y_va, yp, average="macro"),
            "recall": recall_score(y_va, yp, average="macro"),
            "time": t1
        }
    except Exception as e:
        out["logreg"] = {"error": str(e)}
    return out

# ----------------------------
# Plot helpers
# ----------------------------
def plot_metric_vs_C(histC_acc, histC_f1, title_prefix, outdir):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(7,4))
    for label, series in histC_acc.items():
        plt.plot(series["C"], series["acc"], marker="o", label=label)
    plt.xlabel("C"); plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} — Accuracy vs C")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_vs_C.png"), dpi=150); plt.close()

    plt.figure(figsize=(7,4))
    for label, series in histC_f1.items():
        plt.plot(series["C"], series["f1"], marker="o", label=label)
    plt.xlabel("C"); plt.ylabel("Macro-F1")
    plt.title(f"{title_prefix} — Macro-F1 vs C")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "f1_vs_C.png"), dpi=150); plt.close()

def plot_confusion(y_true, y_pred, title, outfile):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4.2,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.colorbar()
    ticks = np.arange(len(np.unique(y_true)))
    plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()

def plot_convergence(traces_list, title, outfile):
    curves = []
    for hist in traces_list:
        f = np.array(hist["f"], dtype=float)
        if f.size == 0:
            continue
        f_norm = f - f.min() + 1e-12
        curves.append(f_norm)
    if len(curves) == 0:
        return
    L = min(map(len, curves))
    mat = np.stack([c[:L] for c in curves], axis=0)
    avg = mat.mean(axis=0)
    plt.figure(figsize=(7,4))
    plt.semilogy(avg, label="avg normalized obj drop")
    plt.xlabel("L-BFGS callbacks"); plt.ylabel("F - Fmin (log)")
    plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(outfile, dpi=150); plt.close()

def plot_runtime_bars(runtime_dict, title, outfile):
    labels = list(runtime_dict.keys())
    vals = [runtime_dict[k] for k in labels]
    plt.figure(figsize=(7,4))
    plt.bar(labels, vals)
    plt.ylabel("Seconds"); plt.title(title)
    plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()

def plot_ovr_convergence_by_class(traces, label_names, title, outfile):
    plt.figure(figsize=(7.5,4.5))
    any_curve = False
    for k, hist in enumerate(traces):
        f = np.array(hist.get("f", []), dtype=float)
        if f.size == 0:
            continue
        f_norm = f - f.min() + 1e-12
        label = label_names[k] if k < len(label_names) else f"class {k}"
        plt.semilogy(f_norm, label=label)
        any_curve = True
    if not any_curve:
        plt.close()
        return
    plt.xlabel("L-BFGS callbacks")
    plt.ylabel("F - Fmin (log)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

# ----------------------------
# Single dataset experiment
# ----------------------------
def run_single_dataset(ds_key, path, test_path, kernel, kparams):
    tag = f"{ds_key}_{kernel}"
    ds_out_dir = os.path.join(RESULTS_DIR, tag)
    ds_plot_dir = os.path.join(PLOTS_DIR, tag)
    os.makedirs(ds_out_dir, exist_ok=True)
    os.makedirs(ds_plot_dir, exist_ok=True)

    fpath = safe_find(path)
    df_raw = safe_read_table(fpath)
    df = clean_dataframe(df_raw)

    if test_path is None or not os.path.exists(safe_find(test_path)):
        raise FileNotFoundError("Unified validation file test_data.csv not found.")
    dft_raw = safe_read_table(safe_find(test_path))
    dft = clean_dataframe(dft_raw)

    target_col = find_target_column(df)
    features_used = map_features(df, target_col)
    X_tr_df = df[features_used].copy()
    y_tr_raw = df[target_col].copy()

    y_tr, inv_mapping = encode_y(y_tr_raw)
    forward_map = {v: k for k, v in inv_mapping.items()}

    for col in features_used:
        if col not in dft.columns:
            dft[col] = np.nan

    y_va = None
    test_has_label = False
    try:
        tcol_test = find_target_column(dft)
        if tcol_test in dft.columns:
            y_va = encode_y_with_mapping(dft[tcol_test], forward_map)
            test_has_label = True
    except Exception:
        pass

    pre, num_cols, cat_cols = build_preprocessor(X_tr_df, features_used)
    X_tr = pre.fit_transform(X_tr_df)
    X_va = pre.transform(dft[features_used])

    results = {
        "meta": {
            "dataset": ds_key, "file": path,
            "kernel": kernel, "kparams": kparams,
            "n_train": int(X_tr.shape[0]), "n_valid": int(X_va.shape[0]),
            "num_features": int(X_tr.shape[1]),
            "num_numeric": len(num_cols),
            "num_categorical": len(cat_cols)
        },
        "historyC_acc": {}, "historyC_f1": {},
        "best": {}, "baselines": {}, "inv_mapping": inv_mapping
    }
    for strat in STRATEGIES:
        results["historyC_acc"][strat] = {"C": [], "acc": []}
        results["historyC_f1"][strat] = {"C": [], "f1": []}

    # --- train/validate for each strategy and C
    for strat in STRATEGIES:
        best = {"acc": -1.0}
        best_details = None
        for Cval in C_GRID:
            if strat == "ovr":
                t0 = time.time()
                models, traces, nit, wall = train_ovr_lbfgs(
                    X_tr, y_tr, C=Cval, maxiter=MAXITER,
                    kernel=kernel, kparams=kparams
                )
                train_time = time.time() - t0
                y_hat, _ = predict_ovr(models, X_va)
            else:
                t0 = time.time()
                models, traces, nit, wall = train_ovo_lbfgs(
                    X_tr, y_tr, C=Cval, maxiter=MAXITER,
                    kernel=kernel, kparams=kparams
                )
                train_time = time.time() - t0
                y_hat, _ = predict_ovo(models, X_va)

            if test_has_label:
                acc = accuracy_score(y_va, y_hat)
                f1m = f1_score(y_va, y_hat, average="macro")
                results["historyC_acc"][strat]["C"].append(Cval)
                results["historyC_acc"][strat]["acc"].append(acc)
                results["historyC_f1"][strat]["C"].append(Cval)
                results["historyC_f1"][strat]["f1"].append(f1m)
                if acc > best["acc"]:
                    best = {
                        "acc": acc, "f1": f1m, "C": Cval,
                        "nit": int(nit), "time": float(train_time),
                        "y_pred": y_hat
                    }
                    best_details = {"models": models, "traces": traces}
            else:
                if best["acc"] < 0:
                    best = {
                        "acc": None, "f1": None, "C": Cval,
                        "nit": int(nit), "time": float(train_time),
                        "y_pred": y_hat
                    }
                    best_details = {"models": models, "traces": traces}

        results["best"][strat] = best

        # plots for best setting
        if best_details is not None:
            if strat == "ovr":
                traces_list = best_details["traces"]
                plot_convergence(
                    traces_list,
                    title=f"{ds_key} [{kernel} {strat}] Convergence (C={best['C']})",
                    outfile=os.path.join(ds_plot_dir, f"convergence_{strat}.png")
                )
                class_names = [str(inv_mapping.get(i, f"class {i}")) for i in range(len(traces_list))]
                plot_ovr_convergence_by_class(
                    traces_list, class_names,
                    title=f"{ds_key} [{kernel} OvR] Per-class Convergence (C={best['C']})",
                    outfile=os.path.join(ds_plot_dir, f"convergence_{strat}_per_class.png")
                )
            else:
                traces_list = list(best_details["traces"].values())
                plot_convergence(
                    traces_list,
                    title=f"{ds_key} [{kernel} {strat}] Convergence (C={best['C']})",
                    outfile=os.path.join(ds_plot_dir, f"convergence_{strat}.png")
                )
            if test_has_label and best["y_pred"] is not None:
                plot_confusion(
                    y_va, best["y_pred"],
                    title=f"{ds_key} [{kernel} {strat}] Confusion (best C={best['C']})",
                    outfile=os.path.join(ds_plot_dir, f"confmat_{strat}.png")
                )

    # --- baselines
    if RUN_BASELINES and test_has_label:
        results["baselines"] = run_baselines(X_tr, y_tr, X_va, y_va)

    plot_metric_vs_C(results["historyC_acc"], results["historyC_f1"],
                     title_prefix=f"{ds_key}-{kernel}", outdir=ds_plot_dir)

    runtime = {}
    for strat in STRATEGIES:
        if "time" in results["best"][strat]:
            runtime[strat] = results["best"][strat]["time"]
    if RUN_BASELINES:
        for k, v in results["baselines"].items():
            if isinstance(v, dict) and "time" in v:
                runtime[k] = v["time"]
    if len(runtime) > 0:
        plot_runtime_bars(runtime, f"{ds_key}-{kernel} Runtime Comparison",
                          os.path.join(ds_plot_dir, "runtime_bars.png"))

    with open(os.path.join(ds_out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=np_json_default)

    # save predictions on validation set
    for strat in STRATEGIES:
        pred = results["best"][strat].get("y_pred", None)
        if test_has_label and pred is not None:
            pd.DataFrame({"true": y_va, f"pred_{strat}": pred}).to_csv(
                os.path.join(ds_out_dir, f"valid_preds_{strat}.csv"), index=False
            )
        else:
            pd.DataFrame({f"pred_{strat}": pred if pred is not None else []}).to_csv(
                os.path.join(ds_out_dir, f"valid_preds_{strat}.csv"), index=False
            )

    return {
        "preprocessor": pre,
        "features_used": features_used,
        "inv_mapping": inv_mapping,
        "X_full": X_tr, "y_full": y_tr,
        "meta": results["meta"],
        "results": results,
        "plot_dir": ds_plot_dir,
        "art_dir": ds_out_dir
    }

# ----------------------------
# MAIN: run all datasets and kernels
# ----------------------------
def main():
    all_results = {}
    for kernel, kparams in KERNEL_CONFIG.items():
        print(f"\n=== Kernel: {kernel}  Params: {kparams} ===")
        for ds_key, path in DATASETS.items():
            print(f"  -> Running dataset {ds_key}: {path}")
            all_results[(ds_key, kernel)] = run_single_dataset(
                ds_key, path, test_path=TEST_FILE,
                kernel=kernel, kparams=kparams
            )

    # aggregate comparison across datasets, kernels, strategies
    rows = []
    for (ds, kernel), pack in all_results.items():
        R = pack["results"]
        for strat in STRATEGIES:
            best = R["best"][strat]
            rows.append({
                "dataset": ds,
                "kernel": kernel,
                "strategy": strat,
                "C": best.get("C"),
                "acc": best.get("acc"),
                "macro_f1": best.get("f1"),
                "time_sec": best.get("time"),
                "nit_total": best.get("nit"),
            })
        if RUN_BASELINES:
            for bname, bres in R["baselines"].items():
                if "acc" in bres:
                    rows.append({
                        "dataset": ds,
                        "kernel": kernel,
                        "strategy": bname,
                        "C": None,
                        "acc": bres.get("acc"),
                        "macro_f1": bres.get("f1"),
                        "time_sec": bres.get("time"),
                        "nit_total": None,
                    })
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(RESULTS_DIR, "overall_comparison.csv"), index=False)

    for metric in ["macro_f1", "acc"]:
        if metric not in comp_df.columns or comp_df[metric].isna().all():
            continue
        plt.figure(figsize=(9,5))
        for strat in sorted(comp_df["strategy"].unique()):
            sub = comp_df[comp_df["strategy"] == strat]
            labels = [f"{r['dataset']}-{r['kernel']}" for _, r in sub.iterrows()]
            plt.plot(labels, sub[metric].values, marker="o", label=strat)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"Overall {metric.upper()} comparison (validated on TEST)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"overall_{metric}.png"), dpi=150)
        plt.close()

    # show best macro-F1 over all kernels/strategies
    choice = None
    best_score = -1
    for (ds, kernel), pack in all_results.items():
        R = pack["results"]
        for strat in STRATEGIES:
            f1m = R["best"][strat].get("f1", -1)
            if f1m is not None and f1m > best_score:
                best_score = f1m
                choice = (ds, kernel, strat, R["best"][strat]["C"])
    print("\n[INFO] Best model (by macro-F1):", choice, "macro-F1 =", best_score)

if __name__ == "__main__":
    main()
