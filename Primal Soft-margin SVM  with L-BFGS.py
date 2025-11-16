# -*- coding: utf-8 -*-
"""
Comprehensive experiments for 3-class Credit Score Classification with SVM (squared-hinge)
optimized by L-BFGS-B (OvR & OvO), across three training datasets:
  - train_data_s.csv
  - train_data_m.csv
  - train_data_l.csv
Optional test set: test_data.csv

What this script does:
- Robust load/clean → preprocess (impute+scale numeric, one-hot categorical)
- Train OvR & OvO L-BFGS SVMs on C ∈ {0.004,0.005, 0.006, 0.007, 0.008, 0.009, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09.0.1.0.2,0.3,0.4,0.5, 1, 10, 100}
- Track convergence (objective per L-BFGS callback), iterations, runtime
- Baselines: Logistic Regression (lbfgs, L2), LinearSVC
- Ablation: remove categorical features
- Sensitivity: metric vs C plots
- Aggregate comparison across datasets & methods
- Pick best “optimized” model (max macro-F1) → save and predict test_data.csv

Outputs (per dataset and overall):
- plots/<ds>/{acc_vs_C.png, f1_vs_C.png, convergence_*.png, confmat_*.png, runtime_bars.png}
- artifacts/<ds>/* (models, metrics.json, predictions.csv)
- plots/overall/* (comparative charts)
- artifacts/optimized_model/* (final model + predictions on test_data.csv)
"""
'''
soft-margin svm
'''

import os, re, glob, json, time
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import __version__ as skver
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

"""
Soft SVM (L2 soft-margin, squared-hinge) — unconstrained primal with L-BFGS

For each binary problem:
  F(w, b) = (1/2)||w||^2 + C * sum_i [ max(0, 1 - y_i (w^T x_i + b)) ]^2  + (lambda_b/2) * b^2

Let f_i = w^T x_i + b, m_i = 1 - y_i f_i, active set I+ = { i : m_i > 0 }.
Gradients:
  ∇_w F = w - 2C * sum_{i in I+} y_i * m_i * x_i
  ∇_b F = lambda_b * b - 2C * sum_{i in I+} y_i * m_i

We use L-BFGS (scipy.optimize.minimize with 'L-BFGS-B') to optimize (w, b).
Multi-class is composed via OvR or OvO.
"""

# ----------------------------
# Config
# ----------------------------
DATASETS = {
    "s": "train_data_s.csv",
    "m": "train_data_m.csv",
    "l": "train_data_l.csv"
}
TEST_FILE = "test_data.csv"  # optional
C_GRID = [0.004,0.005, 0.006, 0.007, 0.008, 0.009, 0.01,0.02,0.03,0.04,0.05,0.06,0.070.08,0.09.0.1.0.2,0.3,0.4,0.5,1,10,100]  # sensitivity
MAXITER = 500
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_REPEATS = 1  # keep 1 for speed; set >1 for mean±std over multiple splits
STRATEGIES = ["ovr", "ovo"]  # which SVM decompositions to run
RUN_BASELINES = True         # LogisticRegression (L2) only
RESULTS_DIR = "artifacts"
PLOTS_DIR = "plots"

USER_FEATURES = [
    'Age','Annual Income','Monthly Inhand Salary','Num Bank Accounts','Num Credit Card',
    'Interest Rate','Num of Loan','Delay from due date','Num of Delayed Payment',
    'changed credit Limit','Num Credit Inquiries','Credit Mix','0utstanding Debt',
    'credit Utilization Ratio','credit History Age','Payment of Min Amount',
    'Amount invested monthly','Monthly Balance','Payment Behaviour'
]
TARGET_NAMES = ["Credit_Score"]  # primary guess for label

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
    # try to locate nearby
    cands = glob.glob(path) + glob.glob(f"*/*{os.path.basename(path)}")
    for p in cands:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Cannot find file: {path}")

def safe_read_table(filepath):
    # try Excel first (in case misnamed), then CSV encodings
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
    # last fallback
    return pd.read_csv(filepath, low_memory=False)

def clean_dataframe(df):
    df = df.copy()
    df.columns = [re.sub(r"[^\w]", "_", str(c)).strip("_") for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r"[^\w\s\.\-\+]", "", x) if pd.notna(x) else x)
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
    # compatibility across sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(X, features_used):
    num_cols = [c for c in features_used if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in features_used if c not in num_cols]
    numeric_tf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", StandardScaler())])
    categorical_tf = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                               ("onehot", make_onehot_dense())])
    transformers = [("num", numeric_tf, num_cols)]
    if len(cat_cols) > 0:
        transformers.append(("cat", categorical_tf, cat_cols))
    pre = ColumnTransformer(transformers)
    return pre, num_cols, cat_cols

# ----------------------------
# Label handling
# ----------------------------
def find_target_column(df):
    # direct match
    for t in TARGET_NAMES:
        ct = canon(t)
        for col in df.columns:
            if canon(col) == ct:
                return col
    # fuzzy
    for col in df.columns:
        cc = canon(col)
        if "credit" in cc and "score" in cc:
            return col
    raise ValueError("Target column not found")

def encode_y(y_raw):
    """
    Encode labels into {0,1,2} following the preferred order:
    Poor -> 0, Standard -> 1, Good -> 2 (when available in data).
    Returns y (int array) and inverse mapping {int: original_label}.
    """
    if y_raw.dtype == object:
        order_pref = ['Poor','Standard','Good']
        uniq = list(pd.unique(y_raw))
        ordered = [c for c in order_pref if c in uniq] + [c for c in uniq if c not in order_pref]
        mapping = {c:i for i,c in enumerate(ordered[:3])}
        y = y_raw.map(mapping).fillna(0).astype(int)
        inv = {v:k for k,v in mapping.items()}
    else:
        y = y_raw.astype(int)
        uniq_vals = sorted(y.unique())
        if len(uniq_vals) > 3:
            y = pd.cut(y, bins=3, labels=[0,1,2]).astype(int)
        inv = {i:str(i) for i in sorted(y.unique())}
    assert y.nunique()==3, f"Expect 3 classes, got {y.nunique()}"
    return y, inv

def map_features(df, target_col):
    """
    Map the preferred USER_FEATURES names to actual columns in df (fuzzy),
    excluding the target column. If nothing maps, fall back to all numeric.
    """
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
                    out.append(col); break
    if not out:
        out = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    return out

# ----------------------------
# SVM objective (squared hinge), with convergence tracing
# ----------------------------
def svm_binary_obj(theta, X, y_pm1, C=1.0, reg_b=1e-5):
    """
    Squared-hinge soft SVM objective & gradient (unconstrained primal).
    theta = [w; b]. Adds tiny b-regularization 'reg_b' for numerical stability.

    F(w,b) = (1/2)||w||^2 + C * sum_i [max(0, 1 - y_i (w^T x_i + b))]^2 + (reg_b/2) * b^2
    ∇_w F = w - 2C * sum_{i: m_i>0} y_i * m_i * x_i
    ∇_b F = reg_b * b - 2C * sum_{i: m_i>0} y_i * m_i
    """
    n, p = X.shape
    w, b = theta[:p], theta[p]
    m = 1.0 - y_pm1*(X.dot(w)+b)
    pos = (m > 0)
    loss = 0.5*np.dot(w,w) + 0.5*reg_b*b*b
    if np.any(pos): loss += C*np.dot(m[pos], m[pos])
    grad_w = w.copy(); grad_b = reg_b*b
    if np.any(pos):
        coeff = -2.0*C*y_pm1[pos]*m[pos]
        grad_w += X[pos].T.dot(coeff)
        grad_b += np.sum(coeff)
    return loss, np.append(grad_w, grad_b)

def train_binary_svm_lbfgs_with_trace(X, y_pm1, C=1.0, maxiter=500):
    """
    Minimize the unconstrained squared-hinge objective using L-BFGS-B.
    Records a convergence trace {f, gnorm, time}.
    """
    n, p = X.shape
    theta0 = np.zeros(p+1)
    history = {"f": [], "gnorm": [], "time": []}
    t0 = time.time()
    def cb(th):
        f, g = svm_binary_obj(th, X, y_pm1, C=C)
        history["f"].append(float(f))
        history["gnorm"].append(float(np.linalg.norm(g)))
        history["time"].append(time.time()-t0)
    res = minimize(lambda th: svm_binary_obj(th, X, y_pm1, C=C),
                   theta0, method="L-BFGS-B", jac=True, callback=cb,
                   options={"maxiter": maxiter, "ftol": 1e-7, "gtol": 1e-5, "disp": False})
    # ensure at least one point is recorded
    if len(history["f"])==0:
        f,g = svm_binary_obj(res.x, X, y_pm1, C=C)
        history["f"] = [float(f)]
        history["gnorm"] = [float(np.linalg.norm(g))]
        history["time"] = [time.time()-t0]
    return res.x, res.nit, time.time()-t0, history

def train_ovr_lbfgs(X, y, C=1.0, maxiter=500):
    K = len(np.unique(y))
    models, traces, iters, wall = [], [], [], []
    for c in range(K):
        y_pm1 = np.where(y==c, 1.0, -1.0).astype(float)
        theta, nit, tsec, hist = train_binary_svm_lbfgs_with_trace(X, y_pm1, C=C, maxiter=maxiter)
        models.append(theta); traces.append(hist); iters.append(nit); wall.append(tsec)
    return np.vstack(models), traces, sum(iters), sum(wall)

def predict_ovr(models, X):
    K, d = models.shape; p = d-1
    scores = np.column_stack([X.dot(models[k,:p]) + models[k,p] for k in range(K)])
    return np.argmax(scores, axis=1), scores

def train_ovo_lbfgs(X, y, C=1.0, maxiter=500):
    classes = sorted(np.unique(y))
    pairs = list(combinations(range(len(classes)), 2))
    models = {}
    traces = {}
    total_it = 0
    total_wall = 0.0
    for (i,j) in pairs:
        mask = (y==i) | (y==j)
        Xp, yp = X[mask], y[mask]
        y_pm1 = np.where(yp==i, 1.0, -1.0).astype(float)
        theta, nit, tsec, hist = train_binary_svm_lbfgs_with_trace(Xp, y_pm1, C=C, maxiter=maxiter)
        models[(i,j)] = theta; traces[(i,j)] = hist
        total_it += nit; total_wall += tsec
    return models, traces, total_it, total_wall

def predict_ovo(models, X):
    classes = sorted(set(k for pair in models.keys() for k in pair))
    K = len(classes); n, p = X.shape
    votes = np.zeros((n, K), dtype=int)
    for (i,j), theta in models.items():
        w, b = theta[:p], theta[p]
        dec = X.dot(w) + b
        votes[dec>0, i] += 1
        votes[dec<=0, j] += 1
    return np.argmax(votes, axis=1), votes

# ----------------------------
# Baselines (only Logistic Regression)
# ----------------------------
def run_baselines(X_tr, y_tr, X_va, y_va):
    out = {}
    # Logistic Regression (lbfgs, L2)
    try:
        t0 = time.time()
        logreg = LogisticRegression(max_iter=200, multi_class="auto", solver="lbfgs")
        logreg.fit(X_tr, y_tr)
        t1 = time.time()-t0
        yp = logreg.predict(X_va)
        out["logreg"] = {"acc": accuracy_score(y_va, yp), "f1": f1_score(y_va, yp, average="macro"),
                         "time": t1}
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
    plt.xlabel("C"); plt.ylabel("Accuracy"); plt.title(f"{title_prefix} — Accuracy vs C")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir,"acc_vs_C.png"), dpi=150); plt.close()

    plt.figure(figsize=(7,4))
    for label, series in histC_f1.items():
        plt.plot(series["C"], series["f1"], marker="o", label=label)
    plt.xlabel("C"); plt.ylabel("Macro-F1"); plt.title(f"{title_prefix} — Macro-F1 vs C")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir,"f1_vs_C.png"), dpi=150); plt.close()

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
    """
    Average normalized objective drop across binaries.
    traces_list: list of histories (each history has 'f').
    """
    curves = []
    for hist in traces_list:
        f = np.array(hist["f"], dtype=float)
        if f.size==0: continue
        f_norm = f - f.min() + 1e-12
        f_norm = f_norm[:100]
        curves.append(f_norm)
    if len(curves)==0:
        return
    L = min(map(len, curves))
    mat = np.stack([c[:L] for c in curves], axis=0)
    avg = mat.mean(axis=0)
    plt.figure(figsize=(7,4))
    plt.semilogy(avg, label="avg normalized obj drop")
    plt.xlabel("L-BFGS callbacks"); plt.ylabel("F - Fmin (log)")
    plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()

def plot_runtime_bars(runtime_dict, title, outfile):
    labels = list(runtime_dict.keys())
    vals = [runtime_dict[k] for k in labels]
    plt.figure(figsize=(7,4))
    plt.bar(labels, vals)
    plt.ylabel("Seconds"); plt.title(title)
    plt.tight_layout(); plt.savefig(outfile, dpi=150); plt.close()

def plot_ovr_convergence_by_class(traces, label_names, title, outfile):
    """
    Overlay per-class convergence (OvR only) on a single figure.
    traces: list of length K; each item is a history dict with 'f'.
    label_names: list of textual class names in the same order as traces (e.g., ['Poor','Standard','Good']).
    """
    plt.figure(figsize=(7.5,4.5))
    any_curve = False
    for k, hist in enumerate(traces):
        f = np.array(hist.get("f", []), dtype=float)
        if f.size == 0:
            continue
        f_norm = f - f.min() + 1e-12
        f_norm = f_norm[:100]
        plt.semilogy(f_norm, label=label_names[k] if k < len(label_names) else f"class {k}")
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
# Label handling (alt encoder with a provided forward map)
# ----------------------------
def encode_y_with_mapping(y_raw, forward_map):
    """Encode y with an existing forward mapping (label->int); unseen labels fallback to 0."""
    if y_raw.dtype == object:
        y = y_raw.map(forward_map).fillna(0).astype(int)
    else:
        # If numeric already, clip to {0,1,2}
        y = y_raw.astype(int).clip(lower=0, upper=2)
    return y

# ----------------------------
# Single dataset experiment (train on train_xxx, validate on TEST_FILE)
# ----------------------------
def run_single_dataset(ds_key, path, test_path=None):
    ds_out_dir = os.path.join(RESULTS_DIR, ds_key)
    ds_plot_dir = os.path.join(PLOTS_DIR, ds_key)
    os.makedirs(ds_out_dir, exist_ok=True)
    os.makedirs(ds_plot_dir, exist_ok=True)

    # --- load training set
    fpath = safe_find(path)
    df_raw = safe_read_table(fpath)
    df = clean_dataframe(df_raw)

    # --- load unified validation set (TEST_FILE)
    if test_path is None or not os.path.exists(safe_find(test_path)):
        raise FileNotFoundError("Unified validation file test_data.csv (TEST_FILE) not found.")
    dft_raw = safe_read_table(safe_find(test_path))
    dft = clean_dataframe(dft_raw)

    # --- target & features (based on training set)
    target_col = find_target_column(df)
    features_used = map_features(df, target_col)
    X_tr_df = df[features_used].copy()
    y_tr_raw = df[target_col].copy()

    # fit encoding on training set (get inv mapping)
    y_tr, inv_mapping = encode_y(y_tr_raw)
    # forward mapping label->int
    forward_map = {v: k for k, v in inv_mapping.items()}

    # --- align test columns
    for col in features_used:
        if col not in dft.columns:
            dft[col] = np.nan

    # if test also has labels, encode them using training mapping for evaluation
    y_va = None
    test_has_label = False
    try:
        tcol_test = find_target_column(dft)
        if tcol_test in dft.columns:
            y_va = encode_y_with_mapping(dft[tcol_test], forward_map)
            test_has_label = True
    except Exception:
        pass

    # --- preprocessing: fit on training, transform test
    pre, num_cols, cat_cols = build_preprocessor(X_tr_df, features_used)
    X_tr = pre.fit_transform(X_tr_df)
    X_va = pre.transform(dft[features_used])

    results = {
        "meta": {"dataset": ds_key, "file": path,
                 "n_train": int(X_tr.shape[0]), "n_valid": int(X_va.shape[0]),
                 "num_features": int(X_tr.shape[1]), "num_numeric": len(num_cols),
                 "num_categorical": len(cat_cols)},
        "historyC_acc": {}, "historyC_f1": {},
        "best": {}, "baselines": {}, "inv_mapping": inv_mapping
    }
    for strat in STRATEGIES:
        results["historyC_acc"][strat] = {"C": [], "acc": []}
        results["historyC_f1"][strat]  = {"C": [], "f1": []}

    # --- train/validate (validation always uses TEST_FILE)
    for strat in STRATEGIES:
        best = {"acc": -1.0}
        best_details = None
        for Cval in C_GRID:
            if strat == "ovr":
                t0 = time.time()
                models, traces, nit, wall = train_ovr_lbfgs(X_tr, y_tr, C=Cval, maxiter=MAXITER)
                train_time = time.time() - t0
                y_hat, _ = predict_ovr(models, X_va)
            else:
                t0 = time.time()
                models, traces, nit, wall = train_ovo_lbfgs(X_tr, y_tr, C=Cval, maxiter=MAXITER)
                train_time = time.time() - t0
                y_hat, _ = predict_ovo(models, X_va)

            # if test has labels, log metrics; otherwise keep placeholders
            if test_has_label:
                acc = accuracy_score(y_va, y_hat)
                f1m = f1_score(y_va, y_hat, average="macro")
                results["historyC_acc"][strat]["C"].append(Cval)
                results["historyC_acc"][strat]["acc"].append(acc)
                results["historyC_f1"][strat]["C"].append(Cval)
                results["historyC_f1"][strat]["f1"].append(f1m)
                if acc > best["acc"]:
                    best = {"acc": acc, "f1": f1m, "C": Cval,
                            "nit": int(nit), "time": float(train_time), "y_pred": y_hat}
                    best_details = {"models": models, "traces": traces}
            else:
                # no labels: pick the first C as placeholder "best"
                if best["acc"] < 0:
                    best = {"acc": None, "f1": None, "C": Cval,
                            "nit": int(nit), "time": float(train_time), "y_pred": y_hat}
                    best_details = {"models": models, "traces": traces}

        results["best"][strat] = best

        # plots: average convergence and confusion (if labels)
        if best_details is not None:
            if strat == "ovr":
                # average across binaries
                traces_list = best_details["traces"]
                plot_convergence(
                    traces_list,
                    title=f"{ds_key} [{strat}] Convergence (C={best['C']})",
                    outfile=os.path.join(ds_plot_dir, f"convergence_{strat}.png")
                )
                # per-class overlay for OvR
                class_names = [str(inv_mapping.get(i, f"class {i}")) for i in range(len(traces_list))]
                plot_ovr_convergence_by_class(
                    traces_list, class_names,
                    title=f"{ds_key} [OvR] Per-class Convergence (C={best['C']})",
                    outfile=os.path.join(ds_plot_dir, f"convergence_{strat}_per_class.png")
                )
            else:
                # for OvO, average trace across pairwise classifiers
                traces_list = list(best_details["traces"].values())
                plot_convergence(
                    traces_list,
                    title=f"{ds_key} [{strat}] Convergence (C={best['C']})",
                    outfile=os.path.join(ds_plot_dir, f"convergence_{strat}.png")
                )
            if test_has_label and best["y_pred"] is not None:
                plot_confusion(y_va, best["y_pred"],
                               title=f"{ds_key} [{strat}] Confusion (best C={best['C']})",
                               outfile=os.path.join(ds_plot_dir, f"confmat_{strat}.png"))

    # --- baselines: Logistic Regression only
    if RUN_BASELINES:
        results["baselines"] = run_baselines(X_tr, y_tr, X_va, y_va if test_has_label else y_tr)

    # --- C curves
    plot_metric_vs_C(results["historyC_acc"], results["historyC_f1"],
                     title_prefix=f"{ds_key}",
                     outdir=ds_plot_dir)

    # --- runtime bars
    runtime = {}
    for strat in STRATEGIES:
        if "time" in results["best"][strat]:
            runtime[strat] = results["best"][strat]["time"]
    if RUN_BASELINES:
        for k, v in results["baselines"].items():
            if "time" in v:
                runtime[k] = v["time"]
    plot_runtime_bars(runtime, f"{ds_key} Runtime Comparison", os.path.join(ds_plot_dir, "runtime_bars.png"))

    # --- persist results
    with open(os.path.join(ds_out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=np_json_default)

    for strat in STRATEGIES:
        # validation predictions (on test set)
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
# MAIN: run all datasets + aggregate + optimized model
# ----------------------------
def main():
    all_results = {}
    for ds_key, path in DATASETS.items():
        print(f"\n=== Running dataset {ds_key}: {path} (train on train, validate on TEST) ===")
        all_results[(ds_key, "full")] = run_single_dataset(
            ds_key, path, test_path=TEST_FILE
        )

    # aggregate comparison (metrics from TEST_FILE)
    rows = []
    for (ds, tag), pack in all_results.items():
        R = pack["results"]
        for strat in STRATEGIES:
            best = R["best"][strat]
            rows.append({
                "dataset": ds,
                "setting": tag,
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
                        "setting": tag,
                        "strategy": bname,
                        "C": None,
                        "acc": bres.get("acc"),
                        "macro_f1": bres.get("f1"),
                        "time_sec": bres.get("time"),
                        "nit_total": None,
                    })
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(RESULTS_DIR, "overall_comparison.csv"), index=False)

    # overall plots
    for metric in ["macro_f1", "acc"]:
        if metric not in comp_df.columns or comp_df[metric].isna().all():
            continue
        plt.figure(figsize=(9,5))
        for strat in sorted(comp_df["strategy"].unique()):
            sub = comp_df[comp_df["strategy"]==strat]
            labels = [f"{r['dataset']}-{r['setting']}" for _, r in sub.iterrows()]
            plt.plot(labels, sub[metric].values, marker="o", label=strat)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(metric.upper()); plt.title(f"Overall {metric.upper()} comparison (validated on TEST)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"overall_{metric}.png"), dpi=150); plt.close()

    # choose "optimized model": among full & SVM strategies by Macro-F1 (from TEST)
    choice = None
    best_score = -1
    for (ds, tag), pack in all_results.items():
        if tag != "full": continue
        R = pack["results"]
        for strat in STRATEGIES:
            f1m = R["best"][strat].get("f1", -1)
            if f1m is not None and f1m > best_score:
                best_score = f1m
                choice = (ds, strat, R["best"][strat]["C"])
    print("\n[INFO] Chosen optimized model:", choice, "macro-F1=", best_score)

def train_on_all_then_test(train_paths, test_path):
    out_dir = os.path.join(RESULTS_DIR, "combined")
    plot_dir = os.path.join(PLOTS_DIR, "combined")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # --- load & concat S/M/L
    dfs = []
    for p in train_paths:
        df_raw = safe_read_table(safe_find(p))
        dfs.append(clean_dataframe(df_raw))
    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    # --- target & features on the combined frame
    target_col = find_target_column(df_all)
    features_used = map_features(df_all, target_col)

    X_all = df_all[features_used].copy()
    y_raw_all = df_all[target_col].copy()
    y_all, inv_mapping = encode_y(y_raw_all)

    # --- preprocessor on combined features
    pre, num_cols, cat_cols = build_preprocessor(X_all, features_used)
    Xp_all = pre.fit_transform(X_all)

    # --- small internal split to choose C and strategy
    X_tr, X_va, y_tr, y_va = train_test_split(
        Xp_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
    )

    best = {"f1": -1.0}
    histC_acc = {s: {"C": [], "acc": []} for s in STRATEGIES}
    histC_f1  = {s: {"C": [], "f1": []} for s in STRATEGIES}

    for strat in STRATEGIES:
        for Cval in C_GRID:
            if strat == "ovr":
                t0 = time.time()
                models, traces, nit, wall = train_ovr_lbfgs(X_tr, y_tr, C=Cval, maxiter=MAXITER)
                t1 = time.time() - t0
                y_hat, _ = predict_ovr(models, X_va)
            else:
                t0 = time.time()
                models, traces, nit, wall = train_ovo_lbfgs(X_tr, y_tr, C=Cval, maxiter=MAXITER)
                t1 = time.time() - t0
                y_hat, _ = predict_ovo(models, X_va)

            acc = accuracy_score(y_va, y_hat)
            f1m = f1_score(y_va, y_hat, average="macro")
            histC_acc[strat]["C"].append(Cval); histC_acc[strat]["acc"].append(acc)
            histC_f1[strat]["C"].append(Cval);  histC_f1[strat]["f1"].append(f1m)

            if f1m > best["f1"]:
                best = {
                    "strategy": strat, "C": Cval, "acc": acc, "f1": f1m,
                    "nit": int(nit), "time": float(t1),
                    "traces": traces
                }

    # --- plot metric vs C on combined
    plot_metric_vs_C(histC_acc, histC_f1, "combined", plot_dir)

    # --- refit best on ALL combined training rows
    strat, Cstar = best["strategy"], best["C"]
    if strat == "ovr":
        models, traces, nit, wall = train_ovr_lbfgs(Xp_all, y_all, C=Cstar, maxiter=MAXITER)
        final_model = {"type": "ovr", "C": Cstar, "models": models}
        traces_list = traces
        # per-class convergence overlay for OvR (combined)
        class_names = [str(inv_mapping.get(i, f"class {i}")) for i in range(len(traces_list))]
        plot_ovr_convergence_by_class(
            traces_list, class_names,
            title=f"combined [OvR] Per-class Convergence (C={Cstar})",
            outfile=os.path.join(plot_dir, f"convergence_ovr_per_class.png")
        )
    else:
        models, traces, nit, wall = train_ovo_lbfgs(Xp_all, y_all, C=Cstar, maxiter=MAXITER)
        final_model = {"type": "ovo", "C": Cstar, "models": {str(k): v for k, v in models.items()}}
        traces_list = list(traces.values())

    # average convergence figure
    plot_convergence(traces_list, f"combined [{strat}] Convergence (C={Cstar})",
                     os.path.join(plot_dir, f"convergence_{strat}.png"))

    # --- test on full test_data.csv
    test_metrics = {}
    if test_path and os.path.exists(test_path):
        dft_raw = safe_read_table(test_path)
        dft = clean_dataframe(dft_raw)

        # ensure all training features exist in test
        for col in features_used:
            if col not in dft.columns:
                dft[col] = np.nan
        Xt = pre.transform(dft[features_used])

        # predict
        if final_model["type"] == "ovr":
            K, d = final_model["models"].shape; p = d - 1
            scores = np.column_stack([Xt.dot(final_model["models"][k, :p]) + final_model["models"][k, p] for k in range(K)])
            y_pred = np.argmax(scores, axis=1)
        else:
            md = {eval(k): v for k, v in final_model["models"].items()}
            n, p = Xt.shape
            votes = np.zeros((n, len(np.unique(y_all))), dtype=int)
            for (i, j), th in md.items():
                w, b = th[:p], th[p]
                dec = Xt.dot(w) + b
                votes[dec > 0, i] += 1
                votes[dec <= 0, j] += 1
            y_pred = np.argmax(votes, axis=1)

        # optional: if test has labels, compute metrics
        try:
            tcol_test = find_target_column(dft)
            y_test, _ = encode_y(dft[tcol_test])
            test_metrics = {
                "test_acc": float(accuracy_score(y_test, y_pred)),
                "test_macro_f1": float(f1_score(y_test, y_pred, average="macro"))
            }
            plot_confusion(y_test, y_pred, f"combined [{strat}] Test Confusion (C={Cstar})",
                           os.path.join(plot_dir, "confmat_test.png"))
        except Exception:
            pass

        # ID column & readable labels
        id_col = next((c for c in ["ID", "id", "Customer_ID", "customer_id"] if c in dft.columns), None)
        if id_col is None:
            dft["ID"] = np.arange(len(dft)); id_col = "ID"
        labels = [inv_mapping.get(int(k), str(k)) for k in y_pred]

        # save predictions
        sub = pd.DataFrame({"ID": dft[id_col], "Credit_Score_Pred": y_pred, "Credit_Score_Label": labels})
        sub.to_csv(os.path.join(out_dir, "test_predictions_combined.csv"), index=False)

    # --- persist combined artifacts
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "strategy": strat, "C": Cstar,
            "valid_acc": float(best["acc"]), "valid_macro_f1": float(best["f1"]),
            **test_metrics
        }, f, indent=2, default=np_json_default)

    # save preprocessor + model
    try:
        import joblib
        joblib.dump(pre, os.path.join(out_dir, "preprocessor.joblib"))
    except Exception:
        pass
    np.save(os.path.join(out_dir, "features.npy"), np.array(features_used, dtype=object))
    np.save(os.path.join(out_dir, "final_model.npy"), np.array([final_model], dtype=object))

    print(f"[INFO] Combined training complete. Artifacts → {out_dir}, Plots → {plot_dir}")

if __name__ == "__main__":
    main()
    train_on_all_then_test(
        train_paths=[DATASETS["s"], DATASETS["m"], DATASETS["l"]],
        test_path=TEST_FILE
    )

