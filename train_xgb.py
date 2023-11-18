import numpy as np

from concrete.ml.sklearn import XGBClassifier

def train_xgb(x_train, y_train, n_bits, emb_name):
    model = XGBClassifier(n_bits=n_bits, max_depth=2, n_estimators=50, n_jobs=1)
    model.fit(x_train, y_train)    
    model.compile(x_train)
    with open(f"models/xgb_{emb_name}_{n_bits}_bits", "w") as f:
        model.dump(f)
    
if __name__ == '__main__':
    x_transf = {'transf': np.load("data/preprocessed_data/x_train_transf_emb.npy")}
    x_gnn = {'gnn': np.load("data/preprocessed_data/x_train_gnn_emb.npy")}
    x_transf_gnn = {'transf_gnn': np.load("data/preprocessed_data/x_train_transf_gnn_emb.npy")}
    y = np.load("data/preprocessed_data/y_train.npy")
    for n_bits in [2, 4, 6, 8]:
        for x in [x_transf, x_gnn, x_transf_gnn]:
            for emb_name, emb in x.items():
                train_xgb(emb, y, n_bits, emb_name)