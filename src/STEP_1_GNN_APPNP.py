'''This code takes in the file 'nuclei_all.xlsx' which is reuiqred to have at least the Z, N, A, AME, M_th, deltaM= AME- M_th for training.
All other features(like S_n etc.) can be removed or new features can be added if required.'''
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import gc
import os
import scipy.sparse as sp

print("Process ID:", os.getpid())

''' This is a .py file which can be called by the STEP_1_GNN_APPNP.ipynb file. The STEP_1_GNN_APPNP.ipynb file will use this code to loop over 22 random groups of nuclei. 
We train over other 21 groups of nuclei and test over 22nd group and repeat this for all the groups and then compile the test data.'''

TEST_GROUP = int(sys.argv[1])

# SETTINGS

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PLATEAU_EPOCHS = 6
'''we are using this because out of hundreds of those neural networks, 90% are reduandant for a given group(but they are random for different groups)
So, when the change in the loss is less than the desired amount, we simply stop the current NN(neural network) and move on to the next one with decreased regularisation.'''

MAX_EPOCHS = 500  #To prevent overfitting.

os.makedirs("results_residual_gnn_singlebin", exist_ok=True)

print("\n" + "="*70)
print(f"MAJOR RUN — TEST GROUP {TEST_GROUP}")
print("="*70)

# LOAD DATA

df = pd.read_excel("nuclei_all.xlsx")
groups_df = pd.read_excel("splits/random_22_groups.xlsx")
df = df.merge(groups_df, on=["Z","N"], how="left")

feature_cols = [
    "Z","N","A",
    "deltaM_local_mean","deltaM_local_var",
    "S_n","S_p","S_2n","S_2p","E_bind",
    "I","abs_I","I2_over_A","absI_over_A",
    "pairing_class",
    "dZ_magic","dN_magic","dZ2_magic","dN2_magic",
    "N-Z",
    "deltaM_local_mean_bous","deltaM_local_var_bous"
]     #these are all the features that we used for training, we may remove them if not needed.

scaler = StandardScaler()
X_all = scaler.fit_transform(df[feature_cols].values).astype(np.float32)

delta_all = df["deltaM"].values.astype(np.float32)

train_mask_np = (df["group"] != TEST_GROUP).values
test_mask_np  = (df["group"] == TEST_GROUP).values

train_mask = tf.constant(train_mask_np)
test_mask  = tf.constant(test_mask_np)

N = len(df)

# BUILD GRAPH (±1, ±2, diagonals)

coords = df[["Z","N"]].values
adj = sp.lil_matrix((N, N))
coord_dict = {(z,n): i for i,(z,n) in enumerate(coords)}

shifts = [
    (1,0),(0,1),(-1,0),(0,-1),
    (2,0),(0,2),(-2,0),(0,-2),
    (1,1),(1,-1),(-1,1),(-1,-1),
    (2,2),(2,-2),(-2,2),(-2,-2)
]

for i,(z,n) in enumerate(coords):
    for dz,dn in shifts:
        neighbor = (z+dz, n+dn)
        if neighbor in coord_dict:
            j = coord_dict[neighbor]

            if abs(dz)+abs(dn) == 1:
                weight = 1.0
            elif abs(dz)+abs(dn) == 2:
                weight = 0.3
            else:
                weight = 0.2

            adj[i,j] = weight

adj = adj + sp.eye(N)

deg = np.array(adj.sum(1)).flatten()
deg_inv_sqrt = np.power(deg, -0.5)
deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
D_inv_sqrt = sp.diags(deg_inv_sqrt)

adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
adj_norm = tf.constant(adj_norm.toarray(), dtype=tf.float32)

# BUILD APPNP MODEL

def build_appnp(input_dim, l2_value):

    gamma = 0.2 * l2_value
    alpha = 0.5
    K = 4

    inputs = tf.keras.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(
        512, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_value)
    )(inputs)

    x = tf.keras.layers.Dense(
        512, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_value)
    )(x)

    x = tf.keras.layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_value)
    )(x)

    x = tf.keras.layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_value)
    )(x)

    x = tf.keras.layers.Dense(
        128, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_value)
    )(x)

    x = tf.keras.layers.Dense(
        128, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_value)
    )(x)

    x = tf.keras.layers.Dense(
        64, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_value)
    )(x)

    h0 = tf.keras.layers.Dense(1)(x)

    class APPNPPropagation(tf.keras.layers.Layer):
        def __init__(self, adj_matrix, alpha, K):
            super().__init__()
            self.adj = adj_matrix
            self.alpha = alpha
            self.K = K

        def call(self, h0):
            h = h0
            for _ in range(self.K):
                h = (1 - self.alpha) * tf.matmul(self.adj, h) + self.alpha * h0
            return h

    h = APPNPPropagation(adj_norm, alpha, K)(h0)

    class SmoothnessLayer(tf.keras.layers.Layer):
        def __init__(self, adj_matrix, gamma):
            super().__init__()
            self.adj = adj_matrix
            self.gamma = gamma

        def call(self, y):
            diff = tf.matmul(self.adj, y) - y
            self.add_loss(self.gamma * tf.reduce_mean(tf.square(diff)))
            return y

    outputs = SmoothnessLayer(adj_norm, gamma)(h)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")

    return model

# L2 SCHEDULE 

'''We begin from large regularisation and slowly decrease it to squeeze out as much as global trend information possible.'''
L2_schedule = []
L2_schedule += list(np.arange(10000e-5, 3000e-5 - 1e-12, -500e-5))
L2_schedule += list(np.arange(2950e-5, 500e-5 - 1e-12, -100e-5))
L2_schedule += list(np.arange(490e-5, 100e-5 - 1e-12, -30e-5))
L2_schedule += list(np.arange(94e-5, 40e-5 - 1e-12, -6e-5))
L2_schedule += list(np.arange(37e-5, 1e-5 - 1e-12, -1e-5))
L2_schedule += list(np.arange(9e-6, 1e-6 - 1e-12, -4e-6))
L2_schedule += list(np.arange(9e-7, 1e-7 - 1e-12, -4e-7))

# TRAINING LOOP
'''we decreased loss threshold because after some point the loss function usually decreases very less so to avoid skipping necessary steps, we decreased the loss threshold'''
for step_index, current_l2 in enumerate(L2_schedule):

    print(f"\nL2 STEP {step_index+1} | L2 = {current_l2:.8e}")
    loss_threshold = 0.02 if current_l2 >= 2e-4 else 0.009

    model = build_appnp(X_all.shape[1], current_l2)

    previous_loss = None
    stable = 0
    epochs = 0

    while epochs < MAX_EPOCHS:

        with tf.GradientTape() as tape:

            preds = model(X_all, training=True)[:,0]

            train_preds = tf.boolean_mask(preds, train_mask)
            train_true  = tf.boolean_mask(delta_all, train_mask)

            loss = tf.reduce_mean(tf.square(train_preds - train_true))
            loss += sum(model.losses)

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        current_loss = float(loss.numpy())
        epochs += 1

        if previous_loss is not None:
            if abs(previous_loss - current_loss) < loss_threshold:
                stable += 1
            else:
                stable = 0

        previous_loss = current_loss

        if stable >= PLATEAU_EPOCHS:
            break

    print(f"Epochs used: {epochs}")

    preds_all = model(X_all, training=False).numpy().flatten()

    train_rmse = sqrt(mean_squared_error(
        delta_all[train_mask_np],
        preds_all[train_mask_np]
    ))

    test_rmse = sqrt(mean_squared_error(
        delta_all[test_mask_np],
        preds_all[test_mask_np]
    ))

    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Test  RMSE: {test_rmse:.6f}")

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    if train_rmse <= 0.025:   # it is mostly noise beyond this point so we stop here.
        print("*** TARGET TRAIN RMSE REACHED — STOPPING GROUP ***")
        break

# ============================================================
# SAVE RESULT
# ============================================================

output_df = df[test_mask_np][["Z","N","A"]].copy()
output_df["deltaM"] = delta_all[test_mask_np]  #deltaM is the original deltaM
output_df["deltaMi_final"] = preds_all[test_mask_np]    #deltaMi_final is the one learnt and predicted by the NN.

save_path = f"results_residual_ffnn_singlebin/Test_Group_{TEST_GROUP}_Final.xlsx"
output_df.to_excel(save_path, index=False)

print("\nGROUP COMPLETE.")
