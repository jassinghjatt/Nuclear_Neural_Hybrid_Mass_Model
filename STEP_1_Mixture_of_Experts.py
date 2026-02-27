'''This code takes in the file 'nuclei_all.xlsx' which is reuiqred to have at least the Z, N, A, AME, M_th, deltaM= AME- M_th for training.
All other features(like S_n etc.) can be removed or new features can be added if required.'''
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import gc
import os

print("Process ID:", os.getpid())

''' This is a .py file which can be called by the STEP_1_FFNN.ipynb file. The STEP_1_FFNN.ipynb file will use this code to loop over 22 random groups of nuclei. 
We train over other 21 groups of nuclei and test over 22nd group and repeat this for all the groups and then compile the test data.'''

TEST_GROUP = int(sys.argv[1])

# SETTINGS

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

PLATEAU_EPOCHS = 6
'''we are using this because out of hundreds of those neural networks, 90% are reduandant for a given group(but they are random for different groups)
So, when the change in the loss is less than the desired amount, we simply stop the current NN(neural network) and move on to the next one with decreased regularisation.'''

MAX_EPOCHS = 500 #To prevent overfitting.

os.makedirs("results_residual_MoE_singlebin", exist_ok=True)

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
] #these are all the features that we used for training, we may remove them if not needed.

train_df = df[df["group"] != TEST_GROUP].copy()
test_df  = df[df["group"] == TEST_GROUP].copy()

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].values)
X_test  = scaler.transform(test_df[feature_cols].values)

delta_train = train_df["deltaM"].values.copy()
delta_test  = test_df["deltaM"].values.copy()

# BUILD MoE

def build_moe(input_dim, l2_value):

    if l2_value >= 2.5e-2:
        entropy_lambda = 0.3 * l2_value
    else:
        entropy_lambda = 2e-2

    inputs = tf.keras.Input(shape=(input_dim,))

    # Gate 
    gate = tf.keras.layers.Dense(
        128, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_value)
    )(inputs)

    gate_logits = tf.keras.layers.Dense(
        5,
        kernel_regularizer=tf.keras.regularizers.l2(l2_value)
    )(gate)

    gate_soft = tf.keras.layers.Softmax()(gate_logits)

    #  Entropy Layer
    class EntropyRegularizer(tf.keras.layers.Layer):
        def call(self, p):
            p = tf.cast(p, tf.float32)
            entropy = -tf.reduce_mean(
                tf.reduce_sum(p * tf.math.log(p + 1e-8), axis=1)
            )
            self.add_loss(entropy_lambda * entropy)
            return p

    gate_soft = EntropyRegularizer()(gate_soft)

    # Experts
    expert_outputs = []

    for _ in range(5):

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

        out = tf.keras.layers.Dense(1)(x)
        expert_outputs.append(out)

    experts_stack = tf.keras.layers.Concatenate(axis=1)(expert_outputs)

    # -------- Weighted Sum --------
    def weighted_sum_fn(tensors):
        g, e = tensors
        return tf.reduce_sum(g * e, axis=1, keepdims=True)

    weighted = tf.keras.layers.Lambda(weighted_sum_fn)(
        [gate_soft, experts_stack]
    )

    model = tf.keras.Model(inputs=inputs, outputs=weighted)
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

for step_index, current_l2 in enumerate(L2_schedule):

    print(f"\nL2 STEP {step_index+1} | L2 = {current_l2:.8e}")

    loss_threshold = 0.02 if current_l2 >= 2e-4 else 0.009
    '''because after some point the loss function usually decreases very less so to avoid skipping necessary steps, we decreased the loss threshold'''
    extreme_weight = 0.2 if current_l2 >= 2e-2 else 0.1
    '''Some nuclei are largely out of bounds and interfere with the learning of other nuclei, so we mask them.'''
    sample_weights = np.ones(len(delta_train))

    mask = (delta_train < -1) | (delta_train > 1)  #1 is in MeV
    sample_weights[mask] = extreme_weight

    model = build_moe(X_train.shape[1], current_l2)

    previous_loss = None
    small_change_count = 0
    epoch_counter = 0

    while epoch_counter < MAX_EPOCHS:

        history = model.fit(
            X_train,
            delta_train,
            sample_weight=sample_weights,
            epochs=1,
            batch_size=64,
            verbose=0
        )

        current_loss = history.history["loss"][0]
        epoch_counter += 1

        if previous_loss is not None:
            if abs(previous_loss - current_loss) < loss_threshold:
                small_change_count += 1
            else:
                small_change_count = 0

        previous_loss = current_loss

        if small_change_count >= PLATEAU_EPOCHS:
            break

    print(f"Epochs used: {epoch_counter}")

    f_train = model.predict(X_train, verbose=0).flatten()
    f_test  = model.predict(X_test, verbose=0).flatten()

    delta_train -= f_train
    delta_test  -= f_test

    train_rmse = sqrt(mean_squared_error(
        np.zeros_like(delta_train),
        delta_train
    ))

    test_rmse = sqrt(mean_squared_error(
        np.zeros_like(delta_test),
        delta_test
    ))

    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Test  RMSE: {test_rmse:.6f}")

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    if train_rmse <= 0.025:  # it is mostly noise beyond this point so we stop here.
        print("*** TARGET TRAIN RMSE REACHED — STOPPING GROUP ***")
        break

# ============================================================
# SAVE RESULT
# ============================================================

output_df = test_df[["Z","N","A"]].copy()
output_df["deltaM"] = test_df["deltaM"]    #deltaM is the original deltaM
output_df["deltaMi_final"] = delta_test    #deltaMi_final is the one learnt and predicted by the NN.

save_path = f"results_residual_ffnn_singlebin/Test_Group_{TEST_GROUP}_Final.xlsx"
output_df.to_excel(save_path, index=False)

print("\nGROUP COMPLETE.")