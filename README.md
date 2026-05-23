
# Vehicle–Pedestrian Interaction Field (Based on Trajectron++)

## 1. Overview

This project builds on **Trajectron++ (Salzmann et al. 2020)** to implement a **vehicle–pedestrian interaction field** (人车交互场) framework.

Compared with the original Trajectron++ implementation, we modify the code so that the model outputs **parameters of trajectory distributions**, rather than only sampled trajectories.

For each prediction step, the model provides the parameters of a **25-component Gaussian mixture**, including:

- Means  
- Standard deviations  
- Correlation coefficients  

These parameters are then used to construct a **probabilistic spatio-temporal field** and to visualize and analyze interaction risk.

---

## 2. Project Structure and Data

The main directory for experiments is:

```text
Probabilistic-Spatio-temporal-Field/experiments
```

Processed datasets are stored under:

```text
Probabilistic-Spatio-temporal-Field/experiments/processed
```

Currently, two types of processed data are included:

- `FC_*`: **Mid-block** scenarios (路中段数据)
- `TJ_*`: **Intersection** scenarios (交叉口数据, derived from **SinD**)

Each `*_train_full.pkl` / `*_test_full.pkl` file follows the standard Trajectron++ data format.

---

## 3. Training the Trajectory Generation Model

Below is an example command to train the trajectory generation model on the **FC (mid-block)** dataset:

```bash
python train.py \
  --eval_every 1 \
  --vis_every 1 \
  --conf ../experiments/nuScenes/models/int_ee/config.json \
  --train_data_dict FC_train_full.pkl \
  --eval_data_dict FC_test_full.pkl \
  --offline_scene_graph yes \
  --preprocess_workers 10 \
  --batch_size 512 \
  --log_dir ../experiments/nuScenes/models \
  --train_epochs 100 \
  --node_freq_mult_train \
  --log_tag _int_ee \
  --site FC
```

**Arguments (key ones):**

- `--conf`: Path to the modified Trajectron++ config used for interaction field experiments.  
- `--train_data_dict` / `--eval_data_dict`: Training and test datasets under `experiments/processed`.  
- `--offline_scene_graph`: Whether to use offline scene graphs (recommended: `yes`).  
- `--site`: Dataset/site identifier (`FC` for mid-block, `TJ` for intersection).  
- `--log_dir`: Directory where models and logs will be saved.  

After training, the model checkpoints will be stored under:

```text
../experiments/nuScenes/models
```

---

## 4. Generating Trajectory Distribution Parameters and Samples

After training, use the following command to:

1. Export the **Gaussian mixture parameters** (means, standard deviations, and correlation coefficients) for each prediction step; and  
2. Sample trajectories for **visualization** and inspection of the interaction field.

Example command for the **FC (mid-block)** test set:

```bash
python evaluate.py \
  --model models/FC_models_01_Nov_2025_07_49_21_int_ee \
  --checkpoint=90 \
  --data ../processed/FC_test_full.pkl \
  --output_path results \
  --output_tag int_ee \
  --node_type VEHICLE \
  --prediction_horizon 12 \
  --site FC
```

**Arguments (key ones):**

- `--model`: Path/name of the trained model directory.  
- `--checkpoint`: Checkpoint index to load (e.g., `90`).  
- `--data`: Test data file (e.g., `FC_test_full.pkl`) under `experiments/processed`.  
- `--output_path`: Directory to save outputs (distribution parameters and sampled trajectories).  
- `--output_tag`: Tag appended to output files.  
- `--node_type`: Node/agent type (e.g., `VEHICLE`).  
- `--prediction_horizon`: Prediction horizon (number of future time steps).  
- `--site`: Dataset/site identifier (`FC` or `TJ`).  

The evaluation script will produce:

- Files containing the **trajectory distribution parameters** (for constructing the probabilistic spatio-temporal field); and  
- **Sampled trajectories**, which can be used directly for plotting and visualizing vehicle–pedestrian interaction patterns.

---

## 5. Notes

- This code is an extension of the original **Trajectron++** framework and assumes a similar environment setup (dependencies, data format, etc.).  
- The 25-component Gaussian mixture parameters are the core outputs used in the subsequent risk field construction and analysis.
