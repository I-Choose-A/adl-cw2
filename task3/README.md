## Task 3: CAM Generation and Evaluation

### Step 1: Generate CAMs and Models

Run the following script to generate CAMs and train the model:

```bash
python merged_train_bbox.py
```
This process can take over 30 minutes.To skip CAM generation, you can copy the precomputed CAMs from ADL-CW2/data/CAM to task3/data/ and then run merged_train_bbox.py, then run

```bash
python merged_train_bbox.py
```
This will train the models based on existing CAMs.

### Step 2: Compare Pseudo Labeling Quality
After training, run the following script to compare the effects of different quality pseudo labels:

```bash
python compare_different_quality.py
```

### Step 3: Ablation Study
Last, if you want to see our ablation experiments, run run_cam_ablation.py
```bash
python run_cam_ablation.py
```