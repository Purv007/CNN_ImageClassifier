# CNN Image Classifier with Parallel Compute Optimization

A deep-learning image classification project focused on **both prediction quality and runtime efficiency**.

This project implements a Convolutional Neural Network (CNN) for binary image classification and benchmarks training performance across multiple execution setups (CPU, GPU, and CPU multithreading). The core goal is to understand how parallel compute configurations impact speed and model behavior.

---

## Project Summary

- Built a CNN-based binary image classification pipeline using TensorFlow/Keras.
- Performed data loading, reshaping to image tensors, normalization, training, and inference.
- Compared execution time and model outcomes across:
  - CPU (1 thread)
  - GPU
  - CPU (2 threads)
  - CPU (4 threads)
- Measured significant runtime acceleration with parallel approaches while maintaining/improving classification performance.

---

## Key Results

| Configuration     | Time (s) | Accuracy | Loss   | Speedup vs CPU |
|-------------------|---------:|---------:|-------:|---------------:|
| CPU (1 thread)    | 150.54   | 0.75     | 0.49   | 1.0×           |
| GPU               | 4.08     | 0.93     | 0.18   | 36.9×          |
| CPU (2 threads)   | 3.73     | 0.88     | 0.30   | 40.3×          |
| CPU (4 threads)   | 3.19     | 1.00     | 0.001  | 47.2×          |

### Highlights

- Runtime reduced from **150.54s (CPU)** to **3.19s (4-thread CPU)**.
- Achieved up to **47.2× speedup** over the CPU baseline.
- Accuracy improved from **0.75** baseline to **0.93 (GPU)** and **1.00 (4-thread run)**.

---

## Repository Structure

- `CAO_cpu.ipynb` — baseline CPU experiment
- `CAO_gpu.ipynb` — GPU-accelerated experiment
- `CAO_2thread.ipynb` — CPU experiment with 2 threads
- `CAO_4thread.ipynb` — CPU experiment with 4 threads
- `imageClassifer.pdf` — project report/documentation

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook / Google Colab

---

## Methodology

1. Load tabular image data and labels from CSV files.
2. Reshape input vectors to `(100, 100, 3)` image tensors.
3. Normalize pixel values.
4. Define and compile CNN architecture for binary classification.
5. Train model with fixed epochs and batch size.
6. Measure execution time and compare performance across hardware/thread configurations.

---

## Practical Takeaways

- Parallel strategy can drastically reduce training time without changing the dataset.
- Hardware-aware experimentation is critical for efficient ML workflows.
- Performance engineering (threads/GPU) can be as impactful as model tuning.

---

## Future Improvements

- Add reproducible train/validation split and unified metrics logging.
- Include confusion matrix, precision, recall, and F1 score.
- Add scripted pipeline (`.py`) for non-notebook execution.
- Evaluate robustness and potential overfitting of the 4-thread configuration.

---

## Author

Purv Patel
