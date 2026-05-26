# 🧠 CNN Image Classifier with Parallel Computing using LENET model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)
![Notebook](https://img.shields.io/badge/Environment-Jupyter%20%7C%20Colab-orange)
![Task](https://img.shields.io/badge/Task-Binary%20Image%20Classification-green)
![Compute](https://img.shields.io/badge/Benchmark-CPU%20vs%20GPU%20vs%20Threads-purple)

A practical deep-learning project that combines **binary image classification** with **performance engineering**.

This repository trains a CNN classifier and compares execution time and model behavior across 4 compute setups: **CPU (1-thread), GPU, CPU (2-thread), and CPU (4-thread)**.

---

## 🚀 Project Highlights

- Built an end-to-end CNN classification workflow (load → preprocess → train → infer).
- Benchmarked runtime across CPU/GPU and thread configurations.
- Achieved up to **47.2× speedup** over CPU baseline.
- Improved observed accuracy from **0.75** (CPU baseline) to **0.93** (GPU) and **1.00** (4-thread run).

---

## 📊 Benchmark Results

| Configuration     | Time (s) | Accuracy | Loss   | Speedup vs CPU |
|-------------------|---------:|---------:|-------:|---------------:|
| CPU (1 thread)    | 150.54   | 0.75     | 0.49   | 1.0×           |
| GPU               | 4.08     | 0.93     | 0.18   | 36.9×          |
| CPU (2 threads)   | 3.73     | 0.88     | 0.30   | 40.3×          |
| CPU (4 threads)   | 3.19     | 1.00     | 0.001  | 47.2×          |

### Key Takeaways

- **Fastest setup:** CPU (4-thread) at **3.19s**.
- **Most balanced setup:** GPU with strong speed (**4.08s**) + high accuracy (**0.93**).
- **Performance lesson:** Compute configuration can influence iteration speed as much as architecture tuning.

---

## 🧩 Repository Structure

```text
CNN_ImageClassifier/
├── CAO_cpu.ipynb        # Baseline CPU experiment
├── CAO_gpu.ipynb        # GPU experiment
├── CAO_2thread.ipynb    # CPU with 2-thread experiment
├── CAO_4thread.ipynb    # CPU with 4-thread experiment
├── imageClassifer.pdf   # Project report / documentation
└── README.md            # Project documentation
```

---

## 🏗️ Methodology

1. Load training and testing CSV data.
2. Reshape vectors into image tensors of shape **(100, 100, 3)**.
3. Normalize pixel values.
4. Build CNN model for binary classification.
5. Train using fixed epochs and batch size.
6. Repeat experiments under different compute/thread configurations.
7. Compare runtime, accuracy, and loss.

---

## ⚙️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Environment:** Jupyter Notebook, Google Colab

---

## 📌 Why This Project Matters

This project demonstrates practical ML engineering skills beyond model design:

- Model training workflow design
- Hardware-aware benchmarking
- Runtime optimization using parallel strategies
- Performance-driven experiment reporting

---

## 🔮 Future Improvements

- Add train/validation split with reproducible seeds.
- Track precision, recall, F1-score, and confusion matrix.
- Convert notebook workflow into modular Python scripts.
- Add experiment logging (e.g., MLflow/W&B) for reproducibility.
- Evaluate overfitting risk for perfect-accuracy runs.

---

## 👨‍💻 Author
