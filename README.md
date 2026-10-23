# 🧠 Deep Learning Practical Exam — Set C

This repository contains the notebook and visual results for five deep-learning practical tasks: ANN classification, OpenCV image processing, CNN digit recognition, GRU temperature forecasting, and Hugging Face Transformer inference.

## 📌 Project Overview

| Module | Objective | Dataset / Input | Main techniques |
|---|---|---|---|
| **1. ANN — Wine Recognition** | Classify wine into one of three cultivars | Scikit-learn Wine Recognition dataset (13 features) | Data scaling, dense neural network, Softmax, accuracy/loss curves, confusion matrix |
| **2. OpenCV — Image Processing** | Apply colour segmentation and image-processing operations to five images | Four built-in `scikit-image` images plus `data/images/RedFlower.jpg` (fallback generated if missing) | BGR/HSV conversion, red-colour masking, erosion/dilation, Sobel edges, Haar-cascade face detection, `minNeighbors` sensitivity check |
| **3. CNN — MNIST Digits** | Recognise handwritten digits (0–9) and compare with a dense ANN baseline | MNIST dataset subset | Normalisation, convolution, pooling, training curves, confusion matrix, feature maps, incorrect-prediction review |
| **4. RNN — GRU Temperature Forecasting** | Predict next-day minimum temperature from the previous 30 days | Daily minimum temperatures in Melbourne (1981–1990) | Sliding windows, GRU, persistence baseline, actual-vs-predicted plot, loss curve |
| **5. Transformers** | Run fill-mask, question-answering and text-generation inference | Short India-related passages and prompts | Hugging Face pretrained pipelines; no fine-tuning |

## 🗂️ Repository Structure

```text
Deep-Learning-Practical-Exam-Set-C/
├── DeepLearning_Practical_Exam_SetC.ipynb
├── README.md
└── output/
    ├── ann_accuracy.png
    ├── ann_confusion_matrix.png
    ├── ann_loss.png
    ├── ann_wine_class_distribution.png
    ├── cnn_accuracy.png
    ├── cnn_confusion_matrix.png
    ├── cnn_feature_maps.png
    ├── cnn_loss.png
    ├── cnn_mnist_samples.png
    ├── cnn_wrong_predictions.png
    ├── opencv_processing_grid.png
    ├── rnn_actual_vs_predicted.png
    ├── rnn_loss.png
    ├── rnn_temperature_1981.png
    ├── rnn_temperature_full_series.png
    └── transformer_qa_confidence.png
```

## 📊 Results & Visualisations

All visualisation files are stored in the `output/` directory and embedded below so they display on GitHub.

### 1. ANN — Wine Recognition

**Training and validation accuracy**

![ANN accuracy](output/ann_accuracy.png)

**Training and validation loss**

![ANN loss](output/ann_loss.png)

**Wine class distribution**

![Wine class distribution](output/ann_wine_class_distribution.png)

**Confusion matrix**

![ANN confusion matrix](output/ann_confusion_matrix.png)

### 2. OpenCV — Image Processing

The grid compares each original image with its HSV view, red-colour mask, morphology output, and Sobel edge map.

![OpenCV image processing results](output/opencv_processing_grid.png)

### 3. CNN — MNIST Digit Recognition

**Training and validation accuracy**

![CNN accuracy](output/cnn_accuracy.png)

**Training and validation loss**

![CNN loss](output/cnn_loss.png)

**Confusion matrix**

![CNN confusion matrix](output/cnn_confusion_matrix.png)

**Example MNIST digits**

![MNIST sample images](output/cnn_mnist_samples.png)

**First-layer feature maps**

![CNN feature maps](output/cnn_feature_maps.png)

**Incorrect predictions for error analysis**

![CNN incorrect predictions](output/cnn_wrong_predictions.png)

### 4. RNN — GRU Temperature Forecasting

**Full temperature time series**

![Full Melbourne temperature series](output/rnn_temperature_full_series.png)

**Temperature series for 1981**

![Melbourne temperature in 1981](output/rnn_temperature_1981.png)

**Actual vs. predicted temperatures**

![Actual versus predicted temperature](output/rnn_actual_vs_predicted.png)

**Training and validation loss**

![GRU loss](output/rnn_loss.png)

### 5. Transformers — Question Answering

The chart shows the model confidence scores for the evaluated question-answering examples. The colours reflect the notebook's correctness labels.

![Transformer QA confidence](output/transformer_qa_confidence.png)

## 🛠️ Requirements

Use the same Python environment/kernel as your Jupyter notebook. The notebook uses Python libraries including:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow` / `keras`
- `opencv-contrib-python`
- `scikit-image`
- `transformers`
- `torch`

A compatible set of versions is recommended because deep-learning and Transformers APIs can change. Avoid installing multiple conflicting OpenCV distributions in the same environment; if OpenCV imports but lacks `CascadeClassifier`, check for a conflicting `cv2.py` file/folder and reinstall a compatible OpenCV package, then restart the kernel.

## ▶️ How to Run

1. Open `DeepLearning_Practical_Exam_SetC.ipynb` in Jupyter Notebook or VS Code.
2. Select the Python environment where the dependencies are installed.
3. Run the notebook cells in order, from top to bottom.
4. The first run may download datasets and pretrained model weights, so internet access is required for those steps.
5. Generated plots and exports should be saved in the `output/` directory when the relevant notebook cells are run.

### OpenCV Haar Cascade note

The Haar cascade XML file must be present and successfully loaded before calling `detectMultiScale`. If your notebook uses a local XML file, update the path to its actual location and verify `face_cascade.empty()` is `False` before detection.

## 📝 Notes

- The provided `output/` images are visual results supplied alongside this README; rerunning the notebook may produce slightly different plots or model results.
- Transformer predictions and confidence scores depend on the pretrained model and library versions.
- The README documents the notebook workflow and visual outputs; it does not claim that every cell has been freshly executed in this packaging step.

## 👤 Author

Deep Learning Practical Exam — Set C
