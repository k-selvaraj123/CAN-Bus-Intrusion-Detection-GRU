# CAN Bus Intrusion Detection using GRU

## Overview

This project develops a lightweight machine-learning-based Intrusion Detection System (IDS) for Controller Area Network (CAN) traffic in automotive systems.

The project combines an offline machine-learning pipeline for preprocessing, training, and model conversion with an embedded ESP32 implementation for real-time CAN traffic monitoring and intrusion detection.

The primary model explored in the embedded implementation is a Gated Recurrent Unit (GRU) network designed to classify CAN traffic using temporal patterns in sequences of CAN frames.

## Project Structure

```text
CAN-Bus-Intrusion-Detection-GRU/
│
├── ml/
│   ├── step1_inspect.py
│   ├── step2_label_merge.py
│   ├── step3_features.py
│   ├── step4_windowing.py
│   ├── step5_normalize_split.py
│   ├── step8_convert_tflite.py
│   ├── step9_to_c_array.py
│   ├── step10_train_all_models.py
│   └── step11_convert_all.py
│
└── embedded/
    └── esp32/
        └── can_ids_gru/
            ├── can_ids_gru.ino
            ├── attack_alert.c
            ├── attack_alert.h
            ├── attack_generator.cpp
            ├── attack_generator.h
            ├── can_handler.c
            ├── can_handler.h
            ├── csv_reporter.cpp
            ├── csv_reporter.h
            ├── feature_extractor.c
            ├── feature_extractor.h
            ├── model_inference.c
            ├── model_inference.h
            └── gru_ids_model.h
```

## Machine Learning Pipeline

The `ml/` directory contains the Python-based data processing and model development pipeline.

The pipeline covers:

1. Dataset inspection
2. Label processing and merging
3. CAN feature extraction
4. Temporal window generation
5. Normalization and train/test splitting
6. Model training
7. TensorFlow Lite conversion
8. Conversion of trained models into C arrays for embedded deployment

## Embedded Implementation

The `embedded/esp32/` directory contains the ESP32 implementation of the IDS.

The embedded system is responsible for:

* Receiving CAN traffic
* Extracting features from incoming CAN frames
* Maintaining temporal input windows
* Running the trained neural-network model on the ESP32
* Detecting potentially malicious CAN traffic
* Generating alerts
* Recording detection results
* Supporting attack-generation/testing functionality

The model is deployed directly to the ESP32 as a C header containing the model parameters.

## Models

The project currently contains implementations/conversions for multiple neural-network architectures, including:

* GRU
* MLP
* 1D CNN

The GRU is the primary architecture for the CAN intrusion-detection system.

## Hardware

Target embedded platform:

* ESP32
* CAN transceiver/interface hardware

## Current Status

* [x] CAN IDS dataset preprocessing pipeline
* [x] Feature extraction
* [x] Temporal windowing
* [x] Neural-network training pipeline
* [x] Model conversion for embedded deployment
* [x] ESP32 CAN communication
* [x] Embedded feature extraction
* [x] ESP32 model inference
* [x] Intrusion alert mechanism
* [x] CAN attack-generation/testing functionality
* [x] CSV-based result logging

## Future Work

* Optimize model size and inference latency
* Benchmark GRU, MLP, and 1D CNN models on the ESP32
* Evaluate detection performance under different CAN attack scenarios
* Improve embedded memory and computational efficiency
* Perform hardware-in-the-loop validation
* Evaluate the IDS on additional automotive CAN datasets


