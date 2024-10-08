# ANNF Specification

## 1. Introduction

### 1.1. Name Reasoning

The **Accelerated Neural Networks Format** (**ANNF**) was designed to provide an efficient and compact way to store and transport neural network models specifically optimized for **accelerated inference** on diverse hardware platforms such as GPUs (CUDA, ROCm), OpenCL, and SIMD (Single Instruction, Multiple Data) CPU instructions.

The name emphasizes the primary focus of the format: **acceleration**, while being simple, intuitive, and easy to pronounce. The `.annf` extension reflects the format's direct relation to accelerated machine learning models.

---

## 2. Purpose of the ANNF File Format

The **ANNF** format is a cross-platform and cross-framework representation of a neural network model specifically tailored for high-performance inference. It is designed to be minimalistic, storing only the essential data necessary for constructing the computational graph and performing efficient inference.

The format targets scenarios where models need to run on a wide variety of hardware platforms, supporting both GPUs and CPUs and potentially different frameworks. It provides enough information to rebuild the network and execute the layers using optimal backend libraries like CUDA, ROCm, OpenCL, or SIMD-optimized routines.

Key features:

- **Model portability**: The file can be used across different platforms and hardware architectures.
- **Acceleration-ready**: Designed with performance in mind, focusing on high-speed inference.
- **Minimal storage footprint**: Only stores the necessary information for inference (network architecture, quantization settings, weights, and biases).

---

## 3 Structure of the ANNF File Format

The **ANNF** file is a binary format. It is structured in three main sections:

1. **File Header**: Contains identification data such as the magic number, versioning, and basic file metadata.
2. **Model Section**: Stores metadata global to the neural network like name, quantization, and number of parameters.
3. **Layers Section**: Contains layers metadata and data arranged in a sequential order.

### 3.1 File Header

The file header is fixed in size and serves to identify the file type and provide essential metadata for parsing the rest of the file.

#### 3.1.1 Layout

| Field Name     | Type       | Size  | Description                                           |
| -------------- | ---------- | ----- | ----------------------------------------------------- |
| Magic Number   | `uint32_t` | 4     | Unique identifier for ANNF (`0xANNF0001` suggested)   |
| Version Number | `uint32_t` | 4     | File format version (e.g., `1` for the first version) |

**Total Size: 8 bytes**

## 3.2 Model Section

The Model Section defines the global properties of the neural network, such as the number of layers, quantization settings, and overall structure. It is relatively compact but essential for parsing and constructing the full computational graph.

### 3.2.1 Layout

| Field Name               | Type       | Size  | Description                                                                |
| ------------------------ | ---------- | ----- | -------------------------------------------------------------------------- |
| Model Name Length        | `uint16_t` | 2     | Length of the model name string                                            |
| Model Name               | `char[]`   | Var.  | UTF-8 encoded model name                                                   |
| Number of Layers         | `uint32_t` | 4     | Total number of layers in the model                                        |
| Total Parameters         | `uint64_t` | 8     | Total number of parameters (weights, biases) in the model                  |
| Quantization Settings    | `uint8_t`  | 1     | Specifies whether quantization is used (0 = none, 1 = quantized)           |
| Quantization Scale       | `float32`  | 4     | If quantized, the global quantization scale (per-layer overrides allowed)  |
| Quantization Zero Point  | `int8_t`   | 1     | Global zero point for quantization (per-layer overrides allowed)           |

**Total Size:** Variable (depending on model name length).

- **Model Name**: Provides a user-friendly name for the model (useful in systems where multiple models are managed).
- **Quantization Settings**: Defines whether the model is stored using integer quantization (common in inference acceleration scenarios). Layer-specific quantization settings might override the global settings.

---

## 3.3 Layer Section

The **Layer Section** describes the structure of each layer in the neural network, including layer-specific configurations like the type of operation, input-output relationships, and activation functions.

Each layer is represented in the **ANNF** file using a unique identifier (UUID) and can connect to other layers via its input/output connections, supporting complex, non-linear architectures.

### 3.3.1 Layout

| Field Name          | Type         | Size  | Description                                                                       |
| ------------------- | ------------ | ----- | --------------------------------------------------------------------------------- |
| Layer UUID          | `uint32_t`   | 4     | Unique identifier for the layer                                                   |
| Layer Type Bitmask  | `uint16_t`   | 2     | Specifies the type of layer using a bitmask (see 3.3.3)                           |
| Input UUIDs         | `uint32_t[]` | Var.  | List of UUIDs representing input layers to this layer (supports multiple inputs)  |
| Output UUIDs        | `uint32_t[]` | Var.  | List of UUIDs representing output layers connected to this layer                  |
| Number of Blocks    | `uint16_t`   | 2     | Number of blocks for specific configurations inside this layer                    |
| Activation Bitmask  | `uint16_t`   | 2     | Activation function bitmask (see 3.3.4)                                           |
| Block Offset        | `uint32_t`   | 4     | Offset in bytes pointing to the first block defining layer properties             |
| Weights Offset      | `uint32_t`   | 4     | Offset to the weights of the layer (if applicable)                                |
| Biases Offset       | `uint32_t`   | 4     | Offset to the biases of the layer (if applicable)                                 |

- **Layer UUID**: Each layer has a unique identifier that allows for dynamic connections between layers, supporting non-linear architectures such as residuals or skip connections.
- **Blocks**: Each layer can contain a set of configuration "blocks" that specify layer details like neuron-specific settings (e.g., different activation functions for groups of neurons).
- **Offsets**: To ensure minimal memory footprint, data like weights, biases, and blocks are stored elsewhere in the file and referenced by their byte offsets.

### 3.3.2 Blocks

Blocks are the fundamental substructures of a layer, containing the specific configuration for neurons (or groups of neurons) within a layer. These blocks can vary in content depending on the layer type. For example, a convolutional layer might have blocks specifying different filter sizes or dilation rates, while a fully connected layer might specify different neuron groups using distinct activation functions.

#### 3.3.2.1 Block Layout

| Field Name         | Type         | Size  | Description                                                            |
| ------------------ | ------------ | ----- | ---------------------------------------------------------------------- |
| Block UUID         | `uint32_t`   | 4     | Unique identifier for the block                                        |
| Number of Neurons  | `uint16_t`   | 2     | Number of neurons governed by this block                               |
| Neuron Indices     | `uint32_t[]` | Var.  | List of neuron indices (identifying which neurons this block affects)  |
| Block Properties   | `Var`        | Var.  | Layer-specific properties such as filter sizes, strides, etc.          |

- **Block UUID**: Used to differentiate between multiple blocks within the same layer.
- **Neuron Indices**: A list of indices representing which neurons (in the layer) this block affects. This enables per-neuron customizations.
- **Block Properties**: These properties will vary based on the layer type (e.g., filter size for convolutional layers or specific activation function overrides).

### 3.3.3 Layer Type Bitmask

The layer type is stored using a `uint16_t` bitmask, allowing multiple types to be combined where necessary (e.g., a layer performing both convolution and batch normalization).

| Layer Type            | Bitmask Value |
| --------------------- | ------------- |
| Fully Connected       | `0x00000001`  |
| Convolutional         | `0x00000002`  |
| Batch Normalization   | `0x00000004`  |
| Pooling (Max/Avg)     | `0x00000008`  |
| Recurrent (LSTM/GRU)  | `0x00000010`  |
| Dropout               | `0x00000020`  |
| ...                   |               |

### 3.3.4 Activation Function Bitmask

The activation function applied to each layer (or groups of neurons within a layer) is stored as a `uint16_t` bitmask.

| Activation Function  | Bitmask Value |
| -------------------- | ------------- |
| ReLU                 | `0x00000001`  |
| Sigmoid              | `0x00000002`  |
| Tanh                 | `0x00000004`  |
| Softmax              | `0x00000008`  |
| None                 | `0x00000000`  |
| Leaky ReLU           | `0x00000010`  |
| ...                  |               |

The activation function can be mixed within the same layer by using blocks (described in section 3.3.2), allowing for more flexible model architectures.
