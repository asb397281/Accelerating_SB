# Quantization Usage Guide

This codebase now supports multiple quantization options to reduce memory usage and potentially speed up training and inference.

## Available Quantization Options

### 1. None (Default)
```bash
python main.py --quantization none
# or simply omit the flag (default behavior)
python main.py
```
- Standard full precision (float32) training
- No memory savings, highest precision

### 2. Float16 (Half Precision)
```bash
python main.py --quantization float16
```
- Uses 16-bit floating point precision
- ~50% memory reduction
- May have slight accuracy trade-offs
- Good balance of performance and precision

### 3. Float8 (Ultra Low Precision)
```bash
python main.py --quantization float8
```
- Uses bfloat16 as approximation (closest available to float8)
- Significant memory savings
- May have more noticeable accuracy trade-offs
- Experimental feature

### 4. Mixed Precision
```bash
python main.py --quantization mixed
```
- Uses PyTorch's Automatic Mixed Precision (AMP)
- Automatically uses float16 for suitable operations, float32 for others
- Good performance with minimal accuracy loss
- Recommended for production use

## Example Commands

### Training ResNet34 on CIFAR-100 with float16:
```bash
python main.py --model ResNet34 --dataset cifar100 --quantization float16 --epochs 200 --batch 128
```

### Training with mixed precision and layer dropback:
```bash
python main.py --model ResNet50 --dataset cifar10 --quantization mixed --epochs 200 --droprate 0.3
```

### Training without layer dropback and float8 quantization:
```bash
python main.py --model MobileNetV2 --dataset cifar100 --quantization float8 --no_ldb --epochs 150
```

## Memory Usage Comparison

| Quantization | Memory Usage | Speed | Accuracy |
|-------------|-------------|-------|----------|
| none        | 100%        | Base  | Highest  |
| mixed       | ~60-70%     | Fast  | High     |
| float16     | ~50%        | Fast  | Good     |
| float8      | ~40%        | Very Fast | Moderate |

## Notes

- The quantization type is automatically included in the experiment name for tracking
- Mixed precision is recommended for most use cases as it provides a good balance
- Float8 is experimental and uses bfloat16 as the closest available approximation
- All quantization options work with both standard training and layer dropback (LDB) method
- GPU support is recommended for optimal performance with quantization
