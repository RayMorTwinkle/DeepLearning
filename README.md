# MNIST MLP (PyTorch)

## 1. Setup (Conda, recommended for your Miniconda environment)

```bash
conda create -n ml_env python=3.11 -y
conda activate ml_env
conda install pytorch torchvision -c pytorch
```

Verify PyTorch and device support:

```bash
python -c "import torch; print(torch.__version__); print('mps built:', torch.backends.mps.is_built()); print('mps available:', torch.backends.mps.is_available())"
```

## 2. Run

From this directory:

```bash
python train_mnist_mlp.py
```

Optional args:

```bash
python train_mnist_mlp.py --epochs 10 --batch-size 64 --lr 0.001 --data-dir ./data --save-path ./best_mlp_mnist.pth
```

## 3. Device Auto-Detection

The script chooses:
1. `mps` on Apple Silicon (if available)
2. else `cuda` on NVIDIA GPU
3. else `cpu`

## 4. Expected Result

- Test accuracy: around `97% ~ 98%` on MNIST
- Best model saved to `./best_mlp_mnist.pth`

## 5. Visual Dashboard (Realtime Training UI)

Run:

```bash
conda run -n ml_env streamlit run /Users/ray/Documents/Fork/DeepLearning/visual_train_dashboard.py
```

If first-run prompt appears, use:

```bash
conda run -n ml_env streamlit run /Users/ray/Documents/Fork/DeepLearning/visual_train_dashboard.py --browser.gatherUsageStats false
```

Detailed guide:

- [VISUAL_DASHBOARD_GUIDE.md](/Users/ray/Documents/Fork/DeepLearning/VISUAL_DASHBOARD_GUIDE.md)
