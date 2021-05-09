Examples for the "From Scratch: Neural Network inference on FPGAs" series
=========================================================================

### Weights

The weights in the `weights/` folder were generated running

```bash
$ python train.py --outdir weights --epochs 10
```

with the final output

```
Accuracy: 0.9370993589743589
Model predictions: [0 1 2 2 4 5 6 7 8 9]
```

Note that the last line outputs predictions on images with labels `0`, ..., to `9`.
