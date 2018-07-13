# Team 100141 *Novacula*

# Environment Requirements
- Python 3.6
- OpenCV2
- PyTorch 0.4.0
- TorchVision

# Run Test
```bash
sh test_submit1.sh
sh test_submit2.sh
sh test_submit3.sh
```

# Train a model 
```bash
sh train_se_resnext101_32x4d.sh
```

# remember to modify the paths in the script!!!

# Pretrain model and fine-tuned models:

[Baidu Netdisk](https://pan.baidu.com/s/1eopgneA7J7ESOEgnc2yF6A) password: ip8f

`se_resnext101_32x4d-3b2fe3d8.pth` is pretrain model on imagenet, it is placed in `./pretrained/` by default.

You can place it anywhere, just remember to modify the path in the script.

Other models in the link are our submitted models. They should be placed in `./checkpoints/se_resnext101_32x4d-Baseline`.

| Submit | Epoch | Acc on TestB |
| :-: | :-: | :-: |
| 1 | 80 | 0.99088 |
| 2 | 74 |  (I forgot this)|
| 3 | 73 | 0.9926 |
