## Optimized Spatiotemporal Masked Autoencoders (ST-MAEs)

A lean, optimized implementation of spatiotemporal masked autoencoders (ST-MAEs). The skeleton of the code is recycled from Facebook's [ST-MAE](https://github.com/facebookresearch/mae_st) repository with various simplifications. The following optimizations are implemented:

- [x] FlashAttention-2
- [x] `torch.compile`
- [x] `fused` AdamW
- [x] mixed precision training (`torch.cuda.amp`)
- [ ] `FSDP` for distributed training

Dependence of model definitions on the `timm` library is also removed in this implementation, so the code is self-contained except for the standard libraries. The code was tested with `pytorch==2.2.0` and `torchvision==0.17.0`.

**Notes:**

- This project is work in progress at the moment.

