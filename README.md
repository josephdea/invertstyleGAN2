# invertstyleGAN2

Supports inversion of the styleGAN2 generator using ILO (intermediate layer optimization).
Official implementation found here: https://github.com/giannisdaras/ilo/tree/master.
This is just the original version that was submitted to ICML.

Given a corrupted image with a styleGAN2 prior, ILO can invert a corrupted image to recover the original image with high probability (also assumes certain knowledge of the corruption operator).
