import torch
from torch.autograd import Function

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook): ### inputs(z_e_x) with (B, H:32, W:32, C:256), codebook (K:512, D:256)
        with torch.no_grad():
            embedding_size = codebook.size(1) ### 256
            inputs_size = inputs.size() ### (B, 32, 32, 256)
            inputs_flatten = inputs.view(-1, embedding_size) ### inputs_flatten (B*32*32, 256)

            codebook_sqr = torch.sum(codebook ** 2, dim=1) ### codebook_sqr (K:512)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True) ### inputs_sqr (B*32*32, 1)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0) ###distances and inputs_flatten @ codebook.t() are both (B*32*32, 512)

            _, indices_flatten = torch.min(distances, dim=1) ### indices_flatten with (B*32*32, 1)
            indices = indices_flatten.view(*inputs_size[:-1]) ### indices with (B, 32, 32)
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook): ### inputs(z_e_x) with (B, H:32, W:32, C:256), codebook (K:512, D:256)
        indices = vq(inputs, codebook) ### indices with (B, 32, 32)
        indices_flatten = indices.view(-1) ### indices_flatten with (B*32*32)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten) ### codes_flatten with (B*32*32, 256)
        codes = codes_flatten.view_as(inputs) ###codes with (B, H:32, W:32, C:256)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors ### indices with (B*32*32), codebook with (K:512, d:256)
            embedding_size = codebook.size(1) ### embedding_size 256

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size)) ### (B*32*32, 256)
            grad_codebook = torch.zeros_like(codebook) ### grad_codebook with (K:512, d:256)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]
