"""Unrolled Compressed Sensing (2D).

This file contains an implementation of the Unrolled Compressed Sensing
framework by CM Sandino, JY Cheng, et al. See paper below for more details.

It is also based heavily on the codebase below:

https://github.com/MRSRL/dl-cs

Implementation is based on:
    CM Sandino, JY Cheng, et al. "Compressed Sensing: From Research to
    Clinical Practice with Deep Neural Networks" IEEE Signal Processing
    Magazine, 2020.
"""
import torch
import torchvision.utils as tv_utils
from torch import nn

import ss_recon.utils.complex_utils as cplx
from ss_recon.utils.events import get_event_storage
from ss_recon.utils.general import move_to_device
from ss_recon.utils.transforms import SenseModel
from ss_recon.modeling.meta_arch.algorithms import ConjugateGradient

from ..layers.layers3D import RevNet
from .build import META_ARCH_REGISTRY

__all__ = ["DGLModl3DUnrolledCNN"]


@META_ARCH_REGISTRY.register()
class DGLModl3DUnrolledCNN(nn.Module):
    """
    PyTorch implementation of Unrolled Compressed Sensing.

    Implementation is based on:
        CM Sandino, JY Cheng, et al. "Compressed Sensing: From Research to
        Clinical Practice with Deep Neural Networks" IEEE Signal Processing
        Magazine, 2020.
    """

    def __init__(self, cfg):
        """
        Args:
            params (dict): Dictionary containing network parameters
        """
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        # Extract network parameters
        num_repeats = cfg.MODEL.NUM_REPEATS
        num_greedy_blocks = cfg.MODEL.NUM_GREEDY_BLOCKS
        num_grad_steps = cfg.MODEL.UNROLLED.NUM_UNROLLED_STEPS
        num_resblocks = cfg.MODEL.UNROLLED.NUM_RESBLOCKS
        num_convblocks = 2*num_resblocks + 1
        num_features = cfg.MODEL.UNROLLED.NUM_FEATURES
        kernel_size = cfg.MODEL.UNROLLED.KERNEL_SIZE
        num_reslayers = cfg.MODEL.MODL.NUM_RESLAYERS #should be 5 to match modl
        num_layers = num_reslayers*num_resblocks
        if len(kernel_size) == 1:
            kernel_size = kernel_size[0]
        drop_prob = cfg.MODEL.UNROLLED.DROPOUT
        circular_pad = cfg.MODEL.UNROLLED.PADDING == "circular"
        fix_step_size = cfg.MODEL.UNROLLED.FIX_STEP_SIZE
        share_weights = cfg.MODEL.UNROLLED.SHARE_WEIGHTS

        # Data dimensions and others
        self.num_emaps = cfg.MODEL.UNROLLED.NUM_EMAPS
        self.num_inf_steps = cfg.MODEL.UNROLLED.NUM_INF_STEPS

        # ResNet parameters
        resnet_params = dict(
            num_resblocks=num_resblocks,
            in_chans=2 * self.num_emaps,
            chans=num_features,
            kernel_size=kernel_size,
            drop_prob=drop_prob,
            circular_pad=circular_pad,
        )
        
        # Declare ResNets and RNNs for each unrolled iteration
        if share_weights:
            self.resnets = nn.ModuleList([RevNet(dims=3,num_channels=num_features,T=5,num_layers=num_layers,device=self.device)] * num_grad_steps)
        else:
            self.resnets = nn.ModuleList([RevNet(dims=3,num_channels=num_features,T=5,num_layers=num_layers,device=self.device) for _ in range(num_grad_steps)])

        # Declare step sizes for each iteration
        init_step_size = torch.tensor([-2.0], dtype=torch.float32)
        if fix_step_size:
            self.step_sizes = [init_step_size] * num_grad_steps
        else:
            self.step_sizes = nn.ParameterList(
                [torch.nn.Parameter(init_step_size) for _ in range(num_grad_steps)]
            )
    
        #Params for CG
        self.num_cg_iter = cfg.MODEL.MODL.CG
        lamb = cfg.MODEL.MODL.LAMBDA
        self.lamda = nn.Parameter(torch.tensor([lamb], dtype=torch.float32),
                                      requires_grad=(not fix_step_size))
        
        # Declare DGL-Unrolled Blocks
        self.dgl_blocks = DGLUnrolledRep(self.resnets, self.step_sizes, self.lamda)
        self.num_greedy_iters = int(num_grad_steps/num_greedy_blocks)
        self.num_repeats = num_repeats
        self.vis_period = cfg.VIS_PERIOD

    def visualize_training(self, kspace, zfs, targets, preds):
        """A function used to visualize reconstructions.

        Args:
            targets: NxHxWx2 tensors of target images.
            preds: NxHxWx2 tensors of predictions.
        """
        storage = get_event_storage()

        with torch.no_grad():
            if cplx.is_complex(kspace):
                kspace = torch.view_as_real(kspace)
            kspace = kspace[0, ..., 0, :].unsqueeze(0).cpu()  # calc mask for first coil only
            targets = targets[0, ...].unsqueeze(0).cpu()
            preds = preds[0, ...].unsqueeze(0).cpu()
            zfs = zfs[0, ...].unsqueeze(0).cpu()

            all_images = torch.cat([zfs, preds, targets], dim=2)

            imgs_to_write = {
                "phases": cplx.angle(all_images),
                "images": cplx.abs(all_images),
                "errors": cplx.abs(preds - targets),
                "masks": cplx.get_mask(kspace),
            }

            for name, data in imgs_to_write.items():
                data = data.squeeze(-1).unsqueeze(1)
                data = tv_utils.make_grid(
                    data,
                    nrow=1,
                    padding=1,
                    normalize=True,
                    scale_each=True,
                )
                storage.put_image("train/{}".format(name), data.numpy(), data_format="CHW")

    def forward(self, inputs, stage = 0, return_pp=False, vis_training=False, upto=False):
        """
            TODO: condense into list of dataset dicts.
            Args:
                inputs: Standard ss_recon module input dictionary
                    * "kspace": Kspace. If fully sampled, and want to simulate
                        undersampled kspace, provide "mask" argument.
                    * "maps": Sensitivity maps
                    * "target" (optional): Target image (typically fully sampled).
                    * "mask" (optional): Undersampling mask to apply.
                    * "signal_model" (optional): The signal model. If provided,
                        "maps" will not be used to estimate the signal model.
                        Use with caution.
                return_pp (bool, optional): If `True`, return post-processing
                    parameters "mean", "std", and "norm" if included in the input.
                vis_training (bool, optional): If `True`, force visualize training
                    on this pass. Can only be `True` if model is in training mode.

            Returns:
                Dict: A standard ss_recon output dict
                    * "pred": The reconstructed image
                    * "target" (optional): The target image.
                        Added if provided in the input.
                    * "mean"/"std"/"norm" (optional): Pre-processing parameters.
                        Added if provided in the input.
                    * "zf_image": The zero-filled image.
                        Added when model is in eval mode.
            """
        if self.training:
            upto = False
            num_greedy_iters = self.num_greedy_iters
        else:
            stage = 0
            upto = True
            num_greedy_iters = self.num_repeats
        
        if vis_training and not self.training:
            raise ValueError("vis_training is only applicable in training mode.")
        # Need to fetch device at runtime for proper data transfer.
        #device = self.resnets[stage].final_layer.weight.device
        device = self.device
        inputs = move_to_device(inputs, device)
        
        kspace = inputs["kspace"]
        target = inputs.get("target", None)
        mask = inputs.get("mask", None)
        A = inputs.get("signal_model", None)
        maps = inputs["maps"]
        num_maps_dim = -2 if cplx.is_complex_as_real(maps) else -1
        if self.num_emaps != maps.size()[num_maps_dim]:
            raise ValueError("Incorrect number of ESPIRiT maps! Re-prep data...")

        # Move step sizes to the right device.
        step_sizes = [x.to(device) for x in self.step_sizes]

        if mask is None:
            mask = cplx.get_mask(kspace)
        kspace *= mask

        # Get data dimensions
        dims = tuple(kspace.size())

        # Declare signal model.
        if A is None:
            A = SenseModel(maps, weights=mask)

        # Compute zero-filled image reconstruction
        zf_image = A(kspace, adjoint=True)
        
        # Begin unrolled HQS blah blah
        model_normal = lambda m: A(A(m), adjoint=True) + self.lamda * m
        cg_solve = ConjugateGradient(model_normal, self.num_cg_iter)
        
        # Begin unrolled proximal gradient descent
        if stage == 0:
            image = zf_image
        else:
            image = inputs["feats"]
        
        for dgl_stage in range(num_greedy_iters):
            cur_stage = stage * num_greedy_iters + dgl_stage 
            image = self.dgl_blocks(image, cur_stage, A, cg_solve, zf_image, dims, self.num_emaps, self.num_inf_steps, upto)
            
        output_dict = {
            "pred": image,  # N x Y x Z x T x 1 x 2
            "target": target,  # N x Y x Z x T x 1 x 2
        }

        if return_pp:
            output_dict.update({k: inputs[k] for k in ["mean", "std", "norm"]})

        if self.training and (vis_training or self.vis_period > 0):
            storage = get_event_storage()
            if vis_training or storage.iter % self.vis_period == 0:
                self.visualize_training(kspace[:,:,:,10,:], zf_image[:,:,:,10,:], target[:,:,:,10,:,], image[:,:,:,10,:])

        if not self.training:
            output_dict["zf_image"] = zf_image
            
        return output_dict
    
class DGLUnrolledRep(nn.Module):
    def __init__(self, resnets, step_sizes, lamda):
        super(DGLUnrolledRep, self).__init__()
        self.resnets = resnets
        self.step_sizes = step_sizes
        self.lamda = lamda
    def forward(self, image, stage, A, cg_solve, zf_image, dims, num_emaps,num_inf_steps,upto=False):
        
        # if upto = True we forward from the input to output of layer n
        # if upto = False we forward just through layer n
        '''
        if upto:
            for i in range(n+1):
                x = self.forward(x,i,upto=False)
            return x
        out = self.blocks[n](x)
        '''
        
        if upto:
            #len(self.resnets)
            for i in range(num_inf_steps):
                #to do: make this for multiprocessing only
                #device = self.resnets[i].final_layer.weight.device
                #device = self.device
                #image = image.to(device)
                #zf_image = zf_image.to(device)
                #A.maps = A.maps.to(device)
                #A.weights = A.weights.to(device)
                image = self.forward(image, i, A, cg_solve, zf_image, dims, num_emaps, num_inf_steps, upto=False)
            return image
        resnet = self.resnets[stage]
        step_size = self.step_sizes[stage]
        
        use_cplx = cplx.is_complex(image)
        if use_cplx:
            image = torch.view_as_real(image)

        # prox update            
        image = image.reshape(dims[0:4]+(num_emaps*2,)).permute(0,4,3,2,1) 
        out_image = resnet(image)

        image = image.permute(0,4,3,2,1).reshape(dims[0:4]+(num_emaps,2))
        if use_cplx:
            image = image.contiguous()
            image = torch.view_as_complex(image)

        out_image = out_image.permute(0,4,3,2,1).reshape(dims[0:4]+(num_emaps,2))
        if use_cplx:
            out_image = out_image.contiguous()
            out_image = torch.view_as_complex(out_image)

        # dc update
        image = cg_solve(image, zf_image + self.lamda * out_image)
        #grad_x = A(A(image), adjoint=True) - zf_image
        #image = image + step_size * grad_x
        #image = image - 2 * grad_x

        
        return image
        
