import logging
import warnings

import numpy as np
import rich
from packaging import version
from scvi import REGISTRY_KEYS
from scvi.model import SCANVI
from shap import Explainer
from torch import Tensor
from tqdm.auto import tqdm

from .utils import get_labels_key, get_layer_key, train_test_group_split

torch = None


class SCANVIDeep(Explainer):
    """SCANVIDeep is extension of DeepExplainer

    Parameters
    ----------
    Explainer
        Main Explainer class from shap package
    """

    def __init__(
        self,
        model: SCANVI,
        train_size: float = 0.8,
        batch_size: int = 128,
    ):
        """Constructor setting up expected values.

        Currently categorical not continuous covariates are not supported.

        Parameters
        ----------
        model
            Trained :class:`~scvi.model.SCANVI` model
        train_size
            :obj:`float` Training size (background), by default 0.8
        batch_size
            :obj:`int` Number of cells used from each group, by default 128
            To ignore the batch_size subsetting, set batch_size=-1
        """
        import torch

        if version.parse(torch.__version__) < version.parse("0.4"):
            warnings.warn("Your PyTorch version is older than 0.4 and not supported.")

        if not isinstance(model, SCANVI):
            raise ValueError(
                f"Provided model is not scANVI! Provided instead: {type(model)}"
            )

        self.labels_key = get_labels_key(model)
        self.layer_key = get_layer_key(model)
        self.data, self.test = train_test_group_split(
            model.adata,
            self.labels_key,
            train_size,
            batch_size,
            self.layer_key,
        )

        self.adata = model.adata
        self.layer = None
        self.train_size = train_size
        self.batch_size = batch_size
        self.input_handle = None
        self.expected_value = None  # to keep the DeepExplainer base happy
        self.model = model.module.eval()

        with torch.no_grad():
            outputs = self.model.classify(
                self.data[REGISTRY_KEYS.X_KEY],
                batch_index=self.data[REGISTRY_KEYS.BATCH_KEY],
                cat_covs=None,
                cont_covs=None,
                use_posterior_mean=True,
            )
            self.device = model.device
            self.multi_output = True
            self.num_outputs = outputs.shape[1]
            self.expected_value = outputs.mean(0).cpu().numpy()
            # cleanup
            del outputs
            torch.cuda.empty_cache()

    def __repr__(self):
        text = f"{self.__class__.__name__} with the following parameters:\n"
        text += f"train_size={self.train_size}, test_size={(1.0 - self.train_size):.1f}, batch_size={self.batch_size}, "
        text += f"labels_key={self.labels_key}, layers_key={self.layer_key}\n"
        text += f"training_on={self.device}"
        rich.print(text)
        return ""

    def get_train_test(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return self.data, self.test

    def memory_stats(self):
        """Helper function to track CUDA memory usage."""
        import torch

        if self.device.type == "cuda":
            print(f"Allocated: {(torch.cuda.memory_allocated(0) / 1024**3):.2f} GB")
            print(f"Cached   : {(torch.cuda.memory_reserved(0) / 1024**3):.2f} GB")
            print()

    def add_target_handle(self, layer):
        input_handle = layer.register_forward_hook(get_target_input)
        self.target_handle = input_handle

    def add_handles(self, model, forward_handle, backward_handle):
        """Add handles to all non-container layers in the model.
        Recursively for non-container layers
        """
        handles_list = []
        model_children = list(model.children())
        if model_children:
            for child in model_children:
                handles_list.extend(
                    self.add_handles(child, forward_handle, backward_handle)
                )
        else:  # leaves
            handles_list.append(model.register_forward_hook(forward_handle))
            handles_list.append(model.register_full_backward_hook(backward_handle))
        return handles_list

    def remove_attributes(self, model):
        """Removes the x and y attributes which were added by the forward handles
        Recursively searches for non-container layers
        """
        for child in model.children():
            if "nn.modules.container" in str(type(child)):
                self.remove_attributes(child)
            else:
                try:
                    del child.x
                except AttributeError:
                    pass
                try:
                    del child.y
                except AttributeError:
                    pass

    def gradient(self, idx, input_x, input_batch):
        import torch

        logging.debug("Starting gradient")
        self.model.zero_grad()
        X = input_x.requires_grad_()

        outputs = self.model.classify(
            input_x,
            batch_index=input_batch,
            cat_covs=None,
            cont_covs=None,
            use_posterior_mean=True,
        )
        selected = [val for val in outputs[:, idx]]

        grad = torch.autograd.grad(selected, X, retain_graph=False, allow_unused=True)[
            0
        ]
        if grad is not None:
            return grad.cpu().numpy()

        return torch.zeros_like(X).cpu().numpy()

    def shap_values(self):
        import torch

        # X ~ self.model_input
        # X_data ~ self.data

        X_batch = self.test[REGISTRY_KEYS.BATCH_KEY].detach()
        X = self.test[REGISTRY_KEYS.X_KEY].detach()

        model_output_ranks = (
            torch.ones((X.shape[0], self.num_outputs)).int()
            * torch.arange(0, self.num_outputs).int()
        )

        # add the gradient handles
        handles = self.add_handles(self.model, add_interim_values, deeplift_grad)

        # compute the attributions
        output_phis = []
        self.memory_stats()

        for i in tqdm(range(model_output_ranks.shape[1])):
            phis = np.zeros(X.shape)

            for j in range(X.shape[0]):
                # tile the inputs to line up with the background data samples
                # correct, it will replicate each row to match the background
                tiled_X = X[j : j + 1].repeat(
                    (self.data[REGISTRY_KEYS.X_KEY].shape[0],)
                    + tuple([1 for _ in range(len(X.shape) - 1)])
                )

                joint_x = torch.cat((tiled_X, self.data[REGISTRY_KEYS.X_KEY]), dim=0)
                joint_batch = X_batch[j].repeat(joint_x.shape[0], 1)
                # run attribution computation graph
                feature_ind = model_output_ranks[j, i]
                sample_phis = self.gradient(feature_ind, joint_x, joint_batch)

                # assign the attributions to the right part of the output arrays
                phis[j] = (
                    torch.from_numpy(
                        sample_phis[self.data[REGISTRY_KEYS.X_KEY].shape[0] :]
                    )
                    * (X[j : j + 1] - self.data[REGISTRY_KEYS.X_KEY])
                ).mean(0)
                # .cpu().detach().numpy().mean(0)

            output_phis.append(phis)
            torch.cuda.empty_cache()

        # cleanup; remove all gradient handles
        for handle in handles:
            handle.remove()
        self.remove_attributes(self.model)
        torch.cuda.empty_cache()
        return output_phis


# Module hooks


def deeplift_grad(module, grad_input, grad_output):
    """The backward hook which computes the deeplift
    gradient for an nn.Module
    """
    # first, get the module type
    module_type = module.__class__.__name__
    # first, check the module is supported
    if module_type in op_handler:
        if op_handler[module_type].__name__ not in ["passthrough", "linear_1d"]:
            return op_handler[module_type](module, grad_input, grad_output)
    else:
        warnings.warn(f"unrecognized nn.Module: {module_type}")
        return grad_input


def add_interim_values(module, input, output):
    """The forward hook used to save interim tensors, detached
    from the graph. Used to calculate the multipliers
    """
    import torch

    try:
        del module.x
    except AttributeError:
        pass
    try:
        del module.y
    except AttributeError:
        pass
    module_type = module.__class__.__name__
    if module_type in op_handler:
        func_name = op_handler[module_type].__name__
        # First, check for cases where we don't need to save the x and y tensors
        if func_name == "passthrough":
            pass
        else:
            # check only the 0th input varies
            for i in range(len(input)):
                if i != 0 and type(output) is tuple:
                    assert input[i] == output[i], "Only the 0th input may vary!"
            # if a new method is added, it must be added here too. This ensures tensors
            # are only saved if necessary
            if func_name in ["maxpool", "nonlinear_1d"]:
                # only save tensors if necessary
                if type(input) is tuple:
                    module.x = torch.nn.Parameter(input[0].detach())
                else:
                    module.x = torch.nn.Parameter(input.detach())
                if type(output) is tuple:
                    module.y = torch.nn.Parameter(output[0].detach())
                else:
                    module.y = torch.nn.Parameter(output.detach())


def get_target_input(module, input, output):
    """A forward hook which saves the tensor - attached to its graph.
    Used if we want to explain the interim outputs of a model
    """
    try:
        del module.target_input
    except AttributeError:
        pass
    module.target_input = input


def passthrough(module, grad_input, grad_output):
    """No change made to gradients"""
    return None


def maxpool(module, grad_input, grad_output):
    import torch

    pool_to_unpool = {
        "MaxPool1d": torch.nn.functional.max_unpool1d,
        "MaxPool2d": torch.nn.functional.max_unpool2d,
        "MaxPool3d": torch.nn.functional.max_unpool3d,
    }
    pool_to_function = {
        "MaxPool1d": torch.nn.functional.max_pool1d,
        "MaxPool2d": torch.nn.functional.max_pool2d,
        "MaxPool3d": torch.nn.functional.max_pool3d,
    }
    delta_in = (
        module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2) :]
    )
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # we also need to check if the output is a tuple
    y, ref_output = torch.chunk(module.y, 2)
    cross_max = torch.max(y, ref_output)
    diffs = torch.cat([cross_max - ref_output, y - cross_max], 0)

    # all of this just to unpool the outputs
    with torch.no_grad():
        _, indices = pool_to_function[module.__class__.__name__](
            module.x,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.ceil_mode,
            True,
        )
        xmax_pos, rmax_pos = torch.chunk(
            pool_to_unpool[module.__class__.__name__](
                grad_output[0] * diffs,
                indices,
                module.kernel_size,
                module.stride,
                module.padding,
                list(module.x.shape),
            ),
            2,
        )

    grad_input = [None for _ in grad_input]
    grad_input[0] = torch.where(
        torch.abs(delta_in) < 1e-7,
        torch.zeros_like(delta_in),
        (xmax_pos + rmax_pos) / delta_in,
    ).repeat(dup0)

    return tuple(grad_input)


def linear_1d(module, grad_input, grad_output):
    """No change made to gradients."""
    return None


def nonlinear_1d(module, grad_input, grad_output):
    import torch

    delta_out = (
        module.y[: int(module.y.shape[0] / 2)] - module.y[int(module.y.shape[0] / 2) :]
    )

    delta_in = (
        module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2) :]
    )
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # handles numerical instabilities where delta_in is very small by
    # just taking the gradient in those cases
    grads = [None for _ in grad_input]
    grads[0] = torch.where(
        torch.abs(delta_in.repeat(dup0)) < 1e-6,
        grad_input[0],
        grad_output[0] * (delta_out / delta_in).repeat(dup0),
    )
    return tuple(grads)


op_handler = {}

# passthrough ops, where we make no change to the gradient
op_handler["Dropout3d"] = passthrough
op_handler["Dropout2d"] = passthrough
op_handler["Dropout"] = passthrough
op_handler["AlphaDropout"] = passthrough

op_handler["Conv1d"] = linear_1d
op_handler["Conv2d"] = linear_1d
op_handler["Conv3d"] = linear_1d
op_handler["ConvTranspose1d"] = linear_1d
op_handler["ConvTranspose2d"] = linear_1d
op_handler["ConvTranspose3d"] = linear_1d
op_handler["Linear"] = linear_1d
op_handler["AvgPool1d"] = linear_1d
op_handler["AvgPool2d"] = linear_1d
op_handler["AvgPool3d"] = linear_1d
op_handler["AdaptiveAvgPool1d"] = linear_1d
op_handler["AdaptiveAvgPool2d"] = linear_1d
op_handler["AdaptiveAvgPool3d"] = linear_1d
op_handler["BatchNorm1d"] = linear_1d
op_handler["BatchNorm2d"] = linear_1d
op_handler["BatchNorm3d"] = linear_1d

op_handler["LeakyReLU"] = nonlinear_1d
op_handler["ReLU"] = nonlinear_1d
op_handler["ELU"] = nonlinear_1d
op_handler["Sigmoid"] = nonlinear_1d
op_handler["Tanh"] = nonlinear_1d
op_handler["Softplus"] = nonlinear_1d
op_handler["Softmax"] = nonlinear_1d
op_handler["SELU"] = nonlinear_1d

op_handler["MaxPool1d"] = maxpool
op_handler["MaxPool2d"] = maxpool
op_handler["MaxPool3d"] = maxpool
