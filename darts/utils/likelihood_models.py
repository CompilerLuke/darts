"""
Likelihood Models
-----------------
"""
import math
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.distributions.negative_binomial import NegativeBinomial


class LikelihoodModel(ABC):

    def __init__(self):
        """
        Abstract class for a likelihood model. It contains all the logic to compute the loss
        and to sample the distribution, given the parameters of the distribution
        """
        pass

    @abstractmethod
    def _compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes a loss from a `model_output`, which represents the parameters of a given probability
        distribution for every ground truth value in `z`, and the `z` itself.
        """
        pass

    @abstractmethod
    def _sample(self, model_output: torch.Tensor) -> torch.Tensor:
        """
        Samples a prediction from the probability distributions defined by the specific likelihood model
        and the parameters given in `model_output`.
        """
        pass

    @property
    @abstractmethod
    def _num_parameters(self) -> int:
        """
        Returns the number of parameters that define the probability distribution for one single
        z value.
        """
        pass


class GaussianLikelihoodModel(LikelihoodModel):
    """
    Gaussian Likelihood
    """
    def __init__(self):
        self.loss = nn.GaussianNLLLoss(reduction='mean')
        self.softplus = nn.Softplus()
        super().__init__()

    def _compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        model_output_means, model_output_vars = self._means_and_vars_from_model_output(model_output)
        return self.loss(model_output_means.contiguous(), target.contiguous(), model_output_vars.contiguous())

    def _sample(self, model_output: torch.Tensor) -> torch.Tensor:
        model_output_means, model_output_vars = self._means_and_vars_from_model_output(model_output)
        sample = torch.normal(model_output_means, model_output_vars)
        return sample

    @property
    def _num_parameters(self) -> int:
        return 2

    def _means_and_vars_from_model_output(self, model_output):
        output_size = model_output.shape[-1]
        output_means = model_output[:, :, :output_size // 2]
        output_vars = self.softplus(model_output[:, :, output_size // 2:])
        return output_means, output_vars


class PoissonLikelihoodModel(LikelihoodModel):
    """
    Poisson Likelihood; can typically be used to model event counts in fixed intervals
    https://en.wikipedia.org/wiki/Poisson_distribution
    """

    def __init__(self):
        self.loss = nn.PoissonNLLLoss(log_input=False)
        self.softplus = nn.Softplus()
        super().__init__()

    def _compute_loss(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        model_output = self._lambda_from_output(model_output)
        return self.loss(model_output, target)

    def _sample(self, model_output: torch.Tensor) -> torch.Tensor:
        model_lambda = self._lambda_from_output(model_output)
        return torch.poisson(model_lambda)

    @property
    def _num_parameters(self) -> int:
        return 1

    def _lambda_from_output(self, model_output):
        return self.softplus(model_output)

N = 400

class NegativeBinomialLikelihoodModel(LikelihoodModel):
    """
    Negative Binomial Likelihood; can typically be used to model positive integers
    https://arxiv.org/pdf/1704.04110.pdf
    """

    def __init__(self):
        self.softplus = nn.Softplus()
        super().__init__()

    def _compute_loss(self, model_output: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        loss = self._distribution(model_output).log_prob(z)

        if False:
            mu, a = self._means_and_shape_from_model_output(model_output)
            inv_a = 1/a
            mu_a = mu * a

            loss = (
                    (z + inv_a).lgamma()
                    - (z + 1).lgamma()
                    - inv_a.lgamma()
                    - (inv_a + z) * (1 + mu_a).log()
                    + z * mu_a.log()
            )

        loss = torch.sum(loss)
        return -loss

    def _sample(self, model_output: torch.Tensor) -> torch.Tensor:
        means, shape = self._means_and_shape_from_model_output(model_output)
        return self._distribution(model_output).sample()

    @property
    def _num_parameters(self) -> int:
        return 2

    def _distribution(self, model_output):
        means, shape = self._means_and_shape_from_model_output(model_output)
        n = 1.0 / shape
        p = means / (means + n)

        return NegativeBinomial(total_count=n, probs=p, validate_args=False)

    def _means_and_shape_from_model_output(self, model_output):
        output_size = model_output.shape[-1]
        output_means = self.softplus(model_output[:, :, :output_size // 2]) * N
        output_shape = self.softplus(model_output[:, :, output_size // 2:]) / math.sqrt(N)
        return output_means, output_shape
