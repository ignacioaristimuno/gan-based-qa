import torch
import torch.nn.functional as F


def hinge_loss(discriminator_output, real=True):
    """Function for returning the Hinge Loss"""

    if real:
        return F.relu(1 - discriminator_output).mean()
    else:
        return F.relu(1 + discriminator_output).mean()


def wasserstein_loss(real_scores, fake_scores):
    """Function for returning the Wasserstein Loss"""
    return torch.mean(fake_scores) - torch.mean(real_scores)
