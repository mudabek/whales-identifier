import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import timm

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device,
        s: float,
        m: float,
        easy_margin: bool,
        ls_eps: float,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class HappyWhaleModel(nn.Module):
    def __init__(self, config, pretrained=True):
        super(HappyWhaleModel, self).__init__()
        embedding_size = config['embedding_size']
        model_name = config['model_name']
        self.device = torch.device(config['device'])

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.embedding = nn.Linear(self.model.get_classifier().in_features, embedding_size)
        self.model.reset_classifier(num_classes=0, global_pool="avg")

        self.arc = ArcMarginProduct(embedding_size, 
                                   config["num_classes"],
                                   device=self.device,
                                   s=config["s"], 
                                   m=config["m"], 
                                   easy_margin=config["ls_eps"], 
                                   ls_eps=config["ls_eps"])        


    def extract(self, images):
        features = self.model(images)
        embedding = self.embedding(features)
        return embedding

    
    def forward(self, images, labels):
        features = self.model(images)
        embedding = self.embedding(features)
        outputs = self.arc(embedding, labels)
        return outputs


class TorchModel(nn.Module):
    def __init__(self, config, pretrained=True):
        super(TorchModel, self).__init__()
        embedding_size = config['embedding_size']
        model_name = config['model_name']
        self.device = torch.device(config['device'])

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.embedding = nn.Linear(self.model.get_classifier().in_features, embedding_size)
        self.model.reset_classifier(num_classes=0, global_pool="avg")       

    # For interface purpose
    def extract(self, images):
        features = self.model(images)
        embedding = self.embedding(features)
        return embedding

    # Kept labels for interface purpose
    def forward(self, images, labels):
        features = self.model(images)
        embedding = self.embedding(features)
        return embedding