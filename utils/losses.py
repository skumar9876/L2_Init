import torch

CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction='mean')



def MSELoss(logits, labels):
    num_classes = logits.shape[-1]
    one_hot = torch.eye(num_classes)[labels]
    
    diff = logits - one_hot
    return torch.mean(torch.sum(diff ** 2, dim=-1))

    