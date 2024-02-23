import torch
import torch.nn as nn

class BinImgEmbedder(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, binimg, force_drop_ids=None):
        # use_dropout = self.dropout_prob > 0
        binimg = self.conv_layers(binimg)
    
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            # drop_ids = torch.rand_like(labels.shape[0], device=labels.device) < self.dropout_prob
            drop_ids = torch.rand_like(labels, dtype=torch.float32, device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        # labels = torch.where(drop_ids, self.num_classes, labels)
        labels = torch.where(drop_ids, torch.full_like(labels, self.num_classes), labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        # labels: torch.Size([1, 1, 200, 200])
        # currently have a batch of single. So no dropout
        # use_dropout = False
        original_shape = labels.shape
        labels = labels.view(-1)  # Flatten the labels

        use_dropout = self.dropout_prob > 0
        
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        embeddings = self.embedding_table(labels)
        embeddings = embeddings.view(*original_shape, -1)  # Reshape back to original dimensions with embedding size


        return embeddings
