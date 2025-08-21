import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.Wav2Vec2_AASIST import AASIST
from nets.LCNN import LCNN


class ADD(nn.Module):
    def __init__(
        self,
        classifier_label_size=2,
        aux_label_size=90,
        backbone="aasist",
        dropout=0.1,
    ):
        super(ADD, self).__init__()

        # Backbone
        if backbone == "lcnn":
            self.feature_extractor = LCNN(flat_dim=384)
            embedding_size = 128
        else:
            self.feature_extractor = AASIST()
            embedding_size = 160

        # Classifier Network (Main Task)
        self.classification_head = nn.Sequential(
            nn.Linear(embedding_size, 80),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(80, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32, classifier_label_size),
        )

        # Label Generation Network (Auxiliary Task)
        self.label_generation_head = nn.Sequential(
            nn.Linear(embedding_size, 80),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(80, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(32, aux_label_size),
        )
        self._initialize_weights(self.label_generation_head)

        self.softmax = nn.Softmax(dim=1)

    def _initialize_weights(self, module):
        for layer in module:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=0.25)
                layer.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)

    def reset_label_generation_head(self):
        self._initialize_weights(self.label_generation_head)

    def mask_softmax(self, x, mask, dim=1):
        logits = (
            torch.exp(x) * mask / torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True)
        )
        return logits

    def forward(self, x, mask=None):
        # ADDM
        x_embeddings = self.feature_extractor(x)
        output = self.classification_head(x_embeddings)
        output = self.softmax(output)

        # Stop gradient for label generation network path
        x_embeddings_detached = x_embeddings.detach()

        if mask is None:
            return output, None, None
        else:
            # AAGM
            label_gen_output_masked = self.label_generation_head(x_embeddings_detached)
            label_gen_output_ori = self.label_generation_head(x_embeddings_detached)

            label_gen_output_masked = self.mask_softmax(label_gen_output_masked, mask)
            label_gen_output_ori = self.softmax(label_gen_output_ori)
            return output, label_gen_output_masked, label_gen_output_ori
