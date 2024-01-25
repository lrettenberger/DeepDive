import lightning as L
from torch import optim

# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_pred = self.encoder(x)
        loss = self.loss(y,y_pred)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer