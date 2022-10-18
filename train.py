from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from TeXid import TeXidDataLoader, LitRobertaTeXid

if __name__ == "__main__":
    # wandb_logger = WandbLogger(project="proj_texid")
    wandb_logger = WandbLogger(project="proj_dummy")

    # model
    model_ck = "roberta-base"
    num_classes = 7
    layers_use_from_last = 4
    method_for_layers = 'sum'
    lit_roberta_texid = LitRobertaTeXid(
        num_classes, model_ck, layers_use_from_last, method_for_layers
    )

    # dataloader
    tokenizer_ck = "roberta-base"
    textid_dataloader = TeXidDataLoader(tokenizer_ck)
    [train_dataloader, valid_dataloader, test_dataloader] = textid_dataloader.get_dataloader(["train", "valid", "test"])

    # train model
    trainer = pl.Trainer(
        max_epochs=1, logger=wandb_logger, devices=2, accelerator="gpu", strategy="ddp",
        callbacks=[EarlyStopping(monitor="valid/acc_epoch", min_delta=0.00, patience=5, verbose=False, mode="max")]
    )
    trainer.fit(model=lit_roberta_texid, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(model=lit_roberta_texid, dataloaders=test_dataloader)
    # lit_roberta_texid.export_model('TeXid_model/model_v1')
    lit_roberta_texid.export_model('proj_testing')
