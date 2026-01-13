# This file aims to train the model
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from dataset_coco import COCOFruitDataset
from model_yolo import YOLOModel
from loss_yolo import YOLOLoss


def main():
    # 1. Basic config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    S = 7
    B = 2
    C = 3

    batch_size = 8
    lr = 1e-4
    num_epochs = 3   # 先跑小一点，确认能收敛

    # 2. Dataset & Dataloader
    train_images_dir = "/kaggle/input/coco-2017-dataset/coco2017/train2017"
    train_annotation_file = (
        "/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_train2017.json"
    )

    dataset = COCOFruitDataset(
        image_dir=train_images_dir,
        annotation_file=train_annotation_file,
        S=S,
        C=C,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Dataset size: {len(dataset)}")

    # 3. Model, Loss, Optimizer
    model = YOLOModel(S=S, B=B, C=C).to(device)
    criterion = YOLOLoss(S=S, B=B, C=C)
    optimizer = Adam(model.parameters(), lr=lr)

    # 4. Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)  # [N,S,S,B*5+C]
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # 5. Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            f"yolo_epoch_{epoch+1}.pth",
        )

    print("Training finished.")


if __name__ == "__main__":
    main()
